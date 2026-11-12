"""Offline audit of the complete pilot: sampling efficiency and paired Dev50."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from difficulty_scheduler import TaskScheduler, canonical, classify, read, sha, write
from prepare_difficulty_scheduler import PACKAGE, SETTINGS, lines


def paired_difference(left, right):
    ids = sorted(left)
    if ids != sorted(right):
        raise ValueError('Evaluation task mismatch')
    differences = [int(right[t]) - int(left[t]) for t in ids]
    rng = random.Random(20261112)
    samples = sorted(sum(rng.choices(differences, k=len(ids))) / len(ids) for _ in range(10000))
    return {'difference': sum(differences) / len(ids), 'task_bootstrap_95': [samples[249], samples[9749]],
            'improved': differences.count(1), 'regressed': differences.count(-1),
            'unchanged': differences.count(0), 'interpretation': 'One paired training seed; exploratory Dev50 only'}


def audit_arm(run, arm):
    folder = Path(run) / arm
    state = read(folder / 'scheduler_state.json')['scheduler']
    end = read(folder / 'training_complete.json')
    if not end['complete'] or end['generated_rollouts'] != SETTINGS['budget']:
        raise ValueError('Arm has not completed the fixed budget: ' + arm)
    events = lines(folder / 'scheduler_events.jsonl')
    replay = TaskScheduler(state['ids'], arm, state['seed'], state['budget'])
    counter, reward_types, invalid_reasons = Counter(), Counter(), Counter()
    visits, selected = Counter(), Counter()
    labels = read(PACKAGE / 'diagnostic_labels.json')
    strata = {name: Counter() for name in ('static_no_gold', 'singleton_axis_screen')}
    response_tokens = input_tokens = 0
    generated_uids, consumed_uids = {}, set()
    pending_ready = None
    for event in events:
        if event['event'] == 'selection':
            expected = replay.reserve()
            if any(event[k] != expected[k] for k in expected):
                raise ValueError('Cannot reproduce selection probability/RNG')
            for task in event['tasks']:
                visits[task['task_id']] += 1
                counter['probability_cap_hits'] += task['at_cap']
        elif event['event'] == 'generation':
            replay.observe(event['rollouts'])
            response_tokens += event['response_tokens']
            input_tokens += event['input_plus_response_tokens'] - event['response_tokens']
            for info in event['rollouts']:
                gold, valid, reasons = classify(info)
                counter['gold_rollouts'] += gold
                counter['valid_rollouts'] += valid
                counter['rollouts'] += 1
                counter['steps'] += info.get('steps', 0)
                counter['guard_rejections'] += info.get('guard_rejections', 0)
                counter['context_limit'] += info.get('termination_reason') in (
                    'context_hard_limit_exceeded', 'context_budget_exhausted',
                    'context_compaction_unsupported_routed_experts')
                counter['max_steps'] += info.get('termination_reason') == 'max_steps'
                reward_types[info['reward_type']] += 1
                invalid_reasons.update(reasons)
                selected[int(info['task_id'])] += 1
                for name, ids in (('static_no_gold', labels['static_no_gold_ids']),
                                  ('singleton_axis_screen', labels['singleton_axis_screen_ids'])):
                    if int(info['task_id']) in ids:
                        strata[name]['rollouts'] += 1
                        strata[name]['gold'] += gold
                        strata[name]['valid'] += valid
            for group in event['groups']:
                if group['uid'] in generated_uids:
                    raise ValueError('Reused generated UID')
                generated_uids[group['uid']] = {'kept': group['kept'], 'update': replay.updates}
                counter['generated_groups'] += 1
                counter['varying_valid_groups'] += bool(group['kept'])
                counter['all_equal_groups'] += group['drop_reason'] == 'constant_reward'
                counter['invalid_groups'] += group['drop_reason'] == 'sampling_invalid'
                # Group identities bind actual public diagnostics, not purchase_success.
                infos = [x for uid, x in zip(event['rollout_uids'], event['rollouts'], strict=True)
                         if uid == group['uid']]
                if len(infos) != 4:
                    raise ValueError('Group identity mismatch')
                num_gold = sum(classify(x)[0] for x in infos)
                counter['gold_mixed_valid_groups'] += bool(group['kept']) and 0 < num_gold < 4
                counter['failure_only_varying_groups'] += bool(group['kept']) and num_gold == 0
        elif event['event'] == 'update':
            if pending_ready is None or event['consumed_uids'] != pending_ready:
                raise ValueError('Optimizer consumption does not match actual ready batch')
            if len(pending_ready) != 2 or any(u in consumed_uids or not generated_uids[u]['kept'] or
                    generated_uids[u]['update'] != replay.updates for u in pending_ready):
                raise ValueError('Invalid, reused or stale-policy group consumed')
            consumed_uids.update(pending_ready)
            pending_ready = None
            replay.commit_update(event['update'])
        elif event['event'] == 'ready':
            if pending_ready is not None:
                raise ValueError('Uncommitted ready batch')
            pending_ready = event['uids']
            counter['consumed_response_tokens'] += event['response_tokens']
            counter['consumed_input_plus_response_tokens'] += event['input_plus_response_tokens']
        elif event['event'] == 'skip':
            counter['skipped_updates'] += 1
        else:
            raise ValueError('Unexpected generation error or unknown audit event')
    if canonical(replay.state_dict()) != canonical(TaskScheduler.from_state(state).state_dict()):
        raise ValueError('Scheduler event replay and saved state differ')
    if replay.pending or pending_ready or replay.generated != 4000 or counter['rollouts'] != 4000:
        raise ValueError('Incomplete/extra generation')
    counter['consumed_groups'] = replay.consumed_groups
    counter['unconsumed_varying_groups'] = counter['varying_valid_groups'] - replay.consumed_groups
    return {'counts': dict(counter), 'reward_types': dict(reward_types), 'invalid_reasons': dict(invalid_reasons),
            'coverage': len(visits), 'visits_per_task': dict(visits), 'optimizer_updates': replay.updates,
            'diagnostic_strata': {k: dict(v) for k, v in strata.items()},
            'response_tokens': response_tokens, 'input_tokens': input_tokens,
            'consumed_groups_per_1000_rollouts': replay.consumed_groups / 4,
            'consumed_groups_per_10000_response_tokens': replay.consumed_groups / max(response_tokens, 1) * 10000,
            'residual_groups_discarded': end['residual_groups_discarded'], 'event_replay_verified': True}


def report(run):
    run = Path(run)
    from run_difficulty_scheduler import checked
    cfg = checked(run)
    import sys
    sys.path.insert(0, str(Path(cfg['bundle']) / 'project/src'))
    from checkpoint_pair_common import outcome
    arms = {arm: audit_arm(run, arm) for arm in SETTINGS['arms']}
    evaluation, scores = {}, {}
    dev_ids = {r['task_id'] for r in lines(PACKAGE / 'dev50_tasks.jsonl')}
    for label in ('sft', 'uniform', 'adaptive'):
        folder = run / ('dev50_' + label)
        summary = read(folder / 'summary.json')
        if not summary['complete'] or summary['tasks'] != 50:
            raise ValueError('Incomplete fixed Dev50 evaluation')
        rows = [read(folder / (str(t) + '.json')) for t in sorted(dev_ids)]
        if summary['protocol'] != cfg['protocol'] or any(outcome(r['trajectory']) != r['outcome'] for r in rows):
            raise ValueError('Evaluation protocol/outcome recomputation differs')
        scores[label] = {r['task_id']: r['outcome']['strict_success'] for r in rows}
        if set(scores[label]) != dev_ids:
            raise ValueError('Dev50 task IDs differ')
        evaluation[label] = {'successes': sum(scores[label].values()), 'tasks': 50,
                              'reward_types': dict(Counter(r['outcome']['reward_type'] for r in rows)),
                              'guard_rejections': sum(r['outcome']['guard_rejections'] for r in rows),
                              'mean_steps': sum(r['outcome']['tool_steps'] for r in rows) / 50}
        if summary['successes'] != evaluation[label]['successes']:
            raise ValueError('Evaluation summary success count differs')
    result = {'complete': True, 'protocol': cfg['protocol'], 'arms': arms, 'dev50': evaluation,
              'adaptive_minus_uniform': paired_difference(scores['uniform'], scores['adaptive']),
              'adaptive_minus_sft': paired_difference(scores['sft'], scores['adaptive']),
              'limits': 'Single training seed, reused development set, original gold reward; no held-out or semantic improvement claim'}
    write(run / 'report/summary.json', result)
    a, b = arms['uniform'], arms['adaptive']
    text = ('# 难度调度小规模对照\n\n仅为单训练seed、已使用Dev50的探索结果；不作为正式泛化或语义能力结论。\n\n'
            '| 指标 | 均匀 | 调度 |\n|---|---:|---:|\n'
            f"| 实际训练rollout | 4000 | 4000 |\n| 实际更新 | {a['optimizer_updates']} | {b['optimizer_updates']} |\n"
            f"| 有效非同分组 | {a['counts'].get('varying_valid_groups',0)} | {b['counts'].get('varying_valid_groups',0)} |\n"
            f"| 消费组／千rollout | {a['consumed_groups_per_1000_rollouts']:.2f} | {b['consumed_groups_per_1000_rollouts']:.2f} |\n"
            f"| 生成token | {a['response_tokens']} | {b['response_tokens']} |\n"
            f"| 任务覆盖 | {a['coverage']} | {b['coverage']} |\n"
            f"| Dev50成功数 | {evaluation['uniform']['successes']} | {evaluation['adaptive']['successes']} |\n\n"
            f"SFT起点：{evaluation['sft']['successes']}/50。抽样RNG、全部概率与计数已按事件日志精确复放。\n\n"
            '先检查有效消费与token成本，再结合开发集和循环／上下文等退化判断；不自动放行正式训练。详见summary.json。\n')
    (run / 'report/分析报告.md').write_text(text, encoding='utf-8', newline='\n')
    print('PASS: exact scheduler event replay and paired pilot report')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    report(p.parse_args().run_dir)
