"""CPU-only structural analysis; existing natural full episodes, never fork continuations.

Outputs are descriptive, not policy-gradient estimates or semantic ground truth.
"""
from collections import Counter, defaultdict
import hashlib
import gzip
import json
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/grpo_structural_analysis_20261116'


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def success(t):
    d = (t.get('terminal_result') or {}).get('reward_detail') or {}
    return bool(d.get('reward_valid') and d.get('purchase_success'))


def valid(t):
    d = (t.get('terminal_result') or {}).get('reward_detail') or {}
    return bool(d.get('reward_valid') and not d.get('sampling_invalid')
                and t['status'] == 'done')


def reward(t):
    return float((t.get('terminal_result') or {}).get('reward', 0))


def purchase(t):
    return (t.get('terminal_result') or {}).get('purchase') or {}


def action(step):
    return {'tool': step['tool_name'], 'parameters': step['parameters']}


def pre_states(t):
    """Only public pre-action observations. Never goal/purchase/reward fields."""
    obs = '__INITIAL_SEARCH_HOME__'
    history = []
    for index, s in enumerate(t['steps']):
        yield index, obs, history, s
        history = history + [{'observation': obs, 'action': action(s)}]
        obs = s['observation']


def peer_stage(t, gold_asins):
    seen, opened = set(), set()
    for s in t['steps']:
        # Public projected search rows and button footer; no hidden result.goal.
        text = s['observation']
        seen.update(a for a in gold_asins if a in text)
        if s['tool_name'] == 'open_product':
            opened.add(str(s['parameters'].get('asin')))
    p = purchase(t)
    if str(p.get('asin')) in gold_asins:
        return 'purchased_peer_success_product'
    if gold_asins & opened:
        return 'opened_peer_success_product_not_bought'
    if gold_asins & seen:
        return 'saw_peer_success_product_not_opened'
    return 'peer_success_product_not_seen'


def anchors(groups, key_mode):
    rows, examples = [], []
    for task, ts in groups.items():
        if not all(valid(t) for t in ts):
            continue
        mean = statistics.fmean(reward(t) for t in ts)
        states = defaultdict(list)
        for t in ts:
            for i, obs, hist, s in pre_states(t):
                if i == 0:
                    continue  # initial group simply reproduces episode baseline
                if key_mode == 'observation':
                    key = obs
                elif key_mode == 'observation_remaining':
                    key = canonical([obs, 35-i])
                elif key_mode == 'observation_tool_history':
                    key = canonical([obs, hist])
                else:
                    raise ValueError(key_mode)
                states[key].append({'path': t['path'], 'index': i,
                    'action': action(s), 'reward': reward(t), 'success': success(t),
                    'episode_adv': reward(t)-mean})
        for key, occurrences in states.items():
            # One first visit per independent episode. Later loop visits are not
            # counted as additional independent supporting trajectories.
            first = {}
            for o in occurrences:
                first.setdefault(o['path'], o)
            os = list(first.values())
            if len(os) < 2:
                continue
            step_mean = statistics.fmean(o['reward'] for o in os)
            reward_var = max(o['reward'] for o in os)-min(o['reward'] for o in os) > 1e-7
            action_var = len({canonical(o['action']) for o in os}) > 1
            changed = abs(step_mean-mean) > 1e-7
            gold_mix = any(o['success'] for o in os) and not all(o['success'] for o in os)
            by_action = defaultdict(list)
            for o in os:
                by_action[canonical(o['action'])].append(o['reward'])
            repeated_actions = [v for v in by_action.values() if len(v) >= 2]
            within_noise = any(max(v)-min(v) > 1e-7 for v in repeated_actions)
            row = {'task_id': ts[0]['task_id'], 'group_id': str(task), 'key_sha256': digest(key.encode()),
                'trajectories': len(os), 'occurrences': len(occurrences),
                'different_actions': action_var, 'different_rewards': reward_var,
                'mixed_success': gold_mix, 'baseline_changed': changed,
                'min_action_support': min(map(len, by_action.values())),
                'same_action_different_returns': within_noise,
                'episode_mean': mean, 'step_mean': step_mean, 'members': os}
            rows.append(row)
            if action_var and reward_var and changed:
                examples.append({**row, 'pre_observation': key if key_mode == 'observation' else None})
    interesting = [r for r in rows if r['different_actions'] and r['different_rewards']]
    novel = [r for r in interesting if r['baseline_changed']]
    return {'valid_task_groups': sum(all(valid(t) for t in ts) for ts in groups.values()),
        'shared_noninitial_keys': len(rows),
        'tasks_with_shared_noninitial_key': len({r['task_id'] for r in rows}),
        'action_and_reward_vary_keys': len(interesting),
        'action_and_reward_vary_tasks': len({r['task_id'] for r in interesting}),
        'vary_keys_with_same_action_return_noise': sum(r['same_action_different_returns'] for r in interesting),
        'vary_keys_all_actions_supported_twice': sum(r['min_action_support'] >= 2 for r in interesting),
        'baseline_changed_keys': len(novel),
        'baseline_changed_groups': len({r['group_id'] for r in novel}),
        'baseline_changed_tasks': sorted({r['task_id'] for r in novel}),
        'baseline_changed_gold_mixed_keys': sum(r['mixed_success'] for r in novel),
        'baseline_changed_decisions': len({(o['path'], o['index']) for r in novel for o in r['members']}),
        'total_valid_noninitial_decisions': sum(max(0, len(t['steps'])-1) for ts in groups.values()
            if all(valid(t) for t in ts) for t in ts)}, examples


def cohort(groups):
    count = Counter(sum(success(t) for t in ts) for ts in groups.values())
    mixed = {i: ts for i, ts in groups.items() if 0 < sum(success(t) for t in ts) < 4}
    stages = []
    for i, ts in mixed.items():
        gold_asins = {str(purchase(t)['asin']) for t in ts if success(t)}
        for t in ts:
            if success(t):
                continue
            p = purchase(t)
            stages.append({'task_id': i, 'path': t['path'], 'valid': valid(t),
                'stage': peer_stage(t, gold_asins), 'peer_success_asins': sorted(gold_asins),
                'purchase': {k: p.get(k) for k in ('asin', 'options', 'price', 'name')},
                'reward': reward(t), 'status': t['status']})
    def describe(ts):
        return {'trajectories': len(ts), 'tool_steps': sum(len(t['steps']) for t in ts),
            'median_tool_steps': statistics.median(len(t['steps']) for t in ts) if ts else None,
            'tools': dict(Counter(s['tool_name'] for t in ts for s in t['steps']))}
    totals = {'tasks': len(groups), 'trajectories': 4*len(groups),
        'success_histogram': dict(sorted(count.items())),
        'success_trajectories': sum(k*v for k, v in count.items()),
        'mixed_tasks': len(mixed), 'mixed_task_failure_traces': len(stages),
        'failure_stages': dict(Counter(r['stage'] for r in stages)),
        'success_paths': describe([t for ts in groups.values() for t in ts if success(t)]),
        'failure_paths': describe([t for ts in groups.values() for t in ts if not success(t)]),
        'anchors': {}}
    examples = {}
    for mode in ('observation', 'observation_remaining', 'observation_tool_history'):
        totals['anchors'][mode], examples[mode] = anchors(groups, mode)
    return totals, stages, examples


def online_windows(path):
    report = {}
    previous_success = set()
    with gzip.open(ROOT/'analysis/online_window_audit_20261116/restored_observations.jsonl.gz',
                   'rt', encoding='utf-8') as f:
        restored = {r['key']: r for r in map(json.loads, f)}
    certified = json.loads((ROOT/'analysis/online_window_audit_20261116/certified_index.json').read_text(encoding='utf-8'))
    certified_keys = {r['key'] for r in certified}
    replay_summary, replay_examples = {}, {}
    with zipfile.ZipFile(path) as z:
        for label in ('reference351_500', 'late651_800'):
            es = [json.loads(l) for l in z.read(label+'/training_diagnostics.jsonl').splitlines()]
            generated, consumed, meta, prior = {}, {}, {}, {}
            for e in es:
                if e['event'] == 'generation_batch':
                    batch = defaultdict(list)
                    for r in e['rollouts']:
                        batch[r['uid']].append(r)
                    for uid, rs in batch.items():
                        assert len(rs) == 4 and uid not in generated
                        generated[uid] = rs
                        prior[uid] = rs[0]['task_id'] in previous_success
                    meta.update({g['uid']: g for g in e['groups']})
                    # Add only after current batch was classified: no future leakage.
                    previous_success.update(r['task_id'] for r in e['rollouts']
                        if r.get('reward_valid') and r['reward']['purchase_success'])
                elif e['event'] == 'optimizer_step':
                    batch = defaultdict(list)
                    for r in e['rollouts']:
                        batch[r['uid']].append(r)
                    for uid, rs in batch.items():
                        assert len(rs) == 4 and uid not in consumed
                        fingerprint = lambda r: canonical({k: v for k, v in r.items() if k != 'rollout_index'})
                        assert Counter(map(fingerprint, rs)) == Counter(map(fingerprint, generated[uid]))
                        consumed[uid] = rs
            histogram = Counter(sum(bool(r['reward']['purchase_success']) for r in rs)
                                for rs in consumed.values())
            nogold = {u: rs for u, rs in consumed.items() if not any(r['reward']['purchase_success'] for r in rs)}
            positives = [r for rs in consumed.values() for r in rs
                if not r['reward']['purchase_success'] and r['reward']['terminal_utility'] >
                statistics.fmean(x['reward']['terminal_utility'] for x in rs)+1e-7]
            eligible_groups = {}
            for uid, rs in consumed.items():
                rebuilt = []
                for ordinal, r in enumerate(rs):
                    # Match the pre-existing replay's documented identity format.
                    key = digest(json.dumps({'task_id': r['task_id'], 'actions': r['actions']},
                        ensure_ascii=False, sort_keys=True).encode())
                    if key not in certified_keys:
                        break
                    rr = restored[key]
                    assert rr['reward_type'] == r['reward_type'] and rr['reward_valid'] == r['reward_valid']
                    assert abs(rr['reward']-r['reward']['terminal_utility']) < 1e-6
                    assert [s['action'] for s in rr['steps']] == r['actions']
                    rebuilt.append({'task_id': r['task_id'], 'path': uid+'/'+str(ordinal),
                        'status': 'done', 'steps': [{'tool_name': s['action']['tool'],
                            'parameters': s['action']['parameters'], 'observation': s['observation']}
                            for s in rr['steps']],
                        'terminal_result': {'reward': r['reward']['terminal_utility'],
                            'reward_detail': {'reward_valid': r['reward_valid'],
                                'purchase_success': bool(r['reward']['purchase_success']),
                                'sampling_invalid': r.get('sampling_invalid', False)}}})
                if len(rebuilt) == 4:
                    eligible_groups[uid] = rebuilt
            replay_summary[label], replay_examples[label] = {}, {}
            for mode in ('observation', 'observation_remaining', 'observation_tool_history'):
                replay_summary[label][mode], replay_examples[label][mode] = anchors(eligible_groups, mode)
            def is_no_success(rs):
                return not any(r['reward']['purchase_success'] for r in rs)
            discarded_constant = {u: rs for u, rs in generated.items()
                if meta[u].get('drop_reason') == 'constant_reward' and is_no_success(rs)}
            report[label] = {'generated_groups': len(generated), 'consumed_groups': len(consumed),
                'consumed_success_histogram': dict(sorted(histogram.items())),
                'non_success_positive_advantage': len(positives),
                'positive_failure_types': dict(Counter(r['reward_type'] for r in positives)),
                'no_success_groups': len(nogold),
                'no_success_groups_with_prior_success': sum(prior[u] for u in nogold),
                'prior_success_task_ids': sorted({rs[0]['task_id'] for u, rs in nogold.items() if prior[u]}),
                'constant_no_success_groups': len(discarded_constant),
                'constant_no_success_with_prior_success': sum(prior[u] for u in discarded_constant),
                'fully_certified_consumed_groups': len(eligible_groups),
                'drop_reasons': dict(Counter(g.get('drop_reason') or 'kept' for g in meta.values()))}
    (OUT/'online_anchor_summary.json').write_text(json.dumps(replay_summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (OUT/'online_anchor_examples.json').write_text(json.dumps(replay_examples, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    archive = ROOT / 'analysis/decision_probe_result_20261116/evidence.zip'
    online = ROOT / 'analysis/online_training_window_return_20261116/evidence.zip'
    assert digest(archive.read_bytes()) == '80d87118fc40480e61a5bef665df257e4c60aff59688e1ab406c8c86f9e832c7'
    assert digest(online.read_bytes()) == '01703af869f5fef97e509c0538b0e3c58910b3508b6298b6c2dccd5396416362'
    tasks_path = ROOT / 'experiments/decision_probe_v1/tasks.jsonl'
    tasks = {r['task_id']: r for r in map(json.loads, tasks_path.read_text(encoding='utf-8').splitlines())}
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('manifest.json'))
        for n, h in manifest['files'].items():
            assert digest(z.read('run/'+n)) == h
        ts = [json.loads(l) for l in z.read('run/trajectories.jsonl').splitlines()]
        cfg = json.loads(z.read('run/run.json'))
        protocol = json.loads(z.read('protocol.json'))
        assert cfg['protocol_sha256'] == digest(z.read('protocol.json'))
    ts = [t for t in ts if t['path'].split('/')[1] == 'A']
    assert len(ts) == len({t['path'] for t in ts}) == 192
    groups = defaultdict(list)
    for t in ts:
        assert not t['branch'] and not t['prefix_steps']
        groups[t['task_id']].append(t)
    assert set(groups) == set(tasks) and all(len(ts) == 4 for ts in groups.values())
    assert set(groups) == set(protocol['task_ids'])
    assert sum(success(t) for t in ts) == 119
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, stages, examples = {}, {}, {}
    for stratum in ('all48', 'uniform32', 'public_requirement16'):
        gg = {i: ts for i, ts in groups.items() if stratum == 'all48' or tasks[i]['stratum'] == stratum}
        summaries[stratum], stages[stratum], examples[stratum] = cohort(gg)
    task_rows = []
    for i, ts in groups.items():
        task_rows.append({'task_id': i, 'query': tasks[i]['query'], 'stratum': tasks[i]['stratum'],
            'success_count': sum(success(t) for t in ts), 'valid_count': sum(valid(t) for t in ts),
            'paths': [{'path': t['path'], 'success': success(t), 'valid': valid(t), 'reward': reward(t),
                'steps': len(t['steps']), 'status': t['status'],
                'reward_type': ((t.get('terminal_result') or {}).get('reward_detail') or {}).get('reward_type'),
                'purchase': purchase(t)} for t in ts]})
    outputs = {'summary.json': {'schema': 'grpo-structure-v1', 'sources': {
        'fixed800_archive_sha256': digest(archive.read_bytes()), 'online_archive_sha256': digest(online.read_bytes()),
        'tasks_sha256': digest(tasks_path.read_bytes()), 'settings': protocol['settings'],
        'model_name': cfg['model_name']}, 'cohorts': summaries, 'online': online_windows(online),
        'limitations': ['Only existing A whole episodes; no B continuations or legacy4192.',
            'Gold is frozen native purchase_success, not an independent semantic label.',
            'Strata separate; policy/harness differ from original online and greedy Dev.',
            'Shared observation is not full actor context; history key omits assistant reasoning.',
            'State comparisons are observational and returns include later actions.',
            'First visit per episode, noninitial, all4valid; gamma1 baseline-change diagnostic only.',
            'Historical success coverage excludes unexported0-350 and501-650.']},
        'mixed_failure_stages.json': stages, 'anchor_examples.json': examples, 'task_paths.json': task_rows}
    for n, obj in outputs.items():
        (OUT/n).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    # Human judgments are explicitly separate from native score-derived tables.
    reviews = [
        ('20628/A/1', 'public_error', '购买页125元，公开上限120元；同题A/2选套餐一成功。', [8]),
        ('20628/A/3', 'public_error', '购买页125元，公开上限120元。', [7]),
        ('18330/A/1', 'public_error', '打开商品最低1700元已超过1350元；规格未体现需求的120W+定时插座+50米。', [1, 5]),
        ('3727/A/0', 'public_error', '选择15ml×2盒，与约20ml、1盒的公开要求不一致；同题有18ml×32杯×1盒路径。', [6]),
        ('5535/A/3', 'public_incomplete_selection', '未选任何规格直接购买，未落实公开要求的2公斤和赠剂；菜单有对应组合。', [1]),
        ('2013/A/3', 'observed_recovery_opportunity', '已到同题成功商品并选69元碳化色，随后改104元大号、离开，最终35步耗尽；颜色映射有语义近似，勿称唯一正确路径。', [15, 16]),
        ('17311/A/2', 'public_candidate_mismatch', '选择349元网球裙，公开需求为浅绿水墨、荷叶边、约200斤；随后搜索循环。同题其他路径打开另一商品成功。', [1]),
        ('4060/A/0', 'public_budget_boundary', '粉色20元不满足严格低于20；随机颜色是原生Gold。颜色不限本身并不要求只能随机色，故不把颜色得分差当语义证据。', [6]),
        ('22462/A/0', 'semantic_preference_unproven', '红杆105元与公开红色笔杆、约110元相符；不能从原生Gold选择V10而V35非Gold推断V35是能力错误。', [8]),
        ('22462/A/3', 'semantic_preference_unproven', '同上：奖励差与公开语义优劣不可直接等同。', [10]),
        ('19827/A/0', 'environment_completion_convention', '已选8成新Z20、700元，少单值套餐官方标配；属于环境规格完成惯例，不当作核心语义推理缺陷。', [4]),
        ('19827/A/2', 'environment_completion_convention', '同上；不能将漏单值轴直接计成严重能力失败。', [4]),
    ]
    trace_by_path = {t['path']: t for members in groups.values() for t in members}
    review_packet = []
    for path, kind, verdict, indices in reviews:
        t = trace_by_path[path]
        review_packet.append({'path': path, 'task_id': t['task_id'], 'query': tasks[t['task_id']]['query'],
            'kind': kind, 'judgment': verdict, 'native_success': success(t),
            'public_observations': [{'after_tool_index': i, 'action': action(t['steps'][i]),
                'observation': t['steps'][i]['observation']} for i in indices]})
    (OUT/'public_case_review.json').write_text(json.dumps(review_packet, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    input_paths = [Path(__file__).resolve(), archive, online, tasks_path,
        ROOT/'analysis/online_window_audit_20261116/restored_observations.jsonl.gz',
        ROOT/'analysis/online_window_audit_20261116/certified_index.json']
    receipt = {'inputs_sha256': {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in input_paths},
        'outputs_sha256': {p.name: digest(p.read_bytes()) for p in sorted(OUT.glob('*.json'))
            if p.name != 'manifest.json'}}
    (OUT/'manifest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(outputs['summary.json'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
