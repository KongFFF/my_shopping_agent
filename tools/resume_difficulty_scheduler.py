"""Explicit, audited crash recovery. Does not alter the frozen training code."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

from difficulty_scheduler import TaskScheduler, canonical, read, write
from prepare_difficulty_scheduler import file_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def replay_checkpoint(raw, checkpoint):
    """Find the exact committed event prefix, preserving even torn tail bytes."""
    saved = checkpoint['scheduler']
    replay = TaskScheduler(saved['ids'], saved['arm'], saved['seed'], saved['budget'])
    runtime = {'consecutive_skips': 0, 'skipped_updates': 0}
    pending_ready = None
    groups, consumed = {}, set()
    offset = boundary = 0
    tail_completed = tail_response = tail_input = 0
    requests = 0
    torn = False
    chunks = raw.splitlines(keepends=True)
    target = canonical(TaskScheduler.from_state(saved).state_dict())
    for index, chunk in enumerate(chunks):
        try:
            event = json.loads(chunk)
        except (ValueError, UnicodeError):
            require(index == len(chunks) - 1 and boundary, 'Invalid event before recoverable boundary')
            torn = True
            break
        offset += len(chunk)
        kind = event['event']
        if boundary:
            if kind == 'generation':
                tail_completed += len(event['rollouts'])
                tail_response += event['response_tokens']
                tail_input += event['input_plus_response_tokens'] - event['response_tokens']
            if kind == 'selection':
                requests = max(requests, event['generated'])
            continue
        if kind == 'selection':
            expected = replay.reserve()
            require(all(event[k] == v for k, v in expected.items()), 'Selection/RNG mismatch')
            require(event['generated'] == replay.generated, 'Generation counter mismatch')
            requests = max(requests, replay.generated)
        elif kind == 'generation':
            replay.observe(event['rollouts'])
            for group in event['groups']:
                require(group['uid'] not in groups, 'Duplicate group UID')
                require(sum(u == group['uid'] for u in event['rollout_uids']) == 4, 'Invalid group identity')
                groups[group['uid']] = (group['kept'], replay.updates)
        elif kind == 'ready':
            require(pending_ready is None, 'Uncommitted ready batch')
            pending_ready = event['uids']
        elif kind == 'update':
            require(pending_ready == event['consumed_uids'] and len(pending_ready) == 2, 'Consumption mismatch')
            require(all(u not in consumed and groups.get(u) == (True, replay.updates)
                        for u in pending_ready), 'Invalid/reused/stale group')
            consumed.update(pending_ready)
            pending_ready = None
            replay.commit_update(event['update'])
            runtime['consecutive_skips'] = 0
        elif kind == 'skip':
            require(event['skipped_updates'] == runtime['skipped_updates'] + 1, 'Skip counter mismatch')
            runtime = {k: event[k] for k in runtime}
        else:
            raise ValueError('Error/unknown event before checkpoint: ' + kind)
        if (canonical(replay.state_dict()) == target and runtime == checkpoint['runtime']
                and pending_ready is None and not replay.pending):
            boundary = offset
    require(boundary > 0, 'No exact event/RNG/count prefix matches checkpoint')
    return {'prefix_bytes': boundary, 'tail_completed_rollouts': tail_completed,
            'tail_response_tokens': tail_response, 'tail_input_tokens': tail_input,
            'requested_from_events': requests, 'torn_final_event': torn}


def assert_idle():
    require(sys.platform == 'linux', 'Cloud recovery preparation requires Linux')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmd = (entry / 'cmdline').read_bytes().replace(b'\0', b' ')
        except (OSError, PermissionError):
            continue
        if (b'verl.trainer.main_ppo' in cmd or
            (b'run_difficulty_scheduler.py' in cmd and b' pipeline ' in cmd) or
            (b'resume_difficulty_scheduler.py' in cmd and b' pipeline ' in cmd)):
            raise ValueError('Training/pipeline process still exists: ' + entry.name)


def prepare(run, step, confirm=False):
    import run_difficulty_scheduler as driver
    run = Path(run).resolve()
    driver.checked(run)
    assert_idle()
    driver.assert_free(5723)
    driver.assert_free(8023)
    marker = run / 'crash_recovery.json'
    require(not marker.exists(), 'Recovery already prepared; do not prepare twice')
    folder = run / 'adaptive'
    require(not (folder / 'training_complete.json').exists(), 'Adaptive arm already has a completion receipt')
    checkpoint_dir = folder / f'global_step_{step}'
    checkpoint = read(checkpoint_dir / 'scheduler_state.json')
    current = read(folder / 'scheduler_state.json')
    require(checkpoint['binding'] == current['binding'], 'Checkpoint/current binding mismatch')
    require(checkpoint['binding'] == __import__('difficulty_scheduler').sha(canonical(read(folder / 'spec.json'))), 'Spec mismatch')
    require(checkpoint['scheduler']['arm'] == 'adaptive', 'Wrong arm')
    require(checkpoint['scheduler']['updates'] == step and not checkpoint['scheduler']['pending'], 'Incomplete checkpoint')
    require(current['scheduler']['generated'] >= checkpoint['scheduler']['generated'], 'Checkpoint is ahead of current state')
    TaskScheduler.from_state(current['scheduler'])
    pointer = folder / 'latest_checkpointed_iteration.txt'
    require(pointer.read_text().strip() == str(step), 'Trainer checkpoint pointer differs')
    for name in ('model', 'optim', 'extra_state'):
        path = checkpoint_dir / 'actor' / f'{name}_world_size_1_rank_0.pt'
        require(path.is_file() and path.stat().st_size > 0, 'Missing actor/optimizer/RNG payload: ' + str(path))
    require((checkpoint_dir / 'data.pt').is_file(), 'Missing dataloader checkpoint')
    require(read(run / 'uniform/training_complete.json')['complete'], 'Uniform arm is incomplete')
    require(read(run / 'dev50_uniform/summary.json')['complete'], 'Uniform Dev50 is incomplete')
    events = folder / 'scheduler_events.jsonl'
    raw = events.read_bytes()
    plan = replay_checkpoint(raw, checkpoint)
    requested = max(current['scheduler']['generated'], plan['requested_from_events'])
    plan.update({'checkpoint_step': step, 'checkpoint_generated': checkpoint['scheduler']['generated'],
                 'interrupted_updates': current['scheduler']['updates'],
                 'tail_requested_rollouts': requested - checkpoint['scheduler']['generated'],
                 'active_chain_rollout_target': checkpoint['scheduler']['budget'],
                 'strict_equal_actual_cost': False})
    require(plan['tail_requested_rollouts'] >= plan['tail_completed_rollouts'], 'Tail accounting mismatch')
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not confirm:
        print('INSPECT_ONLY: no files changed')
        return plan
    archive = run / 'recovery_history' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    archive.mkdir(parents=True, exist_ok=False)
    backups = {}
    sources = [events, folder / 'scheduler_state.json', pointer, checkpoint_dir / 'scheduler_state.json']
    sources += [p for p in run.iterdir() if p.is_file() and p.suffix in ('.json', '.log', '.lock')]
    for source in dict.fromkeys(sources):
        relative = source.relative_to(run)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(file_hash(source) == file_hash(target), 'Archive verification failed')
        backups[target.relative_to(run).as_posix()] = file_hash(target)
    plan.update({'version': 'scheduler-crash-recovery-v1', 'archive': archive.relative_to(run).as_posix(),
                 'backup_sha256': backups, 'recovery_tool_sha256': file_hash(__file__),
                 'preparation_complete': False,
                 'warning': 'Retained learning chain reaches 4000; interrupted tail is extra cost, not a strict equal-cost comparison.'})
    # Persist provenance before changing any active file. Originals are retained.
    write(marker, plan)
    write(folder / 'scheduler_state.json', checkpoint)
    temporary = events.with_suffix('.recovery.tmp')
    with temporary.open('wb') as handle:
        handle.write(raw[:plan['prefix_bytes']])
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(events)
    pipeline_lock = run / 'pipeline.lock'
    if pipeline_lock.exists():
        require(file_hash(pipeline_lock) == backups[(archive / 'pipeline.lock').relative_to(run).as_posix()], 'Lock changed during recovery')
        pipeline_lock.unlink()
    plan['preparation_complete'] = True
    write(marker, plan)
    print('RECOVERY_PREPARED: original tail archived; use this tool pipeline, not the old entry point')
    return plan


def cost_receipt(plan):
    target = plan['active_chain_rollout_target']
    return {'active_chain_rollouts': target,
            'requested_rollouts_including_interruption': target + plan['tail_requested_rollouts'],
            'completed_rollouts_with_saved_results': target + plan['tail_completed_rollouts'],
            'interrupted_unconfirmed_rollouts': plan['tail_requested_rollouts'] - plan['tail_completed_rollouts'],
            'extra_known_response_tokens': plan['tail_response_tokens'],
            'extra_known_input_tokens': plan['tail_input_tokens'],
            'strict_equal_actual_cost': False,
            'token_cost_is_lower_bound': True}


def pipeline(run):
    import run_difficulty_scheduler as driver
    import report_difficulty_scheduler as reporter
    run = Path(run).resolve()
    plan = read(run / 'crash_recovery.json')
    require(plan['preparation_complete'], 'Recovery preparation did not complete')
    require(plan['recovery_tool_sha256'] == file_hash(__file__), 'Recovery tool changed')
    for name, expected in plan['backup_sha256'].items():
        require(file_hash(run / name) == expected, 'Recovery archive changed: ' + name)
    driver.checked(run, models=True)
    with driver.lock(run / 'pipeline.lock'):
        (run / 'STOP').unlink(missing_ok=True)
        for arm in driver.SETTINGS['arms']:
            (run / arm / 'STOP').unlink(missing_ok=True)
        status = {'complete': False, 'exit_code': 1, 'recovered_after_crash': True}
        try:
            driver.evaluate(run, 'sft')
            for arm in driver.SETTINGS['arms']:
                driver.train(run, arm)
                driver.evaluate(run, arm)
            result = reporter.report(run)
            costs = cost_receipt(plan)
            costs['total_known_response_tokens'] = result['arms']['adaptive']['response_tokens'] + plan['tail_response_tokens']
            costs['total_known_input_tokens'] = result['arms']['adaptive']['input_tokens'] + plan['tail_input_tokens']
            costs['consumed_groups_per_1000_requested_including_interruption'] = (
                result['arms']['adaptive']['counts']['consumed_groups'] * 1000 /
                costs['requested_rollouts_including_interruption'])
            result['crash_recovery'] = {'checkpoint_step': plan['checkpoint_step'], 'archive': plan['archive'],
                                        'adaptive_cost': costs}
            result['limits'] += '; crash-recovered adaptive arm has additional interrupted work, not strict equal actual cost'
            write(run / 'report/summary.json', result)
            report_path = run / 'report/分析报告.md'
            text = report_path.read_text(encoding='utf-8').replace('实际训练rollout', '保留学习链rollout')
            warning = ('> 断电恢复实验：B从第100步恢复，原中断尾段完整归档。两组保留学习链各4000条，'
                       f'B含中断的请求额度共{cost_receipt(plan)["requested_rollouts_including_interruption"]}条；'
                       '未完成请求的实际计算量不确定。不得按严格等实际采样成本对照解释效率。\n\n')
            report_path.write_text(warning + text, encoding='utf-8', newline='\n')
            status.update(complete=True, exit_code=0, adaptive_cost=costs)
            print('RECOVERED_SCHEDULER_PILOT_COMPLETE', flush=True)
        except BaseException as exc:
            status['error'] = repr(exc)
            raise
        finally:
            write(run / 'pipeline_exit.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'prepare', 'pipeline'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-step', type=int, default=100)
    args = parser.parse_args()
    if args.command == 'pipeline':
        pipeline(args.run_dir)
    else:
        prepare(args.run_dir, args.checkpoint_step, confirm=args.command == 'prepare')
