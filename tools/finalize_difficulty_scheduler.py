"""CPU-only report repair; never starts training, export, or model servers."""
import argparse
import copy
from pathlib import Path
import shutil
import time

from difficulty_scheduler import read, write
from prepare_difficulty_scheduler import file_hash
import report_difficulty_scheduler as reporter
import resume_difficulty_scheduler as recovery
import run_difficulty_scheduler as driver

MISSING = '__missing_reward_type__'


def repair_reward_counters(result):
    """Normalize only nullable reward-type keys, preserving IDs and counts."""
    result = copy.deepcopy(result)
    missing = {}
    for section in ('arms', 'dev50'):
        for arm, values in result[section].items():
            counts = values['reward_types']
            if None in counts:
                recovery.require(MISSING not in counts, 'Missing-reward label collision')
                missing[section + ':' + arm] = counts[None]
                counts[MISSING] = counts.pop(None)
            recovery.require(all(isinstance(k, str) for k in counts), 'Unexpected reward-type key')
    result['serialization_repair'] = {
        'missing_reward_type_label': MISSING, 'missing_reward_type_counts': missing,
        'meaning': 'Missing reward type, retained in original failure/unusable counts; no rescoring or removal.'}
    return result


def report_with_nullable_rewards(run):
    original_write = reporter.write
    def safe_write(path, value):
        if Path(path) == Path(run) / 'report/summary.json':
            fixed = repair_reward_counters(value)
            value.clear()
            value.update(fixed)
        return original_write(path, value)
    # Keep the byte-frozen reporter untouched; adapt only its output serializer.
    reporter.write = safe_write
    try:
        return reporter.report(run)
    finally:
        reporter.write = original_write


def finalize(run):
    run = Path(run).resolve()
    recovery.assert_idle()
    driver.assert_free(5723)
    driver.assert_free(8023)
    recovery.require(not (run / 'pipeline.lock').exists(), 'Pipeline lock still exists; inspect it first')
    driver.checked(run)  # CPU only; no CUDA or remote model payloads.
    plan = read(run / 'crash_recovery.json')
    recovery.require(plan['preparation_complete'], 'Crash recovery preparation is incomplete')
    recovery.require(plan['recovery_tool_sha256'] == file_hash(recovery.__file__), 'Recovery tool changed')
    for name, expected in plan['backup_sha256'].items():
        recovery.require(file_hash(run / name) == expected, 'Recovery archive changed: ' + name)
    for arm in ('uniform', 'adaptive'):
        end = read(run / arm / 'training_complete.json')
        recovery.require(end['complete'] and end['generated_rollouts'] == 4000, 'Training incomplete: ' + arm)
    for label in ('sft', 'uniform', 'adaptive'):
        summary = read(run / ('dev50_' + label) / 'summary.json')
        recovery.require(summary['complete'] and summary['tasks'] == 50, 'Dev50 incomplete: ' + label)
    marker = run / 'report_repair.json'
    if marker.exists():
        receipt = read(marker)
        recovery.require(receipt['report_sha256'] == file_hash(run / 'report/summary.json'), 'Repaired report changed')
        status = read(run / 'pipeline_exit.json')
        recovery.require(status['complete'] and status['exit_code'] == 0, 'Repaired status differs')
        print('PASS: existing repaired report verified')
        return
    archive = run / 'report_repair_history' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    archive.mkdir(parents=True, exist_ok=False)
    sources = [run / 'pipeline_exit.json'] + list((run / 'report').glob('*'))
    backup_hashes = {}
    for path in sources:
        if path.is_file():
            target = archive / path.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            recovery.require(file_hash(target) == file_hash(path), 'Report backup failed')
            backup_hashes[target.relative_to(run).as_posix()] = file_hash(target)
    # The original full event replay, UID checks and 150 trajectory rescoring run
    # unchanged. If any fails, the old failure receipt remains in place.
    result = report_with_nullable_rewards(run)
    costs = recovery.cost_receipt(plan)
    costs['total_known_response_tokens'] = result['arms']['adaptive']['response_tokens'] + plan['tail_response_tokens']
    costs['total_known_input_tokens'] = result['arms']['adaptive']['input_tokens'] + plan['tail_input_tokens']
    costs['consumed_groups_per_1000_requested_including_interruption'] = (
        result['arms']['adaptive']['counts']['consumed_groups'] * 1000 /
        costs['requested_rollouts_including_interruption'])
    result['crash_recovery'] = {'checkpoint_step': plan['checkpoint_step'], 'archive': plan['archive'],
                                'adaptive_cost': costs}
    result['limits'] += '; crash-recovered adaptive arm has extra interrupted work, not strict equal actual cost'
    write(run / 'report/summary.json', result)
    md = run / 'report/分析报告.md'
    text = md.read_text(encoding='utf-8').replace('实际训练rollout', '保留学习链rollout')
    warning = ('> 断电恢复与离线汇总修复：空奖励类型单列为__missing_reward_type__，未删样本或改分。'
               f'B累计请求额度{costs["requested_rollouts_including_interruption"]}条；'
               '两组保留学习链各4000条，不是严格等实际成本对照。\n\n')
    md.write_text(warning + text, encoding='utf-8', newline='\n')
    receipt = {'complete': True, 'backup_sha256': backup_hashes,
               'tool_sha256': file_hash(__file__), 'report_sha256': file_hash(run / 'report/summary.json'),
               'frozen_reporter_sha256': file_hash(reporter.__file__), 'gpu_used': False,
               'training_or_evaluation_rerun': False}
    status = {'complete': True, 'exit_code': 0, 'recovered_after_crash': True,
              'postprocessing_repaired': True, 'adaptive_cost': costs,
              'prior_failure_archive': archive.relative_to(run).as_posix()}
    write(run / 'pipeline_exit.json', status)
    write(marker, receipt)
    print('PASS: CPU_REPORT_REPAIR_COMPLETE; all training/event/evaluation audits passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    finalize(parser.parse_args().run_dir)
