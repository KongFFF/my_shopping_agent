"""Export existing independent Dev50 evidence; standard library, no inference/network.

Run on the cloud parent project: python tools/export_endpoint_dev.py
The archive includes only Dev50 trajectories/summaries/receipts and its task list.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

RUN = Path('outputs/runs/grpo_baseline_20261022_202816')


def load_bytes(data):
    return json.loads(data.decode('utf-8-sig'))


def query(row):
    return row['initial_result']['instruction']


def strict(row):
    terminal = row.get('terminal_result') or {}
    detail = terminal.get('reward_detail') or {}
    return (row.get('done') is True and row.get('status') == 'done'
            and terminal.get('done') is True and terminal.get('over') is True
            and detail.get('reward_version') == 'shopsimulator-reward-v3'
            and detail.get('reward_valid') is True
            and detail.get('reward_type') == 'gold_purchase'
            and detail.get('purchase_success') is True
            and detail.get('termination_reason') == 'gold_purchase')


def collect(parent):
    project = parent / 'upstream/shopping-grpo-longhorizon'
    run = project / RUN
    blobs, sources, models, summaries = {}, {}, {}, {}

    def add(path, name):
        raw = path.read_bytes()
        blobs[name] = raw
        sources[name] = {'source': str(path), 'bytes': len(raw),
                         'sha256': hashlib.sha256(raw).hexdigest()}
        return raw

    for step in (750, 800):
        label = f'step{step}'
        folder = run / f'evaluation/step{step}_independent_v1/dev50'
        data = {}
        for name in ('trajectories.jsonl', 'summary.json', 'passed.json', 'exit_status.json'):
            data[name] = add(folder / name, f'{label}/{name}')
        rows = [json.loads(line) for line in data['trajectories.jsonl'].decode('utf-8-sig').splitlines() if line.strip()]
        by_id = {int(row['task_id']): row for row in rows}
        summary = load_bytes(data['summary.json'])
        if len(rows) != 50 or len(by_id) != 50:
            raise ValueError(f'{label}: expected 50 unique trajectories')
        if summary['expected_tasks'] != 50 or summary['completed_tasks'] != 50 or summary['missing_tasks']:
            raise ValueError(f'{label}: incomplete summary')
        success_ids = sorted(tid for tid, row in by_id.items() if strict(row))
        if success_ids != sorted(summary['strict_success_task_ids']) or len(success_ids) != summary['strict_successes']:
            raise ValueError(f'{label}: strict-success IDs do not match summary')
        if load_bytes(data['passed.json']).get('passed') is not True or load_bytes(data['exit_status.json']).get('exit_code') != 0:
            raise ValueError(f'{label}: missing successful evaluation receipt')
        if summary['protocol']['model'] != f'shopping-agent-grpo-step{step}':
            raise ValueError(f'{label}: unexpected recorded model identity')
        models[label], summaries[label] = by_id, summary
    if models['step750'].keys() != models['step800'].keys():
        raise ValueError('Endpoint task IDs differ')
    if any(query(models['step750'][tid]) != query(models['step800'][tid]) for tid in models['step750']):
        raise ValueError('Endpoint queries differ')
    protocols = [{k: v for k, v in summary['protocol'].items() if k != 'model'} for summary in summaries.values()]
    if protocols[0] != protocols[1]:
        raise ValueError('Endpoint inference protocols differ beyond model name')
    benchmark = Path(protocols[0]['benchmark'])
    if not benchmark.is_absolute():
        benchmark = project / benchmark
    if not benchmark.resolve().is_relative_to(project.resolve()):
        raise ValueError('Dev benchmark is outside project')
    benchmark_raw = add(benchmark, 'dev50_tasks.jsonl')
    tasks = [json.loads(line) for line in benchmark_raw.decode('utf-8-sig').splitlines() if line.strip()]
    if len(tasks) != 50 or {int(t['task_id']) for t in tasks} != set(models['step750']):
        raise ValueError('Dev benchmark does not match trajectories')
    for record in sources.values():
        if hashlib.sha256(Path(record['source']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Source changed while exporting')
    manifest = {'schema': 'endpoint-dev-evidence-v1', 'files': sources,
                'counts': {label: {'tasks': 50, 'strict_successes': summary['strict_successes']} for label, summary in summaries.items()},
                'scope': 'Existing independent Dev50 only; no Final, weights, training, inference or API calls.',
                'limitations': 'Receipts and protocol identities checked; no live model/weight verification.'}
    return blobs, manifest


def export(parent):
    blobs, manifest = collect(parent)
    output = parent / 'outputs'
    output.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = output / f'endpoint_dev750_800_{stamp}.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as handle:
        for name, data in blobs.items():
            handle.writestr(name, data)
        handle.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest['counts'], ensure_ascii=False))
    print(f'ARCHIVE: {archive}')
    print(f'SHA256: {hashlib.sha256(archive.read_bytes()).hexdigest()}')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path.cwd())
    args = parser.parse_args()
    export(args.parent.resolve())
