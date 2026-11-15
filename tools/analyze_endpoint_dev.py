"""Audit an exported endpoint Dev archive against existing SFT/500 evidence.

Creates descriptive stage counts and review cards, not automatic semantic verdicts.
No environment, inference, API, Final tasks or model weights are accessed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import runpy
import zipfile

from export_endpoint_dev import strict

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'analysis/grpo_baseline_20261022_202816'


def decode(raw):
    return json.loads(raw.decode('utf-8-sig'))


def json_rows(raw):
    return [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def analyze(archive, output):
    # Pure summary implementation only; this does not import environment code.
    summary_source = ROOT / 'upstream/shopping-grpo-longhorizon/src/shopping_grpo/evaluation/summary.py'
    summarizer = runpy.run_path(str(summary_source))
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        expected = {'manifest.json', 'dev50_tasks.jsonl'} | {f'step{step}/{name}' for step in (750, 800)
                    for name in ('summary.json', 'trajectories.jsonl', 'passed.json', 'exit_status.json')}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError('Unexpected or duplicate archive members')
        if sum(i.file_size for i in handle.infolist()) > 256 * 1024 * 1024:
            raise ValueError('Archive exceeds expected Dev evidence size')
        blobs = {name: handle.read(name) for name in names}
    manifest = decode(blobs['manifest.json'])
    if manifest['schema'] != 'endpoint-dev-evidence-v1' or set(manifest['files']) != expected - {'manifest.json'}:
        raise ValueError('Invalid manifest scope')
    for name, record in manifest['files'].items():
        if hashlib.sha256(blobs[name]).hexdigest() != record['sha256'] or len(blobs[name]) != record['bytes']:
            raise ValueError(f'Archive content hash differs: {name}')
    benchmark = json_rows(blobs['dev50_tasks.jsonl'])
    ids = {int(t['task_id']) for t in benchmark}
    if len(ids) != len(benchmark) or len(ids) != 50:
        raise ValueError('Expected 50 unique Dev tasks')
    models, summaries, integrity = {}, {}, {}
    baseline_manifest = decode((BASE / 'manifest.json').read_bytes())
    for label, folder in [('sft', 'sft_dev50'), ('step500', 'grpo_dev50')]:
        for filename in ('summary.json', 'trajectories.jsonl'):
            relative = f'{folder}/{filename}'
            raw = (BASE / relative).read_bytes()
            sha = baseline_manifest['files'][relative]['sha256']
            if sha not in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()):
                raise ValueError(f'Historical snapshot hash differs: {relative}')
            integrity[relative] = hashlib.sha256(raw).hexdigest()
        models[label] = json_rows((BASE / folder / 'trajectories.jsonl').read_bytes())
        summaries[label] = decode((BASE / folder / 'summary.json').read_bytes())
    for step in (750, 800):
        label = f'step{step}'
        models[label] = json_rows(blobs[f'{label}/trajectories.jsonl'])
        summaries[label] = decode(blobs[f'{label}/summary.json'])
        if decode(blobs[f'{label}/passed.json']).get('passed') is not True or decode(blobs[f'{label}/exit_status.json']).get('exit_code') != 0:
            raise ValueError(f'Endpoint receipts failed: {label}')
        if summaries[label]['protocol']['model'] != f'shopping-agent-grpo-step{step}':
            raise ValueError(f'Unexpected endpoint model: {label}')
    reference_protocol = {k: v for k, v in summaries['step500']['protocol'].items() if k != 'model'}
    for label, rows in models.items():
        if len(rows) != 50 or {int(t['task_id']) for t in rows} != ids:
            raise ValueError(f'Task set differs: {label}')
        recomputed = summarizer['summarize_trajectories'](sorted(ids), rows)
        if any(summaries[label].get(k) != v for k, v in recomputed.items()):
            raise ValueError(f'Summary cannot be reproduced: {label}')
        if any(strict(t) != summarizer['_is_strict_success'](t) for t in rows):
            raise ValueError('Strict-success definition differs')
        protocol = {k: v for k, v in summaries[label]['protocol'].items() if k != 'model'}
        if protocol != reference_protocol:
            raise ValueError(f'Inference protocol differs beyond model identity: {label}')
        models[label] = {int(t['task_id']): t for t in rows}
        if any(t['initial_result']['instruction'] != models['sft'][int(t['task_id'])]['initial_result']['instruction'] for t in rows):
            raise ValueError(f'Actor query differs: {label}')
    annotations = decode((BASE / 'derived/gold_annotation_audit.json').read_bytes())
    targets = {int(t['task_id']): str(t['asin']) for t in annotations if t['split'] == 'dev'}
    if set(targets) != ids:
        raise ValueError('Historical diagnostic target set differs')
    reviews, pairs, aggregate = [], [], {}
    for label, by_id in models.items():
        counts = Counter()
        for tid, t in sorted(by_id.items()):
            goal = targets[tid]
            steps = t.get('steps') or []
            visible = [i + 1 for i, s in enumerate(steps) if re.search(r'(?<!\d)' + re.escape(goal) + r'(?!\d)', s.get('observation') or '')]
            opened = [i + 1 for i, s in enumerate(steps) if s.get('tool_name') == 'open_product' and str((s.get('parameters') or {}).get('asin')) == goal]
            terminal = t.get('terminal_result') or {}
            purchase = terminal.get('purchase') or {}
            detail = terminal.get('reward_detail') or {}
            if strict(t):
                stage = 'success'
            elif str(purchase.get('asin')) == goal:
                stage = 'target_purchased_not_success'
            elif opened:
                stage = 'target_opened_purchased_other' if purchase.get('asin') else 'target_opened_no_purchase'
            else:
                stage = 'target_visible_not_opened' if visible else 'target_not_observed'
            counts[stage] += 1
            if stage == 'success':
                continue
            dimensions = detail.get('evidence', {}).get('preference_scoring', {}).get('dimensions', {})
            reviews.append({'model': label, 'task_id': tid, 'stage_proxy': stage,
                'query': t['initial_result']['instruction'], 'reward_type': detail.get('reward_type', 'unknown'),
                'error': t.get('error'), 'release_error': t.get('release_error'),
                'target_asin_offline_only': goal, 'target_visible_turns': visible, 'target_opened_turns': opened,
                'purchase': {k: purchase.get(k) for k in ('asin', 'options', 'price')},
                'hard_gates_original_reward': detail.get('hard_gates'),
                'nonfull_dimensions_original_reward': {k: v for k, v in dimensions.items() if v.get('active') and v.get('score', 0) < 1},
                'guard_rejections': t.get('blocked_tool_calls'),
                'actions_and_visible_evidence': [{'turn': i + 1, 'tool': s.get('tool_name'), 'parameters': s.get('parameters'),
                                                 'observation': s.get('observation')} for i, s in enumerate(steps)],
                'semantic_review_status': 'pending; reward dimensions and target stages are diagnostic proxies'})
        aggregate[label] = {'stages': dict(counts), 'strict_successes': summaries[label]['strict_successes'],
                            'reward_type_counts': summaries[label]['reward_type_counts'], 'average_steps': summaries[label]['average_steps']}
    for tid in sorted(ids):
        pairs.append({'task_id': tid, 'success': {label: strict(rows[tid]) for label, rows in models.items()}})
    transitions = {}
    for a, b in [('sft', 'step800'), ('step500', 'step750'), ('step500', 'step800'), ('step750', 'step800')]:
        transitions[f'{a}_to_{b}'] = {'new_success_ids': [r['task_id'] for r in pairs if not r['success'][a] and r['success'][b]],
                                    'regression_ids': [r['task_id'] for r in pairs if r['success'][a] and not r['success'][b]],
                                    'persistent_failure_ids': [r['task_id'] for r in pairs if not r['success'][a] and not r['success'][b]]}
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'audit.json', {'archive': str(archive), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'summary_source_sha256': hashlib.sha256(summary_source.read_bytes()).hexdigest(), 'baseline_files': integrity,
        'source_manifest': manifest, 'summaries_reproduced': True, 'protocols_and_queries_match': True,
        'limitations': ['Target visibility/opening is a diagnostic proxy, not a semantic or causal verdict.',
                       'Target not observed does not prove retrieval impossibility.', 'Manual semantic review remains pending.']})
    write(output / 'summary.json', aggregate)
    write(output / 'task_pairs.json', pairs)
    write(output / 'transitions.json', transitions)
    write(output / 'failure_review_cards.json', reviews)
    print(json.dumps({'aggregate': aggregate, 'transitions': transitions}, ensure_ascii=False, indent=2))
    print(f'SAVED: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis' / ('endpoint_dev_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    args = parser.parse_args()
    analyze(args.archive, args.output)
