"""Step 1: conservative public-state screening of existing training-task rollouts.

No inference/environment calls. Flags are local comparisons, not whole-task labels.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/checkpoint_pair_v1/run_01'
OUT = ROOT / 'analysis/training_decision_audit_20261115'
CONTRACTS = ROOT / 'analysis/semantic_pilot30_evidence_20261109/contracts1000.jsonl'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def norm(text):
    return unicodedata.normalize('NFKC', text).casefold().strip()


def public_state(observation):
    """Parse only rendered Actor-visible lines, never private result state."""
    state = {}
    for line in (observation or '').splitlines():
        if ': ' not in line:
            continue
        key, value = line.split(': ', 1)
        if key in ('asin', 'title', 'price', 'page_type'):
            state[key] = value
        elif key in ('selected_options', 'available_options'):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    state[key] = parsed
            except json.JSONDecodeError:
                pass
    price = state.get('price', '')
    state['numeric_price'] = float(price) if re.fullmatch(r'\d+(?:\.\d+)?', price) else None
    return state


def requirements(contract):
    q = contract['query']
    result = []
    normalized = contract['normalized']
    for field in ('budget', 'specifications'):
        for r in normalized.get(field, []):
            evidence = r.get('evidence') or []
            if r.get('review_status') != 'approved' or r.get('importance') != 'required' or not evidence:
                continue
            if any(q[e['start']:e['end']] != e['text'] for e in evidence):
                raise ValueError('Requirement evidence offsets differ')
            expected = r.get('expected') or {}
            base = {'id': r['id'], 'evidence': evidence, 'operator': r['operator'], 'expected': expected}
            if field == 'budget' and expected.get('type') == 'number' and expected.get('unit') == 'CNY' and r['operator'] in ('at_most', 'less_than'):
                # Avoid approximate policies and deposit/discount/price-basis ambiguity.
                if re.search(r'左右|大概|约|尽量|最好|到手|券|优惠|定金|订金|预定|预订|尾款|全款|分期', q):
                    continue
                if not any(re.search(r'以内|以下|不超过|不得超|别超|不高于|上限|低于|小于|最多|至多|不多于|不能超', e['text']) for e in evidence):
                    continue
                result.append({**base, 'kind': 'explicit_budget'})
            elif field == 'specifications' and r.get('attribute') == 'mass' and expected.get('unit') in ('g', 'kg', 'jin') and r['operator'] == 'eq' and r.get('subject', {}).get('kind') == 'primary_product':
                result.append({**base, 'kind': 'explicit_mass'})
            elif field == 'specifications' and r.get('attribute') == 'color' and expected.get('value') in ('黑色', '白色', '红色', '绿色', '蓝色', '黄色', '紫色', '灰色') and r['operator'] == 'eq' and r.get('subject', {}).get('kind') == 'primary_product':
                result.append({**base, 'kind': 'explicit_color'})
    return result


def compare(state, req):
    expected = req['expected']
    selected = state.get('selected_options')
    if req['kind'] == 'explicit_budget':
        price = state.get('numeric_price')
        if price is None:
            return None
        limit = float(expected['value'])
        valid = price <= limit if req['operator'] == 'at_most' else price < limit
        return {'passed': valid, 'actual': price, 'expected': limit}
    if not isinstance(selected, dict) or not selected:
        return None
    if req['kind'] == 'explicit_mass':
        masses = []
        for value in selected.values():
            value = norm(str(value))
            if re.search(r'[*×]|\d\s*[xX]\s*\d|每|赠|送', value):
                return None
            for number, unit in re.findall(r'(\d+(?:\.\d+)?)\s*(kg|公斤|千克|斤|克|g)(?![a-z])', value):
                masses.append(float(number) * {'kg': 1000, '公斤': 1000, '千克': 1000, '斤': 500, '克': 1, 'g': 1}[unit])
        if len(masses) != 1:
            return None
        target = float(expected['value']) * {'kg': 1000, 'g': 1, 'jin': 500}[expected['unit']]
        return {'passed': abs(masses[0] - target) < 1e-6, 'actual_g': masses[0], 'expected_g': target}
    colors = set()
    for axis, value in selected.items():
        if '色' in axis:
            colors.update(re.findall(r'黑色|白色|红色|绿色|蓝色|黄色|紫色|灰色', str(value)))
    if len(colors) != 1:
        return None
    actual = next(iter(colors))
    return {'passed': actual == expected['value'], 'actual': actual, 'expected': expected['value']}


def main():
    tasks = rows(ROOT / 'experiments/checkpoint_pair_v1/tasks.jsonl')
    train_ids = {int(t['task_id']) for t in rows(ROOT / 'upstream/shopping-grpo-longhorizon/data/grpo/train.jsonl')}
    task_by_id = {t['task_id']: t for t in tasks}
    assert len(task_by_id) == 262 and set(task_by_id) <= train_ids and len(train_ids) == 1000
    dev_ids = set(load(ROOT / 'analysis/grpo_baseline_20261022_202816/sft_dev50/summary.json')['strict_success_task_ids'])
    dev_ids |= {r['task_id'] for r in rows(ROOT / 'analysis/grpo_baseline_20261022_202816/sft_dev50/trajectories.jsonl')}
    assert not set(task_by_id) & dev_ids
    contracts = {r['task_id']: r for r in rows(CONTRACTS)}
    reqs = {}
    for tid, task in task_by_id.items():
        assert task['query'] == contracts[tid]['query']
        reqs[tid] = requirements(contracts[tid])
    cfg = load(RUN / 'run.json')
    previous_audit = load(ROOT / 'analysis/checkpoint_pair_result_20261112/audit_and_analysis.json')
    assert sha((RUN / 'run.json').read_bytes()) == previous_audit['sources_sha256']['run']
    assert sha((ROOT / 'experiments/checkpoint_pair_v1/manifest.json').read_bytes()) == previous_audit['sources_sha256']['protocol']
    manifest, entries, flags, observations = [], [], [], []
    counts = defaultdict(Counter)
    seen = set()
    for arm in ('sft', 'grpo'):
        for tid in sorted(task_by_id):
            for attempt in range(8):
                folder = RUN / arm / 'attempts' / f'{tid}_{attempt}'
                tries = sorted(folder.glob('try_*'))
                assert len(tries) == 1, str(folder)
                path = tries[0] / 'trajectory.json'
                raw = path.read_bytes()
                receipt = load(tries[0] / 'result.json')
                assert sha(raw) == receipt['trajectory_sha256']
                t = json.loads(raw.decode('utf-8-sig'))
                assert t['task_id'] == tid and t['attempt_index'] == attempt
                assert t['initial_result']['instruction'].removeprefix('Instruction: ') == task_by_id[tid]['query']
                assert t['trajectory_id'] not in seen
                seen.add(t['trajectory_id'])
                relative = path.relative_to(ROOT).as_posix()
                manifest.append({'path': relative, 'sha256': sha(raw), 'result_sha256': sha((tries[0] / 'result.json').read_bytes()), 'arm': arm, 'task_id': tid, 'attempt': attempt})
                counts[arm]['trajectories'] += 1
                buys = [(i, s) for i, s in enumerate(t['steps']) if s['tool_name'] == 'buy_now']
                counts[arm]['with_buy_action'] += bool(buys)
                local_flags = set()
                for index, buy in buys:
                    previous = t['steps'][index - 1]['observation'] if index else ''
                    state = public_state(previous)
                    base = {'arm': arm, 'task_id': tid, 'attempt': attempt, 'path': relative,
                            'trajectory_sha256': sha(raw), 'query': task_by_id[tid]['query'],
                            'buy_turn': index + 1, 'visible_before_buy': previous,
                            'state': state, 'cohort': task_by_id[tid]['cohort']}
                    opts = state.get('available_options')
                    if state.get('selected_options') == {} and isinstance(opts, dict) and any(isinstance(v, list) and len(v) > 1 for v in opts.values()):
                        flags.append({**base, 'kind': 'empty_options_candidate', 'warning': 'No semantic verdict; available axes may be optional.'})
                        local_flags.add('empty_options_candidate')
                    for req in reqs[tid]:
                        cmp = compare(state, req)
                        if cmp is None:
                            counts[arm][req['kind'] + '_unknown_at_buy'] += 1
                            continue
                        counts[arm][req['kind'] + '_comparable_at_buy'] += 1
                        record = {**base, 'kind': req['kind'], 'requirement': req, 'comparison': cmp}
                        observations.append(record)
                        if not cmp['passed']:
                            flags.append(record)
                            local_flags.add(req['kind'])
                        # Same-product earlier local pass -> final local violation.
                        if not cmp['passed']:
                            for prior_index, prior in enumerate(t['steps'][:index]):
                                prior_state = public_state(prior.get('observation'))
                                if prior_state.get('asin') != state.get('asin'):
                                    continue
                                prior_cmp = compare(prior_state, req)
                                if prior_cmp and prior_cmp['passed']:
                                    flags.append({**record, 'kind': 'local_pass_to_fail_candidate',
                                                  'earlier_turn': prior_index + 1, 'earlier_visible': prior['observation'],
                                                  'earlier_comparison': prior_cmp})
                                    local_flags.add('local_pass_to_fail_candidate')
                                    break
                for kind in local_flags:
                    counts[arm][kind + '_trajectories'] += 1
                entries.append({'arm': arm, 'task_id': tid, 'attempt': attempt, 'path': relative,
                                'has_buy': bool(buys), 'screen_flags': sorted(local_flags)})
    # Same-task, same-product, same local requirement contrasts across independent
    # trajectories; these are NOT same-history branch experiments.
    buckets = defaultdict(lambda: {True: [], False: []})
    for r in observations:
        buckets[(r['arm'], r['task_id'], r['state'].get('asin'), r['requirement']['id'])][r['comparison']['passed']].append(r)
    contrasts = []
    for key, outcomes in sorted(buckets.items()):
        if outcomes[True] and outcomes[False]:
            contrasts.append({'arm': key[0], 'task_id': key[1], 'asin': key[2], 'requirement_id': key[3],
                              'pass': outcomes[True][0], 'fail': outcomes[False][0],
                              'pass_occurrences': len(outcomes[True]), 'fail_occurrences': len(outcomes[False]),
                              'warning': 'Local atomic contrast across independent trajectories; not complete utility preference or same-state feasibility.'})
    # Review queue chosen without consulting gold/success: stable hash order,
    # at most four distinct tasks for each screen kind.
    queue, picked = [], set()
    for kind in sorted({r['kind'] for r in flags}):
        candidates = sorted([r for r in flags if r['kind'] == kind], key=lambda r: sha(f'{r["task_id"]}:{r["arm"]}:{r["attempt"]}'.encode()))
        tasks_for_kind = set()
        for r in candidates:
            if r['task_id'] in tasks_for_kind:
                continue
            queue.append(r)
            tasks_for_kind.add(r['task_id'])
            if len(tasks_for_kind) == 4:
                break
    summary = {'scope': '262 preselected original training tasks, 8 attempts each at fixed SFT/GRPO500, not 800 and not population-random.',
               'models': cfg['models'], 'settings': cfg['settings'], 'cohorts': dict(Counter(t['cohort'] for t in tasks)),
               'counts': {k: dict(v) for k, v in counts.items()},
               'eligible_requirement_tasks': {kind: sum(any(r['kind'] == kind for r in rr) for rr in reqs.values()) for kind in ('explicit_budget', 'explicit_mass', 'explicit_color')},
               'flag_tasks': {kind: sorted({r['task_id'] for r in flags if r['kind'] == kind}) for kind in sorted({r['kind'] for r in flags})},
               'contrasts': len(contrasts), 'contrast_tasks': len({r['task_id'] for r in contrasts}),
               'review_queue_entries': len(queue), 'review_queue_tasks': len({r['task_id'] for r in queue}),
               'sources': {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in (CONTRACTS, RUN / 'run.json', ROOT / 'experiments/checkpoint_pair_v1/tasks.jsonl')},
               'limits': ['Screen flags are not independently reviewed error rates.', 'Requirements are locally approved atoms; whole-contract/reward approval is not inferred.',
                          'Only rendered observations used for decisions; no gold ASIN/goal/options used.', 'Unknown does not imply fail; parser has intentional low coverage.']}
    assert len(manifest) == len(seen) == 4192
    OUT.mkdir(parents=True, exist_ok=True)
    for name, obj in [('summary.json', summary), ('source_manifest.json', manifest), ('trajectory_index.json', entries),
                      ('screen_flags.json', flags), ('local_contrasts.json', contrasts), ('review_queue.json', queue)]:
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('models', 'sources', 'flag_tasks')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
