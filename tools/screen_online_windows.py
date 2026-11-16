"""Unified local decision screening for CPU-restored online windows only."""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
from audit_online_windows import ROOT, OUT, inputs, save, sha
# Pure rule functions only. No legacy trajectory files are read by this import.
from public_decision_rules import requirements, public_state, compare


def main():
    manifest, occurrences, _, _ = inputs()
    cache = {}
    names = ('restored_observations.jsonl.gz',) if (OUT / 'restored_observations.jsonl.gz').exists() else ('replays.jsonl.gz', 'replays_parallel.jsonl.gz')
    for name in names:
        p = OUT / name
        if not p.exists(): continue
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            while True:
                try:
                    line = f.readline()
                    if not line: break
                    r = json.loads(line); cache[r['key']] = r
                except (EOFError, json.JSONDecodeError): break
    contract_path = ROOT / 'analysis/semantic_pilot30_evidence_20261109/contracts1000.jsonl'
    contracts = {r['task_id']: r for r in map(json.loads, contract_path.read_text(encoding='utf-8-sig').splitlines())}
    reqs = {tid: requirements(r) for tid, r in contracts.items()}
    counts = defaultdict(Counter); tasks = defaultdict(lambda: defaultdict(set))
    flags, observations, statuses = [], [], []
    for o in occurrences:
        label, original = o['window'], o['rollout']; c = counts[label]
        c['all_generated'] += 1; c['consumed'] += o['consumed']
        prefix = 'consumed_' if o['consumed'] else 'not_consumed_'
        if original.get('guard_rejections', 0) or original.get('error') or original.get('infrastructure_invalid') or original.get('context_compactions', 0):
            c['not_certified_missing_history'] += 1; continue
        p = cache.get(o['key'])
        if not p or p.get('error'):
            c['not_certified_replay_error_or_pending'] += 1; continue
        terminal_match = p['done'] == original['done'] and (not original['done'] or (p['reward_type'] == original['reward_type']
            and p['reward_valid'] == original['reward_valid'] and math.isclose(float(p['reward']), float(original['reward']['terminal_utility']), abs_tol=1e-7)))
        projection_match = all(p['projection'][k] == original.get(k) for k in p['projection'])
        if not terminal_match or not projection_match:
            c['not_certified_terminal_or_projection_mismatch'] += 1; continue
        tid = int(original['task_id'])
        assert p['query'] == contracts[tid]['query']
        c['certified_occurrences'] += 1; c[prefix + 'certified'] += 1
        tasks[label]['certified'].add(tid)
        local_flags = set()
        for index, step in enumerate(p['steps']):
            if step['action']['tool'] != 'buy_now': continue
            c['with_buy'] += 1
            previous = p['steps'][index - 1]['observation'] if index else ''
            state = public_state(previous)
            base = {'window': label, 'step': o['step'], 'uid': original['uid'], 'rollout_index': original['rollout_index'],
                    'key': o['key'], 'consumed': o['consumed'], 'task_id': tid, 'buy_turn': step['turn'],
                    'query': p['query'], 'visible_before_buy_reconstructed': previous, 'state': state,
                    'evidence': 'CPU reconstructed; source/terminal/projection aggregates verified, original per-turn bytes unavailable.'}
            opts = state.get('available_options')
            if state.get('selected_options') == {} and isinstance(opts, dict) and any(isinstance(v, list) and len(v) > 1 for v in opts.values()):
                flags.append({**base, 'kind': 'empty_options_candidate'})
                local_flags.add('empty_options_candidate')
            for req in reqs[tid]:
                cmp = compare(state, req)
                if cmp is None: c[req['kind'] + '_unknown'] += 1; continue
                c[req['kind'] + '_comparable'] += 1
                record = {**base, 'kind': req['kind'], 'requirement': req, 'comparison': cmp}
                observations.append(record)
                if not cmp['passed']:
                    flags.append(record); local_flags.add(req['kind'])
                    for earlier in p['steps'][:index]:
                        earlier_state = public_state(earlier['observation'])
                        if earlier_state.get('asin') != state.get('asin'): continue
                        before = compare(earlier_state, req)
                        if before and before['passed']:
                            flags.append({**record, 'kind': 'local_pass_to_fail_candidate', 'earlier_turn': earlier['turn'],
                                          'earlier_visible_reconstructed': earlier['observation'], 'earlier_comparison': before})
                            local_flags.add('local_pass_to_fail_candidate'); break
        for kind in local_flags:
            c[kind + '_occurrences'] += 1; c[prefix + kind] += 1; tasks[label][kind].add(tid)
        statuses.append({'window': label, 'uid': original['uid'], 'rollout_index': original['rollout_index'], 'key': o['key'],
                         'task_id': tid, 'consumed': o['consumed'], 'screen_flags': sorted(local_flags)})
    buckets = defaultdict(lambda: {True: [], False: []})
    for r in observations:
        buckets[(r['window'], r['task_id'], r['state'].get('asin'), r['requirement']['id'])][r['comparison']['passed']].append(r)
    contrasts = [{'window': k[0], 'task_id': k[1], 'asin': k[2], 'requirement_id': k[3],
                  'pass': v[True][0], 'fail': v[False][0], 'pass_count': len(v[True]), 'fail_count': len(v[False]),
                  'limit': 'Independent online rollouts, potentially different policies/histories; not whole preference or same-state feasibility.'}
                 for k, v in buckets.items() if v[True] and v[False]]
    queue = []
    for window in counts:
        for kind in sorted({r['kind'] for r in flags}):
            rs = sorted([r for r in flags if r['window'] == window and r['kind'] == kind], key=lambda r: sha(f'{r["task_id"]}:{r["key"]}'.encode()))
            picked = set()
            for r in rs:
                if r['task_id'] in picked: continue
                queue.append(r); picked.add(r['task_id'])
                if len(picked) == 3: break
    summary = {'counts': {k: dict(v) for k, v in counts.items()},
               'task_ids': {w: {k: sorted(v) for k, v in by_kind.items()} for w, by_kind in tasks.items()},
               'flag_records': len(flags), 'local_contrasts': len(contrasts), 'review_queue_entries': len(queue),
               'contract_sha256': sha(contract_path.read_bytes()),
               'limits': ['No old 4192 reevaluation records used.', 'Flags are candidates, not independent semantic error rates.',
                          'Unrestorable histories remain in full-generation denominator, not counted as clean.']}
    for name, obj in [('screen_summary.json', summary), ('screen_flags.json', flags), ('local_contrasts.json', contrasts),
                      ('review_queue.json', queue), ('certified_index.json', statuses)]: save(name, obj)
    print(json.dumps({'counts': summary['counts'], 'flag_records': len(flags), 'local_contrasts': len(contrasts), 'review_queue_entries': len(queue)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8'); main()
