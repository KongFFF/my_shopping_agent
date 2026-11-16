"""Preserve every denominator; prepare public review pairs without claiming labels."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import math
from pathlib import Path
from decision_probe_common import PACKAGE, load, rows, save, digest, dump
from decision_probe_rules import public_state, atomic_checks


def outcome(t):
    r = t.get('terminal_result') or {}
    v = r.get('reward')
    valid = t.get('done') is True and r.get('reward_valid') is True and not t.get('error') and not t.get('release_error') and isinstance(v, (int, float)) and math.isfinite(v)
    return {'valid': valid, 'reward': v if valid else None,
            'gold': valid and (r.get('reward_detail') or {}).get('reward_type') == 'gold_purchase',
            'status': t['status']}


def candidate(t, task):
    # Review sees only actual Actor-visible projected observations and actions.
    # Omit all terminal private fields, reward, trigger, and arm identifiers.
    steps = t['steps']
    buys = [i for i, s in enumerate(steps) if s['tool_name'] == 'buy_now']
    before = steps[buys[-1]-1]['observation'] if buys and buys[-1] > 0 else ''
    state = public_state(before)
    return {'candidate_id': digest(t['path'])[:20], 'task_id': t['task_id'], 'query': task['query'],
            'status': t['status'], 'prebuy_public_state': state,
            'atomic_diagnostics': atomic_checks(task['query'], before),
            'public_steps': [{'tool': s['tool_name'], 'parameters': s['parameters'],
                              'visible': s['observation'] if not s['done'] else '<terminal omitted>'} for s in steps]}


def report(run, output):
    if output.exists(): raise ValueError('Report exists; preserve prior results')
    tasks = rows(PACKAGE/'tasks.jsonl'); taskmap = {t['task_id']: t for t in tasks}
    traces = rows(run/'trajectories.jsonl') if (run/'trajectories.jsonl').exists() else []
    if len({t['path'] for t in traces}) != len(traces): raise ValueError('Duplicate paths')
    usage = rows(run/'usage.jsonl') if (run/'usage.jsonl').exists() else []
    groups = defaultdict(list)
    prefixes = {t['task_id']: t for t in traces if t['path'].endswith('/prefix')}
    for t in traces:
        if not t['path'].endswith('/prefix'): groups[(t['task_id'], t['path'].split('/')[1])].append(t)
    table = []
    candidates, private, pairs = [], [], []
    for (tid, arm), ts in sorted(groups.items()):
        os = [outcome(t) for t in ts]; valid = len(os) == 4 and all(o['valid'] for o in os)
        table.append({'task_id': tid, 'stratum': taskmap[tid]['stratum'], 'arm': arm, 'size': len(os),
                      'all_valid': valid, 'nonconstant': valid and max(o['reward'] for o in os)-min(o['reward'] for o in os) > 1e-8,
                      'mixed_gold_failure': valid and 0 < sum(o['gold'] for o in os) < 4, 'outcomes': os})
        cs = [candidate(t, taskmap[tid]) for t in ts]
        candidates += cs
        private += [{'candidate_id': c['candidate_id'], 'path': t['path'], **outcome(t)} for c,t in zip(cs,ts)]
        for left, right in combinations(cs, 2):
            ls, rs = left['prebuy_public_state'], right['prebuy_public_state']
            same = bool(ls.get('asin')) and ls.get('asin') == rs.get('asin')
            pairs.append({'pair_id': digest(':'.join(sorted([left['candidate_id'], right['candidate_id']])))[:20],
                          'task_id': tid, 'left': left['candidate_id'], 'right': right['candidate_id'],
                          'same_product': same, 'different_selected_options': same and ls.get('selected_options') != rs.get('selected_options'),
                          'scope': 'pending', 'preferred': 'pending', 'conflicts': [], 'evidence': []})
    costs = {}
    for arm in ('A','B'):
        us = [u for u in usage if u['path'].split('/')[1] == arm]
        ts = [t for t in traces if t['path'].split('/')[1] == arm]
        costs[arm] = {'model_calls': len(us), 'failed_usage_unknown': sum(u.get('usage') is None for u in us),
                     'prompt_tokens': sum((u.get('usage') or {}).get('prompt_tokens',0) for u in us),
                     'completion_tokens': sum((u.get('usage') or {}).get('completion_tokens',0) for u in us),
                     'trajectory_seconds_including_replay': sum(t['elapsed_seconds'] for t in ts),
                     'replay_steps': sum(len(t['replay']) for t in ts),
                     'replay_seconds': sum(s['seconds'] for t in ts for s in t['replay'])}
    coverage = [{'task_id': t['task_id'], 'stratum': t['stratum'],
                 'prefix_status': prefixes.get(t['task_id'], {}).get('status','not_attempted'),
                 'trigger': (prefixes.get(t['task_id'], {}).get('trigger') or {}).get('kind'),
                 'A_paths': len(groups[(t['task_id'],'A')]), 'B_paths': len(groups[(t['task_id'],'B')])} for t in tasks]
    summary = {'schema': 'decision-probe-report-v1', 'planned_tasks': len(tasks), 'coverage': coverage,
               'trigger_counts': dict(Counter(r['trigger'] or r['prefix_status'] for r in coverage)),
               'groups': table, 'costs': costs, 'reviewed_pairs': 0,
               'limits': ['Prospective trigger conditioned B; no same-cost or training-gain claim.',
                          'Atomic price checks and same-product flags are review queues, not preference labels.',
                          'Uniform32 and enriched16 must be reported separately; all 48 remain in denominator.',
                          'Keep untriggered, unattempted, invalid, all-same and partial tasks.']}
    summary['strata'] = {}
    for stratum in ('uniform32', 'public_requirement16'):
        ids = {t['task_id'] for t in tasks if t['stratum'] == stratum}
        cs = [r for r in coverage if r['task_id'] in ids]
        gs = [g for g in table if g['task_id'] in ids]
        summary['strata'][stratum] = {'planned': len(ids), 'triggered': sum(bool(c['trigger']) for c in cs),
            'attempted_prefixes': sum(c['prefix_status'] != 'not_attempted' for c in cs),
            'arms': {a: {'complete_groups': sum(g['size'] == 4 and g['arm'] == a for g in gs),
                         'valid_groups': sum(g['all_valid'] and g['arm'] == a for g in gs),
                         'nonconstant_groups': sum(g['nonconstant'] and g['arm'] == a for g in gs),
                         'prompt_tokens': sum((u.get('usage') or {}).get('prompt_tokens',0) for u in usage if int(u['path'].split('/')[0]) in ids and u['path'].split('/')[1] == a),
                         'completion_tokens': sum((u.get('usage') or {}).get('completion_tokens',0) for u in usage if int(u['path'].split('/')[0]) in ids and u['path'].split('/')[1] == a)} for a in ('A','B')}}
    save(output/'summary.json', summary)
    save(output/'public_candidates.json', sorted(candidates,key=lambda c:c['candidate_id']))
    save(output/'private_outcomes.json', private)
    save(output/'review_pairs.json', sorted(pairs,key=lambda p:p['pair_id']))
    print('Report written:', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path); a = p.parse_args(); report(a.run_dir, a.output)
