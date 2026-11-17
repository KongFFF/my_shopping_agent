"""Frozen CPU qualification audit. No sampling, training or reward changes."""
from collections import defaultdict, Counter
from itertools import product
from pathlib import Path
import gzip
import hashlib
import json
import statistics as st
import zipfile
import analyze_grpo_structure as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/local_signal_audit_20261117'
EXCLUDED = {22462, 19827}
MODES = ('observation', 'remaining', 'history')
EPS = 1e-7


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def sign(x): return 1 if x > EPS else -1 if x < -EPS else 0
def save(n, x): (OUT/n).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def load():
    old = ROOT/'analysis/grpo_structural_analysis_20261116'
    receipt = json.loads((old/'manifest.json').read_text(encoding='utf-8'))
    for n, h in receipt['inputs_sha256'].items(): assert sha(ROOT/n) == h, n
    with gzip.open(ROOT/'analysis/online_window_audit_20261116/restored_observations.jsonl.gz', 'rt', encoding='utf-8') as f:
        restored = {r['key']: r for r in map(json.loads, f)}
    certified = {r['key'] for r in json.loads((ROOT/'analysis/online_window_audit_20261116/certified_index.json').read_text(encoding='utf-8'))}
    cohorts = {}
    with zipfile.ZipFile(ROOT/'analysis/online_training_window_return_20261116/evidence.zip') as z:
        for label in ('reference351_500', 'late651_800'):
            groups = {}
            for e in map(json.loads, z.read(label+'/training_diagnostics.jsonl').splitlines()):
                if e['event'] != 'optimizer_step': continue
                batch = defaultdict(list)
                for r in e['rollouts']: batch[r['uid']].append(r)
                for uid, rs in batch.items():
                    rebuilt = []
                    for j, r in enumerate(rs):
                        key = hashlib.sha256(json.dumps({'task_id': r['task_id'], 'actions': r['actions']}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                        if key not in certified: break
                        rr = restored[key]
                        assert [s['action'] for s in rr['steps']] == r['actions']
                        assert abs(rr['reward']-r['reward']['terminal_utility']) < 1e-6
                        rebuilt.append({'task_id': r['task_id'], 'path': uid+'/'+str(j), 'status': 'done',
                            'terminal_result': {'reward': rr['reward'], 'reward_detail': {
                                'reward_valid': r['reward_valid'], 'purchase_success': bool(r['reward']['purchase_success']),
                                'sampling_invalid': r.get('sampling_invalid', False)}},
                            'steps': [{'tool_name': s['action']['tool'], 'parameters': s['action']['parameters'],
                                'observation': s['observation']} for s in rr['steps']]})
                    if len(rebuilt) == 4 and all(base.valid(t) for t in rebuilt): groups[uid] = rebuilt
            cohorts[label] = groups
    with zipfile.ZipFile(ROOT/'analysis/decision_probe_result_20261116/evidence.zip') as z:
        traces = [t for t in map(json.loads, z.read('run/trajectories.jsonl').splitlines()) if '/A/' in t['path']]
    tasks = {t['task_id']: t for t in map(json.loads, (ROOT/'experiments/decision_probe_v1/tasks.jsonl').read_text(encoding='utf-8').splitlines())}
    for stratum in ('uniform32', 'public_requirement16'):
        groups = defaultdict(list)
        for t in traces:
            if tasks[t['task_id']]['stratum'] == stratum: groups[str(t['task_id'])].append(t)
        cohorts['fixed800_'+stratum] = {k: v for k, v in groups.items() if all(base.valid(t) for t in v)}
    assert [len(g) for g in cohorts.values()] == [204, 195, 27, 13]
    return cohorts


def ranking(os, binary=False):
    actions = defaultdict(list)
    for o in os: actions[o['action']].append(float(o['success']) if binary else o['reward'])
    if len(actions) != 2 or sorted(map(len, actions.values())) != [2, 2]: return None
    a, b = [actions[k] for k in sorted(actions)]
    # Four choices form two complementary splits; fix a[0] in the first half.
    ds = [(a[0]-b[j], a[1]-b[1-j]) for j in range(2)]
    return ['tie' if 0 in (sign(x), sign(y)) else 'same' if sign(x) == sign(y) else 'reverse' for x, y in ds]


def analyze(groups, mode, exclude):
    rows = []
    decisions = 0
    for uid, ts in groups.items():
        if exclude and ts[0]['task_id'] in EXCLUDED: continue
        states = defaultdict(dict)
        decisions += sum(len(t['steps'])-1 for t in ts)
        for t in ts:
            for i, obs, hist, s in base.pre_states(t):
                if i == 0: continue
                key = obs if mode == 'observation' else base.canonical([obs, 35-i] if mode == 'remaining' else [obs, hist])
                states[key].setdefault(t['path'], {'path': t['path'], 'index': i, 'reward': base.reward(t),
                    'success': base.success(t), 'action': base.canonical(base.action(s))})
        mean = st.fmean(base.reward(t) for t in ts)
        for key, members in states.items():
            os = list(members.values())
            if len(os) < 2: continue
            am = defaultdict(list)
            for o in os: am[o['action']].append(o['reward'])
            if len(am) < 2: continue
            avg = st.fmean(o['reward'] for o in os)
            e = [o['reward']-mean for o in os]
            l = [o['reward']-avg for o in os]
            m = [st.fmean(am[o['action']])-avg for o in os]
            def relation(v):
                if max(abs(a-b) for a, b in zip(e, v)) < EPS: return 'identical'
                if max(abs(b) for b in v) < EPS: return 'zero_signal'
                scale = sum(a*b for a, b in zip(e, v))/sum(a*a for a in e) if sum(a*a for a in e)>EPS else None
                if scale is not None and scale > 0 and max(abs(b-scale*a) for a, b in zip(e, v)) < EPS: return 'positive_rescale'
                return 'changed'
            ranks = ranking(os); gold_ranks = ranking(os, True)
            reliable = ranks == ['same', 'same'] and gold_ranks == ['same', 'same']
            rows.append({'group_id': uid, 'task_id': ts[0]['task_id'], 'state_hash': hashlib.sha256(key.encode()).hexdigest(),
                'support': sorted(map(len, am.values())), 'reward_varies': max(o['reward'] for o in os)-min(o['reward'] for o in os)>EPS,
                'L_relation': relation(l), 'M_relation': relation(m), 'reward_splits': ranks, 'success_splits': gold_ranks,
                'reliable': reliable, 'members': os, 'episode_advantages': e, 'L': l, 'M': m})
    def counts(rs):
        return {'keys': len(rs), 'groups': len({r['group_id'] for r in rs}), 'tasks': len({r['task_id'] for r in rs}),
            'decisions': len({(o['path'], o['index']) for r in rs for o in r['members']})}
    qualified = [r for r in rows if r['reliable']]
    innovative = {kind: counts([r for r in qualified if r[kind+'_relation'] == 'changed']) for kind in ('L', 'M')}
    return {'groups': len({t[0]['path'].rsplit('/',1)[0] for t in groups.values() if not exclude or t[0]['task_id'] not in EXCLUDED}),
        'noninitial_decisions': decisions, 'different_action_states': counts(rows),
        'reward_varies': counts([r for r in rows if r['reward_varies']]),
        'support_histogram': dict(Counter(str(r['support']) for r in rows)),
        'splittable': counts([r for r in rows if r['reward_splits'] is not None]),
        'reward_split_results': dict(Counter(x for r in rows for x in (r['reward_splits'] or []))),
        'success_split_results': dict(Counter(x for r in rows for x in (r['success_splits'] or []))),
        'reliable': counts(qualified), 'reliable_new_information': innovative,
        'all_signal_relations': {kind: dict(Counter(r[kind+'_relation'] for r in rows)) for kind in ('L','M')},
        'gate_pass': {kind: counts(qualified)['groups'] >= 20 and counts(qualified)['tasks'] >= 15 and
            innovative[kind]['groups'] >= 10 and innovative[kind]['tasks'] >= 10 for kind in ('L','M')}}, rows


def main():
    # Disjoint ranking must not label ties or crossing outcomes as confirmation.
    fixture = lambda a, b: [{'action': k, 'reward': v, 'success': bool(v)}
        for k, vs in [('a', a), ('b', b)] for v in vs]
    assert ranking(fixture([1,1], [0,0])) == ['same','same']
    assert ranking(fixture([1,0], [0,1])) == ['reverse','tie']
    assert ranking(fixture([1,-1], [0,0])) == ['reverse','reverse']
    assert ranking(fixture([1], [0,0])) is None
    protocol = OUT/'核查协议.md'
    bound = sha(protocol)
    groups = load(); summary = {}; rows = {}
    for label, gg in groups.items():
        summary[label] = {}; rows[label] = {}
        for mode, sensitivity in product(MODES, (False, True)):
            key = mode+('_exclude_uncertain' if sensitivity else '')
            summary[label][key], rows[label][key] = analyze(gg, mode, sensitivity)
    assert sha(protocol) == bound
    save('summary.json', summary); save('state_evidence.json', rows)
    save('manifest.json', {'protocol_sha256': bound, 'script_sha256': sha(Path(__file__)),
        'upstream_manifest_sha256': sha(ROOT/'analysis/grpo_structural_analysis_20261116/manifest.json'),
        'outputs_sha256': {n: sha(OUT/n) for n in ('summary.json','state_evidence.json')}})
    for label in summary:
        print(label, json.dumps(summary[label]['history'], ensure_ascii=False))


if __name__ == '__main__': main()
