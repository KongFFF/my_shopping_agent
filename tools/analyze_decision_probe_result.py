"""Audit the completed frozen probe from its Git-returned zip; CPU only."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def dump(x):return json.dumps(x,ensure_ascii=False,sort_keys=True)
def sha(b):return hashlib.sha256(b).hexdigest()


def analyze(archive, out):
    with zipfile.ZipFile(archive) as z:
        obj=lambda n:json.loads(z.read(n))
        lines=lambda n:[json.loads(l) for l in z.read(n).splitlines() if l.strip()]
        manifest=obj('manifest.json');protocol=obj('protocol.json')
        for n,h in manifest['files'].items():assert sha(z.read('run/'+n))==h,n
        cfg=obj('run/run.json');summary=obj('run/report/summary.json');ready=obj('run/ready.json')
        assert cfg['protocol_sha256']==sha(z.read('protocol.json'))
        assert ready['identity']['code_sha256']==protocol['code_sha256']
        assert ready['identity']['runtime_sha256']==cfg['runtime_sha256']
        assert ready['identity']['index_physical_sha256']==cfg['index_physical_sha256']
        assert any(m['id']==cfg['model_name'] and m['root']==cfg['model_dir'] for m in ready['models'])
        export=obj('run/original_export_receipt.json')
        assert all(cfg['model_hashes'][n]==h for n,h in export['merged_weights_sha256'].items())
        assert export['merged_model']==cfg['model_dir'] and export['checkpoint'].endswith('/global_step_800')
        ts=lines('run/trajectories.jsonl');us=lines('run/usage.jsonl');events=lines('run/events.jsonl');rs=lines('run/restorations.jsonl')
        assert len({t['path']for t in ts})==len(ts)
        started=[e['task_id']for e in events if e['event']=='task_started']
        completed=[e['task_id']for e in events if e['event']=='task_completed']
        assert started==completed==protocol['task_ids']
        prefixes={t['task_id']:t for t in ts if t['path'].endswith('/prefix')}
        assert len(prefixes)==48
        triggered={i for i,t in prefixes.items()if t['branch']}
        trace={t['path']:t for t in ts}
        for r in rs:
            t=trace[r['path']];p=prefixes[t['task_id']]
            assert r['verified'] and r['state']==p['branch_state']==t['restored_state']
            assert t['prefix_message_sha256']==p['branch_message_sha256']
            assert t['steps'][:t['prefix_steps']]==p['steps']
            assert all(s['matched']for s in t['replay'])
        assert len(rs)==4*len(triggered)
        assert all(len(t['steps'])<=35 for t in ts)
        perpath=Counter()
        for u in us:
            k=u['path'];expected=int(sha((k+':'+str(perpath[k])).encode())[:8],16)%2147483647
            assert u['seed']==expected and u['status']=='ok'
            perpath[k]+=1
        settings=protocol['settings']
        assert len(us)<=settings['max_model_calls']
        assert sum(u['usage']['completion_tokens']for u in us)<=settings['max_completion_tokens']
        groups=summary['groups'];gm={(g['task_id'],g['arm']):g for g in groups}
        assert all(gm[i,'A']['size']==4 for i in completed)
        assert all(gm[i,'B']['size']==4 for i in triggered)
        budget=obj('run/budget.json');stopped=obj('run/services_stopped.json')
        assert not stopped['budget_expired']
        assert 0<stopped['at']-budget['started_at']<settings['gpu_seconds']
        candidates=obj('run/report/public_candidates.json')
        private={c['candidate_id']:c for c in obj('run/report/private_outcomes.json')}
    def costs(ids,a):
        uu=[u for u in us if int(u['path'].split('/')[0])in ids and u['path'].split('/')[1]==a]
        tt=[t for t in ts if t['task_id']in ids and t['path'].split('/')[1]==a]
        return {'model_calls':len(uu),'prompt_tokens':sum(u['usage']['prompt_tokens']for u in uu),
            'completion_tokens':sum(u['usage']['completion_tokens']for u in uu),
            'path_seconds_including_prefix_and_replay':sum(t['elapsed_seconds']for t in tt),
            'replay_steps':sum(len(t['replay'])for t in tt)}
    def table(ids):
        result={}
        for a in ('A','B'):
            gg=[g for g in groups if g['task_id']in ids and g['arm']==a];c=costs(ids,a)
            n=sum(g['nonconstant']for g in gg)
            result[a]={'groups':len(gg),'valid_groups':sum(g['all_valid']for g in gg),
                'nonconstant_groups':n,'mixed_gold_failure':sum(g['mixed_gold_failure']for g in gg),
                'gold_rollouts':sum(o['gold']for g in gg for o in g['outcomes']),
                'tasks_with_any_gold':sum(any(o['gold']for o in g['outcomes'])for g in gg),
                'costs':c,'nonconstant_groups_per_10k_completion_tokens':n/c['completion_tokens']*10000,
                'nonconstant_groups_per_path_minute':n/c['path_seconds_including_prefix_and_replay']*60}
        return result
    variations=[]
    for i in completed:
        for a in ('A','B'):
            cc=[c for c in candidates if c['task_id']==i and private[c['candidate_id']]['path'].split('/')[1]==a]
            pairs=[]
            for l,r in combinations(cc,2):
                ls,rr=l['prebuy_public_state'],r['prebuy_public_state']
                if ls.get('asin') and ls.get('asin')==rr.get('asin') and ls.get('selected_options')!=rr.get('selected_options'):
                    pairs.append({'left':private[l['candidate_id']]['path'],'right':private[r['candidate_id']]['path']})
            if pairs:variations.append({'task_id':i,'arm':a,'pairs':pairs})
    triggers={}
    for kind in sorted({p['trigger']['kind']for p in prefixes.values()if p['branch']}):
        ids={i for i,p in prefixes.items()if p['branch']and p['trigger']['kind']==kind}
        triggers[kind]={'tasks':len(ids),'arms':table(ids)}
    errors=[{'path':t['path'],'status':t['status'],'error_type':(t.get('error')or{}).get('type')}
            for t in ts if t['status']not in ('done','branch')]
    packet={'schema':'decision-probe-audit-v1','archive_sha256':file_hash(archive),
        'return_commit':'523c5314b256c9a968b23fe5915cc0ff9c0a98b7',
        'checks':'Archive hashes, original800 identity, protocol, seeds, 48 ordered task completions, 164 full-prefix/state restores and caps passed.',
        'completed_tasks':len(completed),'triggered_tasks':len(triggered),'restorations':len(rs),
        'elapsed_window_seconds':stopped['at']-budget['started_at'],
        'all_48':table(set(completed)),'paired_41':table(triggered),'by_trigger':triggers,
        'by_stratum':summary['strata'],'errors':errors,'statuses':dict(Counter(t['status']for t in ts)),
        'same_product_different_options':variations,
        'nontriggered':[{'task_id':i,'status':p['status']}for i,p in prefixes.items()if not p['branch']],
        'same_reward_mixed_spec_B_tasks':[v['task_id']for v in variations if v['arm']=='B' and not gm[v['task_id'],'B']['nonconstant']],
        'both_arms_all_valid_tasks':sum(gm[i,'A']['all_valid']and gm[i,'B']['all_valid']for i in triggered),
        'notes':['B prefix and trigger-proposal costs included; nontriggered B costs included in all48.',
                 'Group yield uses original reward, not verified semantic preference.',
                 'Branch rollouts share a prefix; no independent-binomial significance claim.',
                 'Per-task gold counts are conditional sample diagnostics, not single-rollout benchmark scores.']}
    out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf-8')
    # B terminal-option differences exhaustively reviewed; A selected cases only.
    selected={2670,3727,4060,12502,17311,18330,19827,20628,21870,22462,5535,22560}
    public=[{**c,'path':private[c['candidate_id']]['path']}for c in candidates if c['task_id']in selected]
    (out/'public_review_evidence.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:packet[k]for k in ('completed_tasks','triggered_tasks','restorations','elapsed_window_seconds','paired_41')},ensure_ascii=False,indent=2))


def file_hash(p):return sha(Path(p).read_bytes())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'analysis/decision_probe_result_20261116/evidence.zip')
    p.add_argument('--output',type=Path,default=ROOT/'analysis/decision_probe_result_20261116')
    a=p.parse_args();analyze(a.archive,a.output)
