"""Verify supplemental fixed B500 Final and pair against historical A500 IDs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zipfile

from analyze_b500_dev50_return import strict,paired_interval
from step650_evaluation import FINAL_HASH

ROOT=Path(__file__).resolve().parents[1]
POST='preparation/posttrain_step500_v1/'
FOLDER='evaluation/final200_clean/'


def analyze(path):
    errors=[]
    def check(ok,label):
        if not ok:errors.append(label)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    with zipfile.ZipFile(path)as z:
        check(z.testzip()is None,'ZIP CRC')
        manifest=json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        check(sha(path.read_bytes())==manifest['archive_sha256'],'Archive SHA256')
        j=lambda n:json.loads(z.read(n))
        spec=j(POST+'evaluation_spec.json');prep=j(POST+'preparation_receipt.json')
        A=j(POST+'A500_final_summary.json');B=j(FOLDER+'summary.json');passed=j(FOLDER+'passed.json');done=j(POST+'final200_complete.json')
        export=j(POST+'export_receipt.json');service=j(POST+'services_ready.json')
        check(j(POST+'pipeline_exit_status.json')['exit_code']==0 and done['passed'],'Completion')
        check(prep['export_reused']and not prep['Final_is_unseen']and prep['original_Dev_decision_unchanged'],'Supplemental authorization boundary')
        check(set(spec['commands'])=={'final200_clean'}and prep['benchmark_sha256']==spec['benchmarks']['final200_clean']['sha256']==FINAL_HASH,'Original Final-only command and benchmark')
        check(passed['passed']and passed['tasks']==200,'200-task acceptance')
        check(sha(z.read(FOLDER+'trajectories.jsonl'))==passed['trajectory_sha256']and sha(z.read(FOLDER+'summary.json'))==passed['summary_sha256'],'Canonical trajectory/summary hashes')
        rs=[json.loads(l)for l in z.read(FOLDER+'trajectories.jsonl').splitlines()if l.strip()]
        ids=[r['task_id']for r in rs];expected=spec['benchmarks']['final200_clean']['task_ids']
        check(ids==expected and len(ids)==len(set(ids))==200,'Complete unique original200 task order')
        sb={r['task_id']for r in rs if strict(r)};sa=set(A['strict_success_task_ids'])
        check(sorted(sb)==B['strict_success_task_ids']and len(sb)==B['strict_successes']==passed['strict_successes'],'Recomputed strict gold')
        check(len(sa)==A['strict_successes']==137 and sa.issubset(ids),'Historical A137 summary')
        check(B['completed_tasks']==200 and not B['missing_tasks'],'No missing denominator records')
        check(sum(bool(r.get('done'))for r in rs)==B['done_tasks'],'Done count')
        check(math.isclose(sum(r['final_reward']for r in rs)/200,B['mean_final_reward'],abs_tol=1e-12),'Mean reward')
        check(math.isclose(sum(len(r['steps'])for r in rs)/200,B['average_steps'],abs_tol=1e-12),'Mean steps')
        noncontext=[r['task_id']for r in rs if (r.get('error')and r['error'].get('type')!='ContextBudgetError')or r.get('release_error')]
        check(not noncontext,'No non-context execution errors')
        pa=dict(A['protocol']);pb=dict(B['protocol']);pa.pop('model');pb.pop('model');check(pa==pb,'Original A/B Final protocol')
        check(export['checkpoint']==spec['checkpoint']and export['checkpoint'].endswith('/train/global_step_500'),'Fixed500 source')
        m=service['model'];check(service['passed']and m['id']==spec['model_tag']and m['root']==export['merged_model']and m['max_model_len']==24576,'Actual served model identity')
        previous=ROOT/'analysis/b500_independent_dev50_return/20261121T031644Z/evidence.zip'
        with zipfile.ZipFile(previous)as dz:check(json.loads(dz.read(POST+'export_receipt.json'))==export,'Exactly same export receipt/weight hashes as Dev')
        gain=sorted(sb-sa);loss=sorted(sa-sb);n=len(gain)+len(loss)
        check(done['B_only']==gain and done['A_only']==loss and done['net_gain']==len(gain)-len(loss),'Cloud pair recomputation')
        p=min(1.,2*sum(math.comb(n,k)for k in range(min(len(gain),len(loss))+1))/2**n)if n else 1.
        context=[r['task_id']for r in rs if (r.get('error')or{}).get('type')=='ContextBudgetError']
        changed=[{'task_id':r['task_id'],'change':'B_only'if r['task_id']in gain else'A_only',
            'B_reward_type':(r.get('terminal_result')or{}).get('reward_detail',{}).get('reward_type'),
            'B_steps':len(r['steps'])}for r in rs if r['task_id']in gain+loss]
        return {'audit_passed':not errors,'errors':errors,'return_commit':'64a6b9296605cdd1a3a6f6ab95dfd33c3dd5de84',
            'archive_sha256':manifest['archive_sha256'],'SFT_historical_Final':134,'A500_Final':len(sa),'B500_Final':len(sb),
            'B_minus_A_successes':len(sb)-len(sa),'B_minus_A_percentage_points':100*(len(sb)-len(sa))/200,
            'B_minus_SFT_percentage_points':100*(len(sb)-134)/200,'B_only':gain,'A_only':loss,
            'both_success':len(sa&sb),'both_fail':200-len(sa|sb),'exact_McNemar_two_sided_p':p,
            'paired_bootstrap_95_percentile_interval':paired_interval(len(gain),len(loss),200),
            'A_mean_reward':A['mean_final_reward'],'B_mean_reward':B['mean_final_reward'],
            'A_average_steps':A['average_steps'],'B_average_steps':B['average_steps'],
            'A_reward_types':A['reward_type_counts'],'B_reward_types':B['reward_type_counts'],
            'context_overflow_task_ids':context,'all_200_retained_in_denominator':True,'changed_tasks_B_details':changed,
            'paired_A_source':'Cloud-returned historical A500 summary success IDs; original A200 trajectories absent in this archive',
            'model_sha256':export['merged_weights_sha256'],'supplemental_evaluation':True,'Dev_screen_still_failed':done['Dev_screen_still_failed'],
            'interpretation':'Observed positive +2pp on both Dev and supplemental Final; stable method superiority remains unproven; no automatic continuation.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.archive);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['audit_passed']else 1)
