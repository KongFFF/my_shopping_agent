from pathlib import Path
import json,zipfile,hashlib,statistics
R=Path(__file__).resolve().parents[1];OUT=R/'analysis/project_closeout_20261125';OUT.mkdir(parents=True,exist_ok=True);used={}
def load(rel):
 p=R/rel;used[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text(encoding='utf-8'))
def archive(rel):
 p=R/rel;used[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return zipfile.ZipFile(p)
def j(z,p):return json.loads(z.read(p))
def rows(z,p):return [json.loads(l)for l in z.read(p).splitlines()if l.strip()]
evals={};evpaths={
'B500dev':'analysis/b500_independent_dev50_return/20261121T031644Z/evidence.zip','B500final':'analysis/b500_final200_return/20261121T040953Z/evidence.zip',
'B650':'analysis/b650_evaluation_return/20261122T011337Z/evidence.zip','B800':'analysis/b800_evaluation_return/20261122T134221Z/evidence.zip',
'C500':'analysis/c500_evaluation_return/20261124T052234Z/evidence.zip','C650':'analysis/c650_evaluation_return/20261124T132239Z/evidence.zip','C800':'analysis/c800_evaluation_return/20261125T075945Z/evidence.zip'}
for arm in ['A','B','C']:
 for step in [500,650,800]:
  tag=arm+str(step);ss={}
  for suite,n in [('dev50',50),('final200_clean',200)]:
   key=('C'+str(step))if arm=='A'else tag
   if tag=='B500':key='B500dev'if suite=='dev50'else'B500final'
   member=f'preparation/posttrain_step{step}_v1/A{step}_{suite}_summary.json'if arm=='A'else'evaluation/'+suite+'/summary.json'
   with archive(evpaths[key])as z:s=j(z,member)
   assert s['expected_tasks']==s['completed_tasks']==n and not s['missing_tasks']
   assert len(s['strict_success_task_ids'])==s['strict_successes']
   ss[suite]={k:s[k]for k in ['strict_successes','expected_tasks','mean_final_reward','average_steps','reward_type_counts']};ss[suite]['source']=evpaths[key]+':'+member
  evals[tag]=ss
native={};B=load('analysis/b500_training_audit_20261120/audit.json');native['B500']={'mean_reward':next(x['mean_reward']for x in B['internal_mean_reward_curve']if x['step']==500),'score1':None,'source':'analysis/b500_training_audit_20261120/audit.json'}
native['A500']={'mean_reward':B['historical_A500_internal_reward'],'score1':29,'source':'analysis/b500_training_audit_20261120/audit.json + old record'}
native['A650']={'mean_reward':0.508238,'score1':29,'source':'总流程记录_截至20261115原稿.md (historical six-decimal summary)'}
native['A800']={'mean_reward':0.449571,'score1':28,'source':'总流程记录_截至20261115原稿.md (historical six-decimal summary)'}
for arm in ['B','C']:
 for step in [500,650,800]:
  if arm=='B'and step==500:continue
  if step==500:
   x=load('analysis/c500_training_audit_20261124/audit.json');v=next(v for v in x['internal_validation']if v['step']==500);native['C500']={'mean_reward':v['reward_mean'],'score1':v['strict_score1'],'source':'analysis/c500_training_audit_20261124/audit.json'}
  elif arm=='B'and step==800:
   with archive(evpaths['B800'])as z:
    events=rows(z,'training_evidence/training_diagnostics.jsonl');u=next(e for e in events if e['event']=='optimizer_step'and e['global_step']==800)
    val=rows(z,'training_evidence/validation/800.jsonl')if 'training_evidence/validation/800.jsonl'in z.namelist()else []
    native['B800']={'mean_reward':u['metrics']['val-core/shopsimulator/reward/mean@1'],'score1':sum(abs(v['score']-1)<1e-6 for v in val)if val else None,'source':evpaths['B800']+':training_evidence/validation/800.jsonl'}
  else:
   rel=f'analysis/{arm.lower()}{step}_training_audit_'+('20261121'if arm=='B'else('20261124'if step==650 else'20261125'))+'/audit.json';x=load(rel);v=next(v for v in x['internal_validation']if v['step']==step);native[arm+str(step)]={'mean_reward':v['mean_reward'],'score1':v['score_equal1_count'],'source':rel}
# Actual committed batch data, normalized to the same last150-update window.
train={};full={}
with archive(evpaths['B800'])as z:full['B']=rows(z,'training_evidence/training_diagnostics.jsonl');Bbudget=j(z,'training_evidence/generation_budget.json')
with archive('analysis/c800_training_return/20261125T025942Z/evidence.zip')as z:full['C']=rows(z,'training_diagnostics.jsonl')
for arm,es in full.items():
 us=[e for e in es if e['event']=='optimizer_step'];assert [e['global_step']for e in us]==list(range(1,801))
 for step in [500,650,800]:train[arm+str(step)]=[e for e in us if step-149<=e['global_step']<=step]
with archive('analysis/online_training_window_return_20261116/evidence.zip')as z:
 for tag,member in [('A500','reference351_500/training_diagnostics.jsonl'),('A800','late651_800/training_diagnostics.jsonl')]:
  train[tag]=[e for e in rows(z,member)if e['event']=='optimizer_step']
  m=train[tag][-1]['metrics'];native[tag]['mean_reward']=m['val-core/shopsimulator/reward/mean@1'];native[tag]['source']='analysis/online_training_window_return_20261116/evidence.zip:'+member
train_summary={}
for tag,us in train.items():
 assert len(us)==150
 utilities=[r['reward']['terminal_utility']for e in us for r in e['rollouts']];assert len(utilities)==1200
 train_summary[tag]={'start':us[0]['global_step'],'end':us[-1]['global_step'],'updates':150,'trained_rollouts':len(utilities),'terminal_utility_mean':statistics.mean(utilities),'native_purchase_success_rate':statistics.mean(float(r['reward']['purchase_success'])for e in us for r in e['rollouts']),'critic_rewards_mean':statistics.mean(e['metrics']['critic/rewards/mean']for e in us),'grad_norm_mean':statistics.mean(e['metrics']['actor/grad_norm']for e in us),'source_scope':'Actual committed batch; not all generated trajectories, not held-out evaluation'}
train_summary['A650']={'available':False,'reason':'Unified return has only A351..500 and651..800; do not impute501..650'}
cost={'B500':B['generation_budget'],'B650':load('analysis/b650_training_audit_20261121/audit.json')['cumulative_generation'],'B800':Bbudget,'C500':load('analysis/c500_training_audit_20261124/audit.json')['generation_budget'],'C650':load('analysis/c650_training_audit_20261124/audit.json')['cumulative_generation'],'C800':load('analysis/c800_training_audit_20261125/audit.json')['cumulative_generation']}
compositions={}
for arm in ['B','C']:
 with archive(evpaths['B800']if arm=='B'else'analysis/c800_training_return/20261125T025942Z/evidence.zip')as z:
  # Canonical B800 weight log holds all651..800; parent logs cover earlier phases.
  if arm=='B':
   from collections import Counter
   x=load('analysis/group_weight_unified_20261122/training_summary.json')['B']
   for step,names in [(500,['early1_150','middle151_350','late351_500']),(650,['early1_150','middle151_350','late351_500','continuation501_650']),(800,['all1_800'])]:
    c=Counter()
    for name in names:c.update(x[name]['composition'])
    compositions[arm+str(step)]=dict(c)
  else:
   from collections import Counter
   a=load('analysis/c500_training_audit_20261124/audit.json')['composition'];b=load('analysis/c650_training_audit_20261124/audit.json')['new_composition'];c=load('analysis/c800_training_audit_20261125/audit.json')['new_composition'];x=Counter(a)
   for step,new in [(500,{}),(650,b),(800,c)]:x.update(new);compositions[arm+str(step)]=dict(x)
result={'date':'2026-11-25','checkpoints':[500,650,800],'evaluation':evals,'internal_validation':native,'last150_committed_training_reward':train_summary,'physical_generation_ledger':cost,'cumulative_weight_composition':compositions,'sources_sha256':used,'old_record_sha256':hashlib.sha256((OUT/'总流程记录_截至20261115原稿.md').read_bytes()).hexdigest(),'new_training_or_inference':False}
(OUT/'关键实验数据.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:result[k]for k in ['internal_validation','last150_committed_training_reward','physical_generation_ledger','cumulative_weight_composition']},ensure_ascii=False,indent=2))
