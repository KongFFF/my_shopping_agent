"""Audit historical A locally and freeze the B500 specification; never train."""
from pathlib import Path
import csv
import hashlib
import json
import math
import zipfile
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/b500_matching_20261117'
EXP=ROOT/'experiments/group_weight_b500_v1'
BASE=ROOT/'analysis/grpo_baseline_20261022_202816'
UP=ROOT/'upstream/shopping-grpo-longhorizon'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def differences(a,b,path=''):
    if isinstance(a,dict)and isinstance(b,dict):
        return [r for k in sorted(set(a)|set(b))for r in differences(a.get(k),b.get(k),path+'.'+k)]
    return []if a==b else [{'field':path,'before':a,'after':b}]


def main():
    OUT.mkdir(parents=True,exist_ok=True);EXP.mkdir(parents=True,exist_ok=True)
    integrity={}
    for n,meta in read(BASE/'manifest.json')['files'].items():
        raw=(BASE/n).read_bytes();status='exact'if sha(raw)==meta['sha256']else 'CRLF_only'if sha(raw.replace(b'\r\n',b'\n'))==meta['sha256']else 'mismatch'
        assert status!='mismatch',n
        integrity[n]=status
    first=yaml.safe_load((BASE/'configs/formal_memfix_v1.yaml').read_text(encoding='utf-8'))
    resume=yaml.safe_load((BASE/'configs/formal_resume350_v1.yaml').read_text(encoding='utf-8'))
    diff=differences(first,resume)
    assert {r['field']for r in diff}=={'.actor_rollout_ref.rollout.trace.experiment_name','.trainer.default_local_dir',
        '.trainer.experiment_name','.trainer.resume_from_path','.trainer.resume_mode','.trainer.val_before_train','.trainer.validation_data_dir'}
    updates=list(csv.DictReader((BASE/'training_diagnosis/updates.csv').open(encoding='utf-8-sig')))
    assert [int(r['step'])for r in updates]==list(range(1,501))
    for r in updates:
        step=int(r['step']);assert int(float(r['training/optimizer_updated']))==1
        assert math.isclose(float(r['actor/lr']),1e-6*min(step/15,1),rel_tol=1e-9)
    archive=ROOT/'analysis/online_training_window_return_20261116/evidence.zip'
    assert sha(archive.read_bytes())=='01703af869f5fef97e509c0538b0e3c58910b3508b6298b6c2dccd5396416362'
    with zipfile.ZipFile(archive)as z:
        binding=json.loads(z.read('inputs/baseline_data_binding.json'))
        env=json.loads(z.read('inputs/environment.json'))
        launch=json.loads(z.read('reference351_500/launch_binding.json'))
        archived_ids={json.loads(l)['task_id']for l in z.read('inputs/train.jsonl').splitlines()}
    data_check={}
    for n,h in binding['data_sha256'].items():
        raw=(UP/n).read_bytes();status='exact'if sha(raw)==h else 'CRLF_only'if sha(raw.replace(b'\r\n',b'\n'))==h else 'mismatch'
        assert status!='mismatch',n
        data_check[n]={'sha256':h,'local_integrity':status}
    ids=lambda n:{json.loads(l)['task_id']for l in (UP/n).read_text(encoding='utf-8-sig').splitlines()}
    train=ids('data/grpo/train.jsonl');dev=ids('data/grpo/validation.jsonl');final=ids('data/evaluation/tasks.jsonl')
    assert len(train)==1000 and len(dev)==50 and len(final)==200 and train==archived_ids
    assert not(train&dev or train&final or dev&final)
    # Identity metadata only. Neither source's trajectories/outcomes are read.
    receipt=ROOT/'outputs/difficulty_scheduler_v1/return_20261113/run/run.json'
    sft=read(receipt)
    other_path=ROOT/'outputs/checkpoint_pair_v1/run_01/run.json'
    other_receipt=read(other_path);other=other_receipt['models']['sft']
    assert sft['sft_files']==other['files'] and sft['sft']==other['path']==first['actor_rollout_ref']['model']['path']
    evidence=read(BASE/'derived/evidence.json');a=read(BASE/'grpo_dev50/summary.json');s=read(BASE/'sft_dev50/summary.json')
    assert a['strict_successes']==27 and s['strict_successes']==24
    pa=dict(a['protocol']);ps=dict(s['protocol']);pa.pop('model');ps.pop('model');assert pa==ps
    actor=first['actor_rollout_ref']['actor'];model=first['actor_rollout_ref']['model']
    assert actor['optim']['lr']==1e-6 and actor['loss_agg_mode']=='token-mean'
    assert first['data']['seed']==actor['data_loader_seed']==actor['fsdp_config']['seed']==42
    assert first['data']['train_batch_size']==2 and first['actor_rollout_ref']['rollout']['n']==4
    assert actor['ppo_epochs']==1 and actor['entropy_coeff']==0 and not actor['use_kl_loss']
    assert model['lora_rank']==16 and model['lora_alpha']==32
    assert first['algorithm']['norm_adv_by_std_in_grpo']is False
    costs=evidence['training_summary']['counts'];scope=evidence['training_summary']['generation_cost_scope']
    audit={'status':'historical_A_usable_with_documented_limits; cloud_identity_gate_pending',
        'integrity':integrity,'config_differences':diff,'data':data_check,
        'task_boundaries':{'train':len(train),'dev':len(dev),'final_ids_only':len(final),'disjoint':True},
        'effective_updates':500,'warmup_updates':15,'lr_trace_verified':True,
        'A500':{'checkpoint':'formal_resume350_v1/global_step_500','independent_dev_successes':27,
            'training_internal_dev_gold_at_500':29,'sft_independent_dev_successes':24},
        'costs':scope,'counts':costs,'retained_update_timing_hours_lower_bound':sum(float(r['timing_s/step'])for r in updates)/3600,
        'second_identity_receipt':{'source':str(other_path.relative_to(ROOT)),'source_sha256':sha(other_path.read_bytes())},
        'A500_later_export_identity':other_receipt['models']['grpo'],
        'sft_receipt':{'source':str(receipt.relative_to(ROOT)),'source_sha256':sha(receipt.read_bytes()),
            'path':sft['sft'],'files':sft['sft_files'],'independent_receipt_agrees':True},
        'limits':['Original October baseline_plan and real weights are cloud-only: compare before launch.',
            'Original0-350 complete per-token/runtime state not independently archived locally.',
            'Historical restart/async sampling and one training seed preclude exact paired randomness.',
            'Independent Dev50 is old development material; Final has been viewed repeatedly.']}
    save(OUT/'historical_a_audit.json',audit)
    spec={'version':'group-weight-b500-v1','state':'design_frozen_not_execution_ready','method':{
        'F_definition':'all four valid trajectories have purchase_success=false',
        'success_predicate':'reward_valid && reward_detail.purchase_success; includes native valid_alternative_purchase',
        'F_weight':0.5,'S_weight':1.0,'normalization':'global_actor_effective_token_weighted_mean',
        'formula':'L_B = sum_t(w_group(t) * mask_t * clipped_PPO_loss_t) / sum_t(w_group(t) * mask_t)',
        'apply_after_dynamic_filter':True,'group_identity':'uid; preserved through length balancing',
        'all_other_losses':'unchanged; baseline entropy_coeff=0, KL disabled'},
        'baseline':{'commit':'a583dba4c149be09622ce58ad6f7a160db77bb41','run':'grpo_baseline_20261022_202816',
            'primary_endpoint':500,'actor_init':sft['sft'],'sft_files':sft['sft_files'],
            'data_sha256':binding['data_sha256'],'environment':env,'reference_launch_binding':launch},
        'training':{'target_effective_updates':500,'fresh_GRPO_LoRA':True,'lora_rank':model['lora_rank'],'lora_alpha':model['lora_alpha'],
            'optimizer':actor['optim'],'warmup_effective_updates':15,'lr_after_warmup':1e-6,
            'total_epochs':50,'train_batch_size':2,'rollouts_per_task':4,'actor_minibatch_trajectories':8,'ppo_epochs':1,
            'seed_data':42,'seed_actor_loader':42,'seed_fsdp':42,'pythonhashseed':42,
            'vllm_seed':'inherit original exact package/config default; no new seed override',
            'planned_full_resume_at':350,'save_every':50,'early_save_steps':[1,10,25],
            'restore':'model, optimizer, scheduler, extra/RNG and dataloader state; no adapter reinitialization',
            'sampling':{'temperature':0.7,'top_p':0.9,'top_k':-1,'max_model_len':24576,'context_input_budget':16384,
                'max_steps':35,'per_turn_max_tokens':512,'context_compaction':False},
            'dynamic_sampling':first['shopping_dynamic_sampling']},
        'evaluation':{'primary':'fixed B500 independent Dev50 vs archived A500 independent Dev50',
            'A500_successes':27,'sft_successes':24,'protocol':pa,'strict_success':'done && reward_valid && reward_type=gold_purchase',
            'internal_validation_steps':[0]+list(range(50,501,50)),
            'selection':'fixed step500 primary; no best-of-checkpoints substitution',
            'final200_execution':False,'continuation_to800':'not authorized or automatic',
            'continuation_screen':{'net_dev_gain_at_least':3,'minimum_B_successes':30,
                'interpretation':'screening only, not statistical significance; review regressions/costs; no automatic800'}},
        'caps':{'training_generated_trajectories_including_invalid_skipped_rollback':24000,
            'internal_validation_episodes':550,'independent_dev_episodes':50,'smoke_episodes_max':3,
            'allocated_GPU_wall_hours_all_stages':20,'timeout_termination':'save latest safe checkpoint, mark budget_truncated; never call it B500',
            'tokens':'count all sampling prompt/completion, Actor effective and rollout context separately; no invented matched historical token budget'},
        'launch_gates':['cloud historical baseline_plan SFT and data hashes pass',
            'frozen baseline runtime/assets/dependencies verified; B overlay only declared change',
            'B implementation tests: alpha1 equivalence, FF/SS cancellation, FS token weighting, masks/padding, microbatch invariance, uid shuffle',
            'first B update checkpoint and optimizer/weight metrics verified before continuing',
            'no active competing experiment, enough disk/memory and complete return/export plan'],
        'claims':'single-seed historically matched experiment; same500 updates does not imply same generation cost'}
    save(EXP/'protocol.json',spec);save(EXP/'historical_a_composed_config.json',first)
    save(EXP/'sft_identity_receipt.json',audit['sft_receipt'])
    files=['protocol.json','historical_a_composed_config.json','sft_identity_receipt.json']
    if (EXP/'README.md').exists():files.append('README.md')
    save(EXP/'manifest.json',{'files_sha256':{n:sha((EXP/n).read_bytes())for n in files},
        'audit_sha256':sha((OUT/'historical_a_audit.json').read_bytes()),'audit_script_sha256':sha(Path(__file__).read_bytes()),
        'tools_sha256':{'check_b500_history.py':sha((ROOT/'tools/check_b500_history.py').read_bytes())}})
    print(json.dumps({'status':audit['status'],'SFT_model_sha256':sft['sft_files']['model.safetensors'],
        'protocol_sha256':sha((EXP/'protocol.json').read_bytes()),'A500_Dev':27,'B_target_updates':500},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
