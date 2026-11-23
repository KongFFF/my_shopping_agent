"""Opt-in driver hook; shared site-packages stay byte-for-byte unchanged."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from c500_weighting import make_plan,scaled_advantages

TRAINER_SHA='dce8e3d431218888c9cf65ea8a031091699ac4bd573c7c0a50677a14e75cca59'


def validate_config(config):
    a=config.actor_rollout_ref.actor
    checks={'model_engine':(config.model_engine,'dp'),'GPU_count':(config.trainer.n_gpus_per_node,1),
        'nodes':(config.trainer.nnodes,1),'train_batch_size':(config.data.train_batch_size,2),
        'ppo_mini_batch_size':(a.ppo_mini_batch_size,2),'ppo_epochs':(a.ppo_epochs,1),'shuffle':(a.shuffle,False),
        'rollout_n':(config.actor_rollout_ref.rollout.n,4),'loss_agg_mode':(a.loss_agg_mode,'token-mean'),
        'entropy_coeff':(a.entropy_coeff,0),'use_kl_loss':(a.use_kl_loss,False),
        'adv_estimator':(config.algorithm.adv_estimator,'grpo'),
        'norm_adv_by_std':(config.algorithm.norm_adv_by_std_in_grpo,False),
        'use_kl_in_reward':(config.algorithm.use_kl_in_reward,False),
        'dynamic_sampling':(config.shopping_dynamic_sampling.enable,True),'distillation':(config.distillation.enabled,False)}
    errors={k:{'actual':x,'expected':y}for k,(x,y)in checks.items()if x!=y}
    # Pinned trainer applies apply_bypass_mode before _update_actor. Keep the
    # original PPO-clip path; do not disable bypass or permit REINFORCE/RS/IS.
    expected={'bypass_mode':True,'loss_type':'ppo_clip','rollout_is':None,'rollout_rs':None,
        'rollout_is_threshold':2.0,'rollout_rs_threshold':None,'rollout_is_batch_normalize':False}
    rc=config.algorithm.rollout_correction
    for k,v in expected.items():
        if rc.get(k)!=v:errors['algorithm.rollout_correction.'+k]={'actual':rc.get(k),'expected':v}
    mode=a.policy_loss.loss_mode
    if mode not in ('vanilla','bypass_mode'):errors['loss_mode']={'actual':mode,'expected':'vanilla -> bypass_mode/ppo_clip'}
    if mode=='bypass_mode':
        active=a.policy_loss.get('rollout_correction')
        if active is None:errors['actor.rollout_correction']='missing runtime bypass configuration'
        else:
            for k,v in expected.items():
                if active.get(k)!=v:errors['actor.rollout_correction.'+k]={'actual':active.get(k),'expected':v}
    if errors:raise ValueError('C500 runtime configuration differences: '+json.dumps(errors,default=str,sort_keys=True))


def append_record(path,record):
    if not path.is_absolute():raise ValueError('C500_WEIGHT_LOG must be absolute')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8')as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()


def install_on(trainer_class,emit):
    if getattr(trainer_class._update_actor,'_c500_weighted',False):return
    original=trainer_class._update_actor
    @functools.wraps(original)
    def update(self,batch):
        validate_config(self.config)
        coeff,record=make_plan(list(batch.non_tensor_batch['uid']),list(batch.non_tensor_batch['shopping']),batch.batch['response_mask'])
        old=batch.batch['advantages']
        record.update(driver_global_step_before_update=int(self.global_steps),at=time.time())
        for i,row in enumerate(record['trajectories']):
            active=old[i][batch.batch['response_mask'][i].bool()]
            row['original_advantage_mean']=float(active.double().mean().item())
            row['original_positive_tokens']=int((active>0).sum().item())
            row['original_negative_tokens']=int((active<0).sum().item())
        weighted=scaled_advantages(old,coeff)
        emit({**record,'event':'actor_update_started'})
        batch.batch['advantages']=weighted
        try:result=original(self,batch)
        except Exception:
            emit({**record,'event':'actor_update_failed'});raise
        finally:batch.batch['advantages']=old
        emit({**record,'event':'actor_update_returned'})
        metrics=result.meta_info['metrics']
        metrics.update({'c500/F_groups':sum(g['kind']=='F'for g in record['groups']),
            'c500/actor_tokens':record['original_token_count'],'c500/weighted_actor_tokens':record['weighted_token_count'],
            'c500/coefficient_min':float(coeff.min().item()),'c500/coefficient_max':float(coeff.max().item())})
        return result
    update._c500_weighted=True;trainer_class._update_actor=update


def setup_worker():
    # Preserve the original mandatory memory/padding compatibility hook first.
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback
    install_torch_padding_fallback()
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    source=Path(inspect.getsourcefile(RayPPOTrainer))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=TRAINER_SHA:
        raise ValueError('C500 refuses changed historical trainer source')
    log=os.environ.get('C500_WEIGHT_LOG')
    if not log:raise ValueError('C500_WEIGHT_LOG is required; refusing unlogged updates')
    install_on(RayPPOTrainer,lambda record:append_record(Path(log),record))
