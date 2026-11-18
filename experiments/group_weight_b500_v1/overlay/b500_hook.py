"""Opt-in driver hook; shared site-packages stay byte-for-byte unchanged."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from b500_weighting import make_plan,scaled_advantages

TRAINER_SHA='dce8e3d431218888c9cf65ea8a031091699ac4bd573c7c0a50677a14e75cca59'


def validate_config(config):
    a=config.actor_rollout_ref.actor
    if (config.model_engine!='dp' or config.trainer.n_gpus_per_node!=1 or config.trainer.nnodes!=1
        or config.data.train_batch_size!=2 or a.ppo_mini_batch_size!=2 or a.ppo_epochs!=1 or a.shuffle
        or config.actor_rollout_ref.rollout.n!=4 or a.loss_agg_mode!='token-mean'
        or a.policy_loss.loss_mode!='vanilla' or a.entropy_coeff!=0 or a.use_kl_loss
        or config.algorithm.adv_estimator!='grpo' or config.algorithm.norm_adv_by_std_in_grpo
        or config.algorithm.use_kl_in_reward or not config.shopping_dynamic_sampling.enable
        or config.distillation.enabled):
        raise ValueError('B500 configuration differs from supported frozen Actor contract')


def append_record(path,record):
    if not path.is_absolute():raise ValueError('B500_WEIGHT_LOG must be absolute')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8')as f:
        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush()


def install_on(trainer_class,emit):
    if getattr(trainer_class._update_actor,'_b500_weighted',False):return
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
        metrics.update({'b500/F_groups':sum(g['kind']=='F'for g in record['groups']),
            'b500/actor_tokens':record['original_token_count'],'b500/weighted_actor_tokens':record['weighted_token_count'],
            'b500/coefficient_min':float(coeff.min().item()),'b500/coefficient_max':float(coeff.max().item())})
        return result
    update._b500_weighted=True;trainer_class._update_actor=update


def setup_worker():
    # Preserve the original mandatory memory/padding compatibility hook first.
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback
    install_torch_padding_fallback()
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    source=Path(inspect.getsourcefile(RayPPOTrainer))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=TRAINER_SHA:
        raise ValueError('B500 refuses changed historical trainer source')
    log=os.environ.get('B500_WEIGHT_LOG')
    if not log:raise ValueError('B500_WEIGHT_LOG is required; refusing unlogged updates')
    install_on(RayPPOTrainer,lambda record:append_record(Path(log),record))
