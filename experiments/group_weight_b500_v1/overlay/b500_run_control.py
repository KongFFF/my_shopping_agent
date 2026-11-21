"""Driver-only run control, independent of B's frozen loss implementation."""
import json
import hashlib
import math
import os
import re
import shutil
import zipfile
from pathlib import Path
import time

REQUIRED=('actor/model_world_size_1_rank_0.pt','actor/optim_world_size_1_rank_0.pt',
          'actor/extra_state_world_size_1_rank_0.pt','actor/fsdp_config.json','data.pt')
class PlannedBoundary(Exception):pass
class BudgetStop(Exception):pass
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)


def checkpoint(root,step):
    folder=Path(root)/f'global_step_{step}'
    for n in REQUIRED:
        p=folder/n
        if not p.is_file()or p.stat().st_size==0:raise RuntimeError('Incomplete checkpoint: '+str(p))
    if (Path(root)/'latest_checkpointed_iteration.txt').read_text().strip()!=str(step):raise RuntimeError('Checkpoint tracker mismatch')
    return str(folder)


def checked_folder(root,step,crc=False):
    root=Path(root).resolve();folder=root/f'global_step_{step}'
    if folder.is_symlink()or folder.resolve().parent!=root:raise RuntimeError('Unsafe checkpoint path: '+str(folder))
    for p in [folder,*folder.rglob('*')]:
        if p.is_symlink()or getattr(p,'is_junction',lambda:False)():raise RuntimeError('Checkpoint links forbidden: '+str(p))
    for name in REQUIRED:
        p=folder/name
        if not p.is_file()or not p.stat().st_size:raise RuntimeError('Incomplete checkpoint: '+str(p))
        if p.suffix=='.pt':
            with zipfile.ZipFile(p)as archive:
                if not archive.namelist():raise RuntimeError('Empty checkpoint archive: '+str(p))
                if crc and archive.testzip()is not None:raise RuntimeError('Checkpoint CRC failed: '+str(p))
        elif not isinstance(read(p),dict):raise RuntimeError('Invalid FSDP config: '+str(p))
    return folder


def retain_checkpoints(root,step):
    """Only called after synchronous save; never rotate on a failed save."""
    root=Path(root).resolve();checkpoint(root,step);checked_folder(root,step,crc=True)
    complete=[];unmanaged=[]
    for folder in root.iterdir():
        match=re.fullmatch(r'global_step_(\d+)',folder.name)
        if not match:continue
        n=int(match[1])
        # Refuse ambiguous names; failed/foreign directories are preserved.
        if folder.name!=f'global_step_{n}':continue
        if n>step:raise RuntimeError('Future checkpoint present; resolve rollback directory before retention')
        try:checked_folder(root,n)
        except (OSError,ValueError,RuntimeError,zipfile.BadZipFile):unmanaged.append(n);continue
        complete.append(n)
    keep=set(sorted(complete)[-2:]);keep.add(350)
    # Validate retained fallbacks before deleting any other recovery point.
    for n in keep.intersection(complete):
        if n!=step:checked_folder(root,n,crc=True)
    remove=sorted(set(complete)-keep)
    def log(event,**fields):
        with (root/'checkpoint_retention.jsonl').open('a')as f:
            f.write(json.dumps({'event':event,'new_step':step,**fields})+'\n');f.flush();os.fsync(f.fileno())
    log('retention_plan',keep=sorted(keep.intersection(complete)),remove=remove,unmanaged=unmanaged)
    for n in remove:
        # Recheck the exact absolute deletion target immediately before removal.
        folder=checked_folder(root,n)
        if folder.parent!=root or n==350 or n in keep:raise RuntimeError('Unsafe checkpoint deletion')
        shutil.rmtree(folder)
        log('checkpoint_removed',step=n)


def install_checkpoint_retention(trainer_class,size_floor=0):
    original=trainer_class._save_checkpoint
    if getattr(original,'_b500_retention',False):return
    def save(self):
        cfg=self.config.trainer;root=Path(cfg.default_local_dir).resolve();step=int(self.global_steps)
        if self.config.actor_rollout_ref.actor.checkpoint.async_save:raise RuntimeError('Retention requires synchronous save')
        if cfg.get('remove_previous_ckpt_in_save',False):raise RuntimeError('Legacy pre-save deletion forbidden')
        if (root/f'global_step_{step}').exists():raise RuntimeError('Refusing to overwrite an existing checkpoint')
        # Require space for a whole new save, before touching any old directory.
        sizes=[]
        for folder in root.glob('global_step_*'):
            try:checked_folder(root,int(folder.name.rsplit('_',1)[1]))
            except (OSError,ValueError,RuntimeError,zipfile.BadZipFile):continue
            sizes.append(sum(p.stat().st_size for p in folder.rglob('*')if p.is_file()))
        estimate=max([size_floor,*sizes])
        if estimate and shutil.disk_usage(root).free<int(estimate*1.1)+2**30:
            raise RuntimeError('Insufficient space for next full checkpoint plus margin; no old checkpoint deleted')
        old=cfg.max_actor_ckpt_to_keep;cfg.max_actor_ckpt_to_keep=None
        try:result=original(self)
        finally:cfg.max_actor_ckpt_to_keep=old
        retain_checkpoints(root,step)
        return result
    save._b500_retention=True;trainer_class._save_checkpoint=save


def reserve(path,n,validation,deadline,record_only=False):
    if n<=0:raise ValueError('Empty generation batch')
    if not record_only and time.time()>=deadline-120:raise BudgetStop('Wall budget reached; preserve latest safe checkpoint')
    state=read(path)if Path(path).exists()else {'train':0,'validation_physical':0}
    key='validation_physical'if validation else 'train';limit=2400 if validation else 24000
    if not record_only and state[key]+n>limit:raise BudgetStop('Generation budget exhausted: '+key)
    state[key]+=n;write(path,state)


def install(trainer_class,diagnostics_module,control):
    if getattr(trainer_class.fit,'_b500_controlled',False):return
    install_checkpoint_retention(trainer_class,control.get('checkpoint_size_floor',0))
    original_fit=trainer_class.fit
    def fit(self):
        out=Path(control['output']);budget=out/'generation_budget.json'
        original_generate=self.async_rollout_manager.generate_sequences
        original_append=diagnostics_module.append_training_diagnostic
        def generate(batch):
            validation=batch.meta_info.get('validate')is True
            reserve(budget,len(batch),validation,control['deadline'],control.get('budget_record_only',False))
            result=original_generate(batch)
            tokens=result.batch.get('response_mask')
            with (out/'generation_tokens.jsonl').open('a')as f:
                f.write(json.dumps({'validation':validation,'reserved_trajectories':len(batch),
                    'returned_Actor_tokens':int(tokens.sum().item())if tokens is not None else None,
                    'cumulative_multiturn_prompt_tokens':None,'at':time.time()})+'\n')
            return result
        def append(path,event,step,**kwargs):
            original_append(path,event,step,**kwargs)
            if event!='optimizer_step':return
            m=kwargs['metrics']
            for key in ('actor/lr','actor/grad_norm','actor/loss'):
                if key not in m or not math.isfinite(float(m[key])):raise RuntimeError('Nonfinite/missing '+key)
            if m['training/optimizer_updated']!=1 or float(m['actor/grad_norm'])<=0:raise RuntimeError('No finite nonzero Actor update')
            if float(m.get('b500/actor_tokens',0))<=0:raise RuntimeError('B weighting hook did not report actual Actor tokens')
            if not math.isclose(float(m['actor/lr']),1e-6*min(step/15,1),rel_tol=1e-5):raise RuntimeError('Unexpected LR/warmup after resume')
            if step==control.get('first_new_step',151):
                write(out/f'accepted_step_{step}.json',{'step':step,'lr':float(m['actor/lr']),
                    'grad_norm':float(m['actor/grad_norm']),'diagnostics_written':True,'checkpoint_saved':False})
            if step in (1,350,500,control.get('final_step',500)):
                saved=checkpoint(control['train_dir'],step)
                write(out/f'accepted_step_{step}.json',{'step':step,'checkpoint':saved,'lr':float(m['actor/lr']),
                    'grad_norm':float(m['actor/grad_norm']),'diagnostics_written':True})
            if step==control['stop_at']and step==350:
                write(out/'phase350_complete.json',{'step':350,'checkpoint':checkpoint(control['train_dir'],350)})
                raise PlannedBoundary()
        self.async_rollout_manager.generate_sequences=generate
        diagnostics_module.append_training_diagnostic=append
        try:return original_fit(self)
        except PlannedBoundary:
            self._shutdown_dump_executor();return None
        except BudgetStop as e:
            write(out/'budget_truncated.json',{'reason':str(e),'at':time.time()})
            self._shutdown_dump_executor();return None
        finally:
            self.async_rollout_manager.generate_sequences=original_generate
            diagnostics_module.append_training_diagnostic=original_append
    fit._b500_controlled=True;trainer_class.fit=fit


def setup_worker():
    own=Path(__file__);manifest=json.loads((own.parent.parent/'launch_manifest.json').read_text())
    key='experiments/group_weight_b500_v1/overlay/b500_run_control.py'
    if hashlib.sha256(own.read_bytes()).hexdigest()!=manifest['files_sha256'][key]:raise ValueError('Run control source drift')
    from b500_hook import setup_worker as weighting_setup
    weighting_setup()
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from shopping_grpo.training.grpo import dynamic_sampling
    path=os.environ.get('B500_CONTROL_FILE')
    if not path:raise ValueError('Missing run control; training disabled')
    install(RayPPOTrainer,dynamic_sampling,read(path))
