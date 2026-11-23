"""CPU loss/gradient tests against the frozen original veRL vanilla PPO function."""
import ast
from collections import Counter
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import types
import zipfile
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/group_weight_c500_v1/overlay'))
from c500_weighting import make_plan,scaled_advantages
from c500_hook import install_on


class Attr(dict):
    __getattr__=dict.__getitem__
def attrs(x):return Attr({k:attrs(v)for k,v in x.items()})if isinstance(x,dict)else x


def original_loss():
    source=ROOT/'analysis/grpo_mechanism_audit_20261116/package_reference/verl/trainer/ppo/core_algos.py'
    manifest=json.loads((source.parents[3]/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['files_sha256']['verl/trainer/ppo/core_algos.py']
    names={'agg_loss','compute_policy_loss_vanilla'}
    nodes=[n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert len(nodes)==2
    for n in nodes:n.decorator_list=[]
    module=ast.parse('from __future__ import annotations');module.body+=nodes
    helper=SimpleNamespace(masked_sum=lambda x,m:(x*m).sum(),masked_mean=lambda x,m:(x*m).sum()/m.sum())
    scope={'torch':torch,'verl_F':helper,'AlgoConfig':type('UnusedAlgoConfig',(),{})}
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),scope)
    return scope['compute_policy_loss_vanilla']


PPO=original_loss()
def config(tokens):return Attr(clip_ratio=.2,clip_ratio_low=.2,clip_ratio_high=.2,clip_ratio_c=3.,global_batch_info={'batch_num_tokens':tokens,'dp_size':1})


def fixture(kinds='FS'):
    torch.manual_seed(42)
    mask=torch.tensor([[1,1,0,0,0],[1,1,1,0,0],[1,0,0,0,0],[1,1,1,1,0],
                       [1,1,1,1,1],[1,1,1,1,0],[1,1,1,0,0],[1,1,0,0,0]],dtype=torch.float64)
    infos=[];uids=[];advantages=[]
    for g,kind in enumerate(kinds):
        rr=[-.65,-.3,-.85,-.5]if kind=='F'else [1.,-.3,-.65,-.85]
        for j,r in enumerate(rr):
            uids.append('group-'+str(g));advantages.append([r-sum(rr)/4]*5)
            infos.append({'task_id':g,'reward_version':'shopsimulator-reward-v3','reward_valid':True,
                'reward_type':'gold_purchase'if r==1 else 'partial_alternative_purchase',
                'reward':{'purchase_success':r==1,'terminal_utility':r,'sampling_invalid':False}})
    return uids,infos,mask,torch.tensor(advantages,dtype=torch.float64)


def direct_losses(old,new,a):
    ratio=(new-old).clamp(-20,20).exp()
    standard=torch.maximum(-a*ratio,-a*ratio.clamp(.8,1.2))
    return torch.where(a<0,torch.minimum(-a*3.,standard),standard)


class WeightTests(unittest.TestCase):
    def runtime_bypass(self):
        if __import__('importlib.util',fromlist=['find_spec']).find_spec('omegaconf')is None:
            sys.path.insert(0,str(ROOT/'outputs/difficulty_scheduler_v1/cpu_dependencies'))
        from omegaconf import OmegaConf
        cfg=OmegaConf.create(json.loads((ROOT/'experiments/group_weight_c500_v1/historical_a_composed_config.json').read_text()))
        source=ROOT/'experiments/group_weight_c500_v1/reference/rollout_corr_helper.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),'b5b09c8361d587e7f8078fb873766abc2cbf7ce231b9c449450928413ed11ff4')
        names={'apply_bypass_mode','compute_rollout_correction_and_rejection_mask'}
        nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names]
        module=ast.parse('from __future__ import annotations');module.body+=nodes
        scope={'torch':torch,'compute_offpolicy_metrics':lambda **kw:{}}
        exec(compile(ast.fix_missing_locations(module),str(source),'exec'),scope)
        batch=SimpleNamespace(batch={'rollout_log_probs':torch.zeros(8,5,dtype=torch.float64)})
        scope['apply_bypass_mode'](batch,cfg.algorithm.rollout_correction,cfg.actor_rollout_ref.actor.policy_loss)
        return cfg,scope
    def test_real_apply_bypass_runtime_config_accepted(self):
        from c500_hook import validate_config
        cfg,_=self.runtime_bypass();self.assertEqual(cfg.actor_rollout_ref.actor.policy_loss.loss_mode,'bypass_mode');validate_config(cfg)
    def test_unexpected_runtime_reinforce_or_rejection_rejected(self):
        from c500_hook import validate_config
        for key,value in [('loss_type','reinforce'),('rollout_rs','token'),('rollout_is','token')]:
            cfg,_=self.runtime_bypass();cfg.actor_rollout_ref.actor.policy_loss.rollout_correction[key]=value
            with self.assertRaisesRegex(ValueError,key):validate_config(cfg)
    def test_real_bypass_PPO_dispatch_loss_and_gradient(self):
        cfg,helpers=self.runtime_bypass()
        source=ROOT/'analysis/grpo_mechanism_audit_20261116/package_reference/verl/trainer/ppo/core_algos.py'
        fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='compute_policy_loss_bypass_mode');fn.decorator_list=[]
        module=ast.parse('from __future__ import annotations');module.body.append(fn)
        scope=PPO.__globals__;exec(compile(ast.fix_missing_locations(module),str(source),'exec'),scope)
        mod=types.ModuleType('verl.trainer.ppo.rollout_corr_helper');mod.compute_rollout_correction_and_rejection_mask=helpers['compute_rollout_correction_and_rejection_mask']
        u,s,m,a=fixture();c,_=make_plan(u,s,m);old=torch.zeros_like(a);x=torch.randn_like(a).requires_grad_()
        actor=config(m.sum());actor.policy_loss=cfg.actor_rollout_ref.actor.policy_loss
        with patch.dict(sys.modules,{'verl.trainer.ppo.rollout_corr_helper':mod}):
            actual,_=scope['compute_policy_loss_bypass_mode'](old,x,scaled_advantages(a,c),m,config=actor)
        expected,_=PPO(old,x,scaled_advantages(a,c),m,config=actor)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])
    def check_gradient(self,kinds='FS',alpha=.25):
        uid,info,mask,a=fixture(kinds);coeff,plan=make_plan(uid,info,mask,alpha)
        old=torch.randn(8,5,dtype=torch.float64)
        # Include upper/lower clipping and the dual-clip negative-advantage branch.
        delta=torch.tensor([-.9,-.1,.1,.7,1.7],dtype=torch.float64).repeat(8,1)
        new=(old+delta).requires_grad_()
        actual,_=PPO(old,new,scaled_advantages(a,coeff),mask,config=config(mask.sum()))
        raw=torch.tensor([r['raw_weight']for r in plan['trajectories']],dtype=torch.float64)[:,None]
        expected=(direct_losses(old,new,a)*raw*mask).sum()/(raw*mask).sum()
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        g1=torch.autograd.grad(actual,new,retain_graph=True)[0];g2=torch.autograd.grad(expected,new)[0]
        torch.testing.assert_close(g1,g2,rtol=1e-12,atol=1e-12)
        self.assertTrue(torch.equal(g1[mask==0],torch.zeros_like(g1[mask==0])))
        return coeff

    def test_C_default_gamma4_B_gamma2_and_A_gamma1(self):
        from c500_weighting import GAMMA
        self.assertEqual(GAMMA,4.0)
        u,s,m,a=fixture();c,plan=make_plan(u,s,m)
        self.assertEqual(plan['gamma'],4.0)
        self.assertEqual([g['raw_weight']for g in plan['groups']],[.25,1.])
        for gamma in (1.,2.,4.):
            coeff,record=make_plan(u,s,m,fail_weight=1./gamma)
            self.assertAlmostEqual(coeff[4].item()/coeff[0].item(),gamma)
        equal=torch.ones_like(m);coeff,_=make_plan(u,s,equal)
        torch.testing.assert_close(coeff,torch.tensor([.4]*4+[1.6]*4,dtype=torch.float64))

    def test_alpha1_recovers_baseline(self):self.assertTrue(torch.equal(self.check_gradient(alpha=1),torch.ones(8,dtype=torch.float64)))
    def test_FF_cancels(self):self.assertTrue(torch.equal(self.check_gradient('FF'),torch.ones(8,dtype=torch.float64)))
    def test_SS_cancels(self):self.assertTrue(torch.equal(self.check_gradient('SS'),torch.ones(8,dtype=torch.float64)))
    def test_FS_unequal_tokens_dual_clip_gradient(self):
        c=self.check_gradient();self.assertAlmostEqual(c[0].item()/c[4].item(),.25)
        self.assertNotAlmostEqual(c[0].item(),2/3)
    def test_uid_permutation(self):
        u,s,m,a=fixture();c,_=make_plan(u,s,m);p=[5,0,6,3,2,7,4,1]
        cp,_=make_plan([u[i]for i in p],[s[i]for i in p],m[p]);torch.testing.assert_close(cp,c[p])
    def test_masked_values_do_not_change_loss(self):
        u,s,m,a=fixture();c,_=make_plan(u,s,m);old=torch.zeros_like(a);x=torch.zeros_like(a);x[m==0]=12
        l,_=PPO(old,x,scaled_advantages(a,c),m,config=config(m.sum()))
        zero,_=PPO(old,old,scaled_advantages(a,c),m,config=config(m.sum()));torch.testing.assert_close(l,zero)
    def test_microbatch_accumulation(self):
        u,s,m,a=fixture();c,_=make_plan(u,s,m);old=torch.zeros_like(a);x=torch.randn_like(a).requires_grad_()
        full,_=PPO(old,x,scaled_advantages(a,c),m,config=config(m.sum()))
        for chunk in (1,2,3,8):
            parts=[PPO(old[i:i+chunk],x[i:i+chunk],scaled_advantages(a,c)[i:i+chunk],m[i:i+chunk],config=config(m.sum()))[0]for i in range(0,8,chunk)]
            torch.testing.assert_close(sum(parts),full)
            torch.testing.assert_close(torch.autograd.grad(sum(parts),x,retain_graph=True)[0],torch.autograd.grad(full,x,retain_graph=True)[0])
    def test_invalid_mask_or_labels_rejected(self):
        u,s,m,a=fixture()
        for bad in (torch.zeros_like(m),m*.5,m[:4]):
            with self.assertRaises(ValueError):make_plan(u,s,bad)
        for update in ({'reward_valid':False},{'reward_unverifiable':True},{'infrastructure_invalid':True}):
            bad=copy.deepcopy(s);bad[0].update(update)
            with self.assertRaises(ValueError):make_plan(u,bad,m)
        bad=copy.deepcopy(s);bad[0]['task_id']=999
        with self.assertRaises(ValueError):make_plan(u,bad,m)
        bad=copy.deepcopy(s);bad[0]['reward']['purchase_success']=float('nan')
        with self.assertRaises(ValueError):make_plan(u,bad,m)
        for bad_u in (['same']*8,u[:7],['']+u[1:]):
            with self.assertRaises(ValueError):make_plan(bad_u,s,m)
        for alpha in (0,-1,float('nan'),2):
            with self.assertRaises(ValueError):make_plan(u,s,m,alpha)
    def test_valid_alternative_counts_as_success(self):
        u,s,m,a=fixture('FF');s[4]['reward']['purchase_success']=True;s[4]['reward_type']='valid_alternative_purchase'
        _,p=make_plan(u,s,m);self.assertEqual(p['composition'],'FS')
    def test_hook_restores_original_advantages(self):
        from c500_hook import validate_config
        cfg=attrs(json.loads((ROOT/'experiments/group_weight_c500_v1/historical_a_composed_config.json').read_text()))
        validate_config(cfg);u,s,m,a=fixture();seen=[]
        class Trainer:
            config=cfg;global_steps=0
            def _update_actor(self,b):
                seen.append(b.batch['advantages']);return SimpleNamespace(meta_info={'metrics':{}})
        records=[];install_on(Trainer,records.append);install_on(Trainer,records.append)
        batch=SimpleNamespace(batch={'advantages':a,'response_mask':m},non_tensor_batch={'uid':u,'shopping':s})
        Trainer()._update_actor(batch);self.assertIs(batch.batch['advantages'],a);self.assertEqual(len(seen),1)
        self.assertEqual([r['event']for r in records],['actor_update_started','actor_update_returned'])
        self.assertFalse(torch.equal(seen[0],a))
    def test_hook_restores_after_exception(self):
        cfg=attrs(json.loads((ROOT/'experiments/group_weight_c500_v1/historical_a_composed_config.json').read_text()))
        class Trainer:
            config=cfg;global_steps=0
            def _update_actor(self,b):raise RuntimeError('test failure')
        records=[];install_on(Trainer,records.append);u,s,m,a=fixture()
        batch=SimpleNamespace(batch={'advantages':a,'response_mask':m},non_tensor_batch={'uid':u,'shopping':s})
        with self.assertRaises(RuntimeError):Trainer()._update_actor(batch)
        self.assertIs(batch.batch['advantages'],a);self.assertEqual(records[-1]['event'],'actor_update_failed')
    def test_optimizer_scheduler_rng_350_to500_resume(self):
        u,s,m,a=fixture();c,_=make_plan(u,s,m)
        def fresh():
            p=torch.nn.Parameter(torch.zeros_like(a));o=torch.optim.AdamW([p],lr=1e-6)
            scheduler=torch.optim.lr_scheduler.LambdaLR(o,lambda step:min((step+1)/15,1))
            return p,o,scheduler,torch.Generator().manual_seed(42)
        def step(p,o,sched,gen):
            noise=torch.randn(a.shape,generator=gen,dtype=a.dtype)*.1
            o.zero_grad();loss,_=PPO(torch.zeros_like(a),p+noise,scaled_advantages(a,c),m,config=config(m.sum()));loss.backward();o.step();sched.step()
        p,o,sch,gen=fresh()
        for _ in range(350):step(p,o,sch,gen)
        b=io.BytesIO();torch.save({'p':p.detach(),'o':o.state_dict(),'s':sch.state_dict(),'rng':gen.get_state()},b)
        for _ in range(150):step(p,o,sch,gen)
        b.seek(0);r=torch.load(b,weights_only=False);q,oq,sq,gq=fresh();q.data.copy_(r['p']);oq.load_state_dict(r['o']);sq.load_state_dict(r['s']);gq.set_state(r['rng'])
        self.assertEqual(sq.get_last_lr(),[1e-6])
        for _ in range(150):step(q,oq,sq,gq)
        torch.testing.assert_close(p,q,rtol=0,atol=0);self.assertEqual(sch.get_last_lr(),sq.get_last_lr())
        self.assertTrue(torch.equal(gen.get_state(),gq.get_state()))
    def test_historical_group_labels_under_real_uid_order(self):
        counts=Counter();archive=ROOT/'analysis/online_training_window_return_20261116/evidence.zip'
        with zipfile.ZipFile(archive)as z:
            for window in ('reference351_500','late651_800'):
                for e in map(json.loads,z.read(window+'/training_diagnostics.jsonl').splitlines()):
                    if e['event']!='optimizer_step':continue
                    rs=e['rollouts'];_,p=make_plan([r['uid']for r in rs],rs,torch.ones(8,1))
                    counts[p['composition']]+=1
        # Unit-token masks test labels only, not unavailable historical token denominators.
        self.assertEqual(counts,Counter(FF=33,FS=119,SS=148))


if __name__=='__main__':unittest.main()
