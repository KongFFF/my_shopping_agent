"""C500 global-token group weights. No environment, sampler or reward changes."""
from collections import defaultdict
import math
import torch


GAMMA = 4.0  # frozen C arm: S/F; S raw weight stays one


def make_plan(uids,shopping,response_mask,fail_weight=1.0/GAMMA):
    if not math.isfinite(fail_weight) or not 0 < fail_weight <= 1:
        raise ValueError('Invalid F weight')
    if response_mask.ndim!=2 or response_mask.shape[0]!=8 or len(uids)!=8 or len(shopping)!=8:
        raise ValueError('C500 requires one global minibatch of8 trajectories')
    if not torch.isfinite(response_mask).all() or not ((response_mask==0)|(response_mask==1)).all():
        raise ValueError('Actor response mask must be finite binary')
    counts=response_mask.detach().sum(dim=1,dtype=torch.float64)
    if not (counts>0).all():raise ValueError('Every trajectory must have Actor tokens')
    groups=defaultdict(list)
    for i,(uid,info)in enumerate(zip(uids,shopping)):
        if not isinstance(uid,str)or not uid:raise ValueError('Missing stable UID')
        r=info.get('reward',{})
        if info.get('reward_valid')is not True or info.get('reward_version')!='shopsimulator-reward-v3':
            raise ValueError('Invalid or changed reward contract')
        if any(info.get(k)for k in ('infrastructure_invalid','reward_unverifiable','overlong'))or r.get('sampling_invalid'):
            raise ValueError('Invalid rollout reached Actor update')
        success=r.get('purchase_success')
        if not isinstance(success,(bool,int,float))or success not in (0,1):raise ValueError('Missing/invalid purchase_success')
        if not math.isfinite(float(r.get('terminal_utility',float('nan')))):raise ValueError('Invalid terminal utility')
        groups[uid].append(i)
    if len(groups)!=2 or any(len(ix)!=4 for ix in groups.values()):raise ValueError('Expected two UID groups of4')
    raw=torch.ones(8,dtype=torch.float64,device=response_mask.device);rows=[]
    for uid,ix in groups.items():
        if any(shopping[i].get('task_id')is None for i in ix):raise ValueError('Missing task identity')
        tasks={int(shopping[i]['task_id'])for i in ix}
        if len(tasks)!=1:raise ValueError('UID crosses task identities')
        nsuccess=sum(bool(shopping[i]['reward']['purchase_success'])for i in ix)
        w=1. if nsuccess else fail_weight
        raw[ix]=w
        rows.append({'uid':uid,'task_id':next(iter(tasks)),'kind':'S'if nsuccess else 'F','successes':nsuccess,
                     'raw_weight':w,'effective_tokens':int(counts[ix].sum().item()),'indices':ix})
    original_tokens=counts.sum();weighted_tokens=(counts*raw).sum()
    coeff=raw*original_tokens/weighted_tokens
    if not torch.isfinite(coeff).all()or not (coeff>0).all():raise ValueError('Invalid weight denominator')
    for row in rows:row['coefficient']=float(coeff[row['indices'][0]].item())
    return coeff,{'gamma':1.0/fail_weight,'experiment':'C500','original_token_count':int(original_tokens.item()),'weighted_token_count':float(weighted_tokens.item()),
        'composition':''.join(sorted(row['kind']for row in rows)),'groups':rows,
        'trajectories':[{'uid':uid,'task_id':int(info['task_id']),'effective_tokens':int(counts[i].item()),
            'raw_weight':float(raw[i].item()),'coefficient':float(coeff[i].item()),
            'reward_type':info.get('reward_type'),'success':bool(info['reward']['purchase_success']),
            'terminal_utility':float(info['reward']['terminal_utility']),'tool_steps':int(info.get('steps',0))}
            for i,(uid,info)in enumerate(zip(uids,shopping))]}


def scaled_advantages(advantages,coeff):
    if advantages.ndim!=2 or advantages.shape[0]!=len(coeff)or not torch.isfinite(advantages).all():
        raise ValueError('Invalid original advantages')
    # Positive scaling commutes with both standard and dual PPO clipping.
    return advantages*coeff.to(device=advantages.device,dtype=advantages.dtype).unsqueeze(1)
