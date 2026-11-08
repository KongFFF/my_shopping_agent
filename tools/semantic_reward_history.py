"""Evidence-aware offline reward: decisive failures, exact scores or bounds.

Unknown evidence is never silently converted to pass/fail or dropped from an
average. Bounds are diagnostic and cannot be fed to training as scalar rewards.
"""
from semantic_contract_v2_r7_final12 import DIMENSIONS, validate, lint, split_flags
from semantic_product_verifier_v4 import VERSION as VERIFIER_VERSION, contract_fingerprint

VERSION='semantic-reward-history-v2-partial-evidence'


def overlaps(a,b):return a['start']<b['end'] and b['start']<a['end']


def locally_reviewed(contract,d,index,r,findings):
    if r['review_status']!='approved' or r['importance']=='uncertain':return False
    if any(overlaps(p['evidence'],e) for p in contract['pending_mentions'] for e in r['evidence']):return False
    for f in findings:
        if f.get('location')==f'/{d}/{index}':return False
        if f.get('location')=='/query':
            source=f.get('source')
            if not source or not {'start','end'}<=set(source) or any(overlaps(source,e) for e in r['evidence']):return False
    return True


def score_verified(contract,verification,*,source_query):
    # Keep schema/source/contradiction checks. An unrelated pending item cannot
    # change a locally reviewed, proven category/explicit-budget failure.
    validate(contract,source_query=source_query)
    if verification.get('version')!=VERIFIER_VERSION or verification.get('contract_sha256')!=contract_fingerprint(contract):
        raise ValueError('Verification is not bound to this exact contract')
    reqs=[(d,i,r) for d in DIMENSIONS for i,r in enumerate(contract[d])]
    statuses=verification['statuses']
    if set(statuses)!={r['id'] for _,_,r in reqs} or any(s not in ('pass','fail','unknown') for s in statuses.values()):
        raise ValueError('Exactly one verified status per requirement required')
    if not any(r['importance']=='required' for _,_,r in reqs):raise ValueError('Nonempty required objective needed')
    base={'reward':None,'reward_valid':False,'semantic_success':None,'version':VERSION,
          'reward_interval':None,'interval_is_training_reward':False}
    if not verification['execution_complete'] or any(b['reason']!='contract_not_reward_ready' for b in verification['readiness_blockers']):
        return {**base,'type':'not_ready'}
    _,findings=split_flags(contract,lint(contract))
    hard=[r['id'] for d,i,r in reqs if d in ('category','budget') and r['importance']=='required'
          and statuses[r['id']]=='fail' and locally_reviewed(contract,d,i,r,findings)]
    try:validate(contract,source_query=source_query,require_reviewed=True)
    except ValueError as error:approval_error=str(error)
    else:approval_error=None
    if hard:
        return {**base,'reward':-0.85,'reward_valid':True,'semantic_success':False,
                'type':'category_budget_failure','failed_ids':hard,'scope':'decisive_hard_failure',
                'contract_fully_reviewed':approval_error is None,'remaining_contract_blocker':approval_error}
    if approval_error:
        failed=[r['id'] for d,i,r in reqs if r['importance']=='required' and statuses[r['id']]=='fail'
                and locally_reviewed(contract,d,i,r,findings)]
        # A proven necessary-condition failure settles success, even if other
        # requirements still prevent an exact partial score. No scalar reward.
        return {**base,'type':'contract_blocked','contract_blocker':approval_error,
                'semantic_success':False if failed else None,'known_failed_required_ids':failed,
                'reward_interval':[-0.85,0.25] if failed else None,
                'scope':'known_failure_unknown_score' if failed else 'contract_blocked'}

    def dimension_bounds(importance):
        dims={}
        for d in DIMENSIONS:
            rs=[r for dim,_,r in reqs if dim==d and r['importance']==importance]
            if rs:
                passed=sum(statuses[r['id']]=='pass' for r in rs);unknown=sum(statuses[r['id']]=='unknown' for r in rs)
                dims[d]=(passed/len(rs),(passed+unknown)/len(rs),len(rs))
        if not dims:return None,{}
        return (sum(v[0] for v in dims.values())/len(dims),sum(v[1] for v in dims.values())/len(dims)),dims

    mandatory,md=dimension_bounds('required');preference,pd=dimension_bounds('preference')
    required=[(d,r) for d,_,r in reqs if r['importance']=='required']
    failed=[r['id'] for _,r in required if statuses[r['id']]=='fail']
    unknown=[r['id'] for _,_,r in reqs if statuses[r['id']]=='unknown']
    mandatory_unknown=[(d,r) for d,r in required if statuses[r['id']]=='unknown']
    if not mandatory_unknown and failed:
        # Preferences do not enter the approved partial-failure formula.
        return {**base,'reward':-0.30+0.55*mandatory[0],'reward_valid':True,'semantic_success':False,
                'type':'partial_failure','unknown_preference_ids':unknown,'scope':'exact_required_score'}
    if not unknown:
        value=1.0 if preference is None else 0.55+0.45*preference[0]
        return {**base,'reward':value,'reward_valid':True,'semantic_success':True,'type':'semantic_success','scope':'fully_verified'}

    candidates=[]
    if any(d in ('category','budget') for d,_ in mandatory_unknown):candidates.append(-0.85)
    if failed or mandatory_unknown:
        candidates.append(-0.30+0.55*mandatory[0])
        upper=mandatory[1]
        if not failed:
            upper-=min(1/(len(md)*md[d][2]) for d,_ in mandatory_unknown)
        candidates.append(-0.30+0.55*upper)
    if not failed:
        candidates.extend([1.0,1.0] if preference is None else [0.55+0.45*preference[0],0.55+0.45*preference[1]])
    return {**base,'type':'unverifiable','unknown_ids':unknown,
            'semantic_success':False if failed else True if not mandatory_unknown else None,
            'reward_interval':[min(candidates),max(candidates)],'scope':'bounded_unknown_evidence'}
