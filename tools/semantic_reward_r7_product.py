"""Latest-contract aggregation over verified comparisons. Offline, opt-in only."""
from semantic_contract_v2_r7_product import DIMENSIONS, validate
from semantic_product_verifier_v2 import VERSION as VERIFIER_VERSION, contract_fingerprint

VERSION='semantic-reward-r7-product-draft'


def score_verified(contract,verification,*,source_query):
    # Never approve drafts by clearing pending or rewriting their version.
    validate(contract,source_query=source_query,require_reviewed=True)
    if verification.get('version')!=VERIFIER_VERSION or verification.get('contract_sha256')!=contract_fingerprint(contract):
        raise ValueError('Verification is not bound to this exact contract')
    reqs=[(d,r) for d in DIMENSIONS for r in contract[d]]
    statuses=verification['statuses']
    if set(statuses)!={r['id'] for _,r in reqs} or any(s not in ('pass','fail','unknown') for s in statuses.values()):
        raise ValueError('Exactly one verified status per requirement required')
    if not any(r['importance']=='required' for _,r in reqs):raise ValueError('Nonempty required objective needed')
    if verification['readiness_blockers'] or not verification['execution_complete']:
        return {'reward':None,'reward_valid':False,'semantic_success':None,'type':'not_ready','version':VERSION}
    hard=[r['id'] for d,r in reqs if d in ('category','budget') and r['importance']=='required' and statuses[r['id']]=='fail']
    if hard:return {'reward':-0.85,'reward_valid':True,'semantic_success':False,'type':'category_budget_failure','failed_ids':hard,'version':VERSION}
    unknown=[k for k,v in statuses.items() if v=='unknown']
    if unknown:return {'reward':None,'reward_valid':False,'semantic_success':None,'type':'unverifiable','unknown_ids':unknown,'version':VERSION}
    def avg(importance):
        dims={}
        for d in DIMENSIONS:
            rs=[r for dim,r in reqs if dim==d and r['importance']==importance]
            if rs:dims[d]=sum(statuses[r['id']]=='pass' for r in rs)/len(rs)
        return (sum(dims.values())/len(dims) if dims else None),dims
    mandatory,md=avg('required');preference,pd=avg('preference')
    success=all(statuses[r['id']]=='pass' for _,r in reqs if r['importance']=='required')
    value=(1.0 if preference is None else 0.55+0.45*preference) if success else -0.30+0.55*mandatory
    return {'reward':value,'reward_valid':True,'semantic_success':success,'type':'semantic_success' if success else 'partial_failure',
            'required_dimension_scores':md,'preference_dimension_scores':pd,'version':VERSION}
