"""User-approved training objective, separate from complete semantic truth.

Confirmed 2026-11-07: only publicly supported satisfaction earns item credit.
Unknown does not mean false and stays in the fixed denominator. Unknown
required items cannot produce success; unknown preferences only lose bonus.
Review/source/execution gates remain mandatory.
"""
from semantic_contract_v2_r7_final12 import DIMENSIONS,validate
from semantic_reward_history import score_verified

VERSION='semantic-evidence-reward-v1-user-confirmed-20261107'


def score_terminal_failure(row):
    """Retain original fixed penalties; require matching final/last events.

    Does not use final_reward, old success labels or original target answers.
    """
    from web_agent_site.engine.reward import DEFAULT_REWARDS
    terminal=row.get('terminal_result') or {};steps=row.get('steps') or []
    last=steps[-1] if steps else {};result=last.get('result') or {}
    invalid={'reward':None,'reward_valid':False,'semantic_success':None,'evidence_success':None,
             'version':VERSION,'type':'execution_failure_unproven'}
    if row.get('error') or not terminal.get('done') or not result.get('done'):return invalid
    if terminal.get('purchase') or result.get('purchase'):return invalid
    reason=terminal.get('termination_reason')
    if reason!=result.get('termination_reason'):return invalid
    guard=((result.get('reward_detail') or {}).get('evidence') or {}).get('purchase_guard') or {}
    if guard.get('status')=='exhausted' and last.get('tool_name')=='buy_now' and reason=='wrong_purchase':
        kind='wrong_purchase'
    elif reason in ('repeat_loop','max_steps'):kind=reason
    else:return invalid
    return {'reward':float(DEFAULT_REWARDS[kind]),'reward_valid':True,'semantic_success':False,'evidence_success':False,
            'version':VERSION,'type':'fixed_execution_failure','termination_reason':reason,
            'basis':'unchanged_upstream_fixed_penalty','objective':'verified_public_evidence_satisfaction',
            'proof':{'both_done':True,'no_purchase_in_both_events':True,'last_and_terminal_reason_equal':True}}


def score_training_verified(contract,verification,*,source_query):
    truth=score_verified(contract,verification,source_query=source_query)
    base={'version':VERSION,'objective':'verified_public_evidence_satisfaction','semantic_result':truth,
          'semantic_success':truth['semantic_success'],'unknown_ids':[k for k,v in verification['statuses'].items() if v=='unknown'],
          'unknown_is_real_failure':False}
    if truth['reward_valid']:
        return {**base,'reward':truth['reward'],'reward_valid':True,'type':truth['type'],
                'evidence_success':truth['semantic_success'] is True,'basis':'exact_semantic_score'}
    try:validate(contract,source_query=source_query,require_reviewed=True)
    except ValueError:return {**base,'reward':None,'reward_valid':False,'type':'contract_blocked','evidence_success':None}
    if not verification['execution_complete'] or verification['readiness_blockers']:
        return {**base,'reward':None,'reward_valid':False,'type':'not_ready','evidence_success':None}
    statuses=verification['statuses'];reqs=[(d,r) for d in DIMENSIONS for r in contract[d]]
    def ratio(importance):
        dims={}
        for d in DIMENSIONS:
            rs=[r for dim,r in reqs if dim==d and r['importance']==importance]
            if rs:dims[d]=sum(statuses[r['id']]=='pass' for r in rs)/len(rs)
        return sum(dims.values())/len(dims) if dims else None,dims
    required,rd=ratio('required');preference,pd=ratio('preference')
    supported=all(statuses[r['id']]=='pass' for _,r in reqs if r['importance']=='required')
    value=(1 if preference is None else 0.55+0.45*preference) if supported else -0.30+0.55*required
    return {**base,'reward':value,'reward_valid':True,'type':'evidence_success' if supported else 'evidence_partial',
            'evidence_success':supported,'basis':'user_confirmed_evidence_objective',
            'required_dimension_support':rd,'preference_dimension_support':pd}
