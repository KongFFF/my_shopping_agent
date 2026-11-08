"""Source/predicate-bound user decisions, separate from general language rules.

Unanswered entries are inert. A task ID alone never authorizes an override.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/'experiments/semantic_history_review_20261107/user_decisions.json'
PREDICATE_KEYS=('subtype','attribute','subject','operator','expected')


@lru_cache(maxsize=1)
def records():
    result=json.loads(PATH.read_text(encoding='utf-8'))['decisions'] if PATH.exists() else []
    migration=PATH.parents[1]/'semantic_training_policy_migration_v1/user_decisions.json'
    if migration.exists():result+=json.loads(migration.read_text(encoding='utf-8'))['decisions']
    closeout=PATH.parents[1]/'semantic_remaining_closeout_v1/user_decisions.json'
    if closeout.exists():result+=json.loads(closeout.read_text(encoding='utf-8'))['decisions']
    return result


def strength_decision(query,r):
    sha=hashlib.sha256(query.encode()).hexdigest()
    predicate={k:r[k] for k in PREDICATE_KEYS}
    for decision in records():
        if decision.get('kind')!='requirement_strength' or decision.get('status')!='confirmed_by_user':continue
        if decision.get('importance') not in ('required','preference'):raise ValueError('Confirmed strength decision needs required/preference')
        if decision['query_sha256']!=sha or decision['predicate']!=predicate:continue
        if query.count(decision['source_text'])!=1 or not any(e['text']==decision['source_text'] for e in r['evidence']):continue
        return decision
    return None


def verification_decision(query,r):
    sha=hashlib.sha256(query.encode()).hexdigest();predicate={k:r[k] for k in PREDICATE_KEYS}
    for decision in records():
        if decision.get('kind')!='verification_standard' or decision.get('status')!='confirmed_by_user':continue
        if decision['query_sha256']!=sha or decision['predicate']!=predicate:continue
        if query.count(decision['source_text'])==1 and any(e['text']==decision['source_text'] for e in r['evidence']):return decision
    return None


def resolved_strength_source_flag(contract,flag):
    # A global heuristic must not resurrect a preferred-part demand on the
    # exact clause the user explicitly resolved as required. Other clauses
    # and other kinds of diagnostics remain untouched.
    if flag.get('code')!='soft_clause_in_mixed_uncovered' or flag.get('location')!='/query':return False
    source=flag.get('source')
    if not isinstance(source,str):return False
    for d in ('category','budget','functions','specifications','brand','model'):
        for r in contract[d]:
            decision=strength_decision(contract['query'],r)
            if decision and decision['source_text']==source and r['importance']==decision['importance']:return True
    return False


def resolve_strength_mentions(c):
    decisions=[strength_decision(c['query'],r) for d in ('category','budget','functions','specifications','brand','model') for r in c[d]]
    kept=[];audit=[]
    for p in c['pending_mentions']:
        matches=[d for d in decisions if d and p['reason']=='ambiguous_strength' and p['evidence']['text']==d['source_text']]
        if matches:audit.append({'kind':'user_confirmed_source_bound_strength','resolved_pending':p,'decision_ids':[d['id'] for d in matches]})
        else:kept.append(p)
    c['pending_mentions']=kept
    return audit
