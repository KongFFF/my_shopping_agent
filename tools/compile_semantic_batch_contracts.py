"""Audited batch-source fixes and status-only acceptance candidates.

All semantic fixes are offline data edits bound to exact source/predicates.
No target product, ASIN, training outcome or model output is consulted.
This compiler does not itself grant training permission.
"""
import copy
import json
from pathlib import Path

from semantic_contract_v2_r7_final12 import DIMENSIONS, validate, lint, split_flags
import semantic_reviewed_scopes as receipts
from prepare_semantic_batch_acceptance import OUT, SOURCE, LIMITS, digest, lines, approved_copy
from semantic_training_bundle import ROOT, rows, dump, sha256

PACKAGE = ROOT / 'experiments/semantic_batch_release_v1'
EXTRA_RECEIPTS = PACKAGE / 'scope_reviews.jsonl'


def compile_contracts():
    PACKAGE.mkdir(parents=True, exist_ok=True)
    records = rows(SOURCE)
    original = {r['task_id']:copy.deepcopy(r) for r in records}
    edits = []
    findings = []
    held = {r['task_id']:r for r in rows(LIMITS)}
    decisions = json.loads((PACKAGE/'policy.json').read_text(encoding='utf-8'))
    for task, reason in decisions['additional_source_limits'].items():
        task = int(task)
        held[task] = {'task_id':task,'query':original[task]['query'], 'finding':reason,
                      'disposition':'reviewed_frozen_source_limitation','training_policy':'whole_group_invalid',
                      'decision_status':'confirmed_by_user_or_prior_confirmed_policy',
                      'source_contract_sha256':digest(original[task]['normalized'])}

    def add(c, dim, st, attr, expected, quote, *, subject=None, op='eq', identity=None):
        start=c['query'].index(quote)
        r={'id':identity or 'batch_'+str(len(c[dim])+1),'subtype':st,'attribute':attr,
           'subject':subject or {'kind':'primary_product','name':'主商品'},'importance':'required',
           'operator':op,'expected':expected,'price_basis':None,
           'evidence':[{'text':quote,'start':start,'end':start+len(quote)}],'review_status':'pending'}
        c[dim].append(r)

    def move(c, identity, dim, *, subtype, attribute, expected=None):
        for old_dim in DIMENSIONS:
            for r in list(c[old_dim]):
                if r['id'] != identity:continue
                c[old_dim].remove(r);c[dim].append(r)
                r.update(subtype=subtype,attribute=attribute)
                if expected is not None:r.update(operator='eq',expected=expected)
                return r
        raise ValueError('Missing source field '+identity)

    for row in records:
        task=row['task_id'];c=row['normalized'];before=copy.deepcopy(c)
        if task==16481:
            p=next(p for p in c['specifications'] if p['id']=='r006')
            p['operator']=decisions['16481_colors_operator']
            # The user explicitly resolved the relation, not just AND versus
            # OR. Require the corresponding package claim as well. A two-tone
            # garment or two gray sets must not establish one of each color.
            add(c,'specifications','packaging','packaging',
                {'type':'text','value':'灰色和黑色各一套'},'颜色要灰色和黑色',identity='batch_color_allocation')
            c['specifications'][-1]['evidence'].append(copy.deepcopy(next(
                p for p in c['specifications'] if p['id']=='r004')['evidence'][0]))
        if task==827:
            move(c,'r005','functions',subtype='function',attribute='function',
                 expected={'type':'text','value':'可添加果粒'})
        if task==13937:
            move(c,'r006','specifications',subtype='condition',attribute='condition')
            add(c,'specifications','size','size_label',{'type':'text','value':'009'},'009',identity='batch_gauge009')
        if task==882:
            add(c,'functions','function','function',{'type':'text','value':'可以调节长度'},
                '喷杆带机动功能并且可以调节长度',subject={'kind':'component','name':'喷杆'},identity='batch_adjustable')
        if task==9531:
            # Same light/thin claim was duplicated, once incorrectly as mass.
            c['specifications']=[p for p in c['specifications'] if p['id']!='r009']
        if task in (9817,10004):
            identity='r008' if task==9817 else 'r006'
            move(c,identity,'functions',subtype='function',attribute='function')
        if task in (5965,10995):
            for p in c['specifications']:
                if p['subtype']=='condition' and p['expected'].get('value')=='不要开刃':
                    p['expected']['value']='未开刃'
        if task==20632:
            for p in c['specifications']:
                if p['subtype']=='color' and p['expected'].get('value')=='同色':
                    p['expected']['value']='黑色'
                    quote='黑色英伦风西装套装';start=c['query'].index(quote)
                    p['evidence'].append({'text':quote,'start':start,'end':start+len(quote)})
        if task==13751:
            c['specifications']=[p for p in c['specifications'] if p['id']!='migration001']
        if task==4051:
            # Old uncertain monetary slot duplicated the already reviewed
            # non-numeric deposit-payment permission/transaction term.
            c['budget']=[p for p in c['budget'] if p['id']!='r008']
        if task==17761:
            p=next(p for p in c['functions'] if p['id']=='migration001')
            p['subject']={'kind':'component','name':'白玉髓'}
            quote='温润白玉髓';start=c['query'].index(quote)
            p['evidence']=[{'text':quote,'start':start,'end':start+len(quote)}]
        # Keep physical attribute identities distinct; no inferred unit
        # conversions, no percent proof invented for alcohol strength.
        for p in list(c['specifications']):
            if p['attribute']!='other_feature_count':continue
            quote=' '.join(e['text'] for e in p['evidence'])
            if task in (83,698,1588,3562,3908,10596,10950,18807,19581,22999):p['attribute']='color_count'
            elif task in (3676,8804,10591,10696,11106,13016,14318,17932):p['attribute']='layer_count'
            elif task in (882,3868,13142):p['attribute']='hole_count'
            elif task==971 and p['id']=='r005':p['attribute']='pixel_count'
            elif task in (971,1092,3000,5149,10867):
                dim='functions' if task in (971,1092,5149) else 'specifications'
                st='function' if dim=='functions' else 'composition'
                attr=st
                value={971:'8倍变焦',1092:'亮度能调六成',3000:'45度',
                       5149:'每厘米长度内的孔数大于1500',10867:'20倍浓缩'}[task]
                move(c,p['id'],dim,subtype=st,attribute=attr,expected={'type':'text','value':value})
        if c!=before:
            edits.append({'task_id':task,'query_sha256':receipts.query_digest(row['query']),
                          'before':before,'after':copy.deepcopy(c),
                          'basis':'fixed sample review plus targeted same-error-class review; source only'})
        if task not in held:
            c=approved_copy(c);row['normalized']=c
            row['batch_source_accepted']=True
            row['batch_acceptance_kind']='prior_reviews_plus_full_checks_plus_fixed_sample'
            # These historical flags keep their exact original meaning.
            assert row.get('whole_contract_approved')==original[task].get('whole_contract_approved')
        else:
            row['batch_source_accepted']=False
            row['batch_hold_reason']=held[task]['finding']
        row['batch_original_contract_sha256']=digest(original[task]['normalized'])

    # Transport existing query receipts through review_status-only changes.
    # If a semantic predicate changed, do not transport its query receipt.
    # Field receipts remain bound to the exact unchanged predicate as before.
    extra=[]
    by_task={r['task_id']:r for r in records}
    for old in rows(receipts.PATH):
        task=old['task_id'];before=original[task]['normalized'];after=by_task[task]['normalized']
        if old['kind']=='predicate_scope_review':
            # Only the explicitly reviewed layer-count retyping. No generic
            # predicate transport after semantic changes.
            if task==17932 and old['predicate']['attribute']=='other_feature_count':
                new=copy.deepcopy(old);new['predicate']['attribute']='layer_count'
                new['transport_basis']='reviewed 三层设计; typed layer identity only'
                extra.append(new)
            continue
        if digest(before)!=old['contract_sha256']:raise ValueError('Stale original scope receipt')
        related_count_only=task==19581 and all(
            {k:v for k,v in a.items() if k not in ('review_status','attribute')}==
            {k:v for k,v in b.items() if k not in ('review_status','attribute')}
            for d in DIMENSIONS for a,b in zip(before[d],after[d]))
        explicit_order_rechecked=task==16481 and old['code']=='recommendation_quantity_scope'
        if approved_copy(before)==after or related_count_only or explicit_order_rechecked:
            new=copy.deepcopy(old);new['contract_sha256']=digest(after)
            new['transport_basis']=('source recommendation clause rechecked; explicit independent 我想买两套 preserved'
                if explicit_order_rechecked else 'review-status-only transition or explicitly rechecked color-count identity')
            new['previous_contract_sha256']=old['contract_sha256'];extra.append(new)
    lines(EXTRA_RECEIPTS,extra)
    receipts.records.cache_clear()
    for row in records:
        c=row['normalized']
        try:validate(c,source_query=row['query'],require_reviewed=row['batch_source_accepted'])
        except ValueError as exc:findings.append({'task_id':row['task_id'],'error':str(exc)})
    lines(OUT/'contracts1000.jsonl',records)
    lines(OUT/'data_edits.jsonl',edits)
    lines(OUT/'compiler_findings.jsonl',findings)
    lines(OUT/'source_limitations.jsonl',[held[t] for t in sorted(held)])
    dump(OUT/'compile_summary.json',{'original_source_sha256':sha256(SOURCE),'tasks':len(records),
         'batch_accepted':sum(r['batch_source_accepted'] for r in records),'held':len(held),
         'held_ids':sorted(held),'semantic_edits':len(edits),'compiler_findings':len(findings),
         'whole_query_reviews_falsely_added':0,'api_calls':0,'actor_queries_changed':0})
    print(json.dumps({'tasks':len(records),'held':len(held),'edits':len(edits),'findings':len(findings)}))


if __name__=='__main__':compile_contracts()
