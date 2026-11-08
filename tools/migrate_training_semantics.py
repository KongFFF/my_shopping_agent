"""Offline migration of reviewed policies; never blanket-approves train1000.

Every removed pending needs a retained source requirement, a source-checked
existing predicate, or an explicit non-scoring context decision. Old files,
original query/task IDs, and prior whole-query approvals remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import semantic_contract_v2_r7_final12 as b
from semantic_acceptance_representation import MEASURES, validate_approximate_target
from semantic_scope_v7 import direct_scopes, direct_hope_scopes, approximation_evidence
from semantic_source_scope_proofs import resolve_claim_mentions, resolve_selection_mentions
from semantic_review_decisions import resolve_strength_mentions
from semantic_training_bundle import rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'analysis/semantic_training_revision_20261107/validated1000.jsonl'
OUT = ROOT / 'analysis/semantic_policy_migration_20261107'
POLICY = ROOT / 'experiments/semantic_training_policy_migration_v1/policy.json'
LEDGER = POLICY.parent / 'source_review.jsonl'
PRIMARY = {'kind': 'primary_product', 'name': '主商品'}
ORDER = {'kind': 'order', 'name': '本次订单'}
PUNCT = r'[，,。；;！？!?\n]'


def jl(path, values):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values), encoding='utf-8')


def evidence(query, text):
    starts = [i for i in range(len(query)) if query.startswith(text, i)]
    if len(starts) != 1:
        raise ValueError(f'Nonunique source quote: {text}')
    return {'text': text, 'start': starts[0], 'end': starts[0] + len(text)}


def allreq(c):
    return [(d, r) for d in b.DIMENSIONS for r in c[d]]


def overlap(a, e):
    return a['start'] < e['end'] and e['start'] < a['end']


def literal(r):
    v = r['expected']
    return v['value'] if v['type'] == 'text' else None


def clean(text):
    return re.sub(r'[\s，,。；;！？!?]', '', text)


def grammatical_core(text):
    # Only grammatical introducers, never numeric bounds, negation or qualifiers.
    return re.sub(r'^(?:我(?:比较)?(?:想要|喜欢|偏好|追求)|要求|需要|想要|最好|优先|尽量|偏好|喜欢|要)(?:也|还)?(?:是|有|为)?', '', text, count=1).strip()


def req(query, text, subtype, *, dimension=None, attribute=None, value=None, operator='eq', subject=None, importance='required'):
    dimension = dimension or next(d for d, types in b.DIMENSIONS.items() if subtype in types)
    return dimension, {'id': 'migration', 'subtype': subtype, 'attribute': attribute or b.ATTRIBUTES[subtype][0],
        'subject': subject or copy.deepcopy(PRIMARY), 'importance': importance, 'operator': operator,
        'expected': value or {'type': 'text', 'value': grammatical_core(text)}, 'price_basis': None,
        'evidence': [evidence(query, text)], 'review_status': 'approved'}


def add_unique(c, dimension, r):
    for d, old in allreq(c):
        if d == dimension and all(old[k] == r[k] for k in ('subtype', 'attribute', 'subject', 'importance', 'operator', 'expected')):
            for e in r['evidence']:
                if e not in old['evidence']:old['evidence'].append(e)
            return old
    used = {old['id'] for _, old in allreq(c)}
    n = 1
    while f'migration{n:03d}' in used:n += 1
    r['id'] = f'migration{n:03d}'
    c[dimension].append(r)
    return r


def strength_for(c, r, original=None):
    scopes = direct_scopes(r, c['query'])
    cues = {s['cue'] for s in scopes}
    if '希望' in cues and not cues & {'必须', '务必', '一定要', '要求', '最好', '优先', '尽量', '偏好', '喜欢'}:
        return None
    if cues & {'最好', '优先', '尽量', '偏好', '喜欢'} and cues & {'必须', '务必', '一定要', '要求'}:
        return None
    if cues & {'最好', '优先', '尽量', '偏好', '喜欢'}:return 'preference'
    if cues & {'必须', '务必', '一定要', '要求'}:return 'required'
    if original in ('required', 'preference'):return original
    # Modal/permission/hearsay are never promoted by a source-copying rule.
    q = c['query'];e = r['evidence'][0]
    start = max([m.end() for m in re.finditer(PUNCT, q[:e['start']])] or [0])
    end = next((e['start'] + m.start() for m in re.finditer(PUNCT, q[e['start']:])), len(q))
    clause = q[start:end]
    if re.search(r'希望|听说|好像|可能|可以是|可以赠送|或许|据说|最好|优先|尽量|喜欢|偏好', clause):return None
    return 'required'


def claim_subtype(text):
    if re.match(r'搭配|适合|用于|用来',text):return 'purpose'
    if re.search(r'质保|售后',text):return 'warranty'
    if re.search(r'风格|复古|百搭|新潮|潮流|时髦|颜值|显瘦|显腿|气质|设计|国风|美学|萌|可爱|慵懒|奢华|洋气|高贵|大气|时尚|造型|低调|优雅', text):return 'style'
    if re.search(r'质量|品质|靠谱|正宗|高端|高档|原装|热销|热门|上过电视|出口标准|认证', text):return 'condition'
    return 'function'


def matching(c, p):
    core = clean(grammatical_core(p['evidence']['text']))
    return [(d, r) for d, r in allreq(c) if literal(r) is not None and clean(grammatical_core(literal(r))) == core
            and any(overlap(p['evidence'], e) for e in r['evidence'])]


def reviewed_claim_candidates(c, p):
    exact=matching(c,p)
    if exact:return exact
    # Preserve an existing source-bound predicate (e.g. 款式+复古英伦), rather
    # than adding a second copy of the same claim under a different dimension.
    return [(d,r) for d,r in allreq(c) if literal(r) and literal(r) in p['evidence']['text']
            and r['importance']=='uncertain' and p['evidence'] in r['evidence']]


def restore_public_claim(c, p):
    text = p['evidence']['text']
    if re.search(r'今年|近期|当下|最近|听说|好像|可能|据说|[?？]', text):return None
    # Fragment-level unit/quantity/tolerance ambiguities cannot be erased by
    # representing a numeric requirement as a generic qualitative statement.
    if re.search(r'\d|至少|不超过|不低于|价格|划算|实惠|便宜|经济|平价|太贵|低端', text):return None
    hits = reviewed_claim_candidates(c, p)
    if hits:
        d, r = hits[0]
        if r['importance'] == 'uncertain':
            r['importance'] = strength_for(c, r)
            if r['importance'] is None:return None
        r['review_status'] = 'approved'
        return 'retained_existing_public_claim'
    d, r = req(c['query'], text, claim_subtype(text))
    r['importance'] = strength_for(c, r)
    if r['importance'] is None:return None
    # Do not put a clearly named part/recipient into a whole-product predicate.
    for _, other in allreq(c):
        name = other['subject']['name']
        if other['subject']['kind'] in ('component', 'intended_user') and name and name in text:
            r['subject'] = copy.deepcopy(other['subject']);break
    add_unique(c, d, r)
    return 'restored_missing_public_claim'


def resolve_approximation(c, p):
    # Exact-center rule only for the supported source-bound physical attributes.
    candidates = [(d,r) for d,r in allreq(c) if r['subtype'] in ('net_content','weight','dimensions','capacity')
                  and any(overlap(p['evidence'],e) for e in r['evidence'])]
    if len(candidates) != 1:return None
    d,r = candidates[0]
    if r['operator'] == 'approximately':return 'existing_exact_center_rule'
    units = '|'.join(sorted(MEASURES, key=len, reverse=True))
    measurements = list(re.finditer(r'([\d零〇一二两三四五六七八九十百千万]+(?:\.\d+)?)\s*('+units+r')(?![A-Za-z])', p['evidence']['text'], re.I))
    if len(measurements) != 1:return None
    m = measurements[0]; unit = m[2].lower()
    family = MEASURES[unit][0]
    expected_family = {'mass':'mass','liquid_volume':'volume','load_capacity':'mass','length':'length',
                       'width':'length','height':'length','thickness':'length','diameter':'length'}.get(r['attribute'])
    if family != expected_family:return None
    normalized_unit = {'克':'g','千克':'kg','公斤':'kg','斤':'jin','毫升':'ml','升':'l','毫米':'mm','厘米':'cm','米':'m'}.get(unit,unit)
    candidate = copy.deepcopy(r)
    candidate.update(operator='approximately', expected={'type':'number','value':float(b.chinese_number(m[1])), 'unit':normalized_unit})
    if p['evidence'] not in candidate['evidence']:candidate['evidence'].append(copy.deepcopy(p['evidence']))
    candidate['importance'] = strength_for(c, candidate, r['importance'] if r['importance']!='uncertain' else None)
    if candidate['importance'] is None:return None
    candidate['review_status']='approved'
    validate_approximate_target(candidate,b.chinese_number)
    r.clear();r.update(candidate)
    return 'exact_center_supported_other_values_unknown'


def resolve_latest(c, p, policy):
    text = p['evidence']['text']
    if not policy['latest_declaration']['confirmed_by_user']:return None
    if re.search(r'今年|近期|当下|最近',text) and not re.search(r'20\d{2}',text):return None
    if not re.search(r'最新|新款|新出|新上市|新品',text):return None
    hits=reviewed_claim_candidates(c,p)
    if hits:
        _,r=hits[0]
        importance=strength_for(c,r, None if r['importance']=='uncertain' else r['importance'])
        if importance is None:return None
        r.update(importance=importance,review_status='approved')
    else:
        d,r=req(c['query'],text,'condition')
        r['importance']=strength_for(c,r)
        if r['importance'] is None:return None
        add_unique(c,d,r)
    return 'user_confirmed_frozen_latest_declaration'


def resolve_existing_meaning(c,p):
    text=p['evidence']['text'];hits=matching(c,p)
    if not hits:return None
    for d,r in hits:
        if r['importance']=='uncertain':return None
        if re.search(r'希望|听说|好像|可能|可以是|或者|或|及以上|不超过|不少于',text):return None
        if re.search(r'\d|[一二两三四五六七八九十百千万]',text):
            # Packaging is a sales-pack description, never an order count.
            core=grammatical_core(text).removesuffix('的')
            if r['subtype']!='packaging' or not re.fullmatch(r'(?:\d+|[一二两三四五六七八九十]+)(?:瓶|盒|包|袋|支|个)装',core):return None
        r['review_status']='approved'
    return 'already_expressed_source_meaning'


def restore_clear_fragment(c,p):
    text=p['evidence']['text']
    # No blanket conversion of ambiguity into free text: relational counts,
    # references, approximation and undefined strength remain in the queue.
    if re.search(r'希望|听说|好像|可能|或者|\b或\b|可以是|多少|几十|几种|来张|左右|大概|接近|同款|同一种|每|全部|全套|齐全|所有|不超过|以上|不低于|合计|一共|组合装|套装|[?？]',text):return None
    labels={
        '家用':'use_environment','家用小型':'form_factor','工业用的':'use_environment','家用甲级':'condition',
        '正牌':'condition','正版':'condition','原装':'condition','热门款':'condition','热门款式':'condition',
        '新型':'condition','精品款':'condition','升级机械款':'condition','海外版':'version','跨境的':'condition',
        '国潮':'style','网红':'condition','外置':'form_factor','低帮':'form_factor','干燥型':'form_factor',
        '小型的':'form_factor','专业款':'condition','后排':'compatibility',
        '男款':'target_user','女款':'target_user','男子':'target_user','给男生穿':'target_user',
        '男女都适合':'target_user','男女同款':'target_user','男生女生都能穿':'target_user',
        '女生都穿':'target_user','适合初学者':'target_user','初学者使用':'target_user','初学者用的':'target_user',
        '定制':'customization','单股':'form_factor','顶棚':'compatibility','修复':'purpose','背胶':'function',
        '空调显示控制面板':'compatibility','故宫主题':'pattern','经典的军绿色':'color','长管':'form_factor',
        '给葫芦用的':'purpose','制作炒河粉':'purpose','卤的鸭翅中':'condition','新年':'use_environment',
        '鸿运当头':'pattern','抬头见喜':'pattern','代表“福禄”的那种':'pattern',
        '是跨境的':'condition','是冲云霄小蚩尤的组合装':'content','工业用的':'use_environment',
    }
    subtype=labels.get(text)
    if subtype is None and re.match(r'适合|适用|用于|用来|专门用于|奶茶店专用',text):subtype='purpose'
    if subtype is None and re.match(r'可以单独买|可以直接|可拆卸|可改变|能够代替|不能遮盖|可以平替|不用系鞋带|集成好的|不用出运费|邮费由商家承担',text):subtype='function'
    if subtype is None:return None
    if re.search(r'\d|[一二两三四五六七八九十百千万](?:个|件|瓶|袋|包|支|片|盒|套|组|米|克|毫)',text):return None
    d,r=req(c['query'],text,subtype)
    r['importance']=strength_for(c,r, p['importance'] if p['importance']!='uncertain' else None)
    if r['importance'] is None:return None
    # Replace only exact same-source, same-semantic-field renderings; never
    # remove unrelated requirements on the same page or task.
    for dim in b.DIMENSIONS:
        c[dim]=[old for old in c[dim] if not (old['subtype']==subtype and old['evidence']==r['evidence']
                                             and literal(old) is not None and literal(old) in text)]
    add_unique(c,d,r)
    return 'retained_unambiguous_source_fragment'


def resolve_existing_budget(c,p):
    text=p['evidence']['text']
    if not re.search(r'预算|价格|售价|价位|总价|单价',text):return None
    if re.search(r'接近|比|更|打折|优惠|不要钱|三百四五十',text):return None
    if re.search(r'价格[为是].*那(?:一)?款',text):return None
    for r in c['budget']:
        if not any(overlap(p['evidence'],e) for e in r['evidence']):continue
        for e in r['evidence']:
            try:op,value,_=b.budget_for_role(b.budget_source_clause(c['query'],e['text']),r['subtype'])
            except ValueError:continue
            if r['expected']['type']=='text' and r['importance']=='uncertain':
                candidate=copy.deepcopy(r);candidate.update(operator=op,expected=value)
                importance=strength_for(c,candidate)
                if importance is None:continue
                candidate.update(importance=importance,review_status='approved')
                r.clear();r.update(candidate)
                return 'restored_source_parsed_budget'
            if r['importance']!='uncertain' and (op,value)==(r['operator'],r['expected']):
                r['review_status']='approved';return 'existing_confirmed_budget_policy'
    return None


def resolve_supported_unit(c,p,policy):
    text=p['evidence']['text'];hits=matching(c,p)
    if re.search(r'(?<!英)寸',text):return None  # The user's existing explicit decision stays in force.
    if re.search(r'(?:含金量|黄金.*(?:含量|纯度)).*\d+(?:\.\d+)?[%％]',text):
        m=re.search(r'(不低于|不少于|至少|达到|大于|超过)?\s*(\d+(?:\.\d+)?)\s*[%％]',text)
        if not m:return None
        n=float(m[2]);op=m[1]
        if op in ('大于','超过'):
            operator='between';value={'type':'range','lower':n,'upper':None,'unit':'percent','lower_inclusive':False,'upper_inclusive':False}
        else:
            operator='at_least' if op else 'eq';value={'type':'number','value':n,'unit':'percent'}
        component='黄金' if '黄金' in text else '金'
        d,r=req(c['query'],text,'composition',attribute='percentage',operator=operator,value=value,
                subject={'kind':'component','name':component})
        from semantic_acceptance_representation import validate_percentage
        validate_percentage(r)
        for dim in b.DIMENSIONS:
            c[dim]=[old for old in c[dim] if not (literal(old) and clean(literal(old))==clean(text))]
        add_unique(c,d,r)
        return 'source_numeric_gold_percentage'
    m=re.fullmatch(r'能承载(\d+(?:\.\d+)?)吨重物',text)
    if m:
        d,r=req(c['query'],text,'capacity',attribute='load_capacity',operator='at_least',
                value={'type':'number','value':float(m[1])*1000,'unit':'kg'})
        for dim in b.DIMENSIONS:c[dim]=[old for old in c[dim] if literal(old)!=text]
        add_unique(c,d,r);return 'explicit_tonne_to_kg'
    if hits and (re.search(r'\d+(?:\.\d+)?G(?:显存|大内存)|显存要\d+G',text)
                 or re.fullmatch(r'鞋码是\d+',text)):
        for _,r in hits:r['review_status']='approved'
        return 'retained_named_spec_label_no_unit_conversion'
    return None


def confirmed_strength_prepass(c):
    from semantic_review_decisions import records
    audit=[];digest=hashlib.sha256(c['query'].encode()).hexdigest()
    keys=('subtype','attribute','subject','operator','expected')
    for decision in records():
        if decision.get('kind')!='requirement_strength' or decision.get('query_sha256')!=digest or decision.get('status')!='confirmed_by_user':continue
        quote=evidence(c['query'],decision['source_text'])
        if decision.get('add_dimension'):
            r=copy.deepcopy(decision['predicate'])
            r.update(id='migration',price_basis=None,evidence=[quote],review_status='approved',importance=decision['importance'])
            add_unique(c,decision['add_dimension'],r)
            audit.append({'method':'confirmed_missing_requirement_restored','decision_id':decision['id'],
                          'after_requirement':copy.deepcopy(r)})
        if decision.get('before_predicate'):
            for dimension in b.DIMENSIONS:
                for r in list(c[dimension]):
                    if r['id']==decision['requirement_id'] and {k:r[k] for k in keys}==decision['before_predicate']:
                        previous=copy.deepcopy(r)
                        r.update(copy.deepcopy(decision['predicate']))
                        c[dimension].remove(r);c[decision['move_dimension']].append(r)
                        audit.append({'method':'confirmed_requirement_dimension_fix','decision_id':decision['id'],
                                      'before_requirement':previous,'after_requirement':copy.deepcopy(r)})
        for _,r in allreq(c):
            if {k:r[k] for k in keys}!=decision['predicate']:continue
            old=copy.deepcopy(r)
            if quote not in r['evidence']:r['evidence'].append(quote)
            r.update(importance=decision['importance'],review_status='approved')
            if old!=r:audit.append({'method':'confirmed_source_bound_strength','decision_id':decision['id'],
                                    'before_requirement':old,'after_requirement':copy.deepcopy(r)})
    return audit


def resolve_confirmed_strength(c,p):
    from semantic_review_decisions import strength_decision
    hits=[r for _,r in allreq(c) if strength_decision(c['query'],r)
          and any(overlap(p['evidence'],e) for e in r['evidence'])]
    if not hits:return None
    if p['reason']=='ambiguous_strength':return 'confirmed_source_bound_strength_pending_closed'
    if p['reason']=='ambiguous_meaning' and '混装' in p['evidence']['text']:
        return 'confirmed_mixed_pack_requirement'
    return None


def price_remark(c,p,policy):
    if not policy.get('qualitative_price_with_numeric_budget',{}).get('confirmed_by_user'):return None
    if not any(r['expected']['type'] in ('number','range') for r in c['budget']):return None
    text=p['evidence']['text']
    if re.search(r'经济实惠|价格实惠|便宜|平价|划算|经济型|别太贵',text) and not re.search(r'比市面|比.*便宜|更低|听说|希望',text):
        return 'confirmed_non_scoring_qualitative_price_note'
    return None


def apply_manual(c, p, decision):
    if decision['query_sha256'] != hashlib.sha256(c['query'].encode()).hexdigest():
        raise ValueError('Manual source decision query changed')
    if decision['action']=='context':return 'reviewed_non_scoring_context'
    if decision['action']=='hold':return None
    # Drop exact predicate(s) only when the ledger explicitly replaces them.
    for dimension in b.DIMENSIONS:
        c[dimension]=[r for r in c[dimension] if r['id'] not in decision.get('replace_ids',[])]
    for field in decision.get('requirements',[]):
        field=copy.deepcopy(field);d=field.pop('dimension')
        field.setdefault('review_status','approved')
        if 'id' not in field:field['id']='migration'
        field['evidence']=[evidence(c['query'],t) for t in field.pop('source_texts')]
        add_unique(c,d,field)
    if decision.get('existing_ids'):
        for _,r in allreq(c):
            if r['id'] in decision['existing_ids']:
                if r['importance']=='uncertain':raise ValueError('Existing interpretation still uncertain')
                r['review_status']='approved'
    return 'explicit_source_review'


def migrate(row, policy, decisions):
    row=copy.deepcopy(row);c=row['normalized'];changes=[];held=[]
    if row.get('whole_contract_approved') and row.get('reward_approved'):
        b.validate(c,source_query=row['query'],require_reviewed=True)
        return row,changes,held
    for delta in confirmed_strength_prepass(c):changes.append({'task_id':row['task_id'],**delta})
    for r in c['budget']:
        if r['expected']['type']=='text':continue
        for e in r['evidence']:
            try:op,value,rule=b.budget_for_role(b.budget_source_clause(c['query'],e['text']),r['subtype'])
            except ValueError:continue
            if (op,value)!=(r['operator'],r['expected']):
                previous=copy.deepcopy(r)
                r.update(operator=op,expected=value)
                changes.append({'task_id':row['task_id'],'method':'confirmed_budget_policy_replay',
                                'before_requirement':previous,'after_requirement':copy.deepcopy(r),'rule':rule})
            break
    for p in list(c['pending_mentions']):
        before=copy.deepcopy(c);key=(row['task_id'],p['reason'],p['evidence']['text'])
        decision=decisions.get(key);method=None
        try:
            method=resolve_confirmed_strength(c,p) or price_remark(c,p,policy)
            if not method and decision:method=apply_manual(c,p,decision)
            elif not method and p['reason']=='unverifiable_subjective':method=restore_public_claim(c,p)
            elif not method and p['reason']=='undefined_tolerance':method=resolve_approximation(c,p)
            elif not method and p['reason']=='missing_reference_time':method=resolve_latest(c,p,policy)
            elif not method and p['reason']=='ambiguous_meaning':method=resolve_existing_budget(c,p) or resolve_existing_meaning(c,p) or restore_clear_fragment(c,p)
            elif not method and p['reason']=='unsupported_unit':method=resolve_supported_unit(c,p,policy)
            elif not method and p['reason']=='unsupported_type':
                hits=matching(c,p)
                if hits and all(r['importance']!='uncertain' for _,r in hits):
                    for _,r in hits:r['review_status']='approved'
                    method='already_supported_existing_exact_predicate'
            if method:
                c['pending_mentions'].remove(p)
                b.validate(c,source_query=row['query'])
                changes.append({'task_id':row['task_id'],'query_sha256':hashlib.sha256(row['query'].encode()).hexdigest(),
                    'method':method,'resolved_pending':p,'decision':decision,'before_contract':before,'after_contract':copy.deepcopy(c),
                    'whole_contract_approved':False})
            else:
                c.clear();c.update(before)
                held.append({'task_id':row['task_id'],'query':row['query'],'pending':p,
                             'disposition':'requires_source_interpretation_or_supported_representation',
                             'initial_review':decision.get('note') if decision else None})
        except (ValueError,KeyError) as error:
            c.clear();c.update(before)
            held.append({'task_id':row['task_id'],'query':row['query'],'pending':p,
                         'disposition':'candidate_rejected_by_checker','detail':str(error),'proposed_method':method})
    # Preserve real uncertainty reasons, including fields previously missing a pending record.
    for d,r in allreq(c):
        if r['importance']=='uncertain' and not any(any(overlap(p['evidence'],e) for e in r['evidence']) for p in c['pending_mentions']):
            p={'evidence':copy.deepcopy(r['evidence'][0]),'reason':'ambiguous_strength','importance':'uncertain'}
            c['pending_mentions'].append(p)
            changes.append({'task_id':row['task_id'],'method':'restored_missing_uncertainty_reason','requirement_id':r['id'],'pending':p})
    # Confirming strength does not settle an unspecified count range or set.
    for task,text,note in [(247,'希望能包含所有配件','accessory set unspecified'),
                           (11503,'希望可以有10来张替换使用','approximate count interval unspecified')]:
        if c['task_id']==task and policy.get('hope_table_c',{}).get('confirmed_by_user'):
            p={'evidence':evidence(c['query'],text),'reason':'ambiguous_meaning','importance':'required'}
            existing = next((old for old in c['pending_mentions'] if old['reason']==p['reason'] and old['evidence']==p['evidence']), None)
            if existing is not None:
                existing['importance']='required'
            else:
                c['pending_mentions'].append(p)
                changes.append({'task_id':row['task_id'],'method':'retained_independent_interpretation_pending','pending':p,'detail':note})
    b.validate(c,source_query=row['query'])
    _,findings=b.split_flags(c,b.lint(c));row['review_findings']=findings
    row['program_valid']=True
    row['migration_audit']=[{'method':a['method'],'source':(a.get('resolved_pending') or a.get('pending'))} for a in changes]
    # This stage migrates source policy, not the future batch training signoff.
    row['whole_contract_approved']=False;row['reward_approved']=False
    return row,changes,held


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    before_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
    source=rows(args.source);policy=json.loads(POLICY.read_text(encoding='utf-8'))
    decisions={(d['task_id'],d['reason'],d['text']):d for d in rows(LEDGER)} if LEDGER.exists() else {}
    results=[];changes=[];held=[]
    for row in source:
        updated,delta,remaining=migrate(row,policy,decisions)
        results.append(updated);changes.extend(delta);held.extend(remaining)
    # Build the queue from final contracts, including uncertainties restored
    # after the pending loop. Never report a stale or missing review item.
    def pending_key(task, pending):
        return (task, pending['reason'], pending['evidence']['start'], pending['evidence']['end'])
    dispositions = {pending_key(r['task_id'], r['pending']): r for r in held}
    held = [dispositions.get(pending_key(r['task_id'], p), {
        'task_id': r['task_id'], 'query': r['query'], 'pending': p,
        'disposition': 'retained_independent_uncertainty',
    }) for r in results for p in r['normalized']['pending_mentions']]
    assert [r['task_id'] for r in source]==[r['task_id'] for r in results]
    assert all(a['query']==z['query'] for a,z in zip(source,results))
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==before_hash
    reasons=Counter(p['reason'] for r in results for p in r['normalized']['pending_mentions'])
    summary={'tasks':len(results),'program_valid':len(results),'tasks_changed':sum(a['normalized']!=z['normalized'] for a,z in zip(source,results)),
             'source_sha256':before_hash,'source_unchanged':True,'actor_queries_changed':0,'tasks_removed':0,
             'before_pending_tasks':sum(bool(r['normalized']['pending_mentions']) for r in source),
             'after_pending_tasks':sum(bool(r['normalized']['pending_mentions']) for r in results),
             'before_pending_items':sum(len(r['normalized']['pending_mentions']) for r in source),
             'after_pending_items':sum(len(r['normalized']['pending_mentions']) for r in results),
             'resolution_methods':dict(Counter(a['method'] for a in changes)),'remaining_reasons_items':dict(reasons),
             'whole_source_approvals':sum(r['whole_contract_approved'] for r in results),
             'api_calls':0,'training_runs':0,'formal_training_ready':False}
    jl(args.output/'validated1000.jsonl',results);jl(args.output/'edits.jsonl',changes)
    jl(args.output/'remaining_review.jsonl',held)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
