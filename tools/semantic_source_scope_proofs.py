"""Narrow source proofs for promotion and square dimensions; no task IDs.

They resolve only checker scope mistakes. Pending source interpretations remain
pending, and a promotion never supplies an independent order quantity.
"""
import re
from semantic_coverage_v7 import NUM, COUNTERS, UNITS, explicit_counts

DIMS=('category','budget','functions','specifications','brand','model')
PROMOTION=re.compile(r'买(?P<paid>'+NUM+r')(?P<unit>'+COUNTERS+r')?(?:要能|可以|可|能|要)?(?:赠送|送|赠)(?P<gift>'+NUM+r')(?P<giftunit>'+COUNTERS+r')?(?=$|[，,。；;！!？?])')
SQUARE=re.compile(r'('+NUM+r'(?:\.\d+)?)(毫米|厘米|mm|cm|米|m)\s*[×*xX]\s*('+NUM+r'(?:\.\d+)?)(毫米|厘米|mm|cm|米|m)')
SCALE={'毫米':1,'mm':1,'厘米':10,'cm':10,'米':1000,'m':1000}


def promotion_proofs(c,parser):
    proofs=[]
    for d in DIMS:
        for i,r in enumerate(c[d]):
            v=r['expected']
            if r['subtype']!='bundle' or v['type']!='bundle' or r['operator']!='matches_bundle' or r['subject']['kind']!='primary_product':continue
            for e in r['evidence']:
                for m in PROMOTION.finditer(e['text']):
                    if re.search(r'不|别|无需|不用',e['text'][max(0,m.start()-3):m.start()]):continue
                    # With omitted counters only an explicitly reviewed bundle
                    # unit may be used; no counter is invented by this proof.
                    units=[UNITS[u] for u in (m['unit'],m['giftunit']) if u]
                    if units and any(u!=v['unit'] for u in units):continue
                    if (v['paid_quantity'],v['gift_quantity'])!=(int(parser(m['paid'])),int(parser(m['gift']))):continue
                    proofs.append({'kind':'source_scope_proof','rule':'promotion_paid_count_is_not_order_count',
                       'location':f'/{d}/{i}','start':e['start']+m.start(),'end':e['start']+m.end(),
                       'text':m.group(),'paid_quantity':v['paid_quantity'],'gift_quantity':v['gift_quantity']})
    return proofs


def square_proof(r,parser):
    v=r['expected']
    if r['subtype']!='dimensions' or r['attribute'] not in ('length','width') or r['subject']['kind']!='primary_product':return None
    if v['type']!='number' or r['operator']!='eq' or v['unit'] not in SCALE:return None
    for e in r['evidence']:
        m=SQUARE.fullmatch(e['text'].strip())
        if not m:continue
        a=float(parser(m[1]))*SCALE[m[2]];z=float(parser(m[3]))*SCALE[m[4]]
        if a==z==v['value']*SCALE[v['unit']]:
            return {'kind':'source_scope_proof','rule':'equal_planar_dimensions_axis_invariant','source':dict(e)}
    return None


def filter_flags(c,flags,parser):
    promotions=promotion_proofs(c,parser);introductions=selection_introduction_proofs(c,parser);result=[]
    for f in flags:
        from semantic_review_decisions import resolved_strength_source_flag
        if resolved_strength_source_flag(c,f):continue
        candidate=f.get('source',{})
        if f['code']=='quantity_scope_pending' and candidate.get('role')=='purchase':
            from semantic_remaining_policy import approximate_target
            # Exact source-bound approximate order counts have a defined
            # comparison policy; packaging/variant/promotion scopes do not.
            if not re.search(r'装|每|买.+送',candidate.get('source','')):
                resolved=False
                for r in c['specifications']:
                    if r['subtype']!='quantity' or r['attribute']!='purchase_count' or r['operator']!='approximately' or r['review_status']!='approved':continue
                    v=approximate_target(r,parser)
                    if v and v['type']=='number' and v['value']==candidate['count'] and v['unit']==candidate['unit'] and any(e['start']<=candidate['start']<candidate['end']<=e['end'] for e in r['evidence']):resolved=True
                if resolved:continue
            # A declared 三瓶装 package is not a purchase of three sales units.
            # The original numeral/counter and selected package label must agree.
            tail=c['query'][candidate['end']:]
            if tail.startswith('装'):
                package=False
                for r in c['specifications']:
                    v=r['expected']
                    if r['subtype']!='packaging' or r['operator']!='eq' or r['subject']['kind']!='primary_product' or r['review_status']!='approved':continue
                    m=re.fullmatch('('+NUM+')('+COUNTERS+')装',v['value']) if v['type']=='text' else None
                    numeric=(v['type']=='number' and r['attribute']=='pack_count' and v['value']==candidate['count'] and v['unit']==candidate['unit'])
                    if numeric or (m and int(parser(m[1]))==candidate['count'] and UNITS.get(m[2])==candidate['unit']):
                        package=any(e['start']<=candidate['end']-1<e['end'] and '装' in e['text'] for e in r['evidence'])
                order=any(r['subtype']=='quantity' and r['attribute']=='purchase_count' and any(e['start']<=candidate['start']<candidate['end']<=e['end'] for e in r['evidence']) for r in c['specifications'])
                if package and not order:continue
        if f['code']=='quantity_scope_pending' and candidate.get('role') in ('gift','component'):
            from semantic_remaining_policy import colloquial_target
            if any(r['attribute'] in ('gift_count','included_component_count') and r['subject']['kind'] in ('gift','component') and r['subject']['name'] in candidate.get('source','') and any(candidate.get('source','') in e['text'] or e['text'] in candidate.get('source','') for e in r['evidence']) and colloquial_target(r,parser)==(r['operator'],r['expected']) for r in c['specifications']):continue
        if f['code']=='quantity_scope_pending' and candidate.get('role')=='purchase':
            from semantic_review_decisions import strength_decision
            if any(r['attribute']=='received_count' and r['expected'].get('type')=='number' and r['expected']['value']==candidate['count'] and r['expected']['unit']==candidate['unit'] and strength_decision(c['query'],r) for r in c['specifications']):continue
        if f['code']=='purchase_quantity_coverage':
            if any(p['start']<=candidate['start']<candidate['end']<=p['end'] for p in promotions):continue
            if any(p['start']==candidate['start'] and p['end']==candidate['end'] for p in introductions):continue
            # An explicitly retained ambiguity must not provoke retries that
            # manufacture contradictory order counts. It still blocks approval.
            if candidate['count']==1 and re.match(r'^(?:需要|找|要)',candidate['source']) and any(p['reason']=='ambiguous_meaning' and p['evidence']['start']<=candidate['start']<candidate['end']<=p['evidence']['end'] for p in c['pending_mentions']):
                result.append({**f,'code':'selection_quantity_scope'});continue
        parts=f.get('location','').strip('/').split('/')
        if len(parts)==2 and parts[0] in DIMS and parts[1].isdigit():
            r=c[parts[0]][int(parts[1])]
            from semantic_remaining_policy import human_height_target,colloquial_target
            if f['code']=='numeric_unit_without_source' and human_height_target(r,parser)==r['expected']:continue
            if f['code'] in ('feature_duo_boundary','feature_duo_pending') and colloquial_target(r,parser)==(r['operator'],r['expected']):continue
            if f['code']=='accessory_as_purchase' and r['attribute'] in ('gift_count','included_component_count') and r['subject']['kind'] in ('gift','component') and r['review_status']=='approved':
                if colloquial_target(r,parser)==(r['operator'],r['expected']):continue
                from semantic_review_decisions import strength_decision
                if strength_decision(c['query'],r):continue
        if f['code'] in ('hope_strength_pending','soft_clause_mandatory','explicit_hard_clause_optional','mixed_strength','budget_strength_source','weak_strength','permitted_material_strength') and len(parts)==2 and parts[0] in DIMS and parts[1].isdigit():
            from semantic_review_decisions import strength_decision
            if strength_decision(c['query'],c[parts[0]][int(parts[1])]):continue
        if f['code']=='non_budget_tolerance' and len(parts)==2 and parts[0] in DIMS and parts[1].isdigit():
            r=c[parts[0]][int(parts[1])]
            if r['operator']=='matches_label':
                from semantic_remaining_policy import validate_label
                validate_label(r)
                continue
            from semantic_remaining_policy import colloquial_target
            if colloquial_target(r,parser)==(r['operator'],r['expected']):continue
            if r['operator']=='approximately':
                from semantic_acceptance_representation import validate_approximate_target
                validate_approximate_target(r,parser,c['query'])
                continue
        if f['code'] in ('invented_component_count','component_count_source_unproven','unresolved_count_interpretation') and len(parts)==2 and parts[0] in DIMS and parts[1].isdigit():
            from semantic_component_group_proofs import group_proof
            if group_proof(c[parts[0]][int(parts[1])],parser):continue
        if f['code']=='subjective_standard' and len(parts)==2 and parts[0] in DIMS and parts[1].isdigit():
            r=c[parts[0]][int(parts[1])]
            if '最新' not in str(r['expected']) and explicit_claim_standard(r):continue
            from semantic_migration_policy import latest_declaration
            if latest_declaration(r):continue
        if f['code']=='axis_scope_unproven' and len(parts)==2 and parts[0] in DIMS and parts[1].isdigit():
            if square_proof(c[parts[0]][int(parts[1])],parser):continue
        result.append(f)
    return result


def certificates(c,parser):
    result=promotion_proofs(c,parser)+selection_introduction_proofs(c,parser)
    for d in DIMS:
        for i,r in enumerate(c[d]):
            p=square_proof(r,parser)
            if p:result.append({**p,'location':f'/{d}/{i}'})
            from semantic_component_group_proofs import group_proof
            group=group_proof(r,parser)
            if group:result.append({**group,'location':f'/{d}/{i}'})
    return result


def abbreviated_money(phrase,fallback):
    # User confirmed 2026-11-07: 四百五=450, 三千五=3500,
    # in explicit monetary context only. Quantity parsing is unchanged.
    if not re.search(r'预算|价格|售价|价位|单价|总价|元|块钱|块',phrase):return fallback(phrase)
    from semantic_migration_policy import round_tens_duo
    duo=round_tens_duo(phrase)
    if duo is not None:return duo
    from semantic_migration_policy import adjacent_tens_money
    ranged=adjacent_tens_money(phrase)
    if ranged is not None:
        result=fallback(ranged)
        return result[0],result[1],'user_confirmed_adjacent_tens_upper/'+result[2]
    from semantic_migration_policy import elided_scale_money
    elided=elided_scale_money(phrase)
    if elided is not None:
        result=fallback(elided)
        return result[0],result[1],'user_confirmed_elided_money/'+result[2]
    from semantic_migration_policy import explicit_money_multiplier
    expanded=explicit_money_multiplier(phrase)
    if expanded!=phrase:
        result=fallback(expanded)
        return result[0],result[1],'explicit_money_multiplier/'+result[2]
    pattern=r'(?<![零〇一二两三四五六七八九十百千万])([一二两三四五六七八九])([百千])([一二两三四五六七八九])(?![零〇一二两三四五六七八九十百千万])'
    hits=list(re.finditer(pattern,phrase))
    if len(hits)!=1:return fallback(phrase)
    m=hits[0];digits={c:i for i,c in enumerate('零一二三四五六七八九')};digits['两']=2
    scale=100 if m[2]=='百' else 1000
    amount=digits[m[1]]*scale+digits[m[3]]*(scale//10)
    result=fallback(phrase[:m.start()]+str(amount)+phrase[m.end():])
    return result[0],result[1],'user_confirmed_elided_money/'+result[2]


def monetary_context(query,phrase,context):
    if re.search(r'[一二两三四五六七八九][百千][一二两三四五六七八九]',phrase) and not re.search(r'预算|价格|售价|价位|单价|元|块',context):
        pos=query.find(phrase)
        previous=re.search(r'(?:预算|价格|售价|价位|单价)[^，,。；;！？!?\n]{0,12}[，,]\s*$',query[:pos])
        if previous:return query[previous.start():pos+len(phrase)]
    return context


def explicit_claim_standard(r):
    """User confirmed: qualitative claims can be checked as public assertions.

    Requires the same literal property in the source. It does not settle hope,
    approximation, numeric standards, damaged text, or unstated predicates.
    """
    if r['subtype'] not in ('function','purpose','style','craft','pattern','form_factor','condition','customization','warranty'):return False
    if r['expected']['type']=='text_list' and r['operator']=='contains_all':
        values=r['expected']['values']
        return any(all(v in e['text'] for v in values) and not re.search(r'或|任选|不|非|别|无需|不用|禁止|[?？]',e['text']) for e in r['evidence'])
    if r['expected']['type']!='text' or r['operator']!='eq':return False
    value=r['expected']['value']
    from semantic_fact_language_v3 import canonical
    for e in r['evidence']:
        text=e['text']
        if re.search(r'[?？]|听说|传闻',text):continue
        if value in text:
            prefix=text[max(0,text.index(value)-4):text.index(value)]
            if re.search(r'不|非|别|无需|不用|禁止',prefix):continue
            return True
        if canonical(r['subtype'],value)==canonical(r['subtype'],text):return True
    return False


def resolve_claim_mentions(c):
    audit=[];kept=[]
    for p in c['pending_mentions']:
        matching=[r for d in DIMS for r in c[d] if r['importance']!='uncertain' and explicit_claim_standard(r)
                  and any(e==p['evidence'] for e in r['evidence'])]
        if p['reason']=='unverifiable_subjective' and matching:
            audit.append({'kind':'user_confirmed_public_claim_standard','resolved_pending':p,
                          'requirement_ids':[r['id'] for r in matching],
                          'standard':'explicit public assertion of the same property; no implicit quality inference',
                          'whole_contract_approved':False})
        else:kept.append(p)
    c['pending_mentions']=kept
    return audit


def postfix_preference_proof(r):
    if r['importance']!='preference' or r['expected']['type']!='text':return False
    pattern=re.escape(r['expected']['value'])+r'(?:一类的|这类的|这类|一类|的)?优先考虑'
    for e in r['evidence']:
        text=e['text']
        if re.fullmatch(pattern,text):return True
        # 也/还 are intervening particles, not a new requirement or a reason
        # to discard the directly attached 最好 modifier.
        prefix=r'(?:最好|优先|尽量|偏好|喜欢)(?:也|还)?(?:是|为|有)?'+re.escape(r['expected']['value'])
        if re.search(prefix,text) and not re.search(r'必须|务必|一定要|要求|不|非|别|禁止',text):return True
    return False


def filter_review_findings(c,findings):
    result=[]
    for f in findings:
        from semantic_reviewed_scopes import reviewed
        if reviewed(c,f):continue
        parts=f.get('location','').strip('/').split('/')
        if f['code'] in ('preference_scope_unproven','hope_strength_pending','soft_clause_mandatory','explicit_hard_clause_optional','mixed_strength') and len(parts)==2 and parts[0] in DIMS and parts[1].isdigit():
            from semantic_review_decisions import strength_decision
            if strength_decision(c['query'],c[parts[0]][int(parts[1])]):continue
        if f['code']=='preference_scope_unproven' and len(parts)==2 and parts[0] in DIMS and parts[1].isdigit():
            if postfix_preference_proof(c[parts[0]][int(parts[1])]):continue
        result.append(f)
    return result


def selection_introduction_proofs(c,parser):
    # User confirmed: an introductory "需要一个 <primary product>" selects an
    # offer when a separately explicit same-item promotion sets its contents.
    # Only the same primary category and physical counter are eligible. An
    # explicit "只需要/只买/订单数量" or a separate named gift is not covered.
    promotions=promotion_proofs(c,parser)
    categories=[r for r in c['category'] if r['expected']['type']=='text' and r['operator']=='eq'
                and r['subject']['kind']=='primary_product']
    if not promotions or len(categories)!=1:return []
    name=categories[0]['expected']['value'];proofs=[]
    for candidate in explicit_counts(c['query'],parser):
        if candidate['role']!='purchase' or candidate['count']!=1 or candidate['unit']!='piece' or candidate['ambiguous']:continue
        if not re.fullmatch(r'(?:需要|找)(?:一|1)个',candidate['source']):continue
        before=c['query'][max(0,candidate['start']-8):candidate['start']]
        if re.search(r'只|仅|恰好|只能|订单|数量|不|非|别',before):continue
        tail=re.split(r'[，,。；;！？!?\n]',c['query'][candidate['end']:],maxsplit=1)[0]
        if name not in tail or re.search(r'和|以及|另外|另一个|再买',tail):continue
        if any(p['start']>candidate['end'] and c['specifications'][int(p['location'].split('/')[-1])]['expected']['unit']=='piece' for p in promotions if p['location'].startswith('/specifications/')):
            proofs.append({'kind':'source_scope_proof','rule':'user_confirmed_selection_intro_with_explicit_primary_bundle',
                           'start':candidate['start'],'end':candidate['end'],'source':candidate['source'],'category':name})
    return proofs


def resolve_selection_mentions(c,parser):
    proofs=selection_introduction_proofs(c,parser);kept=[];audit=[]
    for p in c['pending_mentions']:
        matching=[proof for proof in proofs if p['reason']=='ambiguous_meaning' and p['evidence']['text']==proof['source']
                  and p['evidence']['start']<=proof['start']<proof['end']<=p['evidence']['end']]
        if matching:audit.append({'kind':'user_confirmed_selection_intro_resolution','resolved_pending':p,'proofs':matching})
        else:kept.append(p)
    c['pending_mentions']=kept
    return audit


def resolve_reviewed_claim_mentions(c):
    """Apply confirmed assertion standard after an explicit field review.

    Never invoked by the LLM extractor. Numeric/tolerance/strength/source-loss
    pending reasons cannot be resolved by this path.
    """
    qualitative={'function','purpose','style','craft','pattern','form_factor','condition','color','brand_positioning'}
    kept=[];audit=[]
    for p in c['pending_mentions']:
        candidates=[r for d in DIMS for r in c[d] if r['review_status']=='approved' and r['importance']!='uncertain'
                    and r['subtype'] in qualitative and r['expected']['type'] in ('text','text_list')
                    and any(e['start']<=p['evidence']['start']<p['evidence']['end']<=e['end'] for e in r['evidence'])]
        if p['reason']=='unverifiable_subjective' and candidates:
            audit.append({'kind':'reviewed_public_assertion_standard','resolved_pending':p,
                          'reviewed_requirement_ids':[r['id'] for r in candidates],
                          'standard':'public assertion of the reviewed property; absence remains unknown'})
        else:kept.append(p)
    c['pending_mentions']=kept
    return audit
