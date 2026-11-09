"""Offline source-backed product/SKU comparisons; no training or LLM calls.

Public product fields are the only automatic evidence. instructions, annotated
target options, reviews, and unselected variants never become purchased facts.
Optional reviewed catalog facts supply semantics beyond literal structured data.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import sys
import unicodedata
import copy
from pathlib import Path

VERSION = 'semantic-product-verifier-v4-pilot-evidence-r3'
ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env'
if str(ENGINE) not in sys.path:
    sys.path.insert(0,str(ENGINE))
from web_agent_site.engine.variant_price import resolve_variant_price

PRIMARY = {'kind':'primary_product','name':'主商品'}
ORDER = {'kind':'order','name':'本次订单'}
NUM = r'(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)'
CONVERSIONS = {'g':('mass',1), 'kg':('mass',1000), 'jin':('mass',500),
               'mm':('length',1), 'cm':('length',10), 'm':('length',1000), 'inch':('length',25.4),
               'ml':('volume',1), 'l':('volume',1000), 'CNY':('money',1)}


def normalized(value):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(value))).casefold()


def fingerprint(product):
    # Instructions/target answers have no role in evidence or identity hashing.
    public = {k:v for k,v in product.items() if k not in ('instructions','reviews')}
    return hashlib.sha256(json.dumps(public,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def contract_fingerprint(contract):
    return hashlib.sha256(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def pointer(document, path):
    if not isinstance(path,str) or not path.startswith('/'):
        raise ValueError('Evidence requires a JSON pointer')
    tokens = [t.replace('~1','/').replace('~0','~') for t in path[1:].split('/')]
    if not tokens or tokens[0] in ('instructions','reviews'):
        raise ValueError('Task answers/reviews are not product facts')
    value = document
    for token in tokens:
        value = value[int(token)] if isinstance(value,list) else value[token]
    return value


def source(path, value):
    return {'pointer':path, 'value':value}


def fact(dimension, subtype, attribute, value, evidence, *, subject=None, scope='product', complete=True):
    return {'dimension':dimension,'subtype':subtype,'attribute':attribute,'subject':subject or PRIMARY,
            'value':value,'evidence':evidence,'scope':scope,'complete':complete,'origin':'deterministic_public_field'}


def snapshot(product, selected_options):
    if not isinstance(selected_options,dict):
        raise ValueError('Selected options must be an object')
    axes = product.get('customization_options') or {}
    if not isinstance(axes,dict):
        raise ValueError('Malformed option catalog')
    selected = []; errors = []
    for axis,value in selected_options.items():
        entries = axes.get(axis)
        if not isinstance(entries,list):
            errors.append({'reason':'unknown_axis','axis':axis}); continue
        matches = [(i,e) for i,e in enumerate(entries) if isinstance(e,dict) and e.get('value') == value]
        if len(matches) != 1:
            errors.append({'reason':'unknown_or_duplicate_value','axis':axis,'value':value}); continue
        i,entry = matches[0]
        if entry.get('is_available') is False:
            errors.append({'reason':'unavailable_value','axis':axis,'value':value}); continue
        escaped = str(axis).replace('~','~0').replace('/','~1')
        selected.append({'axis':axis,'value':value,'entry':entry,'pointer':f'/customization_options/{escaped}/{i}/value'})
    price = resolve_variant_price(product,selected_options) if not errors else {'status':'unverifiable','price':None,'method':'invalid_selection'}
    return {'product_asin':str(product.get('asin','')), 'product_sha256':fingerprint(product),
            'selected_options':dict(selected_options), 'selected':selected, 'selection_errors':errors,
            'missing_axes':[a for a in axes if a not in selected_options], 'price_resolution':price}


def automatic_facts(product, state):
    from semantic_contract_v2_r7_final12 import chinese_number
    facts = []
    for field,dimension,subtype,attribute in (('brand','brand','brand','brand'),('model','model','product_model','product_model')):
        if isinstance(product.get(field),str) and product[field].strip():
            facts.append(fact(dimension,subtype,attribute,{'type':'text','value':product[field]},[source('/'+field,product[field])]))
    # An exact common catalog declaration supports snapshot market status.
    # Platform names, recommendations, old dated claims and negated labels do
    # not become hot-selling evidence through substring matching.
    for index,claim in enumerate(product.get('attribute') or []):
        if claim in ('网络热卖款','热卖款','热卖'):
            facts.append(fact('specifications','condition','condition',{'type':'text','value':claim},
                              [source(f'/attribute/{index}',claim)],complete=False))
    # A completeness assertion is credited only for this product without
    # variant choices, or for an explicitly selected SKU below. A generic
    # title advertising optional accessories is never sufficient.
    complete_claims=('配件齐全','全套配件','全套附件')
    if not product.get('customization_options'):
        for index,claim in enumerate(product.get('attribute') or []):
            if claim in complete_claims:
                facts.append(fact('functions','function','function',{'type':'text','value':'配件齐全'},
                                  [source(f'/attribute/{index}',claim)],complete=False))
    if state['selection_errors']:
        return facts
    for selected in state['selected']:
        if selected['axis']=='鞋码' and re.fullmatch(r'\d+(?:\.\d+)?',selected['value']):
            facts.append(fact('specifications','size','size_label',{'type':'text','value':selected['value']},
                              [source(selected['pointer'],selected['value'])],scope='sku'))
        text = selected['value']; evidence = [source(selected['pointer'],text)]
        if text in complete_claims:
            facts.append(fact('functions','function','function',{'type':'text','value':'配件齐全'},evidence,scope='sku',complete=False))
        if selected['axis'] in ('尺寸','大小','规格'):
            facts.append(fact('specifications','dimensions','dimensions',{'type':'text','value':text},evidence,scope='sku'))
            facts.append(fact('specifications','size','size_label',{'type':'text','value':text},evidence,scope='sku'))
        promotions = list(re.finditer('买('+NUM+')送('+NUM+')',text))
        if len(promotions)==1:
            match = promotions[0]; paid,gift = [int(chinese_number(x)) for x in match.groups()]
            # A promotion label states quantities, but not a physical unit or
            # that gifts are necessarily identical. Those stay unknown.
            facts.append(fact('specifications','bundle','promotion',{'type':'bundle','paid_quantity':paid,'gift_quantity':gift,
                              'received_quantity':paid+gift,'unit':None,'same_item':None},evidence,scope='sku'))
        counts = list(re.finditer('('+NUM+')(件套|小包|支装|瓶装|袋装|盒装)',text))
        if len(counts)==1:
            match=counts[0]; unit={'件套':'piece','小包':'bag','支装':'piece','瓶装':'bottle','袋装':'bag','盒装':'box'}[match[2]]
            facts.append(fact('specifications','packaging','pack_count',{'type':'number','value':float(chinese_number(match[1])),'unit':unit},evidence,scope='sku'))
        # A single item label can disprove a promotion's quantity, but must not
        # invent a paid/gift split or a purchase count for a multi-item pack.
        one = re.search('('+NUM+')(个|支|件)$',text)
        if one and not promotions:
            facts.append(fact('specifications','bundle','promotion',{'type':'bundle','paid_quantity':None,'gift_quantity':None,
                              'received_quantity':int(chinese_number(one[1])),'unit':'piece','same_item':None},evidence,scope='sku'))
        # Exact, simple color labels only. A complex 颜色分类 label may be a
        # bundle; no substring color extraction or use of other SKU colors.
        if selected['axis'] in ('颜色','颜色分类','color') and re.fullmatch(r'(?:黑色|白色|红色|蓝色|绿色|黄色|金色|银色|粉色|灰色|紫色|橙色|棕色)',text):
            facts.append(fact('specifications','color','color',{'type':'text','value':text},evidence,scope='sku'))
    from semantic_catalog_rules_v2 import selected_facts
    facts.extend(selected_facts(product,state))
    from semantic_catalog_rules_v3 import extended_facts
    facts.extend(extended_facts(product,state))
    from semantic_public_matching import automatic_pack_facts
    facts.extend(automatic_pack_facts(product,state))
    return facts


def reviewed_facts(product,state,records):
    from semantic_product_facts_v3 import validate_fact
    from jsonschema import ValidationError
    accepted=[]; rejected=[]
    for index,record in enumerate(records):
        try:
            validate_fact(record)
            if record.get('review_status')!='approved' or record.get('product_asin')!=state['product_asin']:
                raise ValueError('Fact is unreviewed or belongs to another product')
            if record.get('product_sha256')!=state['product_sha256']:
                raise ValueError('Catalog fingerprint differs')
            if record.get('scope') not in ('product','sku'):
                raise ValueError('Fact needs explicit product/SKU scope')
            if record.get('polarity','positive') not in ('positive','negative'):
                raise ValueError('Unknown fact polarity')
            if not isinstance(record.get('complete'),bool):
                raise ValueError('Fact completeness must be explicit boolean')
            # SKU claims must bind all selected axes to prevent carry-over
            # between variants. Product-scope facts must be reviewed as common.
            if record['scope']=='sku':
                applies=(all(state['selected_options'].get(k)==v for k,v in record['options'].items())
                         if record.get('selector_mode')=='contains' else record['options']==state['selected_options'])
                # A cache contains facts for many variants. A valid fact for a
                # different variant is not an input error and cannot leak in.
                if not applies:continue
            if not record.get('evidence'):
                raise ValueError('Fact has no public source')
            for e in record['evidence']:
                if e['pointer'].startswith('/customization_options/'):
                    allowed=[s['pointer'].rsplit('/',1)[0]+'/' for s in state['selected']]
                    if record['scope']!='sku' or not any(e['pointer'].startswith(prefix) for prefix in allowed):
                        raise ValueError('Unselected option is not a purchased fact')
                if e['pointer'].startswith('/variant_combinations/'):
                    index=int(e['pointer'].split('/')[2])
                    combination=product['variant_combinations'][index]
                    if record['scope']!='sku' or combination.get('options')!=state['selected_options']:
                        raise ValueError('Unselected combination is not a purchased fact')
                actual=pointer(product,e['pointer'])
                if 'text' in e:
                    if not isinstance(actual,str) or actual[e['start']:e['end']]!=e['text'] or not 0<=e['start']<e['end']<=len(actual):
                        raise ValueError('Fact evidence span differs')
                elif actual!=e.get('value'):
                    raise ValueError('Fact evidence value differs')
            if record['value'].get('unit')=='percent':
                value=record['value']
                if value['type']!='number':raise ValueError('Percentage interval facts need a separately supported interval source')
                texts=[e.get('text',pointer(product,e['pointer'])) for e in record['evidence']]
                literals=[float(n) for text in texts if isinstance(text,str) for n in re.findall(r'(\d+(?:\.\d+)?)\s*[%％]',text)]
                if value['value'] not in literals:raise ValueError('Numeric percentage is not stated in the public evidence')
            for key in ('dimension','subtype','attribute','subject','value','complete'):
                if key not in record:raise ValueError('Missing fact '+key)
            accepted.append({**record,'origin':'reviewed_public_catalog_fact'})
        except (KeyError,ValueError,TypeError,IndexError,ValidationError) as error:
            rejected.append({'index':index,'reason':str(error)})
    return accepted,rejected


def convert(value, unit):
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
        raise ValueError('Non-finite or nonnumeric fact')
    return CONVERSIONS.get(unit,(unit,1))[0],float(value)*CONVERSIONS.get(unit,(unit,1))[1]


def compare_value(requirement, actual, *, complete=True):
    from semantic_fact_language_v3 import canonical
    requirement=copy.deepcopy(requirement);actual=copy.deepcopy(actual)
    if requirement['operator']=='matches_label':
        from semantic_remaining_policy import label_normalized
        values=actual.get('values', [actual.get('value')]) if actual.get('type') in ('text','text_list') else []
        return 'pass' if any(isinstance(v,str) and label_normalized(v)==label_normalized(requirement['expected']['value']) for v in values) else 'unknown'
    subtype=requirement.get('subtype','')
    for value in (requirement['expected'],actual):
        if value['type']=='text':value['value']=canonical(subtype,value['value'])
        elif value['type']=='text_list':value['values']=[canonical(subtype,x) for x in value['values']]
    expected=requirement['expected']; op=requirement['operator']; kind=expected['type']
    if (requirement.get('attribute')=='dimensions' and op=='eq'
            and kind==actual.get('type')=='text'):
        from semantic_public_matching import ordered_dimensions
        if normalized(expected['value'])==normalized(actual['value']):return 'pass'
        want,got=ordered_dimensions(expected['value']),ordered_dimensions(actual['value'])
        return 'pass' if want is not None and got is not None and want==got else 'unknown'
    if (kind==actual.get('type')=='range' and op=='approximately'
            and requirement.get('attribute') in ('wearer_height','wearer_weight_limit')):
        probe={**requirement,'operator':'covers_range'}
        return 'pass' if compare_value(probe,actual,complete=complete)=='pass' else 'unknown'
    if op=='covers_range':
        if actual.get('type')!='range' or not complete:return 'unknown'
        for name,is_lower in (('lower',True),('upper',False)):
            if actual.get(name) is None:return 'unknown'
            family,want=convert(expected[name],expected['unit']);other,got=convert(actual[name],actual['unit'])
            if family!=other:return 'unknown'
            if (got>want if is_lower else got<want):return 'fail'
            if got==want and expected[name+'_inclusive'] and not actual[name+'_inclusive']:return 'fail'
        return 'pass'
    if kind=='member_ranges' and actual.get('type')=='member_values' and op=='matches_members':
        if actual.get('member_attribute')!=expected['member_attribute'] or actual['unit']!=expected['unit']:return 'unknown'
        if not complete:return 'unknown'
        values=actual['values'];slots=[g for g in expected['groups'] for _ in range(g['count'])]
        if len(values)!=len(slots):return 'fail'
        assigned={}
        def augment(i,seen):
            group=slots[i]
            for j,value in enumerate(values):
                if j in seen or not group['lower']<=value<=group['upper']:continue
                seen.add(j)
                if j not in assigned or augment(assigned[j],seen):assigned[j]=i;return True
            return False
        return 'pass' if all(augment(i,set()) for i in range(len(slots))) else 'fail'
    if kind=='number' and actual.get('type')=='range' and requirement.get('attribute') in ('wearer_height','wearer_weight_limit') and op in ('eq','approximately'):
        # A wearer's point measurement must lie inside the SKU fit interval.
        point=requirement['expected']
        probe={'expected':actual,'operator':'between'}
        result=compare_value(probe,point,complete=complete)
        return 'unknown' if op=='approximately' and result!='pass' else result
    if kind=='number' and actual.get('type')=='number':
        family,want=convert(expected['value'],expected['unit']); other,got=convert(actual['value'],actual['unit'])
        if family!=other:return 'unknown'
        if op=='approximately':return 'pass' if got==want else 'unknown'
        return 'pass' if {'eq':got==want,'at_most':got<=want,'at_least':got>=want}.get(op,False) else 'fail' if op in ('eq','at_most','at_least') else 'unknown'
    if kind=='range' and actual.get('type')=='number':
        family,got=convert(actual['value'],actual['unit'])
        for bound, lower in (('lower',True),('upper',False)):
            if expected[bound] is None:continue
            other,limit=convert(expected[bound],expected['unit'])
            if family!=other:return 'unknown'
            inclusive=expected[bound+'_inclusive']
            if (got<limit or got==limit and not inclusive) if lower else (got>limit or got==limit and not inclusive):return 'unknown' if op=='approximately' else 'fail'
        return 'pass'
    if kind=='text' and actual.get('type')=='text' and op=='eq':
        return 'pass' if normalized(expected['value'])==normalized(actual['value']) else 'fail' if complete else 'unknown'
    if kind=='text' and actual.get('type')=='text_list' and op=='eq':
        return 'pass' if normalized(expected['value']) in {normalized(x) for x in actual['values']} else 'fail' if complete else 'unknown'
    if kind=='text_list' and actual.get('type') in ('text','text_list'):
        values={normalized(x) for x in actual.get('values',[actual.get('value')])}
        required={normalized(x) for x in expected['values']}
        if op=='one_of':return 'pass' if values & required else 'fail' if complete else 'unknown'
        if op=='contains_all':return 'pass' if required<=values else 'fail' if complete else 'unknown'
        if op=='excludes':return 'fail' if required & values else 'pass' if complete else 'unknown'
    if kind=='components' and actual.get('type')=='components':
        items={normalized(i['name']):i for i in actual['items']}
        if op=='excludes':
            return 'fail' if any(normalized(i['name']) in items for i in expected['items']) else 'pass' if complete else 'unknown'
        if op!='contains_all':return 'unknown'
        unknown=False
        for item in expected['items']:
            got=items.get(normalized(item['name']))
            if got is None:
                if complete:return 'fail'
                unknown=True;continue
            if item.get('quantity') is not None:
                if got.get('quantity') is None:unknown=True
                elif got.get('unit')!=item.get('unit'):unknown=True
                elif got['quantity']!=item['quantity']:return 'fail'
        return 'unknown' if unknown else 'pass'
    if kind=='bundle' and actual.get('type')=='bundle' and op=='matches_bundle':
        unknown=False
        for key in ('paid_quantity','gift_quantity','received_quantity','unit','same_item'):
            want=expected.get(key)
            # Paid + gift determines a required total, even if not repeated.
            if key=='received_quantity' and want is None and expected.get('paid_quantity') is not None and expected.get('gift_quantity') is not None:
                want=expected['paid_quantity']+expected['gift_quantity']
            if want is None:continue
            got=actual.get(key)
            if got is None:unknown=True
            elif want!=got:return 'fail'
        return 'unknown' if unknown else 'pass'
    return 'unknown'


def compare_requirement(requirement,dimension,product,state,facts):
    from semantic_catalog_rules_v2 import explicit_requirement_value
    rewrite=explicit_requirement_value(requirement)
    if rewrite:requirement={**requirement,**rewrite}
    if requirement['importance']=='uncertain':return {'status':'unknown','reason':'uncertain_requirement','evidence':[]}
    if state['selection_errors']:return {'status':'unknown','reason':'invalid_selection','evidence':state['selection_errors']}
    if dimension=='budget':
        # The simulator has no independent cart quantity. Its price is for
        # the selected sale variant; do not multiply by a requested count.
        price=state['price_resolution']
        if price['status']!='pass':return {'status':'unknown','reason':'price_unverifiable','evidence':[price]}
        if requirement['subtype']=='unit_price':
            basis=requirement.get('price_basis')
            if not basis:return {'status':'unknown','reason':'unit_price_basis_missing','evidence':[]}
            if basis['unit']=='sale_unit':
                quantities=[f for f in facts if f['subject']==PRIMARY and (f['subtype'],f['attribute'])==('transaction_terms','sale_quantity') and f['value']['type']=='number']
                units=[f for f in facts if f['subject']==PRIMARY and (f['subtype'],f['attribute'])==('transaction_terms','sale_unit_quantity') and f['value']['type']=='number']
                candidates=[]
                for quantity in quantities:
                    qfamily,qvalue=convert(quantity['value']['value'],quantity['value']['unit'])
                    for unit in units:
                        ufamily,uvalue=convert(unit['value']['value'],unit['value']['unit'])
                        if qfamily==ufamily and qvalue>0 and uvalue>0:
                            candidates.append((price['price']/qvalue*uvalue*basis['amount'],quantity,unit))
                rates={round(v,10) for v,_,_ in candidates}
                if len(rates)!=1:return {'status':'unknown','reason':'sale_unit_evidence_missing_or_conflicting','evidence':[price]}
                actual={'type':'number','value':next(iter(rates)),'unit':'CNY'}
                return {'status':compare_value(requirement,actual),'reason':'catalog_supported_sale_unit_price','actual':actual,
                        'evidence':[price]+[{'sale_quantity':q['evidence'],'sale_unit_quantity':u['evidence']} for _,q,u in candidates]}
            candidates=[]
            for f in facts:
                if f['subject']==PRIMARY and (f['subtype'],f['attribute']) in (('quantity','purchase_count'),('packaging','pack_count'),('transaction_terms','sale_quantity')) and f['value']['type']=='number':
                    family,amount=convert(f['value']['value'],f['value']['unit'])
                    other,denom=convert(basis['amount'],basis['unit'])
                    if family==other and amount>0:candidates.append((price['price']/amount*denom,f))
            prices={round(value,10) for value,_ in candidates}
            if len(prices)!=1:return {'status':'unknown','reason':'unit_price_sale_quantity_missing_or_conflicting','evidence':[price]}
            actual={'type':'number','value':next(iter(prices)),'unit':'CNY'}
            return {'status':compare_value(requirement,actual),'reason':'source_supported_unit_price','actual':actual,
                    'evidence':[price]+[f['evidence'] for _,f in candidates]}
        if requirement['subtype']!='total_price' or requirement['subject']!=ORDER or requirement.get('price_basis') is not None:
            return {'status':'unknown','reason':'unsupported_price_scope','evidence':[]}
        actual={'type':'number','value':price['price'],'unit':'CNY'}
        return {'status':compare_value(requirement,actual),'reason':'resolved_selected_variant_price','actual':actual,'evidence':[price]}
    from semantic_public_matching import subject_matches, public_claim, selected_component_claim
    matches=[f for f in facts if (f['dimension'],f['subtype'],f['attribute'])==
             (dimension,requirement['subtype'],requirement['attribute']) and subject_matches(requirement,f)]
    gift=selected_component_claim(requirement,product,state)
    if gift is not None and not matches:return gift
    if requirement['subtype']=='components':
        from semantic_component_comparison import compare_components
        joined=compare_components(requirement,matches)
        if joined is not None:return joined
    claim_like=requirement['subtype'] in ('function','purpose','target_user','use_environment','material','compatibility',
                'condition','craft','style','pattern','form_factor','transaction_terms','brand_positioning')
    if claim_like and requirement['operator']=='contains_all' and requirement['expected']['type']=='text_list':
        from semantic_fact_language_v3 import canonical
        want={canonical(requirement['subtype'],v) for v in requirement['expected']['values']}
        positive=set();negative=set();evidence=[]
        for f in matches:
            if f['value']['type'] not in ('text','text_list'):continue
            values={canonical(requirement['subtype'],v) for v in f['value'].get('values',[f['value'].get('value')])}
            if f.get('polarity','positive')=='negative':negative.update(values)
            else:positive.update(values)
            evidence.append({'actual':f['value'],'evidence':f['evidence'],'polarity':f.get('polarity','positive')})
        conflict=want&positive&negative
        status='unknown' if conflict else 'fail' if want&negative else 'pass' if want<=positive else 'unknown'
        return {'status':status,'reason':'conflicting_facts' if conflict else 'conjunctive_public_claims',
                'satisfied_parts':sorted(want&positive),'unknown_parts':sorted(want-positive-negative),
                'failed_parts':sorted(want&negative),'evidence':evidence}
    decisions=[]
    for f in matches:
        try:
            # 防水 and 防尘 are independent positive claims, not mutually
            # exclusive values of a single field. Only a complete set or an
            # explicit reviewed negative can establish missing capability.
            claim_like=requirement['subtype'] in ('function','purpose','target_user','use_environment','material','compatibility',
                'condition','craft','style','pattern','form_factor','transaction_terms','brand_positioning')
            complete=f['complete'] and not (claim_like and f['value']['type']=='text')
            if requirement['subtype']=='condition' and requirement['expected'].get('value') in ('未开刃','已开刃') and f['value'].get('value') in ('未开刃','已开刃'):
                complete=f['complete']
            if requirement['subtype']=='color' and requirement['expected']['type']=='text':
                from semantic_fact_language_v2 import canonical
                from semantic_catalog_rules_v2 import COLORS
                # A free description such as 偏白/好气色 is not an exact
                # shade code. Different wording alone cannot disprove it.
                complete=complete and canonical('color',requirement['expected']['value']) in {canonical('color',v) for v in COLORS}
            status=compare_value(requirement,f['value'],complete=complete)
            if f.get('polarity','positive')=='positive' and requirement['operator']=='eq' and requirement['expected']['type']==f['value']['type']=='text':
                from semantic_fact_language_v3 import implies
                if implies(requirement['subtype'],f['value']['value'],requirement['expected']['value']):status='pass'
            if requirement['subtype']=='color' and status=='fail' and requirement['operator']=='eq' and requirement['expected']['type']==f['value']['type']=='text':
                # Different shades of a family are not proven incompatible.
                # This conservative exception never invents a successful match.
                families=[{'白色','米白色','象牙白','象牙色'}, {'灰色','浅灰色','深灰色'},
                          {'蓝色','浅蓝色','深蓝色','暗蓝色'}, {'绿色','浅绿色','深绿色','墨绿色'}]
                want=normalized(requirement['expected']['value']);got=normalized(f['value']['value'])
                if any(want in family and got in family for family in families):status='unknown'
            if f.get('polarity','positive')=='negative':status='fail' if status=='pass' else 'unknown'
        except (KeyError,TypeError,ValueError):status='unknown'
        decisions.append({'status':status,'actual':f['value'],'evidence':f['evidence'],'origin':f['origin']})
    known={x['status'] for x in decisions if x['status']!='unknown'}
    if len(known)>1:return {'status':'unknown','reason':'conflicting_facts','evidence':decisions}
    if known:return {'status':next(iter(known)),'reason':'typed_source_fact','evidence':decisions}
    literal=public_claim(requirement,dimension,product,state)
    if literal is not None:return literal
    # Taxonomy membership supports exact category names only. Absence is not
    # a mismatch without a reviewed semantic category mapping.
    if dimension=='category' and requirement['expected']['type']=='text' and requirement['operator']=='eq':
        value=product.get('category','')
        if normalized(requirement['expected']['value']) in {normalized(x) for x in re.split('[›>]',value)}:
            return {'status':'pass','reason':'literal_category_node','evidence':[source('/category',value)]}
    # Product attributes are explicit common claims, not the title containing
    # every available variant. Only exact whole-value equality is supported.
    if (requirement['subject']==PRIMARY and requirement['subtype'] in ('function','material')
            and requirement['expected']['type']=='text' and requirement['operator']=='eq'
            and normalized(requirement['expected']['value']) not in ('配件齐全','全套配件','全套附件')):
        for i,value in enumerate(product.get('attribute') or []):
            if isinstance(value,str) and normalized(value)==normalized(requirement['expected']['value']):
                return {'status':'pass','reason':'literal_structured_attribute','evidence':[source(f'/attribute/{i}',value)]}
    return {'status':'unknown','reason':'no_matching_supported_fact','evidence':decisions}


def verify(contract, product, selected_options, *, catalog_facts=(), purchase_executed=False):
    from semantic_contract_v2_r7_final12 import DIMENSIONS, validate
    validate(contract,source_query=contract['query'])
    state=snapshot(product,selected_options)
    extra,rejected=reviewed_facts(product,state,catalog_facts)
    facts=automatic_facts(product,state)+extra
    comparisons={}
    for dimension in DIMENSIONS:
        for requirement in contract[dimension]:
            result=compare_requirement(requirement,dimension,product,state,facts)
            comparisons[requirement['id']]={'dimension':dimension,'subtype':requirement['subtype'],'attribute':requirement['attribute'],**result}
    blockers=[]
    try:validate(contract,source_query=contract['query'],require_reviewed=True)
    except ValueError as error:blockers.append({'reason':'contract_not_reward_ready','detail':str(error)})
    if rejected:blockers.append({'reason':'rejected_catalog_facts','detail':rejected})
    if state['selection_errors']:blockers.append({'reason':'invalid_selection'})
    return {'version':VERSION,'task_id':contract['task_id'],'contract_sha256':contract_fingerprint(contract),'product_asin':state['product_asin'],
            'snapshot':state,'comparisons':comparisons,'statuses':{k:v['status'] for k,v in comparisons.items()},
            'contract_pending_mentions':contract['pending_mentions'],'readiness_blockers':blockers,
            'full_contract_reward_ready':not blockers and purchase_executed is True and all(r['status']!='unknown' for r in comparisons.values()),
            'selection_complete':not state['selection_errors'] and not state['missing_axes'],
            'purchase_executed':purchase_executed is True,
            'execution_complete':purchase_executed is True and not state['selection_errors'],
            'rejected_catalog_facts':rejected,'reward':None}
