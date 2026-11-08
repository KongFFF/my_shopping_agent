"""General audited r7 representation adapters. No task IDs or reward approval."""
import copy
import re
from semantic_repairs_v6 import preprocess as legacy_preprocess, extra_flags as legacy_extra_flags
from semantic_relations_v7 import NUM

DIMS = ('category','budget','functions','specifications','brand','model')
UNIT_WORDS = {'m':r'米|m', 'cm':r'厘米|公分|cm', 'mm':r'毫米|mm',
              'kg':r'千克|公斤|kg', 'g':r'克|g', 'jin':r'斤', 'ml':r'毫升|ml', 'l':r'升|l',
              'W':r'瓦|W'}


def length_unit_in_source(source):
    return bool(re.search(NUM+r'\s*(?:毫米|厘米|公分|英寸|寸|米|mm|cm|inch|m)(?![A-Za-z])',source,re.I))


def quote(r): return '，'.join(e['text'] for e in r['evidence'])


def add_pending(c, evidence, reason, importance):
    m={'evidence':copy.deepcopy(evidence),'reason':reason,'importance':importance}
    if m not in c['pending_mentions']: c['pending_mentions'].append(m)


def literal_measure(source, value, unit, parser, name=None):
    expr=UNIT_WORDS.get(unit)
    if not expr: return None
    pattern=rf'(?P<n>{NUM})\s*(?:{expr})(?![A-Za-z])'
    if name: pattern+=rf'\s*(?:的)?{re.escape(name)}'
    for m in re.finditer(pattern,source,re.I):
        try: same=float(parser(m['n']))==float(value)
        except ValueError: continue
        if same: return m.group()
    return None


def preprocess(raw, task, number_parser):
    c=copy.deepcopy(raw);audit=[]
    for dim in DIMS:
        additions=[]
        for index,r in enumerate(c.get(dim,[])):
            added_start=len(additions)
            path=f'/{dim}/{index}';before=copy.deepcopy(r);v=r.get('expected',{});source=quote(r)
            if not all(e['text'] in task['query'] for e in r['evidence']): continue
            rules=[]
            if r['subtype']=='components' and v.get('type')=='components':
                for item in v['items']:
                    # An explicit 50 metres of pipe has no component count 50.
                    # Preserve existence + physical measure, never fabricate 1.
                    if item['quantity'] is None or item['unit'] not in ('m','cm','mm'): continue
                    span=literal_measure(source,item['quantity'],item['unit'],number_parser,item['name'])
                    if span is None: continue
                    added=copy.deepcopy(r)
                    added.update(subtype='dimensions',attribute='dimensions',operator='eq',
                                 subject={'kind':'component','name':item['name']},
                                 expected={'type':'number','value':item['quantity'],'unit':item['unit']})
                    if task['query'].count(span)==1: added['evidence']=[{'text':span}]
                    additions.append(added);item.update(quantity=None,unit=None)
                    rules.append('physical_component_measure_not_piece_count')
            if (r['subtype']=='components' and v.get('type')=='number' and v.get('unit')=='W'
                    and literal_measure(source,v['value'],'W',number_parser)):
                r.update(subtype='function',attribute='power');rules.append('component_power_attribute')
            if (r['subtype']=='quantity' and r['attribute']=='purchase_count' and v.get('type') in ('number','range')
                    and r['subject']['kind'] in ('primary_product','order')
                    and v.get('unit') in ('kg','g','jin','l','ml')
                    and re.search(r'需求量|总量|总共|需要|购买|买|订购', source)
                    and not re.search(r'每|装',source)):
                r['attribute']='purchase_mass' if v['unit'] in ('kg','g','jin') else 'purchase_volume'
                rules.append('physical_order_amount_attribute')
            if (r['subtype']=='quantity' and r['attribute']=='purchase_count' and v.get('type')=='number'
                    and re.search(r'每(?:种|款)\s*'+NUM, source)):
                r['attribute']='per_variant_count';rules.append('explicit_per_variant_not_order_total')
            if r['subtype']=='attribute_count' and v.get('type') in ('number','range'):
                duo=list(re.finditer(r'('+NUM+r')多(?:个|种|档)',source))
                if len(duo)==1:
                    try: lower=float(number_parser(duo[0][1]))
                    except ValueError: lower=None
                    existing=v.get('value') if v['type']=='number' else v.get('lower')
                    if lower is not None and existing==lower:
                        r.update(operator='between',importance='uncertain',expected={'type':'range','lower':lower,'upper':None,
                                 'unit':'count','lower_inclusive':False,'upper_inclusive':False})
                        for e in r['evidence']: add_pending(c,e,'ambiguous_meaning','uncertain')
                        rules.append('confirmed_feature_duo_strict_lower_upper_pending')
            if (r['subtype']=='attribute_count' and r['attribute']=='tone_type_count'
                    and re.search(NUM+r'\s*种(?:不同)?(?:颜色|口味)', source)
                    and '音色' not in source):
                r['attribute']='other_feature_count';rules.append('color_flavor_count_not_audio_tone')
            if r['subtype']=='merchant' and re.search(r'由.+生产|.+(?:企业|厂家|工厂)生产',source) and not re.search(r'直接销售|直销|直供|直营',source):
                r.update(subtype='manufacturer',attribute='manufacturer');rules.append('manufacturer_not_merchant')
            if r['subtype']=='warranty' and v.get('type')=='text' and v.get('value') in ('正品保障','正品保证','正品'):
                r.update(subtype='condition',attribute='condition');rules.append('authenticity_not_warranty')
            if r['subtype']=='material' and v.get('type')=='text' and re.fullmatch(r'(?:可用水清洗|可水洗|支持水洗|可清洗)',v['value']):
                r.update(subtype='function',attribute='function');rules.append('washability_not_material')
            # Percent ranges also cannot borrow the dimensionless count unit.
            if v.get('type')=='range' and v.get('unit')=='count' and re.search(r'[%％]',source):
                r['operator']='eq';r['expected']={'type':'text','value':source}
                for e in r['evidence']: add_pending(c,e,'unsupported_unit',r['importance'])
                rules.append('percentage_range_preserve_source')
            if (v.get('type') in ('number','range') and v.get('unit') in ('mm','cm','m','inch')
                    and r['attribute'] in ('wearer_height','length','width','height','diameter','thickness')
                    and not length_unit_in_source(source)):
                r['operator']='eq';r['expected']={'type':'text','value':source}
                for e in r['evidence']: add_pending(c,e,'unsupported_unit',r['importance'])
                rules.append('missing_unit_preserve_source_pending')
            if rules:
                audit.append({'kind':'r7_representation_adapter','path':path,'rules':rules,'before':before,
                              'after':copy.deepcopy(r),'added':copy.deepcopy(additions[added_start:])})
        c[dim].extend(additions)
    from semantic_contract_v2_r7 import DIMENSIONS
    destinations={s:d for d,subs in DIMENSIONS.items() for s in subs}
    routed={d:[] for d in DIMS}
    for d in DIMS:
        for r in c[d]: routed[destinations.get(r['subtype'],d)].append(r)
    c.update(routed)
    c,old_audit=legacy_preprocess(c,task,number_parser)
    return c,audit+old_audit


def extra_flags(c):
    flags=legacy_extra_flags(c)
    positive=set();excluded=set()
    for dim in DIMS:
        for i,r in enumerate(c[dim]):
            path=f'/{dim}/{i}';v=r['expected'];source=quote(r)
            if v['type']=='components':
                for item in v['items']:
                    key=(r['subject']['kind'],r['subject']['name'],r['attribute'],item['name'])
                    if r['operator']=='excludes' and r['importance']=='required': excluded.add(key)
                    elif r['operator'] in ('contains_all','eq') and r['importance']=='required': positive.add(key)
                    negative=re.search(r'(?:不需要|无需|不要|不接受|不得含|不含|不带|禁止|排除)\s*(?:任何|附带的|随附的)?'+re.escape(item['name']),source)
                    if r['operator'] in ('contains_all','eq') and negative:
                        flags.append({'code':'negated_component_as_positive','location':path,
                                      'detail':'A negative source cannot establish a positive required component; do not automatically turn 不需要 into a prohibition either'})
                    if r['operator']=='excludes' and not negative:
                        flags.append({'code':'exclusion_source_unproven','location':path,
                                      'detail':'Each excluded item needs its own negative source scope; do not borrow negation from another object'})
                    if item['quantity'] is not None and re.search(r'(?:加|配|带|有|送)个',source):
                        flags.append({'code':'unresolved_count_interpretation','location':path,
                                      'detail':'Elided one classifier remains a user-policy question; no automatic count inference or whole-contract approval'})
            if r['operator']=='excludes' and v['type']=='text_list':
                for name in v['values']:
                    if not re.search(r'(?:不需要|无需|不要|不接受|不得含|不含|不带|禁止|排除)\s*'+re.escape(name),source):
                        flags.append({'code':'exclusion_source_unproven','location':path,
                                      'detail':'Excluded values require their own explicit negative source scope'})
            if r['subtype']=='quantity' and r['attribute']=='per_variant_count' and v['type']=='number':
                from semantic_contract_v2_r7 import chinese_number
                from semantic_coverage_v7 import UNITS, COUNTERS
                matches=list(re.finditer(r'每(?:种|款)\s*('+NUM+r')\s*('+COUNTERS+r')',source))
                valid=any(float(chinese_number(m[1]))==v['value'] and UNITS[m[2]]==v['unit'] for m in matches)
                if not valid:
                    flags.append({'code':'per_variant_count_source_mismatch','location':path,
                                  'detail':'Each-kind count and unit must agree with the explicit 每种/每款 source; do not infer the order total'})
            if v['type'] in ('number','range') and r['attribute'] in ('wearer_height','length','width','height','diameter','thickness'):
                if v['unit'] in ('mm','cm','m','inch') and not length_unit_in_source(source):
                    flags.append({'code':'numeric_unit_without_source','location':path,
                                  'detail':'A numeric dimension with no source unit must not acquire cm/mm/m/inch implicitly'})
                if r['attribute'] in ('diameter','length','width','height','thickness'):
                    axis={'diameter':r'直径|孔径|口径|径','length':r'长','width':r'宽','height':r'高','thickness':r'厚'}[r['attribute']]
                    if not re.search(axis,source):
                        flags.append({'code':'axis_scope_unproven','location':path,
                                      'detail':'Unit alone does not name a physical axis; preserve source and review the inferred axis'})
    if positive & excluded:
        flags.append({'code':'positive_exclusion_conflict','location':'/specifications',
                      'detail':'The same required component cannot simultaneously be included and excluded'})
    return flags
