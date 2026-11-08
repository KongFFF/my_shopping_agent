"""User-confirmed remaining-source policies; no task-ID language rules."""
from pathlib import Path
import json
import re
import unicodedata

PATH=Path(__file__).resolve().parents[1]/'experiments/semantic_remaining_closeout_v1/policy.json'
NUM=r'(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)'
ALIASES={'千克':'kg','公斤':'kg','克':'g','斤':'jin','毫升':'ml','升':'l','毫米':'mm','厘米':'cm','公分':'cm','米':'m',
         '%':'percent','％':'percent','平方米':'m2','m2':'m2','瓦':'W','瓦特':'W','w':'W','千瓦时':'kWh','度电':'kWh','千瓦':'kW','kw':'kW',
         '小袋':'bag','枝':'piece','颗':'piece','粒':'piece','支':'piece','根':'piece','条':'piece','个':'piece','片':'sheet','张':'sheet',
         '瓶':'bottle','袋':'bag','包':'bag','盒':'box','套':'set','双':'pair','本':'book','档':'count','种':'count'}
for _unit in ('g','kg','jin','ml','l','mm','cm','m','W','Wh','kWh','mAh','Ah','V','count'):
    ALIASES[_unit]=_unit
FAMILIES={'kg':'mass','g':'mass','jin':'mass','ml':'volume','l':'volume','mm':'length','cm':'length','m':'length',
          'W':'power','Wh':'energy','kWh':'energy','mAh':'charge','Ah':'charge','V':'voltage','m2':'area','percent':'percentage'}
UNITS='|'.join(re.escape(u) for u in sorted(ALIASES,key=len,reverse=True))
APPROX=re.compile(r'大概|大约|差不多|约|左右|上下|接近|这样|样子')

def confirmed(key):
    return PATH.exists() and json.loads(PATH.read_text(encoding='utf-8')).get(key,{}).get('confirmed_by_user') is True

def family_for(r):
    if r['subtype']=='attribute_count':return 'feature_count'
    if r['attribute'] in ('purchase_count','pack_count','sale_unit_quantity','gift_count','included_component_count','received_count'):return 'purchase_count'
    return {'percentage':'percentage','cooling_power':'power','coverage_area':'area','circumference':'length','power':'power','energy_capacity':'energy','electric_charge':'charge','voltage':'voltage',
            'mass':'mass','purchase_mass':'mass','wearer_weight_limit':'mass','load_capacity':'mass',
            'liquid_volume':'volume','purchase_volume':'volume','length':'length','width':'length',
            'height':'length','diameter':'length','thickness':'length','wearer_height':'length'}.get(r['attribute'])

def measurements(text,parser):
    # Only explicitly spelled units. Bare 寸, G, 匹 and missing units are not converted.
    text=re.sub(r'('+NUM+r')米([一二两三四五六七八九1-9])(?![\d一二两三四五六七八九十百])',
                lambda m:str(float(parser(m[1]))+float(parser(m[2]))/10)+'米',text)
    results=[];occupied=[]
    range_pattern=r'(?<![\d.一二两三四五六七八九十百千万])('+NUM+r')(?:\s*('+UNITS+r'))?\s*(?:到|至|～|~|—|-)\s*('+NUM+r')\s*('+UNITS+r')(?![A-Za-z])'
    for m in re.finditer(range_pattern,text,re.I):
        a=ALIASES.get(m[2],m[2]) if m[2] else None;z=ALIASES.get(m[4],ALIASES.get(m[4].lower()))
        if a is not None and a!=z:continue
        results.append({'type':'range','lower':float(parser(m[1])),'upper':float(parser(m[3])),
                        'unit':z,'lower_inclusive':True,'upper_inclusive':True})
        occupied.append(m.span())
    for m in re.finditer(r'(?<![\d.一二两三四五六七八九十百千万])('+NUM+r')\s*('+UNITS+r')(?![A-Za-z])',text,re.I):
        if any(a<=m.start()<z for a,z in occupied):continue
        # Never read 一100包 as 100包, or 500多升 as an exact 500升.
        if m.start() and re.match(r'[\d一二两三四五六七八九十百千万]',text[m.start()-1]):continue
        unit=ALIASES.get(m[2],ALIASES.get(m[2].lower()))
        if unit=='kW':value=float(parser(m[1]))*1000;unit='W'
        else:value=float(parser(m[1]))
        results.append({'type':'number','value':value,'unit':unit})
    return results


def human_height_target(r,parser):
    if not confirmed('bare_human_height_cm') or r['attribute']!='wearer_height':return None
    targets=[]
    for e in r['evidence']:
        text=e['text']
        if re.search(r'cm|厘米|公分|米',text):continue
        for match in re.finditer(r'(?:身高(?:在|是)?\s*('+NUM+r')|('+NUM+r')身高)',text):
            value=float(parser(match[1] or match[2]))
            if value not in targets:targets.append(value)
    return {'type':'number','value':targets[0],'unit':'cm'} if len(targets)==1 else None

def approximate_target(r,parser):
    if not confirmed('approximate_numeric'):return None
    height=human_height_target(r,parser)
    if height and any(APPROX.search(e['text']) for e in r['evidence']):return height
    word_target=colloquial_target(r,parser)
    if word_target and word_target[0]=='approximately':return word_target[1]
    family=family_for(r)
    if not family:return None
    values=[]
    for e in r['evidence']:
        text=e['text']
        if not APPROX.search(text) or re.search(r'多一点|多些|来[张包个]|\d+多|几十|二十多|百来|[xX×*]',text):continue
        # Bind the approximation to its local numeric phrase, not a later
        # denominator such as 一袋 in 9颗左右一袋.
        cue=APPROX.search(text)
        numeric_text=text
        if cue:
            if cue.group() in ('左右','上下','这样','样子'):
                numeric_text=text[:cue.end()]
            else:
                numeric_text=text[cue.end():]
                suffix=re.search(r'左右|上下|这样|样子|差不多',numeric_text)
                if suffix:numeric_text=numeric_text[:suffix.end()]
        if not measurements(numeric_text,parser) and cue and cue.group()=='差不多' and not text[cue.end():].strip('的。 ，,'):
            numeric_text=text[:cue.start()]
        candidates=measurements(numeric_text,parser)
        eligible=[]
        for v in candidates:
            if family=='feature_count' and v['unit'] in ('count','piece'):
                v={**v,'unit':'count'};eligible.append(v)
            elif family=='purchase_count' and v['unit'] in ('piece','bottle','bag','box','pair','set','book','sheet','roll','tube'):eligible.append(v)
            elif FAMILIES.get(v['unit'])==family:eligible.append(v)
        if len(eligible)==1 and eligible[0] not in values:values.append(eligible[0])
    return values[0] if len(values)==1 else None

def colloquial_target(r,parser):
    if not confirmed('table_c'):return None
    family=family_for(r)
    if family not in ('feature_count','purchase_count','mass','volume'):return None
    values=[]
    def unit_for(raw):
        unit=ALIASES.get(raw)
        if family=='feature_count' and unit in ('count','piece'):return 'count'
        if family=='purchase_count' and unit in ('piece','sheet','bag','box','bottle','book','set','pair'):return unit
        if FAMILIES.get(unit)==family:return unit
    for e in r['evidence']:
        text=e['text']
        for match in re.finditer('('+NUM+r')多('+UNITS+')',text):
            unit=unit_for(match[2]);n=float(parser(match[1]))
            if unit is None or n<=0 or not n.is_integer() or int(n)%10:continue
            zeros=len(str(int(n)))-len(str(int(n)).rstrip('0'));step=10**zeros
            discrete=family in ('feature_count','purchase_count')
            value={'type':'range','lower':n+1 if discrete else n,'upper':n+step-1 if discrete else n+step,
                   'unit':unit,'lower_inclusive':discrete,'upper_inclusive':discrete}
            values.append(('between',value))
        for match in re.finditer('('+NUM+r')来('+UNITS+')',text):
            unit=unit_for(match[2])
            if unit:values.append(('approximately',{'type':'number','value':float(parser(match[1])),'unit':unit}))
        for match in re.finditer(r'几十('+UNITS+')',text):
            unit=unit_for(match[1])
            if unit:values.append(('between',{'type':'range','lower':20,'upper':99,'unit':unit,'lower_inclusive':True,'upper_inclusive':True}))
    unique=[]
    for value in values:
        if value not in unique:unique.append(value)
    return unique[0] if len(unique)==1 else None

def validate_colloquial(r,parser):
    target=colloquial_target(r,parser)
    if target is None or target!=(r['operator'],r['expected']):raise ValueError('Colloquial quantity disagrees with the confirmed source policy')

def validate_approximate(r,parser,source_query=None):
    if source_query is not None:
        from semantic_review_decisions import verification_decision
        decision=verification_decision(source_query,r)
        if decision and decision.get('standard')=='explicit_source_numeric_correction':return
    expected=approximate_target(r,parser)
    if expected is None or r['expected']!=expected:
        raise ValueError('Approximation must match a unique explicit source target, unit, attribute and range')

def label_normalized(text):
    # Formatting only; no unit conversion, ordering/sorting, axis swapping or inference.
    text=unicodedata.normalize('NFKC',text)
    return re.sub(r'\s+','',text).replace('×','x').replace('*','x')

def validate_label(r):
    if not confirmed('literal_spec_label') or r['expected']['type']!='text' or r['subtype'] not in ('size','dimensions','compatibility','capacity','function','version'):
        raise ValueError('Specification-label comparison requires the confirmed policy and a text specification')
    value=r['expected']['value']
    if not re.search(r'[0-9一二两三四五六七八九十半]',value):raise ValueError('Specification label must retain its source numbers')
    if not any(label_normalized(value) in label_normalized(e['text']) for e in r['evidence']):
        raise ValueError('Specification label must be a complete retained source span')


def validate_covering_range(r,parser):
    if not confirmed('table_b') or r['subtype']!='dimensions' or r['attribute']!='height':
        raise ValueError('Range coverage requires an approved adjustment-range interpretation')
    value=r['expected']
    if value['type']!='range' or value['lower'] is None or value['upper'] is None:
        raise ValueError('Adjustment coverage needs a bounded requested range')
    if not any('升降范围' in e['text'] and value in measurements(e['text'],parser) for e in r['evidence']):
        raise ValueError('Adjustment-range endpoints must match their original clause')
