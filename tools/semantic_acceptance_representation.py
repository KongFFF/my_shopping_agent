"""Source-bound percentage representation; never infer numeric purity from 足金."""
import re

MEASURES={'g':('mass',1),'克':('mass',1),'kg':('mass',1000),'千克':('mass',1000),'公斤':('mass',1000),
          '斤':('mass',500),'jin':('mass',500),'ml':('volume',1),'毫升':('volume',1),'l':('volume',1000),'升':('volume',1000),
          'mm':('length',1),'毫米':('length',1),'cm':('length',10),'厘米':('length',10),'m':('length',1000),'米':('length',1000)}


def validate_approximate_target(r,parser,source_query=None):
    from semantic_remaining_policy import confirmed,validate_approximate
    if confirmed('approximate_numeric'):
        return validate_approximate(r,parser,source_query)
    v=r['expected']
    if r['subtype'] not in ('net_content','weight','dimensions','capacity') or v['type']!='number' or v['unit'] not in MEASURES:
        raise ValueError('Approximate target requires a supported physical attribute and explicit unit')
    from semantic_scope_v7 import approximation_evidence
    if not approximation_evidence(r):raise ValueError('Approximately requires a source approximation cue')
    family,scale=MEASURES[v['unit']];target=v['value']*scale;supported=False
    units='|'.join(sorted(MEASURES,key=len,reverse=True))
    for e in approximation_evidence(r):
        for m in re.finditer(r'([\d零〇一二两三四五六七八九十百千万]+(?:\.\d+)?)\s*('+units+')',e['text'],re.I):
            other,factor=MEASURES[m[2].lower()]
            around=e['text'][max(0,m.start()-4):m.start()]+e['text'][m.end():m.end()+2]
            if re.search(r'约|大概|差不多|左右|上下',around):
                supported |= family==other and float(parser(m[1]))*factor==target
    if not supported:raise ValueError('Approximation center/unit must match its own source quantity')


def validate_percentage(r):
    v=r['expected']
    if r['subtype']!='composition' or r['attribute']!='percentage' or v.get('unit')!='percent':
        raise ValueError('Percent needs composition/percentage and a percent unit')
    if r['subject']['kind']!='component':raise ValueError('Composition percentage must name its component')
    if v['type']=='number':
        if not 0<=v['value']<=100:raise ValueError('Percentage outside 0..100')
    elif v['type']=='range':
        if any(x is not None and not 0<=x<=100 for x in (v['lower'],v['upper'])):raise ValueError('Percentage bound outside 0..100')
    else:raise ValueError('Numeric percentage required')
    if r['operator']=='approximately':
        from semantic_remaining_policy import validate_approximate
        from semantic_contract_v2_r7_final12 import chinese_number
        return validate_approximate(r,chinese_number)
    supported=False
    for e in r['evidence']:
        text=e['text']
        if r['subject']['name'] not in text or not re.search(r'含量|含金量|纯度',text):continue
        for m in re.finditer(r'(?P<op>不要超过|不能超过|不得超过|不大于|不低于|不少于|不超过|不高于|大于|超过|小于|低于|至少|达到|至多|为|是)?\s*(\d+(?:\.\d+)?)\s*[%％]',text):
            n=float(m[2]);op=m['op'] or '为'
            if re.search(r'不|非|别|禁止',text[max(0,m.start()-3):m.start()]):continue
            if op in ('大于','超过'):expected=('between',{'type':'range','lower':n,'upper':None,'unit':'percent','lower_inclusive':False,'upper_inclusive':False})
            elif op in ('小于','低于'):expected=('between',{'type':'range','lower':None,'upper':n,'unit':'percent','lower_inclusive':False,'upper_inclusive':False})
            elif op in ('不低于','不少于','至少','达到'):expected=('at_least',{'type':'number','value':n,'unit':'percent'})
            elif op in ('不要超过','不能超过','不得超过','不大于','不超过','不高于','至多'):expected=('at_most',{'type':'number','value':n,'unit':'percent'})
            else:expected=('eq',{'type':'number','value':n,'unit':'percent'})
            supported |= (r['operator'],v)==expected
    if not supported:raise ValueError('Percentage threshold/operator/component must match the exact source')
