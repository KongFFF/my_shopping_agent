"""Experimental source-checked fact compiler; no task-ID branches or API calls.

Proposals are untrusted. Actual values are parsed from the full public clause;
model interpretations and completeness flags never become proof by themselves.
Unsupported constructions remain unknown. Frozen training code is untouched.
"""
from __future__ import annotations
from collections import Counter
import copy
import html
import re
from decimal import Decimal
import unicodedata

from jsonschema import Draft202012Validator
from semantic_product_verifier_v4 import compare_value, normalized
from semantic_public_matching import claim_terms, occurrences, selected_colors
from semantic_fact_language_v3 import canonical

VERSION = 'purchase-fact-proofs-v2'
NUM = r'\d+(?:\.\d+)?'
COUNT_UNITS = {'片':'piece','个':'piece','件':'piece','支':'piece','只':'piece','罐':'piece',
               '瓶':'bottle','袋':'bag','包':'bag','盒':'box','套':'set','双':'pair','本':'book'}
UNIT_MAP = {'kg':'kg','g':'g','千克':'kg','公斤':'kg','克':'g','ml':'ml','毫升':'ml','l':'l','升':'l',
            'mm':'mm','cm':'cm','m':'m','毫米':'mm','厘米':'cm','公分':'cm','米':'m','w':'W','瓦':'W','瓦特':'W'}
MEASURE = re.compile(r'(?<![\d.])('+NUM+r')\s*(千克|公斤|毫米|厘米|公分|毫升|瓦特|kg|ml|mm|cm|克|升|米|瓦|g|l|m|w)(?![a-zA-Z])', re.I)
COUNTS = re.compile(r'(?<![\d.])('+NUM+r')\s*(片|个|件|支|只|罐|瓶|袋|包|盒|套|双|本)')
COMPONENT_MARKERS = ('外壳','框架','面材','芯材','刀身','镜片','接触面','蘸料','赠品','附件','配件')
OPTIONAL = re.compile(r'选配|可选|另购|加购|不含|没有|未含|不赠|不送|无赠|随机')
SIZE_NAMES = {'s':'小码','m':'中码','l':'大码'}


def schema():
    def fact(kind, roles, actual):
        return {'type':'object','additionalProperties':False, 'properties':{
            'kind':{'const':kind},'role':{'enum':roles},'subject':{'type':['string','null']},
            'pointer':{'type':'string'},'quote':{'type':'string','minLength':1},'actual':actual},
            'required':['kind','role','subject','pointer','quote','actual']}
    def obj(props):
        return {'type':'object','additionalProperties':False,'properties':props,'required':list(props)}
    return {'type':'object','additionalProperties':False,'properties':{'facts':{'type':'array','maxItems':60,'items':{'oneOf':[
        fact('claim',['declaration'],obj({'text':{'type':'string','minLength':1}})),
        fact('label',['size_label','color','model'],obj({'text':{'type':'string','minLength':1}})),
        fact('scalar',['quantity','net_content','capacity','power','length','width','height','diameter'],
             obj({'value':{'type':'number','minimum':0},'unit':{'type':'string'}})),
        fact('members',['variant_members'],obj({'values':{'type':'array','minItems':2,'items':{'type':'number'}},'unit':{'const':'count'}})),
        fact('component',['component'],obj({'name':{'type':'string','minLength':1},'quantity':{'type':['number','null']},'unit':{'type':['string','null']}}))
    ]}}},'required':['facts']}


def text(value):
    return unicodedata.normalize('NFKC', html.unescape(str(value))).strip()


def norm(value):
    return re.sub(r'\s+', '', text(value)).casefold()


def clause_for(full, quote):
    """Do not allow clipping 2g out of a named accessory to erase the owner."""
    full, quote = text(full), text(quote)
    if quote == full:
        return full
    parts = [s.strip() for s in re.split(r'[+＋|｜，,；;]', full) if s.strip()]
    exact = [s for s in parts if quote == s]
    if len(exact) != 1:
        raise ValueError('numeric/component quote must retain a complete public clause')
    return exact[0]


def source_span(proposal, request):
    pointer = proposal['pointer']
    if pointer not in request['evidence_table']:
        raise ValueError('unselected or nonpublic pointer')
    full = request['evidence_table'][pointer]
    if not isinstance(full,str):
        raise ValueError('public text required')
    decoded, quote = text(full), text(proposal['quote'])
    if not quote or quote not in decoded:
        raise ValueError('quote is not an exact source span')
    if decoded.count(quote) != 1 and quote != decoded:
        raise ValueError('ambiguous repeated quote')
    return {'pointer':pointer,'value':full,'quote':quote,
            'quote_normalization':'NFKC_HTML_decode','start_in_decoded':decoded.index(quote),
            'end_in_decoded':decoded.index(quote)+len(quote)}


def compile_fact(p, request):
    evidence = source_span(p,request)
    full, quote = text(evidence['value']), evidence['quote']
    selected = p['pointer'].startswith('/customization_options/')
    subject = p['subject']
    if subject is not None and (not subject or text(subject) not in quote):
        raise ValueError('subject must be literally bound in the quoted clause')
    context = next((s for s in re.split(r'[+＋|｜，,；;]', full) if quote in s), full)
    if subject is None and any(marker in context for marker in COMPONENT_MARKERS) and p['kind'] != 'members':
        # Whole component inventories are handled below; their primary subject
        # is the item including the component, not a component's own material.
        if p['kind'] != 'component':
            raise ValueError('component-scoped text cannot automatically become a primary fact')
    base = {'kind':p['kind'],'role':p['role'],'subject':subject,'evidence':evidence,
            'scope':'sku' if selected else 'product','review_status':'program_checked_candidate_not_human_approved'}
    if p['kind'] == 'members':
        if not selected or quote != full:
            raise ValueError('member inventory requires the whole selected option')
        # Numeric variant chain only, never the whole title's advertised sizes.
        match = re.search(r'(?<![\d.])('+NUM+r'(?:段|节|cm|mm)?(?:\s*\+\s*'+NUM+r'(?:段|节|cm|mm)?){1,})',quote,re.I)
        if not match or OPTIONAL.search(quote) or re.search(r'价格|加价|运费|优惠|赠|送\d|[元块]|(?:kg|ml|[gwl])\b',quote,re.I):
            raise ValueError('unsupported or optional numeric inventory')
        values = [float(v) for v in re.findall(NUM,match[0])]
        # Other numbered content prevents claiming this is the complete list.
        if re.search(NUM,quote[:match.start()]+quote[match.end():]):
            raise ValueError('other numbers outside member inventory')
        if values != p['actual']['values']:
            raise ValueError('proposed members differ from parsed selected inventory')
        return {**base,'actual':{'values':values,'unit':'count'},'complete':True,
                'proof':'entire_selected_numeric_inventory'}
    if p['kind'] in ('claim','label'):
        value = text(p['actual']['text'])
        if not value or value not in quote:
            raise ValueError('claim/label cannot paraphrase its actual source value')
        if p['kind']=='label':
            if not selected:
                raise ValueError('variant labels require selected evidence')
            axis = p['pointer'].split('/')[2]
            if p['role']=='size_label' and not re.search(r'尺码|鞋码|size',axis,re.I):
                raise ValueError('size label needs an explicit size axis')
            if p['role']=='color':
                colors=selected_colors(full)
                if colors and norm(value) not in {norm(c) for c in colors} or not colors and value!=full:
                    raise ValueError('color label cannot shorten or invent a shade')
        checks=request.get('_checks',{})
        product=checks.get('product',{})
        if not selected and product:
            advertised=any(norm(value) in norm(e.get('value','')) for entries in (product.get('customization_options') or {}).values() for e in entries)
            selected_support=any(norm(value) in norm(v) for v in request['selected_options'].values())
            if advertised and not selected_support:
                raise ValueError('common title/claim cannot lend an unselected variant property')
        polarities={p for _,_,p in occurrences(full,[norm(value)])}
        polarity=next(iter(polarities)) if len(polarities)==1 else 'ambiguous'
        return {**base,'actual':{'text':value},'complete':False,'proof':'literal_public_span','polarity':polarity}
    if p['kind']=='component':
        if not selected:
            raise ValueError('component inventories currently require selected SKU evidence')
        clause = clause_for(full,quote)
        name = text(p['actual']['name'])
        if not name or name not in clause or OPTIONAL.search(clause):
            raise ValueError('component name not positively stated in selected clause')
        amount, unit = p['actual']['quantity'],p['actual']['unit']
        if amount is not None:
            matches = [m for m in COUNTS.finditer(clause) if float(m[1])==amount and COUNT_UNITS[m[2]]==unit
                       and re.match(r'(?:装)?'+re.escape(name),clause[m.end():].strip())]
            if not matches:
                raise ValueError('component count must directly bind its named item')
        elif unit is not None:
            raise ValueError('unproven component unit without count')
        return {**base,'actual':{'name':name,'quantity':amount,'unit':unit},'complete':False,
                'proof':'named_selected_component'}
    if not selected:
        raise ValueError('automatic scalar proof currently requires selected evidence')
    clause = clause_for(full,quote)
    if OPTIONAL.search(clause) or re.search(r'每|/[a-z秒分时]|总共|合计|总量',clause,re.I):
        raise ValueError('rate/total/optional scope requires a separate proof')
    if p['role']=='quantity':
        matches = list(COUNTS.finditer(clause))
        if len(matches)!=1 or re.search(r'送|赠',clause):
            raise ValueError('ambiguous or promotional count')
        m=matches[0]
        if subject is None and re.search(r'[\u4e00-\u9fff]',clause[m.end():].strip('装)）】] ')):
            raise ValueError('named following object is a component, not primary quantity')
        actual={'value':float(m[1]),'unit':COUNT_UNITS[m[2]]}
    else:
        matches=list(MEASURE.finditer(clause))
        if len(matches)!=1:
            raise ValueError('need one unambiguous measurement in its complete clause')
        m=matches[0];actual={'value':float(m[1]),'unit':UNIT_MAP[m[2].lower()]}
        unit=actual['unit'];role=p['role']
        if role=='power' and unit!='W' or role=='capacity' and unit not in ('ml','l') or role=='net_content' and unit not in ('g','kg','ml','l'):
            raise ValueError('measurement dimension and role disagree')
        if role in ('length','width','height','diameter'):
            cues={'length':r'长|长度','width':r'宽|宽度','height':r'高|高度|跟高','diameter':r'直径|[φΦ]'}
            axis=p['pointer'].split('/')[2]
            if unit not in ('mm','cm','m') or not re.search(cues[role],clause[:m.start()]+axis):
                raise ValueError('dimension axis is not explicitly bound')
    if actual != p['actual']:
        raise ValueError('proposed scalar differs from parsed public value/unit')
    return {**base,'actual':actual,'complete':True,'proof':'literal_numeric_value_unit_and_scope'}


def compile_answer(answer,request):
    Draft202012Validator(schema()).validate(answer)
    accepted,rejected=[],[]
    for index,p in enumerate(answer['facts']):
        try:
            accepted.append(compile_fact(p,request))
        except (ValueError,KeyError,IndexError,TypeError) as error:
            rejected.append({'index':index,'reason':str(error),'proposal':p})
    return {'facts':accepted,'rejected':rejected,'proposal_count':len(answer['facts'])}


def subject_ok(r,f):
    if r['subject']['kind']=='component':
        return f['subject'] is not None and norm(f['subject'])==norm(r['subject']['name'])
    return r['subject']['kind']=='primary_product' and f['subject'] is None


def compare_requirement(r,facts):
    """Return a proof result or unknown. No model status or reason is consumed."""
    candidates=[f for f in facts if subject_ok(r,f)]
    outcomes=[]
    for f in candidates:
        actual=None;kind=f['kind'];role=f['role'];want=r['expected']
        if kind=='members' and r['subtype']=='attribute_count' and r['attribute']=='other_feature_count':
            source=''.join(e['text'] for e in r['evidence'])
            if not re.search(r'规格|尺寸|档',source):
                continue
            actual={'type':'number','value':len(set(f['actual']['values'])),'unit':'count'}
        elif kind=='scalar':
            roles={'quantity':('quantity',),'net_content':('net_content','weight'),
                   'capacity':('capacity',),'power':('function','power'),
                   'length':('dimensions',),'width':('dimensions',),'height':('dimensions',),'diameter':('dimensions',)}
            if r['subtype'] not in roles.get(role,()):
                continue
            if role in ('length','width','height','diameter') and r['attribute']!=role:
                continue
            actual={'type':'number',**f['actual']}
        elif kind=='label' and r['subtype']=='size' and role=='size_label' and want['type']=='text':
            got=canonical('size',f['actual']['text']);expected=canonical('size',want['value'])
            got=canonical('size',SIZE_NAMES.get(got,got));expected=canonical('size',SIZE_NAMES.get(expected,expected))
            status='pass' if got==expected else 'fail'
            outcomes.append({'status':status,'fact':f,'actual':{'type':'text','value':got}})
            continue
        elif kind=='label' and r['subtype']=='color' and role=='color' and want['type']=='text':
            actual={'type':'text','value':f['actual']['text']}
        elif kind=='component' and r['subtype']=='components' and want['type']=='components':
            # Join items below; absence never means this SKU lacks an item.
            continue
        elif kind=='claim' and want['type']=='text' and r['operator']=='eq':
            if r['subtype'] in ('color','size','dimensions','quantity','attribute_count','components','bundle'):
                continue
            terms=claim_terms(r)
            expected=canonical(r['subtype'],want['value'])
            got=canonical(r['subtype'],f['actual']['text'])
            raw=text(f['evidence']['value']);value=f['actual']['text'];pos=raw.find(value)
            before,after=norm(raw[:pos]),norm(raw[pos+len(value):])
            if r['subtype']=='material' and (re.search(r'(?:人造|仿(?:制|真)?|合成|pu|pvc|贴面?)$',before) or re.match(r'色|质感|效果|纹理|风格',after)):
                continue
            if r['subtype'] in ('brand','product_model'):
                path=f['evidence']['pointer']
                explicitly_typed=(path in ('/brand','/model') or path.startswith('/attribute/') and (norm(raw)==norm(value) or norm(raw)==norm(value)+'型号'))
                if not explicitly_typed or re.search(r'适用|适配|兼容|用于',before[-8:]):
                    continue
            # A category mention in a phone case or an adapter is not a claim
            # that the purchased primary product is a phone or the host device.
            if r['subtype']=='product_category':
                raw=f['evidence']['value'];value=f['actual']['text'];pos=text(raw).find(value)
                before=text(raw)[:pos];after=text(raw)[pos+len(value):]
                if re.search(r'适用|适配|兼容|用于',before[-8:]) or re.match(r'(?:专用)?(?:保护套|收纳盒|配件|支架|屏幕|电池|充电器|贴膜|保护膜|壳|套|盒|袋)',after):
                    continue
            if expected==got or any(p=='positive' for _,_,p in occurrences(f['actual']['text'],terms)):
                status={'positive':'pass','negative':'fail'}.get(f.get('polarity'),'unknown')
                outcomes.append({'status':status,'fact':f,'actual':{'type':'text','value':f['actual']['text']}})
            continue
        if actual is not None:
            status=compare_value(r,actual,complete=f['complete'])
            outcomes.append({'status':status,'fact':f,'actual':actual})
    if r['subtype']=='components' and r['expected']['type']=='components' and r['operator']=='contains_all':
        components=[f for f in candidates if f['kind']=='component']
        items=[{'name':f['actual']['name'],'quantity':f['actual']['quantity'],'unit':f['actual']['unit']} for f in components]
        if items:
            status=compare_value(r,{'type':'components','items':items},complete=False)
            outcomes.append({'status':status,'fact':{'joined':components},'actual':{'type':'components','items':items}})
    known={x['status'] for x in outcomes if x['status']!='unknown'}
    status=next(iter(known)) if len(known)==1 else 'unknown'
    return {'status':status,'reason':'source_checked_fact_comparison' if len(known)==1 else 'unsupported_or_conflicting_fact_proof','evidence':outcomes}


def score_case(case,compiled):
    from semantic_evidence_reward import score_training_verified
    verification=copy.deepcopy(case['rule_verification'])
    for r in case['request']['requirements']:
        rid=r['id'];result=compare_requirement(r,compiled['facts'])
        # A proven size-axis alias can correct the known S/small representation
        # issue in this experimental comparator; original training code stays.
        replace=(verification['statuses'][rid]=='unknown' or r['subtype']=='size' and result['status']!='unknown')
        if replace and result['status']!='unknown':
            verification['statuses'][rid]=result['status']
            verification['comparisons'][rid]={**verification['comparisons'][rid],**result}
    score=score_training_verified(case['contract'],verification,source_query=case['contract']['query'])
    return score,verification


def automatic_proposals(request):
    """CPU-only control: same proof grammar, no task text or model output."""
    result=[]
    def emit(kind,role,pointer,quote,actual,subject=None):
        result.append(dict(kind=kind,role=role,pointer=pointer,quote=quote,actual=actual,subject=subject))
    for pointer,raw in request['evidence_table'].items():
        if not pointer.startswith('/customization_options/') or not isinstance(raw,str):
            continue
        label=text(raw);axis=pointer.split('/')[2]
        chain=re.search(r'(?<![\d.])('+NUM+r'(?:段|节|cm|mm)?(?:\s*\+\s*'+NUM+r'(?:段|节|cm|mm)?){1,})',label,re.I)
        if chain:
            emit('members','variant_members',pointer,label,{'values':[float(v) for v in re.findall(NUM,chain[0])],'unit':'count'})
        if re.search(r'尺码|鞋码|size',axis,re.I):
            size=re.match(r'(?:\d*XL|XXXL|XXL|XL|XS|S|M|L|\d+(?:\.\d+)?)(?![a-z])',label,re.I)
            if size:
                emit('label','size_label',pointer,label,{'text':size[0]})
        for clause in [s.strip() for s in re.split(r'[+＋|｜，,；;]',label) if s.strip()]:
            counts=list(COUNTS.finditer(clause))
            if len(counts)==1:
                m=counts[0]
                emit('scalar','quantity',pointer,clause,{'value':float(m[1]),'unit':COUNT_UNITS[m[2]]})
                name=clause[m.end():].strip('装)）】] ')
                if re.fullmatch(r'[\u4e00-\u9fff]+',name or ''):
                    emit('component','component',pointer,clause,{'name':name,'quantity':float(m[1]),'unit':COUNT_UNITS[m[2]]})
            measurements=list(MEASURE.finditer(clause))
            if len(measurements)!=1:
                continue
            m=measurements[0];unit=UNIT_MAP[m[2].lower()];actual={'value':float(m[1]),'unit':unit}
            if unit=='W':role='power'
            elif unit in ('g','kg'):role='net_content'
            elif unit in ('ml','l'):role='capacity' if re.search(r'容积|容量|升|冰柜|冷柜',axis+clause) else 'net_content'
            else:
                role=next((r for r,cue in [('diameter',r'直径|[φΦ]'),('length',r'长'),('width',r'宽'),('height',r'高')] if re.search(cue,axis+clause[:m.start()])),None)
            if role:
                emit('scalar',role,pointer,clause,actual)
    return {'facts':result}
