"""Confirmed-policy facts from public catalog evidence, independent of queries."""
import re
from semantic_catalog_rules_v2 import PRIMARY,NUMBER,number


def extended_facts(product,state):
    records=[]
    def emit(d,st,attr,value,evidence,*,subject=None,complete=True):
        records.append({'dimension':d,'subtype':st,'attribute':attr,'subject':subject or PRIMARY,
            'value':value,'evidence':evidence,'scope':'sku' if any(e['pointer'].startswith('/customization_options/') for e in evidence) else 'product',
            'complete':complete,'origin':'source_local_confirmed_policy_v3'})
    def typed(st,attr,val,selected,subject=None):emit('specifications',st,attr,val,[{'pointer':selected['pointer'],'value':selected['value']}],subject=subject)
    # Titles/attributes can state a common blade condition; an explicit state
    # choice axis prevents that claim being lent to another blade variant.
    state_options=any(re.search(r'(?<!未)开刃|未开刃',str(e.get('value',''))) for entries in (product.get('customization_options') or {}).values() for e in entries)
    if not state_options:
        for path,text in [('/title',product.get('title',''))]+[(f'/attribute/{i}',s) for i,s in enumerate(product.get('attribute') or [])]:
            if isinstance(text,str) and '未开刃' in text:
                emit('specifications','condition','condition',{'type':'text','value':'未开刃'},[{'pointer':path,'value':text}]);break
    # Only explicit qualitative claims; 铁艺/金色/TPU alone do not prove them.
    common=[(f'/attribute/{i}',s) for i,s in enumerate(product.get('attribute') or [])]
    description=product.get('full_description') or ''
    if isinstance(description,str):common += [('/full_description',description)]
    for path,text in common:
        if not isinstance(text,str):continue
        for clause in re.split(r'[，,。；;！!？?\n]',text):
            clause=clause.strip()
            if re.fullmatch(r'(?:本品|本产品|产品|商品)?(?:的)?(?:表面)?(?:具有|呈现)?金属质感',clause):
                emit('specifications','style','style',{'type':'text','value':'金属质感'},[{'pointer':path,'value':text}])
            if re.fullmatch(r'(?:本品|本产品|产品|商品)?(?:的)?(?:质感)?柔软',clause):
                emit('functions','function','function',{'type':'text','value':'质感柔软'},[{'pointer':path,'value':text}])
    for selected in state['selected']:
        label=selected['value'];compact=re.sub(r'\s+','',label)
        if '未开刃' in label:typed('condition','condition',{'type':'text','value':'未开刃'},selected)
        elif re.search(r'(?:已开刃|开刃款)',label):typed('condition','condition',{'type':'text','value':'已开刃'},selected)
        # An explicitly labelled sale unit is distinct from a requested
        # quantity. A 20-piece offer remains one sale lot of 20 pieces.
        wire=re.fullmatch(r'[φΦ](\d+(?:\.\d+)?)mm\*('+NUMBER+r')(米|m)',compact,re.I)
        sleeve=re.fullmatch(r'(\d+(?:\.\d+)?)mm宽('+NUMBER+r')(米|m)',compact,re.I)
        lot=re.match(r'^\(('+NUMBER+r')个\)',compact)
        if wire or sleeve:
            m=wire or sleeve
            typed('dimensions','diameter' if wire else 'width',{'type':'number','value':float(m[1]),'unit':'mm'},selected)
            typed('dimensions','length',{'type':'number','value':number(m[2]),'unit':'m'},selected)
            for attr in ('sale_quantity','sale_unit_quantity'):
                typed('transaction_terms',attr,{'type':'number','value':number(m[2]),'unit':'m'},selected)
        if lot:
            value={'type':'number','value':number(lot[1]),'unit':'piece'}
            typed('quantity','purchase_count',value,selected)
            for attr in ('sale_quantity','sale_unit_quantity'):typed('transaction_terms',attr,value,selected)
        sheets=list(re.finditer('('+NUMBER+r')片装',label))
        if len(sheets)==1:
            value={'type':'number','value':number(sheets[0][1]),'unit':'piece'}
            typed('packaging','pack_count',value,selected);typed('quantity','purchase_count',value,selected)
        ports=re.search(r'(\d+)\s*(?:usb|u)\s*\+?\s*(\d+)\s*(?:type-c|c)(?!\w)',label,re.I)
        if ports and re.search(r'usb',product.get('title',''),re.I):
            for value,name in [(int(ports[1]),'usb接口'),(int(ports[2]),'type-c接口')]:
                typed('attribute_count','other_feature_count',{'type':'number','value':value,'unit':'count'},selected,subject={'kind':'component','name':name})
        # Full single-domain numeric member list, with every value's segment
        # unit evidenced by the public title. Mixed-product bundles stay unknown.
        if '魔尺' in product.get('title',''):
            core=re.sub(r'【?送教程书】?$','',compact)
            core=re.sub(r'^(?:超值套餐|超值彩虹全套|超值|套餐|全套|彩虹|【|】)+','',core).strip('【】')
            core=re.sub('彩虹','',core)
            if re.fullmatch(r'\d+(?:段)?(?:\+\d+(?:段)?)*',core):
                values=[int(x) for x in re.findall(r'\d+',core)]
                title=product['title']
                if all(re.search(r'(?<!\d)'+str(v)+r'段',title) for v in values):
                    evidence=[{'pointer':selected['pointer'],'value':label},{'pointer':'/title','value':title}]
                    emit('specifications','composition','member_groups',{'type':'member_values','member_attribute':'segment_count','unit':'count','values':values},evidence)
    return records
