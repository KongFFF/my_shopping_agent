"""Source-local SKU facts. Does not read tasks or desired requirements."""
import re
from semantic_contract_v2_r7_closeout import chinese_number

NUMBER=r'(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)'
UNITS={'g':'g','克':'g','kg':'kg','千克':'kg','公斤':'kg','ml':'ml','毫升':'ml','l':'l','升':'l',
       'mm':'mm','毫米':'mm','cm':'cm','厘米':'cm','公分':'cm','m':'m','米':'m','斤':'jin'}
COUNT_UNITS={'瓶':'bottle','袋':'bag','包':'bag','盒':'box','支':'piece','个':'piece','件':'piece','片':'piece','只':'piece','本':'book','套':'set'}
from semantic_public_matching import COLORS
PRIMARY={'kind':'primary_product','name':'主商品'}


def number(text):return float(chinese_number(text))


def selected_facts(product,state):
    facts=[]
    def emit(st,attr,val,selected,subject=None):
        facts.append({'dimension':'specifications','subtype':st,'attribute':attr,'subject':subject or PRIMARY,
          'value':val,'evidence':[{'pointer':selected['pointer'],'value':selected['value']}],
          'scope':'sku','complete':True,'origin':'source_local_sku_rule_v2'})
    for selected in state['selected']:
        text=selected['value'];axis=selected['axis'];compact=re.sub(r'\s+','',text)
        # Labels are parsed only in explicit sizing axes. No model digits,
        # colors, or package counts can be borrowed as wearer measurements.
        if axis in ('尺码','大小','size'):
            size=re.match(r'^(XS|S|M|L|XL|XXL|XXXL|[2-5]XL)(?=$|\s|[（(\d])',text,re.I)
            if size:emit('size','size_label',{'type':'text','value':size[1].upper()},selected)
            fit=re.search(r'(\d+(?:\.\d+)?)\s*[-~～至]\s*(\d+(?:\.\d+)?)\s*(斤|kg|公斤|厘米|cm)',text,re.I)
            if fit:
                unit=UNITS[fit[3].lower()];a,z=map(float,fit.group(1,2))
                if a<=z:emit('size','wearer_height' if unit=='cm' else 'wearer_weight_limit',
                    {'type':'range','lower':a,'upper':z,'unit':unit,'lower_inclusive':True,'upper_inclusive':True},selected)
        if axis in ('参考身高','适合身高','身高'):
            match=re.fullmatch('('+NUMBER+')(cm|厘米|m|米)',compact,re.I)
            if match:emit('size','wearer_height',{'type':'number','value':number(match[1]),'unit':UNITS[match[2].lower()]},selected)
        # A physical net-content label has an explicit quantity, unit, and
        # optionally a single container. Multipliers/gifts are not summed.
        unit_pattern='|'.join(sorted((u for u in UNITS if UNITS[u] in ('g','kg','ml','l')),key=len,reverse=True))
        mass=list(re.finditer('('+NUMBER+')('+unit_pattern+r')(?=$|[*/×（(【\s])',text,re.I))
        if len(mass)==1:
            match=mass[0];unit=UNITS[match[2].lower()]
            # A 2-bottle label does not say whether 15g is per bottle or total.
            # For non-net-content axes require an explicit one-container form.
            if axis in ('净含量','净重','容量') or re.search(r'(?:\*|×)1(?:盒|瓶|袋)',text):
                emit('net_content','mass' if unit in ('g','kg') else 'liquid_volume',
                     {'type':'number','value':number(match[1]),'unit':unit},selected)
        counted=re.fullmatch('('+NUMBER+')('+'|'.join(COUNT_UNITS)+')',compact)
        if counted:
            val={'type':'number','value':number(counted[1]),'unit':COUNT_UNITS[counted[2]]}
            emit('packaging','pack_count',val,selected)
            emit('quantity','purchase_count',val,selected)
        multiple=re.fullmatch('('+NUMBER+r')(g|克|kg|千克|ml|毫升)(?:\*|×)('+NUMBER+')('+'|'.join(COUNT_UNITS)+')',compact,re.I)
        if multiple:
            val,count=number(multiple[1]),number(multiple[3]);unit=UNITS[multiple[2].lower()]
            emit('quantity','purchase_mass' if unit in ('g','kg') else 'purchase_volume',
                 {'type':'number','value':val*count,'unit':unit},selected,subject={'kind':'order','name':'本次订单'})
            emit('quantity','purchase_count',{'type':'number','value':count,'unit':COUNT_UNITS[multiple[4]]},selected)
            emit('packaging','pack_count',{'type':'number','value':count,'unit':COUNT_UNITS[multiple[4]]},selected)
        # Only explicit labels, not title-wide brand or color guesses. 白光
        # remains a light source color and never becomes a white body color.
        if axis in ('颜色','颜色分类','color'):
            from semantic_public_matching import selected_colors
            colors=selected_colors(text)
            if len(colors)==1:
                pos=text.index(colors[0]);prefix=text[max(0,pos-3):pos]
                if not re.search(r'非|不|无|不要',prefix):emit('color','color',{'type':'text','value':colors[0]},selected)
        square=re.search(r'边长('+NUMBER+r')(cm|厘米|mm|毫米)\*高('+NUMBER+r')(cm|厘米|mm|毫米)',compact,re.I)
        if square:
            for attr in ('length','width'):
                emit('dimensions',attr,{'type':'number','value':number(square[1]),'unit':UNITS[square[2].lower()]},selected)
            emit('dimensions','height',{'type':'number','value':number(square[3]),'unit':UNITS[square[4].lower()]},selected)
    return facts


def explicit_requirement_value(r):
    """Small source-literal normalizations, never a semantic guess or mutation."""
    expected=r['expected']
    if expected['type']!='text':return None
    text=expected['value'].strip()
    if r['subtype']=='packaging' and r['attribute']=='packaging':
        match=re.fullmatch('(单|'+NUMBER+')('+'|'.join(COUNT_UNITS)+')(?:装|套)',text)
        if match:return {'subtype':'packaging','attribute':'pack_count','operator':'eq',
            'expected':{'type':'number','value':1 if match[1]=='单' else number(match[1]),'unit':COUNT_UNITS[match[2]]}}
    if r['subtype']=='size' and r['attribute'] in ('wearer_height','wearer_weight_limit'):
        match=re.fullmatch(r'(?:适合)?('+NUMBER+')(斤|kg|公斤|cm|厘米|米)(?:的)?(?:人穿|穿|身高)?',text)
        if match:return {'expected':{'type':'number','value':number(match[1]),'unit':UNITS[match[2]]}}
    return None
