"""Reviewed public claims for the frozen history sample, never answer labels.

Recipes are catalog annotations. Every claim is bound to an exact source span;
SKU claims apply only to the selected value. No requirement or old reward is read.
"""
import gzip
import json
from collections import Counter
from pathlib import Path
from semantic_product_verifier_v4 import fingerprint, PRIMARY
from semantic_product_facts_v3 import validate_fact
from freeze_historical_reward_audit import ARCHIVE, OUT, rows, write, lines

# dimension, subtype, canonical value, exact public quote. Nonexclusive claims
# are partial; even a reviewed title does not enumerate all product properties.
COMMON = {
 '867510242711': [('category','product_category','羽毛球鞋','羽毛球鞋'),('brand','brand','亚瑟士','亚瑟士'),('functions','function','耐磨','耐磨'),('functions','function','防滑','防滑'),('functions','function','缓震','缓震')],
 '788067596051': [('category','product_category','跑鞋','跑鞋'),('brand','brand','锐步','锐步'),('functions','target_user','男女通用','男女款'),('functions','function','充气','充气'),('specifications','style','复古','复古')],
 '693457110723': [('category','product_category','外套','外套'),('specifications','material','人造兔毛','仿兔毛'),('specifications','pattern','笑脸','笑脸'),('functions','target_user','情侣','情侣'),('specifications','style','宽松','宽松'),('functions','function','加厚','加厚')],
 '802358627174': [('category','product_category','杀虫剂','杀虫剂'),('functions','purpose','适用于玉米地','玉米地适用'),('functions','purpose','适用于花生地','花生地适用'),('specifications','material','有机肥料','有机肥料')],
 '587870280557': [('category','product_category','狗粮','狗粮'),('functions','target_user','成年犬','成犬'),('specifications','condition','处方级','全处方粮'),('functions','function','调理肠胃','调理肠胃')],
 '596456996718': [('category','product_category','压脚','压脚'),('specifications','material','全钢','全钢'),('functions','function','工业型电脑平车用','工业电脑平车'),('functions','function','单边隐形拉链','单边隐形拉链')],
 '922654068359': [('category','product_category','黄金手镯','足金古法工艺轮胎手镯'),('brand','brand','中国黄金','中国黄金'),('specifications','craft','古法','古法'),('specifications','form_factor','轮胎状','轮胎状')],
 '867817275960': [('category','product_category','窗帘','窗帘'),('specifications','style','法式','法式'),('functions','use_environment','卧室和客厅','客厅卧室'),('specifications','customization','可定制','定制')],
 '823217094971': [('category','product_category','窗帘','窗帘'),('specifications','style','法式','法式'),('functions','use_environment','卧室和客厅','卧室全遮光奶油风客厅'),('functions','function','100%遮光','全遮光')],
 '616736502624': [('category','product_category','铝板','铝板'),('specifications','material','铝','铝板'),('functions','purpose','车用','车用'),('functions','function','防滑','防滑'),('specifications','pattern','花纹','花纹'),('specifications','customization','可定制','定制')],
 '710173079220': [('category','product_category','景观座椅','景观座椅'),('specifications','material','玻璃钢','玻璃钢'),('specifications','customization','可定做','可定制'),('functions','use_environment','园林','园林')],
 '878676207091': [('category','product_category','茶具套装','茶具套装'),('functions','purpose','新婚礼物','新婚礼物'),('functions','purpose','下午茶','下午茶')],
 '914010124817': [('category','product_category','短裙','短裙'),('brand','brand','北面','北面'),('model','version','2025春夏新款','25春夏新款'),('functions','use_environment','户外','户外'),('functions','function','轻量','轻量'),('functions','function','透气','透气'),('functions','function','防晒','防晒'),('functions','function','抑臭','抑臭')],
 '822662354285': [('category','product_category','枕头','枕头')],
 # This product has filled and empty variants: never lend a common title's
 # pillow/category/medicinal filling to the empty case SKU.
 '791366306668': [],
}


def main(directory=OUT):
    OUT=directory
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as f: catalog={str(p['asin']):p for p in json.load(f)}
    selected=set(json.loads((OUT/'sample_lock.json').read_text(encoding='utf-8'))['manual_acceptance_tasks'])
    asins={r['asin'] for r in rows(OUT/'actual_purchases.jsonl') if r['task_id'] in selected}
    assert asins==set(COMMON)
    records=[]
    for asin in sorted(asins):
        p=catalog[asin]
        def add(d,st,value,quote,*,attribute=None,axis=None,label=None,subject=None,negative=False,complete=False):
            if axis is None:
                candidates=[('/title',p.get('title','')),('/full_description',p.get('full_description','')),('/category',p.get('category',''))]
                candidates += [(f'/attribute/{i}',v) for i,v in enumerate(p.get('attribute') or [])]
            else:
                candidates=[('/customization_options/'+axis.replace('~','~0').replace('/','~1')+f'/{i}/value',e['value']) for i,e in enumerate(p['customization_options'][axis]) if e['value']==label]
            found=[(path,v) for path,v in candidates if isinstance(v,str) and quote in v]
            assert found,(asin,quote,label)
            path,v=found[0];start=v.index(quote)
            r={'product_asin':asin,'product_sha256':fingerprint(p),'scope':'sku' if axis else 'product',
               'dimension':d,'subtype':st,'attribute':attribute or st,'subject':subject or PRIMARY,
               'value':{'type':'text','value':value} if isinstance(value,str) else value,'complete':complete,
               'review_status':'approved','review_note':'逐项审阅公开商品文字及SKU；仅证明字面声明，不证明主观品质、实际疗效或未选规格。',
               'evidence':[{'pointer':path,'text':quote,'start':start,'end':start+len(quote)}]}
            if axis:r.update(options={axis:label},selector_mode='contains')
            if negative:r['polarity']='negative'
            validate_fact(r);records.append(r)
        for d,st,value,quote in COMMON[asin]:
            add(d,st,value,quote,attribute='product_category' if st=='product_category' else None)
        def sku(st,value,axis,label,quote=None,attribute=None,**kw):
            add('specifications',st,value,quote or label,axis=axis,label=label,attribute=attribute,**kw)
        def number(attr,value,unit,axis,label,quote=None,st='dimensions'):
            sku(st,{'type':'number','value':value,'unit':unit},axis,label,quote,attribute=attr,complete=True)
        if asin=='587870280557':
            number('mass',2,'kg','食品口味','犬肠道2kg','2kg',st='net_content')
            add('functions','function','改善软便','调理肠胃软便')
            add('functions','function','改善消化不良','调理肠胃软便消化不粮')
            # Title's typo is disambiguated by its explicit structured target
            # attribute, not by looking at task labels or medical knowledge.
            records[-1]['evidence'].append({'pointer':f'/attribute/{p["attribute"].index("消化不良")}','value':'消化不良'})
        if asin=='788067596051':
            add('specifications','condition','官方','官方')
            add('specifications','condition','新品','新品')
        if asin=='802358627174':
            number('mass',500,'g','颜色分类','500g-1瓶','500g',st='net_content')
        if asin=='693457110723':
            add('specifications','components',{'type':'components','items':[{'name':'帽子','quantity':None,'unit':None}]},'连帽',attribute='included_components')
            add('functions','function','质量有保障','品质保证')
            add('brand','brand_positioning','潮流品牌','潮牌')
            add('model','version',{'type':'text_list','values':['杨博文同款','陈奕恒同款']},'杨博文陈奕恒同款')
        if asin=='596456996718':
            for label,paid,gift,total in [('隐形单边拉链压脚1个',1,None,1),('单边拉链压脚买10送1',10,1,11)]:
                sku('bundle',{'type':'bundle','paid_quantity':paid,'gift_quantity':gift,'received_quantity':total,'unit':'piece','same_item':True},'T168',label,attribute='promotion',complete=True)
        if asin=='616736502624':
            label='200mm*200mm*1.5mm（3片装）'
            for attr,n in [('length',200),('width',200),('thickness',1.5)]:number(attr,n,'mm','颜色分类',label)
        if asin=='823217094971':
            for e in p['customization_options']['尺寸']:
                label=e['value']
                if label.startswith('宽3.0米高2.7米1片'):
                    number('width',3,'m','尺寸',label,'宽3.0米');number('height',2.7,'m','尺寸',label,'高2.7米')
                    number('purchase_count',1,'piece','尺寸',label,'1片',st='quantity')
                    sku('customization','可改短','尺寸',label,'高度可改短')
        if asin in ('823217094971','867817275960'):
            add('specifications','components',{'type':'components','items':[{'name':'蕾丝装饰','quantity':None,'unit':None}]},'蕾丝',attribute='included_components')
        if asin=='710173079220':
            add('specifications','components',{'type':'components','items':[{'name':'树坑','quantity':None,'unit':None}]},'树池',attribute='included_components')
        if asin=='914010124817':sku('color','奶咖棕','颜色分类','5K2/奶咖棕','奶咖棕',complete=True)
        if asin=='878676207091':
            label='繁花朵朵-6杯碟勺+鸟架+荷花荷口壶-红色礼盒装'
            sku('components',{'type':'components','items':[{'name':n,'quantity':None,'unit':None} for n in ('杯','碟','勺')]},
                '颜色分类',label,'6杯碟勺',attribute='included_components')
            sku('components',{'type':'components','items':[{'name':'荷花壶','quantity':None,'unit':None}]},
                '颜色分类',label,'荷花荷口壶',attribute='included_components')
        if asin=='822662354285':sku('color','墨绿色','颜色分类','古法草本-墨绿（全家可用）','墨绿',complete=True)
        if asin=='791366306668':
            label='圆柱颈椎枕 枕套+内袋【没有填充物】'
            add('category','product_category','枕套','枕套+内袋【没有填充物】',axis='颜色分类',label=label,complete=True)
            sku('form_factor','圆筒','颜色分类',label,'圆柱')
            sku('material','中药填充','颜色分类',label,'没有填充物',subject={'kind':'component','name':'内胆'},negative=True)
    lines(OUT/'reviewed_catalog_facts.jsonl',records)
    write(OUT/'fact_pack_summary.json',{'products':len(asins),'approved_facts':len(records),
          'by_product':dict(Counter(r['product_asin'] for r in records)),'api_calls':0,
          'source':'reviewed public catalog assertions; not external truth certification'})
    print(json.dumps({'products':len(asins),'approved_facts':len(records)},ensure_ascii=False))


if __name__=='__main__':main()
