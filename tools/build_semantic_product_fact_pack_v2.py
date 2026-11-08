"""Reviewed public facts for the fixed pilot products, independent of queries.

The recipe uses public catalog text and selected-axis labels only. It never
loads requirement drafts, instruction_options, target ASIN labels or rewards.
Pilot roster is reused from the prior fixed audit, without outcome selection.
"""
import gzip
import hashlib
import json
import re
from pathlib import Path
from semantic_product_verifier_v2 import fingerprint
from semantic_product_facts_v2 import validate_fact

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/semantic_product_facts_v2_20261107'
ARCHIVE=ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'
ROSTER=ROOT/'analysis/semantic_product_verifier_20261107/train20_target_probes.jsonl'
PRIMARY={'kind':'primary_product','name':'主商品'}

# Human-reviewed literal claims from the catalog read in this turn. Values are
# facts, not satisfaction labels. They do not depend on any task's wording.
# Every token must be located in title/category/attribute or a chosen SKU.
COMMON={
 '750684323117': [('category','product_category','product_category','自动浇水器','自动浇水器'),
                 ('functions','function','function','雾化','雾化')],
 '866812292170': [('category','product_category','product_category','床头板','床头靠背'),
                 ('specifications','style','style','简约','简约'),('specifications','form_factor','form_factor','超薄','超薄')],
 '930036074362': [('category','product_category','product_category','鱼肝油','鱼肝油'),
                 ('specifications','material','material','鳕鱼','鳕鱼'),('specifications','form_factor','form_factor','胶囊','胶囊'),
                 ('functions','function','function','补充DHA','补充DHA')],
 '927581317566': [('category','product_category','product_category','电吉他弦','电吉他琴弦')],
 '940170389666': [('category','product_category','product_category','气垫','气垫'),
                 ('functions','function','function','不脱妆','不脱妆'),('functions','function','function','不卡粉','不卡粉')],
 '928444258388': [('specifications','compatibility','compatibility','适用于珑骧mini','适用于珑骧mini')],
 '634419769628': [('functions','function','function','耐摔','耐摔耐折'),('functions','function','function','耐折','耐摔耐折')],
 '930932136287': [('category','product_category','product_category','中阮','中阮'),
                 ('functions','target_user','target_user','初学者','初学者')],
 '641872361298': [('specifications','style','style','现代简约','简约'),
                 ('functions','function','function','可落地','落地'),('specifications','customization','customization','可定制','定制')],
 '709708685186': [('functions','target_user','target_user','男童','男童'),('functions','use_environment','use_environment','夏季','夏季')],
 '941479509510': [('brand','brand','brand','尤尼克斯','尤尼克斯')],
 '715265851689': [('category','product_category','product_category','户外柱头灯','柱头灯'),
                 ('functions','function','function','防水','防水'),('specifications','material','material','全铜','全铜')],
 '853242068063': [('category','product_category','product_category','喷头','喷头'),
                 ('functions','function','function','360度旋转','360度自动旋转'),('functions','function','function','自动','自动'),
                 ('functions','function','function','摇臂','摇臂')],
 '917664309640': [('category','product_category','product_category','运动九分裤','九分裤'),
                 ('functions','use_environment','use_environment','春夏','春夏'),('functions','purpose','purpose','跑步','跑步'),
                 ('functions','function','function','速干','速干'),('specifications','craft','craft','梭织','梭织')],
 '933519049173': [('category','product_category','product_category','口腔冲洗器','口腔牙科手动弯头注射器'),
                 ('specifications','condition','condition','手动','手动'),('specifications','compatibility','compatibility','牙科用','牙科'),
                 ('functions','purpose','purpose','清理牙洞','牙齿洞冲牙器')],
 '730495298721': [('functions','use_environment','use_environment','户外运动','户外健身'),
                 ('functions','use_environment','use_environment','秋季','秋季')],
 '906779405987': [('functions','target_user','target_user','男款','男款'),('functions','function','function','速干','速干'),
                 ('specifications','material','material','冰丝面料','冰丝面料'),('specifications','style','style','紧身','紧身')],
 '885733766281': [('functions','function','function','防水','防水')],
 '678702333328': [('category','product_category','product_category','底妆产品','气垫bb霜'),
                 ('functions','function','function','防水','防水'),('functions','function','function','防汗','防汗'),
                 ('functions','function','function','不易脱妆','不易脱妆'),('functions','function','function','持久','持久')],
}


def main():
    OUT.mkdir(exist_ok=True)
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as handle:products=json.load(handle)
    products={str(p['asin']):p for p in products}
    asins=list(dict.fromkeys(str(json.loads(x)['asin']) for x in ROSTER.read_text(encoding='utf-8').splitlines()))
    assert len(asins)==19 and set(asins)==set(COMMON)
    records=[]
    for asin in asins:
        p=products[asin]
        def evidence(quote,selection=None):
            if selection is not None:
                axis,index=selection;value=p['customization_options'][axis][index]['value']
                pointer='/customization_options/'+axis.replace('~','~0').replace('/','~1')+'/'+str(index)+'/value'
            else:
                fields=[('/title',p.get('title','')),('/full_description',p.get('full_description','')),('/category',p.get('category',''))]
                fields += [(f'/attribute/{i}',v) for i,v in enumerate(p.get('attribute') or [])]
                candidates=[(path,v) for path,v in fields if isinstance(v,str) and quote in v]
                assert candidates,(asin,quote)
                pointer,value=candidates[0]
            assert quote in value,(asin,quote,value)
            start=value.index(quote)
            return {'pointer':pointer,'text':quote,'start':start,'end':start+len(quote)}
        def add(d,st,attr,value,quote,*,selection=None,subject=None,note=None,complete=False):
            value={'type':'text','value':value} if isinstance(value,str) else value
            record={'product_asin':asin,'product_sha256':fingerprint(p),'scope':'sku' if selection else 'product',
                    'dimension':d,'subtype':st,'attribute':attr,'subject':subject or PRIMARY,'value':value,'complete':complete,
                    'review_status':'approved','review_note':note or '人工核对公开商品表述；不从需求或正确答案反推事实。',
                    'evidence':[evidence(quote,selection)]}
            if selection:
                axis,index=selection;record.update(options={axis:p['customization_options'][axis][index]['value']},selector_mode='contains')
            validate_fact(record);records.append(record)
        for d,st,attr,value,quote in COMMON[asin]:add(d,st,attr,value,quote)
        if asin=='930036074362':
            add('functions','target_user','target_user',{'type':'text_list','values':['婴儿','幼儿','儿童']},'婴幼儿宝宝鳕鱼肝油胶囊 儿童',note='公开标题同时声明婴幼儿与儿童适用；未声明具体月龄或临床功效。')
        if asin=='933519049173':
            add('specifications','components','included_components',{'type':'components','items':[{'name':'弯曲头','quantity':None,'unit':None}]},'弯头')
        for axis,entries in p.get('customization_options',{}).items():
            for index,entry in enumerate(entries):
                text=entry['value'];sel=(axis,index)
                def claim(d,st,attr,val,quote=None,**kw):add(d,st,attr,val,quote or text,selection=sel,**kw)
                if asin=='750684323117' and '大屏' in text:claim('functions','function','function','大屏','大屏')
                if asin=='927581317566':
                    if 'M1' in text:
                        claim('functions','function','function','不易锈','防锈')
                        claim('specifications','craft','craft','离子覆膜','覆膜',note='选中M1覆膜款；公共标题明确M1离子覆膜，未把该工艺借给A1。')
                        records[-1]['evidence'].append(evidence('M1离子覆膜'))
                    size=re.search(r'【(\d{2}-\d{2})】',text)
                    if size:claim('specifications','size','size_label',size[1],size[0],complete=True)
                    if '赠品' in text:claim('specifications','components','gift_presence',{'type':'components','items':[{'name':'赠品','quantity':None,'unit':None}]},'赠品')
                    # The public title explicitly describes each set as six
                    # strings. Actual sale-unit packaging is reviewed here.
                    claim('specifications','quantity','purchase_count',{'type':'number','value':1,'unit':'set'})
                    records[-1]['evidence'].append(evidence('一套6根'))
                    claim('specifications','packaging','pack_count',{'type':'number','value':6,'unit':'piece'})
                    records[-1]['evidence'].append(evidence('一套6根'))
                if asin=='940170389666' and '1盒' in text:
                    claim('specifications','packaging','pack_count',{'type':'number','value':1,'unit':'box'},'1盒',complete=True)
                    if '送同款替换装' in text:claim('specifications','components','gift_presence',{'type':'components','items':[{'name':'替换芯','quantity':None,'unit':None}]},'送同款替换装',note='气垫商品的同款替换装表示补充芯，未推额外赠品数量。')
                if asin=='928444258388' and '斜挎套餐' in text:claim('specifications','form_factor','form_factor','斜挎','斜挎套餐')
                if asin=='634419769628' and '魔尺' not in text and re.match(r'\d+段',text):
                    # Category comes from actual product kind plus the
                    # selected label; segment numbers are not model names.
                    if '+' in text:claim('category','product_category','product_category','魔尺套装')
                    if '送教程书' in text:claim('specifications','components','gift_presence',{'type':'components','items':[{'name':'教学手册','quantity':None,'unit':None}]},'送教程书')
                if asin=='930932136287' and '中阮' in text:
                    if '直头' in text:claim('specifications','form_factor','form_factor','直头','直头')
                    # 尼龙品 is not a statement about nylon strings.
                if asin=='641872361298' and text=='单床头':claim('category','product_category','product_category','床头靠背')
                if asin=='715265851689' and '太阳能接电两用款' in text:
                    claim('functions','function','power','太阳能和外接电源这两种方式供电','太阳能接电两用款')
                if asin=='853242068063' and '可调' in text:claim('functions','function','function','可调','可调')
                if asin=='730495298721' and text=='白色上衣':
                    claim('category','product_category','product_category','上衣','上衣')
                    claim('specifications','style','style','宽松');records[-1]['evidence'].append(evidence('宽松'))
                if asin=='906779405987' and text.startswith('长袖三件套'):
                    claim('category','product_category','product_category','运动服长袖三件套','长袖三件套')
                    records[-1]['evidence'].append(evidence('运动套装男'))
                if asin=='885733766281' and '灯' in text:claim('category','product_category','product_category','太阳能灯')
    for r in records:validate_fact(r)
    (OUT/'catalog_facts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    summary={'products':len(asins),'facts':len(records),'product_facts':sum(r['scope']=='product' for r in records),
             'sku_facts':sum(r['scope']=='sku' for r in records),'queries_read':0,'requirement_files_read':0,
             'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ARCHIVE,ROSTER)},
             'api_calls':0,'training_runs':0,'review_scope':'public claims, not external truth/medical efficacy/authenticity certification'}
    (OUT/'fact_pack_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
