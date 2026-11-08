"""Human query-only annotations for fixed hash sample and known regressions.

No target product, original reward features, ASIN, options or result is read.
Unresolved wording is retained; a validation failure is an acceptance finding.
"""
import hashlib
import json
from pathlib import Path
import semantic_contract_v2_r7_final12 as b

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/semantic_history_acceptance_20261107'


def main(directory=OUT):
    OUT=directory
    queries={r['task_id']:r['query'] for r in (json.loads(x) for x in (OUT/'dev50_queries.jsonl').read_text(encoding='utf-8').splitlines())}
    lock=json.loads((OUT/'sample_lock.json').read_text(encoding='utf-8'))
    contracts={t:{'schema_version':b.VERSION,'task_id':t,'query':queries[t],**{d:[] for d in b.DIMENSIONS},'pending_mentions':[]} for t in lock['manual_acceptance_tasks']}
    def add(t,d,st,attr,value,quote,op='eq',importance='required',subject=None):
        assert queries[t].count(quote)==1,(t,quote)
        contracts[t][d].append({'subtype':st,'attribute':attr,'subject':subject or {'kind':'primary_product','name':'主商品'},
            'importance':importance,'operator':op,'expected':value,'price_basis':None,'evidence':[{'text':quote}],'review_status':'pending'})
    def text(t,d,st,value,quote=None,importance='required',subject=None,attr=None):
        add(t,d,st,attr or st,{'type':'text','value':value},quote or value,importance=importance,subject=subject)
    def cat(t,value,quote=None):text(t,'category','product_category',value,quote)
    def fn(t,value,quote=None,importance='required'):text(t,'functions','function',value,quote,importance)
    def conjunction(t,d,st,values,quote,attr=None,subject=None):
        add(t,d,st,attr or st,{'type':'text_list','values':values},quote,'contains_all',subject=subject)
    def spec(t,st,value,quote=None,importance='required',subject=None):text(t,'specifications',st,value,quote,importance,subject,attr='size_label' if st=='size' else st)
    def num(t,st,attr,n,unit,quote,op='eq',subject=None):add(t,'specifications',st,attr,{'type':'number','value':n,'unit':unit},quote,op,subject=subject)
    def budget(t,quote,importance='required'):text(t,'budget','total_price',quote,quote,importance,{'kind':'order','name':'本次订单'})
    def pending(t,quote,reason='unverifiable_subjective',importance='uncertain'):
        assert queries[t].count(quote)==1,(t,quote)
        contracts[t]['pending_mentions'].append({'evidence':{'text':quote},'reason':reason,'importance':importance})
    def component(t,name,quote,quantity=None,unit=None,importance='required',gift=False):
        add(t,'specifications','components','gift_presence' if gift else 'included_components',
            {'type':'components','items':[{'name':name,'quantity':quantity,'unit':unit}]},quote,'contains_all',importance)

    t=652
    cat(t,'羽毛球鞋');budget(t,'我的预算在500块左右')
    num(t,'quantity','purchase_count',1,'pair','一双');spec(t,'size','40','40码')
    text(t,'functions','purpose','比赛和训练','参加比赛和训练');fn(t,'耐磨','耐磨和防滑性能好一点的');fn(t,'防滑','耐磨和防滑性能好一点的')
    text(t,'functions','target_user','男女通用','男女生都能穿')
    spec(t,'color','冰川一样的蓝色和白色搭配','比较喜欢冰川一样的蓝色和白色搭配的效果','preference');pending(t,'冰川一样的','unverifiable_subjective','preference')
    t=2898
    cat(t,'跑鞋');budget(t,'大概1400块');num(t,'quantity','purchase_count',1,'pair','一双')
    text(t,'functions','target_user','男女通用','适合男女穿着');component(t,'气囊','配备气囊')
    spec(t,'style','复古风');pending(t,'复古风')
    add(t,'specifications','condition','condition',{'type':'text_list','values':['官方','新品']},'希望是官方的新品','contains_all',importance='uncertain');pending(t,'希望是官方的新品','ambiguous_strength')
    spec(t,'color','深蓝色','很喜欢深蓝色的鞋','preference');spec(t,'size','41','41码')
    t=3485
    cat(t,'外套','白色外套');budget(t,'预算在400元以下')
    spec(t,'color','白色','白色外套');spec(t,'material','人造兔毛','材质是人造兔毛的');spec(t,'pattern','笑脸','有笑脸元素')
    text(t,'functions','target_user','情侣','适合情侣一起穿');spec(t,'style','宽大','版型宽大点');fn(t,'厚实','厚实一些')
    component(t,'帽子','要有帽子');spec(t,'size','S','小码');spec(t,'form_factor','标准款式','标准款式的')
    conjunction(t,'model','version',['杨博文同款','陈奕恒同款'],'要杨博文和陈奕恒穿过的款式')
    # User-confirmed public-assertion standard makes the quality request a
    # retained claim predicate, rather than an unrepresented pending string.
    fn(t,'质量有保障','质量要有保障');text(t,'brand','brand_positioning','潮流品牌','时髦品牌的')
    t=4128
    cat(t,'杀虫剂');budget(t,'价格大概十块钱的')
    text(t,'functions','purpose','适用于玉米地','适用于玉米地和花生地的杀虫剂');text(t,'functions','purpose','适用于花生地','适用于玉米地和花生地的杀虫剂')
    fn(t,'补充天然养分','能同时补充天然养分');num(t,'net_content','mass',500,'g','规格500g左右','approximately')
    t=5293
    cat(t,'狗粮');budget(t,'价格要在270元以内');text(t,'functions','target_user','成年犬','针对成年犬设计的')
    spec(t,'condition','处方级','兽医用处方级狗粮');text(t,'functions','purpose','兽医用','兽医用')
    fn(t,'调理肠胃','让这类狗的肠胃问题得到改善');fn(t,'改善软便','主要针对狗狗的大便偏软以及消化不良问题');fn(t,'改善消化不良','主要针对狗狗的大便偏软以及消化不良问题')
    num(t,'net_content','mass',5,'jin','重量不要超过5斤','at_most')
    t=8666
    cat(t,'压脚');budget(t,'预算100元以内');spec(t,'material','全钢','全钢材质')
    fn(t,'工业型电脑平车用');fn(t,'单边隐形拉链','针对单边隐形拉链进行设计的款式')
    add(t,'specifications','bundle','promotion',{'type':'bundle','paid_quantity':10,'gift_quantity':1,'received_quantity':11,'unit':'piece','same_item':True},'买10个要能送1个','matches_bundle')
    pending(t,'需要一个','ambiguous_meaning')
    t=9316
    cat(t,'黄金手镯');budget(t,'价格要一万元左右');spec(t,'craft','传统工艺','传统工艺制作');spec(t,'form_factor','轮胎宽面','像轮胎一样的宽面设计')
    add(t,'specifications','composition','percentage',{'type':'range','lower':99.9,'upper':None,'unit':'percent','lower_inclusive':False,'upper_inclusive':False},
        '黄金含量要大于99.9%','between',subject={'kind':'component','name':'黄金'})
    text(t,'functions','purpose','结婚三金','适合充当结婚用的三金');text(t,'functions','purpose','传家','当传家宝传下去');pending(t,'当传家宝传下去')
    t=12703
    cat(t,'窗帘');budget(t,'预算在三百三十元');spec(t,'style','法式','法式窗帘');spec(t,'style','低调奢华的公主风');pending(t,'低调奢华的公主风')
    text(t,'functions','use_environment','卧室和客厅','适合卧室和客厅');num(t,'attribute_count','other_feature_count',2,'count','两层纱',subject={'kind':'component','name':'纱'})
    fn(t,'100%遮光','100%遮光');component(t,'蕾丝装饰','蕾丝装饰');spec(t,'color','白色','颜色为白色')
    num(t,'dimensions','width',3,'m','宽3m');num(t,'dimensions','height',2.7,'m','高2.7m');num(t,'quantity','purchase_count',1,'piece','一片')
    spec(t,'customization','可改短','能改短的');spec(t,'condition','网络热卖款','网络热卖款的')
    t=15432
    cat(t,'铝板');budget(t,'价格不要超过20元');spec(t,'material','铝','铝板');text(t,'functions','purpose','车用','车上用的')
    spec(t,'pattern','凹凸花纹','多条凹凸花纹');fn(t,'防滑','要能防滑');num(t,'dimensions','length',20,'cm','20厘米×20厘米')
    num(t,'dimensions','width',20,'cm','20厘米×20厘米');num(t,'dimensions','thickness',1.5,'mm','厚度1.5毫米');num(t,'quantity','purchase_count',3,'piece','要3片')
    fn(t,'切口平整光滑','切口平整光滑');fn(t,'无毛刺或飞边','无毛刺或飞边');spec(t,'customization','量身定做','能量身定做');fn(t,'可自行组装','可以自己动手组装')
    t=16252
    cat(t,'投料机','鱼塘专用投料机');budget(t,'价格在800-820元之间')
    text(t,'functions','use_environment','鱼塘','鱼塘专用投料机')
    for value,q in [('自动操作','无需人工操作'),('按时快速投喂','能按时快速投喂'),('无料自动关机','鱼料投放完毕后可自动关机')]:fn(t,value,q)
    spec(t,'form_factor','鱼料分仓','采用鱼料分开存放设计');num(t,'capacity','storage_capacity',3,'bag','一次能存放3包以上饲料','at_least')
    fn(t,'储存空间和机器体积都比较大');pending(t,'储存空间和机器体积都比较大')
    num(t,'warranty','warranty',1,'year','至少1年的质保','at_least')
    t=17825
    cat(t,'景观座椅');budget(t,'大概四百五左右');spec(t,'customization','可定做','能定做的');component(t,'树坑','带树坑')
    text(t,'functions','use_environment','园林','放园林里的');conjunction(t,'functions','function',['结实','耐晒'],'材质要结实耐晒');pending(t,'材质要结实耐晒')
    spec(t,'material','玻璃钢','玻璃钢一类的优先考虑','preference')
    t=18751
    cat(t,'茶具套装','茶具有');budget(t,'预算1800元左右');text(t,'functions','purpose','新婚礼物','当作新婚礼物')
    spec(t,'version','繁花系列','喜欢繁花系列的','preference');spec(t,'packaging','红绸包装礼盒');component(t,'荷花壶','荷花壶')
    component(t,'双鸟架','双鸟架')
    add(t,'specifications','components','included_components',{'type':'components','items':[{'name':n,'quantity':6,'unit':'piece'} for n in ('杯','碟','勺')]},'六套杯碟勺齐全','contains_all')
    spec(t,'material','骨瓷');conjunction(t,'functions','function',['厚实','透光'],'骨瓷厚实透光');fn(t,'用半辈子');pending(t,'用半辈子')
    spec(t,'style','喜气','喜气');spec(t,'style','精致','精致又实用');fn(t,'实用','精致又实用');text(t,'functions','purpose','下午茶','喝下午茶')
    t=21613
    cat(t,'短裙');budget(t,'预算大概550元');text(t,'brand','brand','北面')
    num(t,'quantity','purchase_count',1,'piece','一条')
    spec(t,'version','2025春夏新款','2025年的春夏新款')
    text(t,'functions','use_environment','户外','室外活动比较多')
    conjunction(t,'functions','function',['轻','薄'],'轻薄透气');fn(t,'透气','轻薄透气');fn(t,'防晒','能防晒');fn(t,'抑臭','出汗了不容易有异味')
    spec(t,'color','奶咖棕','奶咖棕色');spec(t,'size','M','M码')
    t=23126
    cat(t,'枕头');budget(t,'价格最好在220块钱左右','preference');fn(t,'放松脖子','晚上睡觉能放松脖子')
    fn(t,'自然支撑脖子','希望材质能自然支撑脖子','uncertain');pending(t,'希望材质能自然支撑脖子','ambiguous_strength')
    spec(t,'form_factor','圆筒','最好圆筒形状的设计','preference')
    spec(t,'material','中药填充','内胆最好也是中药填充的','preference',{'kind':'component','name':'内胆'})
    fn(t,'草药气味助眠','闻着草药的味道能让我有更好的睡眠');pending(t,'闻着草药的味道能让我有更好的睡眠')
    spec(t,'color','灰色','要灰色的')

    drafts=[];results=[]
    for task,c in sorted(contracts.items()):
        drafts.append({'task_id':task,'query':queries[task],'raw_draft':c,'whole_contract_approved':False})
        record={'task_id':task,'query':queries[task],'program_valid':False,'review_scope':'whole query was read; ambiguity and missing standards remain pending',
                'whole_contract_approved':False,'source_query_only':True}
        try:n,a=b.normalize(c,{'task_id':task,'query':queries[task]})
        except Exception as error:record['error']=getattr(error,'message',str(error))
        else:
            # The full selected query and every field were read in this audit.
            # Approve only determinate annotations; pending standards remain.
            for d in b.DIMENSIONS:
                for r in n[d]:
                    if r['importance']!='uncertain':r['review_status']='approved'
            from semantic_source_scope_proofs import resolve_reviewed_claim_mentions
            a.extend(resolve_reviewed_claim_mentions(n))
            from semantic_review_decisions import verification_decision
            for dim in b.DIMENSIONS:
                for requirement in n[dim]:
                    decision=verification_decision(queries[task],requirement)
                    if decision:
                        if requirement['importance']!=decision['importance']:raise ValueError('Verification standard changed required strength')
                        a.append({'kind':'user_confirmed_source_bound_verification_standard','id':requirement['id'],
                                  'decision_id':decision['id'],'standard':decision['standard']})
            try:b.validate(n,source_query=queries[task],require_reviewed=True)
            except ValueError as error:record['approval_blocker']=str(error)
            else:record['whole_contract_approved']=True
            record.update(program_valid=True,normalized=n,audit=a)
        results.append(record)
    (OUT/'dev14_drafts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in drafts),encoding='utf-8')
    (OUT/'dev14_contracts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    print(json.dumps({'contracts':len(results),'program_valid':sum(r['program_valid'] for r in results),
        'blocked':[{'task_id':r['task_id'],'error':r['error']} for r in results if not r['program_valid']]},ensure_ascii=False))

if __name__=='__main__':main()
