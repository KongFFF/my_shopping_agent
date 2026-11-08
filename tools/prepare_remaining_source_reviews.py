"""Materialize inspected source edits as data, never runtime task-ID rules."""
import copy
import hashlib
import json
from pathlib import Path
import migrate_training_semantics as m
import semantic_remaining_policy as p
from close_remaining_semantics import SOURCE,LEDGER

SOURCE_ROWS={r['task_id']:r for r in m.rows(SOURCE)}
reviews=[]

def review(task,note):
    row=SOURCE_ROWS[task]
    d={'id':f'remaining-source-{task}-{len(reviews)+1}','task_id':task,'query_sha256':hashlib.sha256(row['query'].encode()).hexdigest(),
       'query':row['query'],'note':note,'reviewer':'assistant_source_review_not_whole_query_approval',
       'changes':[],'resolves':[]}
    working=copy.deepcopy(row['normalized']);d['_working']=working
    reviews.append(d)
    return working,d

def patch(c,d,rid,*,dimension=None,source_texts=None,**updates):
    dim,r=next((dim,r) for dim,r in m.allreq(c) if r['id']==rid)
    op={'op':'patch','id':rid,'before':copy.deepcopy(r),'updates':updates}
    if source_texts is not None:op['source_texts']=source_texts
    if dimension:op['dimension']=dimension
    d['changes'].append(op);r.update(copy.deepcopy(updates))
    if source_texts is not None:r['evidence']=[m.evidence(c['query'],t) for t in source_texts]
    if dimension and dim!=dimension:c[dim].remove(r);c[dimension].append(r)

def remove(c,d,rid):
    dim,r=next((dim,r) for dim,r in m.allreq(c) if r['id']==rid)
    d['changes'].append({'op':'remove','id':rid,'before':copy.deepcopy(r)});c[dim].remove(r)

def add(c,d,text,subtype,**kwargs):
    dim,r=m.req(c['query'],text,subtype,**kwargs)
    q=copy.deepcopy(r);q.pop('evidence');q['source_texts']=[text]
    d['changes'].append({'op':'add','dimension':dim,'requirement':q});m.add_unique(c,dim,r)

def resolve(c,d,text=None,reason=None):
    for item in c['pending_mentions']:
        if (text is None or item['evidence']['text']==text) and (reason is None or item['reason']==reason):
            d['resolves'].append({'reason':item['reason'],'text':item['evidence']['text']})

def existing_pending(task,text,subtype,*,value=None,importance='required',note='',operator='eq'):
    c,d=review(task,note or '保留明确的原文属性，不推断品牌、型号、数值或隐含清单。')
    add(c,d,text,subtype,value=value,importance=importance,operator=operator)
    resolve(c,d,text)
    return c,d

def main():
    c,d=review(15386,'明确2025年8月发售月份，只按同一公开发售月份标签核验，其他月份未知，不与发货时间混用。')
    patch(c,d,'r003',dimension='model',subtype='version',attribute='release_month',operator='matches_label',expected={'type':'text','value':'2025年8月'},importance='required',review_status='approved')
    resolve(c,d,'在2025年8月左右发售')
    c,d=review(16609,'同款指代已明确为徐璐；按既定公开同款声明标准核验，不把它当缺失年份。')
    add(c,d,'徐璐的润唇膏','condition',value={'type':'text','value':'徐璐同款'})
    d['changes'][-1]['requirement']['source_texts']=['徐璐的润唇膏','同款的唇部精华']
    resolve(c,d,'同款')
    c,d=review(17831,'8寸仅作原文相框规格标签；白色明确属于相框，不是画面主色。')
    patch(c,d,'r006',operator='matches_label',expected={'type':'text','value':'8寸'},importance='required',review_status='approved')
    patch(c,d,'r007',subject={'kind':'component','name':'白框'},review_status='approved')
    resolve(c,d,'8寸左右');resolve(c,d,reason='unsupported_unit')
    for task,rid,text in [(3599,'r003','行程是80'),(12117,'r010','1.9长的床')]:
        c,d=review(task,'仅保留原文完整规格标签，不把缺失单位补成毫米或米。')
        patch(c,d,rid,operator='matches_label',expected={'type':'text','value':text},importance='required',review_status='approved');resolve(c,d,text)
    c,d=review(3727,'30杯保留为原文规格声明，不推断是30个小杯包装还是可冲30杯的容量。')
    add(c,d,'有30杯左右','function',operator='matches_label',value={'type':'text','value':'有30杯左右'});resolve(c,d,'有30杯左右')
    c,d=review(1926,'不要钱的话最好明确为免费偏好；不是必须免费，也不丢失该偏好。')
    patch(c,d,'r006',operator='at_most',expected={'type':'number','value':0,'unit':'CNY'},importance='preference',review_status='approved');resolve(c,d,'不要钱的话最好')
    c,d=review(19395,'价格个位数表示小于10元，不强行舍弃小数金额；今年的年份仍未知。')
    target={'type':'range','lower':None,'upper':10,'unit':'CNY','lower_inclusive':False,'upper_inclusive':False}
    if c['budget']:
        for r in list(c['budget']):patch(c,d,r['id'],operator='between',expected=target,importance='required',review_status='approved')
    else:add(c,d,'不用很贵，个位数可以了','total_price',subject=m.ORDER,operator='between',value=target)
    resolve(c,d,'不用很贵，个位数可以了')
    if p.confirmed('last_two_clauses'):
        c,d=review(4049,'用户确认：儿童床兼容目标保留；加宽垫和可折叠地铺两种用途任选一个，不保留旧双必需。')
        remove(c,d,'r005');remove(c,d,'r006')
        add(c,d,'要能铺在幼儿园小床边当加宽垫，或者折叠起来当地铺用的','purpose',operator='one_of',value={'type':'text_list','values':['铺在幼儿园小床边当加宽垫','折叠起来当地铺用的']})
        add(c,d,'儿童床的','compatibility',subject={'kind':'compatible_target','name':'儿童床'},value={'type':'text','value':'儿童床'})
        resolve(c,d)
        c,d=review(19256,'用户确认：老人久坐不硌为偏好，按明确公开声明核验，不证明主观坐感。')
        add(c,d,'最好是头层牛皮材质的老人坐久了不会觉得硌得慌','function',value={'type':'text','value':'老人坐久了不会觉得硌得慌'},importance='preference')
        resolve(c,d,'老人坐久了不会觉得硌得慌')
    c,d=review(10086,'保留半世纪的薄荷香味这一完整原文名称，不将半世纪解释为必须陈放50年，也不猜品牌。')
    patch(c,d,'r005',expected={'type':'text','value':'半世纪的薄荷香味'},source_texts=['半世纪的薄荷香味'],review_status='approved')
    resolve(c,d,'半世纪的薄荷香味')
    c,d=review(3951,'每遍涂刷面积只按完整原文技术声明保留，不把裸平方擅自换算平方米。')
    add(c,d,'刷一遍大概能涂二十平方','function',operator='matches_label',value={'type':'text','value':'刷一遍大概能涂二十平方'});resolve(c,d,'刷一遍大概能涂二十平方')
    c,d=review(12062,'约10档的完整组合声明保留，不假定是三类各10档；已有三种可调功能保持。')
    add(c,d,'有10档左右温度、震动强度和微电档位可调节','function',operator='matches_label',value={'type':'text','value':'有10档左右温度、震动强度和微电档位可调节'})
    resolve(c,d,reason='undefined_tolerance')
    c,d=review(21690,'1.5匹为制冷规格标签，不是Wh电池能量，也不猜额定耗电瓦数。')
    patch(c,d,'r009',dimension='functions',subtype='function',attribute='cooling_power',operator='matches_label',expected={'type':'text','value':'1.5匹'},importance='required',review_status='approved');resolve(c,d,'容量1.5匹左右')
    c,d=review(8139,'256G保留为原文存储规格标签，不猜GB/GiB；只接受同一完整标签。')
    patch(c,d,'r006',operator='matches_label',expected={'type':'text','value':'256G'},review_status='approved');resolve(c,d,'256G左右')
    c,d=review(8784,'用户C表已给出21至29的失真效果数，清除同句剩余的关系待审标记。')
    patch(c,d,'r006',operator='between',expected={'type':'range','lower':21,'upper':29,'unit':'count','lower_inclusive':True,'upper_inclusive':True},importance='required',review_status='approved');resolve(c,d)
    if p.confirmed('table_b'):
        c,d=review(53,'用户B表：25米保留为完整原文规格，既不认作管长，也不认作射程。')
        patch(c,d,'r006',review_status='approved');resolve(c,d,'25米快插式地埋喷头')
        c,d=review(5588,'用户B表：按可读的注射器式外形保留，央视作为原文损坏审计，不补成认证或品牌。')
        add(c,d,'注射器央视的','form_factor',value={'type':'text','value':'注射器式'});resolve(c,d,'注射器央视的')
        c,d=review(9286,'用户B表：损坏数字一100修订为100包，原query不改，其他数量未知。')
        patch(c,d,'r006',operator='approximately',expected={'type':'number','value':100,'unit':'bag'},importance='required',review_status='approved')
        d['numeric_source_correction']=True;resolve(c,d,'需要大概一100包')
        c,d=review(9294,'用户B表：两袋为最终实收2袋，同种商品付1赠1，每袋250g。')
        patch(c,d,'r008',review_status='approved')
        patch(c,d,'r009',attribute='received_count',operator='eq',expected={'type':'number','value':2,'unit':'bag'},importance='required',review_status='approved')
        patch(c,d,'r010',operator='matches_bundle',expected={'type':'bundle','paid_quantity':1,'gift_quantity':1,'received_quantity':2,'unit':'bag','same_item':True},review_status='approved')
        resolve(c,d)
        c,d=review(11851,'用户B表：约3包，合计约150粒；不认作每包150粒。')
        patch(c,d,'r007',importance='required',operator='approximately',expected={'type':'number','value':3,'unit':'bag'},review_status='approved')
        patch(c,d,'r008',attribute='received_count',importance='required',operator='approximately',expected={'type':'number','value':150,'unit':'piece'},review_status='approved')
        resolve(c,d)
        c,d=review(14916,'用户B表：9头是款式标签，花枝数不超过3，两个条件分别保留。')
        patch(c,d,'r004',expected={'type':'text','value':'9头蝴蝶兰'},review_status='approved')
        add(c,d,'粉色的9头蝴蝶兰不超过3枝','quantity',attribute='included_component_count',subject={'kind':'component','name':'蝴蝶兰'},operator='at_most',value={'type':'number','value':3,'unit':'piece'})
        resolve(c,d,'粉色的9头蝴蝶兰不超过3枝')
        c,d=review(15845,'用户B表：整包4支、两把牙刷各可分配2个替换头，不强求外观为两个不同版本。')
        patch(c,d,'r005',review_status='approved')
        add(c,d,'4支装刚好两把牙刷各有两个替换头','function',value={'type':'text','value':'两把牙刷各有两个替换头'})
        resolve(c,d,'4支装刚好两把牙刷各有两个替换头')
        c,d=review(19437,'用户B表：芝士淡奶油或炼乳二选一；已定总120g、10包保持。')
        patch(c,d,'r002',operator='one_of',expected={'type':'text_list','values':['芝士淡奶油','炼乳']},review_status='approved');resolve(c,d,'芝士淡奶油或炼乳')
        c,d=review(21514,'用户B表：可调区间覆盖40到80厘米为偏好，固定高度不能满足。')
        patch(c,d,'r005',dimension='specifications',subtype='dimensions',attribute='height',operator='covers_range',expected={'type':'range','lower':40,'upper':80,'unit':'cm','lower_inclusive':True,'upper_inclusive':True},review_status='approved')
        resolve(c,d,'升降范围最好是在40到80厘米之间')
        c,d=review(23082,'用户B表：三种背法都支持，作为一个组合功能避免拆成三项加权。')
        add(c,d,'既可斜挎着背也能单边肩背，或者横着在腰部使用','function',operator='contains_all',value={'type':'text_list','values':['斜挎','单肩背','腰部使用']})
        resolve(c,d,'既可斜挎着背也能单边肩背，或者横着在腰部使用')
        c,d=review(12941,'用户B表：350元是明确报价目标；两面同色为独立关系要求。')
        patch(c,d,'r008',operator='eq',expected={'type':'number','value':350,'unit':'CNY'},importance='required',review_status='approved')
        add(c,d,'两面同一种颜色','function',value={'type':'text','value':'两面同一种颜色'})
        resolve(c,d,'价格为350元的那一款');resolve(c,d,'两面同一种颜色')
    if p.confirmed('table_c'):
        for task in (9227,12564,13016,13806):
            c,d=review(task,'用户C表：预估不另加价，接近按已定左右预算规则。')
            for r in list(c['budget']):
                for e in r['evidence']:
                    try:op,v,_=m.b.budget_for_role(m.b.budget_source_clause(c['query'],e['text']),r['subtype'])
                    except ValueError:continue
                    patch(c,d,r['id'],operator=op,expected=v,importance='required',review_status='approved');resolve(c,d,reason='ambiguous_meaning');break
        c,d=review(11503,'用户C表：10来张只支持明确中心10张，其他数未知；必需强弱沿用C表旧裁定。')
        patch(c,d,'r005',operator='approximately',expected={'type':'number','value':10,'unit':'sheet'},importance='required',review_status='approved');resolve(c,d)
        c,d=review(2277,'用户C表：几十个夹子按20至99个；该数量是赠品数量，不是订单数。')
        remove(c,d,'r007')
        add(c,d,'附赠几十个木质夹子','quantity',attribute='gift_count',subject={'kind':'gift','name':'木质夹子'},operator='between',value={'type':'range','lower':20,'upper':99,'unit':'piece','lower_inclusive':True,'upper_inclusive':True});resolve(c,d,'几十个')
        c,d=review(12769,'用户C表：10多个随附晾衣架为11至19个，不改主商品订单数量。')
        remove(c,d,'r010')
        add(c,d,'10多个铝制晾衣架','quantity',attribute='included_component_count',subject={'kind':'component','name':'铝制晾衣架'},operator='between',value={'type':'range','lower':11,'upper':19,'unit':'piece','lower_inclusive':True,'upper_inclusive':True});resolve(c,d,'10多个铝制晾衣架')
        c,d=review(5837,'用户C表：草本成分种数为20至99，已点名的三种成分仍保留；不冒充完整成分表。')
        add(c,d,'等几十种草本植物','attribute_count',attribute='herbal_ingredient_count',operator='between',value={'type':'range','lower':20,'upper':99,'unit':'count','lower_inclusive':True,'upper_inclusive':True});resolve(c,d,'等几十种草本植物')
        c,d=review(11037,'用户C表：500多升范围保留为(500,600)升，不误记存储字节容量。')
        patch(c,d,'r009',attribute='liquid_volume',importance='required',operator='between',expected={'type':'range','lower':500,'upper':600,'unit':'l','lower_inclusive':False,'upper_inclusive':False},review_status='approved');resolve(c,d,'大概500多升的容量')
        c,d=review(14555,'用户C表：100多个是国家数101至199，不冒充语言总数。')
        patch(c,d,'r004',dimension='specifications',subtype='attribute_count',attribute='supported_country_count',operator='between',expected={'type':'range','lower':101,'upper':199,'unit':'count','lower_inclusive':True,'upper_inclusive':True},review_status='approved');resolve(c,d,'支持100多个国家的语言')
    # Explicit bare human height is the only newly approved missing-unit inference.
    if p.confirmed('bare_human_height_cm'):
        for task,row in SOURCE_ROWS.items():
            fields=[r for _,r in m.allreq(row['normalized']) if p.human_height_target(r,m.b.chinese_number)]
            if not fields:continue
            c,d=review(task,'用户确认：明确人体身高语境的110/160/85等裸数按厘米，其他缺单位不套此规则。')
            for r in fields:
                v=p.human_height_target(r,m.b.chinese_number)
                approx=any(p.APPROX.search(e['text']) for e in r['evidence'])
                patch(c,d,r['id'],operator='approximately' if approx else 'eq',expected=v,importance='required',review_status='approved')
            resolve(c,d,reason='unsupported_unit');resolve(c,d,reason='undefined_tolerance')
    # Generic attribute slots must not compare unrelated counts or physical quantities.
    for task,rid,updates in [(4979,'r005',{'attribute':'light_count'}),(11016,'r005',{'attribute':'language_count'}),
        (16944,'r006',{'attribute':'hole_count'}),(7035,'r010',{'attribute':'coverage_area'}),
        (22198,'r003',{'attribute':'cooling_power'}),(22735,'r005',{'attribute':'circumference'})]:
        c,d=review(task,'修正通用属性槽，避免孔数/语言数、制冷量/耗电功率、覆盖面积/存储容量相互冒充。')
        patch(c,d,rid,importance='required',review_status='approved',**updates)
    c,d=review(5774,'纯度是磷酸二氢钾所占百分比，不是净含量；近似100%按确认的精确中心口径。')
    patch(c,d,'r001',subtype='composition',attribute='percentage',subject={'kind':'component','name':'磷酸二氢钾'},operator='approximately',expected={'type':'number','value':100,'unit':'percent'},source_texts=['纯度接近100%','磷酸二氢钾肥料'],review_status='approved')
    resolve(c,d,'纯度接近100%')
    # An explicitly written scalar specification need not acquire a guessed axis.
    for task,rid,text in [(1115,'r001','6mm'),(4280,'r005','五十毫米以上'),
        (4971,'r001','14mm'),(10950,'r007','25公分'),(12126,'r006','不超过0.5毫米'),
        (15050,'r005','120厘米'),(19256,'r006','3.23米'),(21607,'r002','20mm'),(22300,'r006','12厘米以上')]:
        c,d=review(task,'用户确认规格标签标准：保留完整原文数值/边界，不猜长度或直径，也不换算。')
        patch(c,d,rid,operator='matches_label',expected={'type':'text','value':text},importance='required',review_status='approved')
        for item in c['pending_mentions']:
            if item['reason']=='ambiguous_meaning' and any(m.overlap(item['evidence'],e) for _,r in m.allreq(c) if r['id']==rid for e in r['evidence']):resolve(c,d,item['evidence']['text'],item['reason'])
    for task in (6193,17581):
        c,d=review(task,'完整规格已作为组件名称原样保留；按同名公开组件核验，不解释寸的物理单位。')
        resolve(c,d,reason='unsupported_unit')
    c,d=review(7777,'分开相框本体和明示属性，避免完整组件名再重复包含尺寸/颜色/材质。寸只作原文标签。')
    patch(c,d,'r006',expected={'type':'components','items':[{'name':'相框','quantity':None,'unit':None}]},review_status='approved')
    patch(c,d,'r007',subject={'kind':'component','name':'相框'},operator='matches_label',expected={'type':'text','value':'7寸正方形'},review_status='approved')
    for rid in ('r008','r009'):patch(c,d,rid,subject={'kind':'component','name':'相框'},review_status='approved')
    resolve(c,d,reason='unsupported_unit')
    c,d=review(16457,'适配16寸童车是完整兼容对象标签；去除重复的主商品尺码条件，不转英寸。')
    remove(c,d,'r004');patch(c,d,'r001',operator='matches_label',review_status='approved');resolve(c,d,reason='unsupported_unit')
    c,d=review(7077,'不新增猜测的数据速率单位，保留不低于10Gbps完整技术声明，只接受相同声明。')
    patch(c,d,'r006',operator='matches_label',review_status='approved');resolve(c,d,'传输速度不低于10Gbps')
    c,d=review(15412,'最大处理量为技术规格；保留最大处理量400毫升完整标签，不误当成至少400毫升。')
    patch(c,d,'r005',operator='matches_label',review_status='approved');resolve(c,d,'最大处理量400毫升')
    c,d=review(4768,'原文已明确赠送4件配件，不需要补造四个附件名称。')
    patch(c,d,'r014',review_status='approved');resolve(c,d,'送赠送4件套配件')
    c,d=review(5837,'三盒包装合计18支，不把支数当成功能种类，也不推断每盒必为6支。')
    patch(c,d,'r003',operator='eq',expected={'type':'number','value':3,'unit':'box'},importance='required',review_status='approved')
    patch(c,d,'r004',subtype='packaging',attribute='pack_count',expected={'type':'number','value':18,'unit':'piece'},review_status='approved')
    resolve(c,d,'要三盒装');resolve(c,d,'要三盒装一共有十八支的')
    c,d=review(16730,'三件套只给出套装件数，未列组件名称不等于三件数值不明确。')
    patch(c,d,'r003',attribute='pack_count',expected={'type':'number','value':3,'unit':'piece'},importance='required',review_status='approved');resolve(c,d,'三件套')
    c,d=review(9984,'起订量是商家销售条件，不能当作用户订单买20个。')
    patch(c,d,'r005',subtype='transaction_terms',attribute='sale_unit_quantity',review_status='approved')
    c,d=review(20808,'不太光滑明确修饰套装内的衬衫；按该部件的公开声明核验。')
    add(c,d,'不是很光滑','function',subject={'kind':'component','name':'衬衫'},value={'type':'text','value':'不是很光滑'})
    d['changes'][-1]['requirement']['source_texts']=['不是很光滑','白色短袖衬衫']
    resolve(c,d,'不是很光滑')
    c,d=review(17795,'天然木材和无特殊气味是两种属性；气味按明确公开声明，不做主观闻味判断。')
    patch(c,d,'r006',expected={'type':'text','value':'自然木材'},review_status='approved')
    add(c,d,'最好是用没有特殊气味的自然木材','function',value={'type':'text','value':'没有特殊气味'},importance='preference')
    resolve(c,d,'没有特殊气味')
    c,d=review(10144,'保留非低端的明确否定属性，不将广告高价或高品质自动作为满足证明。')
    add(c,d,'不要低端的','condition',value={'type':'text','value':'非低端'})
    resolve(c,d,'不要低端的')
    if p.confirmed('table_a'):
        for task,rid,text,importance in [(3398,'r005','好像是天蓝色包装的','preference'),
            (5225,'r004','听说广源出的奶黄味很稳定','preference'),
            (19006,'r011','听说格力这个品牌不错','preference'),
            (22089,'r003','听朋友说CPB家的长管隔离很好用','required')]:
            c,d=review(task,'用户采纳A表：仅裁定本句的品牌/属性强弱，不推广传闻词规则。')
            r=next(r for _,r in m.allreq(c) if r['id']==rid)
            texts=[e['text'] for e in r['evidence']]
            if text not in texts:texts.append(text)
            patch(c,d,rid,importance=importance,review_status='approved',source_texts=texts)
            if task==3398:patch(c,d,rid,subject={'kind':'component','name':'包装'})
            if task==22089:remove(c,d,'migration002')
            resolve(c,d,text)
        c,d=review(3844,'用户A表：黑色与耐脏是偏好，安装简单和踩稳仍按已确认必需。')
        for rid in ('r005','r006'):
            r=next(r for _,r in m.allreq(c) if r['id']==rid)
            texts=[e['text'] for e in r['evidence']]
            if '黑色也比较耐脏' not in texts:texts.append('黑色也比较耐脏')
            patch(c,d,rid,importance='preference',review_status='approved',source_texts=texts)
        resolve(c,d,'黑色也比较耐脏')
        for task,text,sub,value,importance in [
            (4151,'推荐高光泽釉面与金属质感结合的设计','style','高光泽釉面与金属质感结合的设计','preference'),
            (8124,'金维肽的效果可能会好一点','condition','金维肽','preference'),
            (10086,'听说这种雕刻是非遗手艺','craft','非遗手艺','preference'),
            (10181,'发货速度快一点','delivery','快速发货','required'),
            (11037,'一级节能的那种','condition','一级节能','required'),
            (19296,'朋友很喜欢日本的粉霜','origin','日本','preference'),
            (21100,'听说兴牧康畜牧疫苗企业店不错','merchant','兴牧康畜牧疫苗企业店','preference')]:
            c,d=existing_pending(task,text,sub,value={'type':'text','value':value},importance=importance,note='用户采纳A表逐项强弱；公开声明核验，未声明仍未知。')
            if task==11037:
                # Include 希望 in the exact source proof, not just a fragment.
                r=next(r for _,r in m.allreq(c) if r['expected'].get('value')==value)
                # The new requirement is already recorded as an add operation.
                d['changes'][-1]['requirement']['source_texts']=['希望是一级节能的那种']
                r['evidence']=[m.evidence(c['query'],'希望是一级节能的那种')]
        c,d=review(13918,'用户A表：训练或排球任一适用，为一个用途偏好。')
        patch(c,d,'r007',importance='preference',operator='one_of',review_status='approved');resolve(c,d,'我可能会用来训练或者打排球')
        c,d=review(20711,'用户A表：3+32G为必需原文配置标签，不推断GB/GiB或最低配置。')
        patch(c,d,'r007',importance='required',review_status='approved',dimension='specifications',subtype='capacity',attribute='storage_capacity',operator='matches_label',expected={'type':'text','value':'3+32G'},source_texts=['内存3+32G够用就行'])
        resolve(c,d,'够用就行')
        c,d=review(21203,'用户A表：后续购买请求明确指代大疆；传闻质量仅保留审计备注。')
        patch(c,d,'r001',importance='required',review_status='approved');resolve(c,d,'听说大疆品牌的移动电源质量不错')
    if p.confirmed('complete_accessory_claim'):
        for task,text,ids in [(247,'希望能包含所有配件',['r008']),
            (3877,'超多全套配件',['r008']), (5423,'配套设施要齐全',[]),
            (8711,'很齐全的配件',['r002','migration001']),
            (9329,'配件要齐全',['r004']), (20590,'七件套配件齐全',['r005'])]:
            c,d=review(task,'用户确认：配件齐全只按当前商品/套餐的明确齐全声明核验，具体点名配件保持独立。')
            for rid in ids:remove(c,d,rid)
            add(c,d,text,'function',value={'type':'text','value':'配件齐全'})
            resolve(c,d,text)
            if task==20590:
                add(c,d,'七件套','packaging',attribute='pack_count',value={'type':'number','value':7,'unit':'piece'})
    # Explicitly named dimensional axes are separable; unlabelled axes are not guessed.
    c,d=review(77,'宽、高为原文明示的两个轴，分别保留近似中心，不套统一容差。')
    patch(c,d,'r010',source_texts=['尺寸大概1米5宽'],operator='approximately',expected={'type':'number','value':1.5,'unit':'m'},review_status='approved')
    patch(c,d,'r011',source_texts=['2米高左右'],operator='approximately',expected={'type':'number','value':2,'unit':'m'},review_status='approved')
    resolve(c,d,'尺寸大概1米5宽、2米高左右')
    c,d=review(2061,'原文明示2米宽、1.8米高，分轴保留，不猜未命名的尺寸顺序。')
    remove(c,d,'r010')
    add(c,d,'尺寸约为2米宽','dimensions',attribute='width',operator='approximately',value={'type':'number','value':2,'unit':'m'})
    add(c,d,'1.8米高','dimensions',attribute='height',value={'type':'number','value':1.8,'unit':'m'})
    resolve(c,d,'尺寸约为2米宽、1.8米高')
    c,d=review(20050,'原文明示宽和长；近似前缀统摄两个尺寸，分别保留来源。')
    remove(c,d,'r006')
    add(c,d,'尺寸大概90公分宽','dimensions',attribute='width',operator='approximately',value={'type':'number','value':90,'unit':'cm'})
    add(c,d,'1米5长的','dimensions',attribute='length',value={'type':'number','value':1.5,'unit':'m'})
    resolve(c,d,'尺寸大概90公分宽1米5长的')
    # Named source strings remain opaque names, not corrected zoology/brands.
    for task,text,sub in [(2358,'数量有限款','condition'),(3527,'半度','condition'),
        (4280,'索摩里鬼艳锹甲','content'),(4505,'蝴蝶盛宴剧粉蝶','content'),
        (6289,'重量级','condition'),(13751,'富晒','composition'),(22602,'特研','condition')]:
        existing_pending(task,text,sub,note='按原文明确名称/定性声明保留；不改错别字，不把“重量级”转成重量，不猜物种或品牌。')
    existing_pending(5973,'精品款','condition',importance='preference')
    existing_pending(9028,'正版','condition',importance='preference')
    existing_pending(16274,'男女同款','target_user')
    c,d=existing_pending(4051,'可以先付订金的','transaction_terms',value={'type':'text','value':'可以先付订金'},note='要求支持订金付款这一交易能力；不同于允许某种颜色/材质。')
    # Clearly explicit package counts, not an order of N separately sold packs.
    for task,rid,count,unit in [(2610,'r004',1,'book'),(7163,'r004',2,'bottle'),(9558,'r009',2,'box'),(19859,'r002',3,'piece')]:
        c,d=review(task,'原文N装为包装数量；不补成下单N个销售单位。')
        patch(c,d,rid,subtype='packaging',attribute='pack_count',operator='eq',expected={'type':'number','value':count,'unit':unit},importance='required',review_status='approved')
        for item in c['pending_mentions']:
            if item['reason']=='ambiguous_meaning' and any(m.overlap(item['evidence'],e) for _,r in m.allreq(c) if r['id']==rid for e in r['evidence']):resolve(c,d,item['evidence']['text'],item['reason'])
        if task==2610:
            patch(c,d,'r005',subtype='packaging',attribute='pack_count',expected={'type':'number','value':80,'unit':'sheet'},review_status='approved')
    # Source wording already fixes the budget. Resolve stale non-budget flags.
    for task in (2465,8049,15386,21011,22987):
        c,d=review(task,'预算已按既定货币规则解析；清除误挂在预算上的非预算容差标记。')
        for item in c['pending_mentions']:
            if item['reason']=='undefined_tolerance' and any(any(m.overlap(item['evidence'],e) for e in r['evidence']) for r in c['budget']):resolve(c,d,item['evidence']['text'],item['reason'])
    # Repair dimensions vs mass, power, usable volume, and total order mass.
    for task,rid,updates in [(2506,'r002',{'subtype':'net_content','attribute':'mass'}),
        (5418,'r005',{'attribute':'liquid_volume'}),(5774,'r010',{'attribute':'purchase_mass'}),
        (17955,'r005',{'attribute':'purchase_mass'}),(22678,'r004',{'subtype':'function','attribute':'power'})]:
        c,d=review(task,'按原文明示单位修正量纲；随后由通用近似目标规则核验。')
        patch(c,d,rid,dimension='functions' if task==22678 else None,importance='required',review_status='approved',**updates)
        if task==22678:resolve(c,d,'30瓦左右')
    # Explicit omitted meter-decimals, with source axis retained.
    for task,rid in [(966,'r005'),(8477,'r011'),(20808,'r008')]:
        c,d=review(task,'米后单个位为小数，如1米4=1.4m；仍不设近似容差。')
        patch(c,d,rid,review_status='approved')
    # Exact per-container count was omitted from extraction.
    c,d=review(2184,'恢复每盒约90颗的包装数量，保持订单3盒独立。')
    add(c,d,'每盒里边有90颗左右','packaging',attribute='pack_count',operator='approximately',value={'type':'number','value':90,'unit':'piece'})
    resolve(c,d,'每盒里边有90颗左右')
    # A short qualitative statement is not a fabricated numeric threshold.
    qualitative=[(4147,'r007','要很轻薄','function','required'),
        (7520,'r001','梳齿多','function','required'),(7682,'r004','材质要厚一点的','function','required'),
        (7764,'r009','稍厚一点的','function','required'),(7764,'r010','袖子长一些','function','required'),
        (9095,'r004','要短小一点的半帘','form_factor','required'),(9239,'r004','底部要厚一点','function','required'),
        (9329,'r006','功率要大一点','function','required'),(13016,'r006','容量大些','function','required'),
        (17932,'r007','体积不用太大','form_factor','required'),(20482,'r008','厚一些','function','required'),
        (23206,'r006','功率大一点的','function','required')]
    grouped={}
    for task,rid,text,sub,importance in qualitative:
        if task not in grouped:grouped[task]=review(task,'定性要求按明确商品声明核验；撤去错误的数值/材质归类，不补比较基准或阈值。')
        c,d=grouped[task]
        dim=next(dim for dim,types in m.b.DIMENSIONS.items() if sub in types)
        patch(c,d,rid,dimension=dim,subtype=sub,attribute=m.b.ATTRIBUTES[sub][0],expected={'type':'text','value':m.grammatical_core(text)},importance=importance,operator='eq',review_status='approved')
        resolve(c,d,text)
    for task,text in [(9140,'脂肪含量少'),(9140,'热量低'),(22134,'能覆盖车间厂房')]:
        if task==22134:
            c,d=review(task,'适用场景已明确表达，未要求未知的覆盖面积阈值。');resolve(c,d,text)
        else:existing_pending(task,text,'function',note='按明确的低脂/低热量公开声明核验，不编造营养学阈值。')
    # Adjacent explanatory wording has an explicit numerical requirement.
    for task,text in [(882,'喷杆要长一点'),(6401,'容量别太小'),(11941,'尺寸不用太大'),(18008,'量不用太多')]:
        c,d=review(task,'相邻数值已具体化该描述，保留为非重复计分说明。');resolve(c,d,text)
    # Required zero buyer shipping cost; remove duplicate generic-function labels.
    for task,rid,text in [(8711,'r003','邮费由商家承担'),(10355,'r006','不用出运费')]:
        c,d=review(task,'明确买方无需运费；统一为运费上限0，不另加一个功能奖励。')
        patch(c,d,rid,operator='at_most',expected={'type':'number','value':0,'unit':'CNY'},importance='required',review_status='approved')
        for _,r in list(m.allreq(c)):
            if r['id']!=rid and r['subtype']=='function' and r['expected'].get('value')==text:remove(c,d,r['id'])
        resolve(c,d,text)
    c,d=review(14600,'包含邮费为总价含运费的交易条件，不是商品价格0或运费必定0。')
    remove(c,d,'r007');resolve(c,d,'包含邮费')
    c,d=review(11841,'最好明确为偏好；保留原文通用性要求。')
    add(c,d,'最好是通用性好的','compatibility',value={'type':'text','value':'通用性好'},importance='preference');resolve(c,d,'最好是通用性好的')
    existing_pending(12380,'有特价优惠活动','condition',note='明确要求有特价活动；与临期状态分开保留。')
    c,d=review(12614,'一般款式就行没有提出额外可计分规格，作为许可性说明保留。');resolve(c,d,'一般的款式就行')
    c,d=review(20753,'在指定品牌中找商品的直接请求，品牌为必需；问句不自动降级为偏好。')
    patch(c,d,'r009',importance='required',review_status='approved');resolve(c,d,'有CRUML&KECC家的推荐吗？')
    # Permission cases are activated only after the user's explicit reply.
    if p.confirmed('permission_notes'):
        for task,rid,text in [(16807,'r005','颜色可以是玫瑰金'),(22210,'r003','可以是石膏材质的')]:
            c,d=review(task,'用户确认：两处允许表述均按偏好计分；不接受旧推荐的不计分备注方案。');patch(c,d,rid,importance='preference',review_status='approved',source_texts=[text]);resolve(c,d,text)
    if p.confirmed('gift_15032'):
        c,d=review(15032,'用户确认：赠品为偏好，不补件数。')
        patch(c,d,'r009',importance='preference',review_status='approved');resolve(c,d,'可以赠送耳机，转接头和OTG线')
    # Use a literal complete source label without deciding real unit/axes.
    if p.confirmed('literal_spec_label'):
        for task,row in SOURCE_ROWS.items():
            ps=[x for x in row['normalized']['pending_mentions'] if x['reason'] in ('unsupported_unit','ambiguous_meaning','undefined_tolerance')]
            if not ps:continue
            chosen=[]
            modified={(d['task_id'],op.get('id')) for d in reviews for op in d['changes'] if op['op'] in ('patch','remove')}
            for dim,r in m.allreq(row['normalized']):
                if (task,r['id']) in modified:continue
                if r['subtype'] not in ('dimensions','size','compatibility'):continue
                v=r['expected'];text=v.get('value') if v['type']=='text' else None
                es=[e['text'] for e in r['evidence']]
                # Labels only for explicitly written 寸/英寸 or dimensions;
                # missing units and approximate scalar targets stay separate.
                if text and any(__import__('re').search(r'(?:[0-9一二两三四五六七八九十]+(?:\.\d+)?)(?:英)?寸',e) for e in es) and any(any(m.overlap(x['evidence'],e) for e in r['evidence']) for x in ps):chosen.append((r,text))
                elif r['attribute']=='dimensions' and text and __import__('re').search(r'\d+(?:\.\d+)?[xX×*]\d',text):chosen.append((r,text))
                elif r['attribute']=='dimensions' and text and __import__('re').search(r'[0-9一二两三四五六七八九十半]',text) and any(u in text for u in ('mm','cm','CM','厘米','毫米','公分','米')) and not any(e['text'].endswith('长') or '宽' in e['text'] or '高' in e['text'] for e in r['evidence']):chosen.append((r,text))
            if not chosen:continue
            c,d=review(task,'用户确认规格标签口径：原单位/尺寸轴仍未知，只接受同一完整原文标签，不作物理换算。')
            for old,text in chosen:
                if not any(text in e['text'] for e in old['evidence']):continue
                # Retain the exact complete target label while discarding only
                # grammatical introducers/approximation wording. Bounds stay.
                import re
                if not re.search(r'超过|以下|以上|少于|至少|最多|以内|大于|小于',text):
                    unit=r'(?:毫米|厘米|公分|英寸|寸|mm|cm|CM|米|m)?'
                    labels=list(re.finditer(r'(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半]+)(?:(?:[xX×*])\d+(?:\.\d+)?){0,2}'+unit,text))
                    if len(labels)==1:text=labels[0].group()
                if old['importance']=='uncertain':
                    # An unspecified unit/axis does not itself make strength unknown.
                    importance=m.strength_for(c,old)
                    if importance is None:continue
                else:importance=old['importance']
                patch(c,d,old['id'],operator='matches_label',expected={'type':'text','value':text},importance=importance,review_status='approved')
                for x in ps:
                    if any(e['start']<=x['evidence']['start']<x['evidence']['end']<=e['end'] for e in old['evidence']):resolve(c,d,x['evidence']['text'],x['reason'])
    # Exact source/predicate bindings only for clauses the user actually decided.
    user_decisions=[]
    for d in reviews:
        authorized=('用户采纳A表' in d['note'] or '用户A表' in d['note'] or '用户确认' in d['note'] or '用户B表' in d['note'] or '用户C表' in d['note'])
        if not authorized:continue
        c=d['_working'];ids={op['id'] for op in d['changes'] if op['op']=='patch'}
        added=[op['requirement'] for op in d['changes'] if op['op']=='add']
        for _,r in m.allreq(c):
            keys=('subtype','attribute','subject','operator','expected')
            if r['id'] not in ids and not any(all(r[k]==a[k] for k in keys) for a in added):continue
            if r['importance'] not in ('required','preference'):continue
            quote=max(r['evidence'],key=lambda e:len(e['text']))['text']
            if d.get('numeric_source_correction') and r['id']=='r006':
                user_decisions.append({'id':d['id']+'-numeric-proof','kind':'verification_standard','standard':'explicit_source_numeric_correction',
                    'status':'confirmed_by_user','query_sha256':d['query_sha256'],'source_text':quote,
                    'predicate':{k:copy.deepcopy(r[k]) for k in keys},'basis':d['note']})
            user_decisions.append({'id':d['id']+'-'+r['id'],'kind':'requirement_strength',
                'status':'confirmed_by_user','query_sha256':d['query_sha256'],'source_text':quote,
                'predicate':{k:copy.deepcopy(r[k]) for k in keys},
                'importance':r['importance'],'basis':d['note']})
    (LEDGER.parent/'user_decisions.json').write_text(json.dumps({'decisions':user_decisions},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    for d in reviews:
        c=d.pop('_working')
        for rid in [r['id'] for _,r in m.allreq(c)]:
            r=next(r for _,r in m.allreq(c) if r['id']==rid)
            if r['importance']=='uncertain' or r['operator'] in ('approximately','matches_label'):continue
            v=p.approximate_target(r,m.b.chinese_number)
            if v is not None:patch(c,d,rid,operator='approximately',expected=v,review_status='approved')
    m.jl(LEDGER,reviews)
    print(json.dumps({'source_review_records':len(reviews),'source_tasks':len({d['task_id'] for d in reviews})}))

if __name__=='__main__':main()
