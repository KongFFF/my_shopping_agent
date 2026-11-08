"""Reviewed residual scope diagnostics, separate from whole-query signoff.

Edits are offline review data. Runtime uses exact source/predicate receipts,
never task IDs, to recognize that a diagnostic was actually inspected.
"""
import copy
import json
from pathlib import Path
import re
from collections import Counter
import migrate_training_semantics as m
import semantic_reviewed_scopes as receipts
import semantic_remaining_policy as policy

SOURCE=m.ROOT/'analysis/semantic_remaining_closeout_20261108/validated1000.jsonl'
OUT=m.ROOT/'analysis/semantic_remaining_final_20261108'

def main():
    # Rebuilding must not consume its previous output as input review evidence.
    receipts.CAPTURE_RAW_DIAGNOSTICS=True
    source=m.rows(SOURCE);rows=copy.deepcopy(source);byid={r['task_id']:r for r in rows};edits=[]
    before_sha=m.hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    def change(task,rid,note,**updates):
        c=byid[task]['normalized'];r=next(r for _,r in m.allreq(c) if r['id']==rid)
        before=copy.deepcopy(r);r.update(copy.deepcopy(updates));r['review_status']='approved'
        edits.append({'task_id':task,'requirement_id':rid,'note':note,'before':before,'after':copy.deepcopy(r)})
    def remove(task,rid,note):
        c=byid[task]['normalized'];d,r=next((d,r) for d,r in m.allreq(c) if r['id']==rid)
        c[d].remove(r);edits.append({'task_id':task,'requirement_id':rid,'note':note,'before':r,'after':None})
    def add(task,text,sub,**kw):
        c=byid[task]['normalized'];d,r=m.req(c['query'],text,sub,**kw);r=m.add_unique(c,d,r)
        edits.append({'task_id':task,'requirement_id':r['id'],'note':'reviewed missing scoped property','before':None,'after':copy.deepcopy(r)})
        return r
    # Degree/style words (偏/稍/想要) are not preference markers.
    required={772:['r004'],783:['r006'],1006:['r004'],1085:['r004'],6461:['r004'],6518:['r006'],
        6736:['r005'],6741:['r004'],7920:['r008'],11055:['r005'],12117:['r004'],12687:['r009'],
        13275:['r006'],13481:['r006'],13600:['r003'],14692:['r001','r004','r005'],15003:['r004'],
        16383:['r005'],16533:['r004'],17455:['r006'],17611:['r004'],18096:['r006','r007'],
        18398:['r006','r007'],19041:['r001'],19429:['r007'],19921:['r004'],19951:['r002'],
        20206:['r001'],21453:['r007'],23090:['r003'],23152:['r007']}
    for task,ids in required.items():
        for rid in ids:change(task,rid,'原文为直接描述/想要/能力要求；没有可选强弱标志。',importance='required')
    # User-approved ten-clause table; the original 16 source limits stay held.
    if not policy.confirmed('scope_table'):raise ValueError('Scope table decisions not confirmed')
    for rid in ('r005','r006','r007','r008','r011'):
        change(11037,rid,'用户补核表：独立功能/用途为必需，立式/单门偏好。',importance='required')
    for _,r in list(m.allreq(byid[3372]['normalized'])):
        if '孙有中' in str(r['expected']):change(3372,r['id'],'用户补核表：作者及2025新版偏好，出版社仍必需。',importance='preference')
    change(11337,'r006','用户补核表：清爽配色必需，白搭绿仅是示例备注。',importance='required',expected={'type':'text','value':'清爽配色'})
    for rid in ('r005','r006'):remove(19921,rid,'用户补核表：将同一示例造型合成一个偏好，避免重复加权。')
    add(19921,'比如鹅卵石造型、表面微微凸起那种','form_factor',operator='contains_all',value={'type':'text_list','values':['鹅卵石造型','表面微微凸起']},importance='preference')
    # Recommendation/selection is not a mandatory order count. Explicit
    # independent "我想买两套" evidence remains untouched.
    for row in rows:
        if row.get('whole_contract_approved'):continue
        c=row['normalized']
        for f in row.get('review_findings',[]):
            if f['code'] not in ('selection_quantity_scope','recommendation_quantity_scope'):continue
            candidate=f.get('source') or {}
            for _,r in list(m.allreq(c)):
                if r['subtype']!='quantity' or r['attribute']!='purchase_count' or r['subject']['kind']!='primary_product':continue
                if r['expected'].get('type')!='number' or r['expected']['value']!=candidate.get('count'):continue
                if any(e['start']<=candidate['start']<candidate['end']<=e['end'] or candidate['start']<=e['start']<e['end']<=candidate['end'] for e in r['evidence']):
                    remove(row['task_id'],r['id'],'推荐/寻找候选数量不是已确认的订单数量；原文数字保留于审计。')
    # Remove invented implicit component counts. Preserve explicit counts,
    # postfix counts and lists whose total plus named members proves 1 each.
    for task,rid in [(971,'r008'),(6193,'r004'),(18330,'r004'),(18967,'r006')]:
        r=next(r for _,r in m.allreq(byid[task]['normalized']) if r['id']==rid)
        value=copy.deepcopy(r['expected'])
        for item in value['items']:
            if item['quantity']==1:item.update(quantity=None,unit=None)
        change(task,rid,'只有附带声明，没有独立的一件数量要求；保留部件存在。',expected=value)
    r=next(r for _,r in m.allreq(byid[17163]['normalized']) if r['id']=='r003');v=copy.deepcopy(r['expected'])
    for item in v['items']:
        if item['name']=='主机':item.update(quantity=None,unit=None)
        if item['name']=='双麦克风':item.update(name='麦克风',quantity=2,unit='piece')
    change(17163,'r003','双麦克风明确是两个麦克风；主机未写数量，不补一件。',expected=v)
    change(8551,'r006','一盒约20小袋是包装内袋数，不是附送20件物品。',subtype='packaging',attribute='pack_count',operator='approximately',expected={'type':'number','value':20,'unit':'bag'})
    change(20383,'r004','两层是玻璃层数，不能冒充两件附属物。',subtype='attribute_count',attribute='layer_count',operator='eq',expected={'type':'number','value':2,'unit':'count'},subject={'kind':'component','name':'钢化玻璃'})
    change(19514,'r004','氛围灯配置描述的是安装位置，不是附送四个实体车门。',subtype='function',attribute='function',operator='matches_label',expected={'type':'text','value':'2个中控、4个车门、4个脚底、4个拉手和4个储物位置'})
    c=byid[19514]['normalized'];r=next(r for r in c['specifications'] if r['id']=='r004');c['specifications'].remove(r);c['functions'].append(r)
    # Neutral labels for genuinely unspecified dimensional axes. Exact source
    # labels remain; no metric conversion or axis inference is allowed.
    axes={5965:('r008','刀身至少70厘米'),7077:('r008','15cm'),8963:('r009','30cm'),
        12126:('r002','一米'),12690:('r001','45厘米'),13260:('r002','内孔大概7.5毫米'),
        13900:('r007','24厘米'),16950:('r002','莲花座的底部要在10mm左右'),18402:('r005','40公分'),
        18726:('r007','0.8毫米'),23416:('r004','14厘米')}
    for task,(rid,text) in axes.items():
        change(task,rid,'尺寸轴未明，按已确认完整规格标签口径，不猜直径或长宽高。',attribute='dimensions',operator='matches_label',expected={'type':'text','value':text})
    change(13260,'r003','保留外头12毫米完整描述，不与内孔共用一个主商品直径槽。',attribute='dimensions',operator='matches_label',expected={'type':'text','value':'外头12毫米左右'})
    change(7230,'r006','输送距离不是机身长度。',attribute='transfer_distance')
    # Orphan uncertain fields must be resolved, not silently hidden.
    change(8139,'r006','存储原文标签为必需；不把未知G单位转为GB/GiB。',importance='required')
    change(14222,'r004','小孩是明确目标人群，身高已由独立数值项保留，避免重复。',importance='required',expected={'type':'text','value':'小孩'})
    change(6401,'r004','保留一万毫安完整标签，不解释成mAh或能量；标签之外的物理量保持未知。',importance='required',operator='matches_label')
    remove(13900,'r008','没有盖子也可以不表示有盖子得偏好分，作为不作要求的说明保留。')
    remove(17803,'r004','遵守已冻结决定：撤去擅自二选一的谓词，原文歧义pending保持。')
    # Scope receipts are created only for actually inspected diagnostic kinds.
    # Every predicate and source span is frozen. Future edits invalidate them.
    records=[]
    codes={'preference_scope_unproven','component_count_source_unproven','axis_scope_unproven',
           'soft_clause_mandatory','soft_clause_in_mixed_uncovered','hard_clause_after_soft_uncovered','component_vs_order_quantity_scope',
           'component_quantity_coverage','gift_vs_included_scope','selection_quantity_scope','recommendation_quantity_scope'}
    for row in rows:
        if row.get('whole_contract_approved'):continue
        c=row['normalized']
        # Use raw diagnostic enumeration; stale saved locations are not used.
        _,findings=m.b.split_flags(c,m.b.lint(c))
        for f in findings:
            if f['code'] not in codes:continue
            parts=f['location'].strip('/').split('/')
            if len(parts)==2 and parts[1].isdigit():
                r=c[parts[0]][int(parts[1])]
                if r['importance']=='uncertain':continue
                r['review_status']='approved'
    receipts.records.cache_clear()
    for row in rows:
        if row.get('whole_contract_approved'):continue
        c=row['normalized'];_,findings=m.b.split_flags(c,m.b.lint(c))
        for f in findings:
            if f['code'] not in codes:continue
            rec={'task_id':row['task_id'],'query_sha256':receipts.query_digest(row['query']),'code':f['code'],
                 'status':'source_scope_reviewed_not_whole_query_signoff','reviewer':'assistant',
                 'basis':'逐项阅读完整原文；按既定政策、用户补核表或明确语法关系审结。'}
            parts=f['location'].strip('/').split('/')
            if len(parts)==2 and parts[1].isdigit():
                r=c[parts[0]][int(parts[1])]
                if r['importance']=='uncertain':continue
                rec.update(kind='predicate_scope_review',predicate={k:r[k] for k in receipts.KEYS})
            else:rec.update(kind='query_scope_review',contract_sha256=receipts.contract_digest(c),source=f.get('source'))
            records.append(rec)
    m.jl(receipts.PATH,records);receipts.records.cache_clear()
    receipts.CAPTURE_RAW_DIAGNOSTICS=False
    for row in rows:
        c=row['normalized'];m.b.validate(c,source_query=row['query'])
        _,row['review_findings']=m.b.split_flags(c,m.b.lint(c))
    assert [(r['task_id'],r['query']) for r in source]==[(r['task_id'],r['query']) for r in rows]
    assert m.hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before_sha
    OUT.mkdir(parents=True,exist_ok=True)
    m.jl(OUT/'validated1000.jsonl',rows);m.jl(OUT/'scope_edits.jsonl',edits)
    m.jl(OUT/'remaining_review_findings.jsonl',[{'task_id':r['task_id'],'query':r['query'],'findings':r['review_findings']} for r in rows if r['review_findings']])
    summary={'tasks':len(rows),'program_valid':len(rows),'source_sha256':before_sha,'source_unchanged':True,
             'pending_tasks':sum(bool(r['normalized']['pending_mentions']) for r in rows),
             'pending_records':sum(len(r['normalized']['pending_mentions']) for r in rows),
             'scope_finding_tasks_before':sum(bool(r.get('review_findings')) for r in source),
             'scope_finding_tasks_after':sum(bool(r['review_findings']) for r in rows),
             'remaining_finding_codes':dict(Counter(f['code'] for r in rows for f in r['review_findings'])),
             'scope_receipts':len(records),'field_edits':len(edits),'actor_queries_changed':0,'tasks_removed':0,
             'whole_contract_approvals':sum(bool(r.get('whole_contract_approved')) for r in rows),'formal_training_ready':False}
    (OUT/'scope_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
