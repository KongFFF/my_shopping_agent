"""Audited source-bound representation repair; never approves or invents facts."""
import copy
import re
from semantic_coverage_v6 import explicit_counts, NUM, COUNTERS
DIMS=('category','budget','functions','specifications','brand','model')


def hits(query,text):return [i for i in range(len(query)) if query.startswith(text,i)] if text else []
def clause(query,pos):
    return next((m for m in re.finditer(r'[^，,。；;！？!?\n]+',query) if m.start()<=pos<m.end()),None)


def preprocess(raw,task,number_parser):
    c=copy.deepcopy(raw);audit=[];q=task['query']
    if c.get('query')!=q or c.get('task_id')!=task['task_id']:raise ValueError('Original task ID/query changed')
    candidates=explicit_counts(q,number_parser)
    for d in DIMS:
        for i,r in enumerate(c.get(d,[])):
            path=f'/{d}/{i}';ev=r.get('evidence',[]);sub=r.get('subject',{})
            # Resolve a repeated quote only when existing object evidence or an
            # explicit order-count span identifies exactly one occurrence.
            for j,e in enumerate(ev):
                occurrences=hits(q,e['text'])
                if len(occurrences)<=1:continue
                eligible=[]
                for pos in occurrences:
                    cl=clause(q,pos)
                    if cl is None:continue
                    object_anchor=(sub.get('kind') in ('component','gift','compatible_target','intended_user') and sub.get('name') in cl.group())
                    other_anchor=any(x is not e and len(hits(q,x['text']))==1 and cl.start()<=q.index(x['text'])<cl.end() for x in ev)
                    v=r.get('expected',{})
                    count_anchor=(r.get('subtype')=='quantity' and v.get('type')=='number' and any(x['role']=='purchase' and not x['ambiguous'] and x['count']==v['value'] and x['unit']==v['unit'] and pos<=x['start'] and x['end']<=pos+len(e['text']) for x in candidates))
                    if object_anchor or other_anchor or count_anchor:eligible.append(cl)
                if len(eligible)==1 and len(hits(q,eligible[0].group()))==1:
                    before=copy.deepcopy(e);e['text']=eligible[0].group()
                    audit.append({'kind':'evidence_unique_context','path':f'{path}/evidence/{j}','before':before,'after':copy.deepcopy(e)})
            # Add a missing literal object only from the SAME uniquely anchored
            # clause. Never borrow a similarly named object elsewhere in query.
            if sub.get('kind') in ('component','gift','compatible_target','intended_user') and not any(sub.get('name','') in e['text'] for e in ev):
                anchors=[]
                for e in ev:
                    positions=hits(q,e['text'])
                    if len(positions)==1:
                        cl=clause(q,positions[0])
                        if cl and sub['name'] in cl.group() and cl.group().count(sub['name'])==1:anchors.append(cl.group())
                if len(set(anchors))==1:
                    e={'text':anchors[0]}
                    if e not in ev:
                        ev.append(e);audit.append({'kind':'subject_same_clause_evidence','path':path+'/evidence','after':copy.deepcopy(e)})
            v=r.get('expected',{})
            source='，'.join(e['text'] for e in ev)
            bare_cun=bool(re.search(r'(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)\s*寸',source)) and not re.search(r'英寸|inch|[″"]',source,re.I)
            bound_cun=(r.get('subtype') in ('dimensions','size') and ((v.get('type') in ('number','range') and v.get('unit')=='inch') or (v.get('type')=='text' and '寸' in v.get('value',''))))
            if bare_cun and bound_cun:
                before=copy.deepcopy(v);r['operator']='eq';r['expected']={'type':'text','value':source}
                for e in ev:
                    if re.search(r'(?<!英)寸',e['text']) and len(hits(q,e['text']))==1:
                        mention={'evidence':copy.deepcopy(e),'reason':'unsupported_unit','importance':r['importance']}
                        if mention not in c['pending_mentions']:c['pending_mentions'].append(mention)
                audit.append({'kind':'confirmed_bare_cun_pending','path':path,'before':before,'after':copy.deepcopy(r['expected'])})
            if bare_cun and not bound_cun:
                # A dimension embedded in a component's literal name remains
                # structured; record its unresolved unit without flattening it.
                for e in ev:
                    if re.search(r'(?:\d+(?:\.\d+)?|[一二两三四五六七八九十百千万]+)\s*寸',e['text']) and len(hits(q,e['text']))==1:
                        mention={'evidence':copy.deepcopy(e),'reason':'unsupported_unit','importance':r['importance']}
                        if mention not in c['pending_mentions']:
                            c['pending_mentions'].append(mention)
                            audit.append({'kind':'embedded_bare_cun_pending','path':path,'source':copy.deepcopy(e)})
            # A schema lacking storage/% units must retain source text rather
            # than relabel a physical amount as a dimensionless count.
            v=r.get('expected',{})
            if v.get('type')=='number' and v.get('unit')=='count':
                source='，'.join(e['text'] for e in ev)
                literal=re.escape(str(v['value']))
                if isinstance(v['value'],(int,float)) and float(v['value']).is_integer():literal=rf'{int(v["value"])}(?:\.0+)?'
                percent=bool(re.search(rf'(?<![\d.]){literal}\s*[%％]',source))
                storage=r.get('attribute')=='storage_capacity' and bool(re.search(rf'(?<![\d.]){literal}\s*(?:[KMGT](?:i?[Bb])?)(?![A-Za-z])',source,re.I))
                if percent or storage:
                    before=copy.deepcopy(v);r['operator']='eq';r['expected']={'type':'text','value':source}
                    # Keep requirement strength, and explicitly block scoring
                    # until its unsupported unit is reviewed. No GB/Gbit guess.
                    for e in ev:
                        if len(hits(q,e['text']))==1:
                            mention={'evidence':copy.deepcopy(e),'reason':'unsupported_unit','importance':r['importance']}
                            if mention not in c['pending_mentions']:c['pending_mentions'].append(mention)
                    audit.append({'kind':'unsupported_unit_preserve_source','path':path,'before':before,'after':copy.deepcopy(r['expected'])})
    return c,audit


def extra_flags(c):
    flags=[]
    for d in DIMS:
        for i,r in enumerate(c[d]):
            v=r['expected'];source='，'.join(e['text'] for e in r['evidence'])
            if r['attribute']=='storage_capacity' and v['type'] in ('number','range') and v['unit']=='count':
                flags.append({'code':'storage_as_count','location':f'/{d}/{i}','detail':'Storage amount is not count; preserve exact source and unresolved unit without guessing GB/Gbit'})
            if r['subtype']=='components' and v['type']=='components':
                # Numeric dimensions elsewhere are not proof of accessory counts.
                from semantic_coverage_v6 import PART
                for item in v['items']:
                    if item['quantity'] is None:continue
                    from semantic_contract_v2_r6 import chinese_number
                    pattern=rf'(?P<n>{NUM})(?:{COUNTERS})\s*(?P<name>[^，,。；;、]{{1,24}})'
                    justified=any(item['name'] in m['name'] and not re.search(r'和|以及|及|与|并',m['name'].split(item['name'],1)[0]) and item['quantity']==float(chinese_number(m['n'])) for m in re.finditer(pattern,source))
                    if not justified:
                        flags.append({'code':'component_count_source_unproven','location':f'/{d}/{i}',
                                      'detail':'A number in a dimension/other field does not establish this accessory quantity; retain null unless a source count is bound to this item'})
                        break
    return flags
