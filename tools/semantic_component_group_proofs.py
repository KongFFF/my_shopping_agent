"""One reviewed component-group requirement; no duplicate reward objectives.

Only a literal N套 + complete ordered member names + 齐全 construction is
accepted. No numbers inside member names, optional items or separate counts.
"""
import re


def group_proof(r,parser):
    v=r['expected']
    if r['subtype']!='components' or r['operator']!='contains_all' or v['type']!='components':return None
    items=v['items'];names=[i['name'] for i in items]
    if not 2<=len(items)<=6 or len(set(names))!=len(names):return None
    if any(re.search(r'[\d一二两三四五六七八九十百千万或与和、，,]',n) for n in names):return None
    if any(i['unit']!='piece' or i['quantity'] is None for i in items):return None
    pattern=r'([\d零〇一二两三四五六七八九十百千万]+)套'+re.escape(''.join(names))+'齐全'
    for e in r['evidence']:
        m=re.fullmatch(pattern,e['text'])
        if m and all(i['quantity']==int(parser(m[1])) for i in items):
            return {'kind':'source_scope_proof','rule':'reviewed_complete_component_sets','source':e,
                    'sets':int(parser(m[1])),'members':names,'one_reward_requirement':True}
    return None
