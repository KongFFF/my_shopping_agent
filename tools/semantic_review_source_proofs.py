"""Source proofs for explicitly confirmed review policies, never task-ID rules.

These justify already asserted counts. They neither create counts nor approve a
whole contract, and never suppress unrelated source/schema/quantity failures.
"""
import re
from semantic_coverage_v7 import NUM, COUNTERS, UNITS

DIMS=('category','budget','functions','specifications','brand','model')


def item_proof(c,r,item,parser):
    name=re.escape(item['name'])
    for e in r['evidence']:
        for m in re.finditer(r'(?:加|配|有)个(?P<qual>[^，,。；;、：:！？!?\n和与及]{0,16}?)'+name,e['text']):
            # A friend needing a box is not one box. Only a direct head noun
            # or an explicitly adjectival 的 phrase is licensed automatically.
            if m['qual'] and not m['qual'].endswith('的'):continue
            pos=e['start']+m.start()
            if re.search(r'不想|不要|不能|不再|无需|不用|禁止|别',c['query'][max(0,pos-5):pos]):continue
            if item['quantity']==1 and item['unit']=='piece':
                return {'kind':'user_confirmed_quantity_source_proof','rule':'elided_one_clear_object',
                        'object':item['name'],'quantity':1,'source':{'text':m.group(),'start':pos,'end':e['start']+m.end()},
                        'whole_contract_approved':False}
        for m in re.finditer(r'('+NUM+r')('+COUNTERS+r')[^，,。；;、！？!?\n和与及]{0,16}?'+name,e['text']):
            if item['unit']==UNITS[m[2]] and item['quantity']==float(parser(m[1])):
                return {'kind':'literal_quantity_source_proof','rule':'explicit_count_same_item',
                        'object':item['name'],'quantity':item['quantity'],'source':{'text':m.group(),
                        'start':e['start']+m.start(),'end':e['start']+m.end()},'whole_contract_approved':False}
    return None


def certificates(c,parser):
    result=[]
    for dim in DIMS:
        for i,r in enumerate(c[dim]):
            if r['expected']['type']!='components' or r['operator']=='excludes':continue
            needed=[item for item in r['expected']['items'] if item['quantity'] is not None]
            if not needed:continue
            proofs=[item_proof(c,r,item,parser) for item in needed]
            if all(proofs) and any(p['rule']=='elided_one_clear_object' for p in proofs):
                for p in proofs:result.append(dict(p,location=f'/{dim}/{i}'))
    return result


def filter_flags(c,flags,parser):
    proven={p['location'] for p in certificates(c,parser)}
    count_codes={'invented_component_count','component_count_source_unproven','unresolved_count_interpretation'}
    return [f for f in flags if not (f['code'] in count_codes and f.get('location') in proven)]
