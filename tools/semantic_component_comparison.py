"""Join independently evidenced components without summing repeated claims."""
from semantic_fact_language_v3 import base


def compare_components(r,facts):
    if r['expected']['type']!='components' or r['operator'] not in ('contains_all','excludes'):return None
    records=[f for f in facts if f['value']['type']=='components']
    if not records:return None
    positive={};negative=set();closed=[]
    for f in records:
        items=f['value']['items']
        if f.get('polarity','positive')=='negative':negative.update(base(i['name']) for i in items);continue
        for i in items:positive.setdefault(base(i['name']),[]).append(i)
        if f['complete']:closed.append({base(i['name']) for i in items})
    details=[]
    for want in r['expected']['items']:
        name=base(want['name']);got=positive.get(name,[])
        if got and name in negative:status='unknown'
        elif r['operator']=='excludes':
            status='fail' if got else 'pass' if name in negative or any(name not in names for names in closed) else 'unknown'
        elif name in negative:status='fail'
        elif not got:status='fail' if closed else 'unknown'
        elif any(name not in names for names in closed):status='unknown'
        elif want['quantity'] is None:status='pass'
        else:
            amounts={i['quantity'] for i in got if i['quantity'] is not None and i['unit']==want['unit']}
            status='unknown' if len(amounts)!=1 else 'pass' if next(iter(amounts))==want['quantity'] else 'fail'
        details.append({'component':want['name'],'status':status})
    status='fail' if any(d['status']=='fail' for d in details) else 'unknown' if any(d['status']=='unknown' for d in details) else 'pass'
    return {'status':status,'reason':'joined_component_evidence','component_comparisons':details,
            'evidence':[{'actual':f['value'],'evidence':f['evidence']} for f in records]}
