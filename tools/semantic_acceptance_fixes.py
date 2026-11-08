"""Two source-bound acceptance fixes; no task-ID branches or new reward rules."""
import re
from semantic_coverage_v7 import NUM


def money_context(query,phrase,legacy):
    context=legacy(query,phrase)
    if context.count(phrase)!=1:return context
    pos=context.index(phrase)
    boundaries=[m.end() for m in re.finditer(r'(?:并且|而且|同时|且)(?=总价|预算|价格|售价|单价|价位)',context) if m.end()<=pos]
    return context[boundaries[-1]:] if boundaries else context


def filter_structure_flags(c,flags,parser):
    kept=[]
    for f in flags:
        source=f.get('source',{})
        structural=False
        if f['code']=='purchase_quantity_coverage' and re.match(r'^(?:要|需要)',source.get('source','')):
            for r in c['specifications']:
                if r['subtype']!='form_factor' or r['expected']['type']!='text':continue
                value=r['expected']['value']
                literal=[m for m in re.finditer(r'('+NUM+r')(?:个|根|只|片)?[^，,。；;]{1,10}(?:设计|结构|构型)',value)]
                same=any(float(parser(m[1]))==source['count'] for m in literal)
                covered=any(e['start']<=source['start']<source['end']<=e['end'] and value in e['text'] for e in r['evidence'])
                structural|=same and covered
        if not structural:kept.append(f)
    return kept
