"""Explicitly confirmed semantic policies, with source proofs, never task IDs."""
import re

NUM=r'(?:\d+|[零〇一二两三四五六七八九十百千万]+)'


def sale_basis(phrase,literal_parser):
    try:return literal_parser(phrase)
    except ValueError:
        # Do not hide an unsupported *explicit* denominator as a sale unit.
        if '单价' in phrase and not re.search(r'每|/|一(?:支|只|个|件|瓶|袋|盒|米|公斤|千克|克)',phrase):
            return {'amount':1.0,'unit':'sale_unit'}
        raise


def coordinated_pairs(r,parser):
    result=[]
    for e in r['evidence']:
        source=e['text']
        if re.search(r'但|仅|恰好|最多|不超过|不要|不能|最好|优先',source):continue
        match=re.search(r'(?:至少|不少于)('+NUM+r')(?:个|种|档)?([^，,。；;和与及]{1,24})(?:和|与|以及)('+NUM+r')(?:个|种|档)?([^，,。；;]{1,24})$',source,re.I)
        if match:result.extend([(float(parser(match[1])),match[2]),(float(parser(match[3])),match[4])])
    return result


def coordination_proof(r,parser):
    return (r['subtype']=='attribute_count' and r['operator']=='at_least' and r['expected']['type']=='number' and r['subject']['kind']=='component'
        and any(r['subject']['name'] in name and r['expected']['value']==n for n,name in coordinated_pairs(r,parser)))


def checked_flags(contract,flags,parser):
    kept=[]
    for flag in flags:
        parts=flag.get('location','').strip('/').split('/')
        r=None
        if len(parts)==2 and parts[0] in contract and parts[1].isdigit():r=contract[parts[0]][int(parts[1])]
        if flag['code']=='bare_feature_exact_count' and r and coordination_proof(r,parser):continue
        kept.append(flag)
    for d in ('category','budget','functions','specifications','brand','model'):
        for i,r in enumerate(contract[d]):
            value=r['expected']
            if r['subtype']=='attribute_count' and r['subject']['kind']=='component':
                matched=[n for n,name in coordinated_pairs(r,parser) if r['subject']['name'] in name]
                if matched and not coordination_proof(r,parser):
                    kept.append({'code':'confirmed_coordinated_lower_bound','location':f'/{d}/{i}',
                                 'detail':'A confirmed shared at-least cue must retain its own count and lower-bound operator'})
            if r['subtype']=='quantity' and value['type']=='range':
                source='，'.join(e['text'] for e in r['evidence'])
                pair=re.search('('+NUM+r')、('+NUM+r')(片|个|件|支)',source)
                if pair:
                    a,z=sorted([int(parser(pair[1])),int(parser(pair[2]))])
                    if z-a!=1 or value['lower']!=a or value['upper']!=z or not (value['lower_inclusive'] and value['upper_inclusive']):
                        kept.append({'code':'confirmed_adjacent_quantity_pair','location':f'/{d}/{i}',
                            'detail':'Adjacent stated counts preserve both endpoints; a nonadjacent list is not a continuous range'})
            if value['type']!='member_ranges':continue
            quotes='，'.join(e['text'] for e in r['evidence'])
            bases=[int(parser(n)) for n in re.findall('('+NUM+r')多段',quotes)]
            expected=[{'lower':n+1,'upper':n+9,'count':1} for n in bases if n%10==0]
            if not bases or len(expected)!=len(bases) or value['groups']!=expected:
                kept.append({'code':'member_range_source','location':f'/{d}/{i}',
                             'detail':'Member intervals must follow confirmed whole-tens-duo segment policy and preserve one member per source group'})
            total=re.search(r'包含('+NUM+r')个单品',quotes)
            if total and int(parser(total[1]))!=sum(g['count'] for g in value['groups']):
                kept.append({'code':'member_total_source','location':f'/{d}/{i}', 'detail':'Member group cardinality differs from the explicitly stated total'})
    return kept
