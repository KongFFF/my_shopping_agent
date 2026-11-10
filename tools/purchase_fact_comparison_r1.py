"""Offline comparison revision; preserves the originally frozen v2 compiler.

Check imitation/appearance modifiers around the REQUIRED material token, not
only around the model's proposed whole phrase. Multi-shade labels cannot prove
a single whole-product shade through a clipped substring.
"""
import copy
import re

import purchase_fact_proofs as base
from semantic_evidence_reward import score_training_verified

VERSION='purchase-fact-comparison-r1'


def comparison(r,facts):
    allowed=[]
    for f in facts:
        if r['subtype']=='material' and r['expected']['type']=='text' and f['kind']=='claim':
            full=base.text(f['evidence']['value']).casefold()
            unsafe=False
            for term in base.claim_terms(r):
                for match in re.finditer(re.escape(term),full):
                    before,after=base.norm(full[:match.start()]),base.norm(full[match.end():])
                    if re.search(r'(?:人造|仿(?:制|真)?|合成|pu|pvc|贴面?)$',before) or re.match(r'色|质感|效果|纹理|风格',after):
                        unsafe=True
            if unsafe:continue
        if r['subtype']=='color' and r['operator']=='eq' and f['kind']=='label':
            shades=base.selected_colors(base.text(f['evidence']['value']))
            if len(shades)>1 and base.norm(f['actual']['text'])!=base.norm(f['evidence']['value']):continue
        allowed.append(f)
    return base.compare_requirement(r,allowed)


def score(case,compiled):
    verification=copy.deepcopy(case['rule_verification'])
    for r in case['request']['requirements']:
        rid=r['id'];result=comparison(r,compiled['facts'])
        replace=(verification['statuses'][rid]=='unknown' or r['subtype']=='size' and result['status']!='unknown')
        if replace and result['status']!='unknown':
            verification['statuses'][rid]=result['status']
            verification['comparisons'][rid]={**verification['comparisons'][rid],**result}
    reward=score_training_verified(case['contract'],verification,source_query=case['contract']['query'])
    return reward,verification
