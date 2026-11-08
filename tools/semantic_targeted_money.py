"""Money-only spans, with surrounding denominators and limits preserved."""
import re
from semantic_policy_v4 import money_only
from semantic_acceptance_fixes import money_context as prior_context
NUM=r'(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)'
UNCLEAR=r'十几|几十|几百|几千|三四|二三|两三|七八|八九|十来|百来'

def money_context(query,phrase,legacy):
    if query.count(phrase)!=1:raise ValueError('Money phrase needs unique source')
    try:context=prior_context(query,phrase,legacy)
    except ValueError:context=phrase
    simple=money_only(phrase.replace('别超了','不超过'))
    if simple and not re.search(UNCLEAR,phrase):
        # Never clip a real denominator or explicit monetary qualifier.
        outside=context.replace(phrase,'',1)
        if re.search(r'每|单价|运费|定金|尾款|不超过|上限|最多|别超',outside):return context
        return phrase
    return context

def bounded_money(phrase,parser,fallback):
    if re.search(UNCLEAR,phrase):raise ValueError('Colloquial budget interval has no confirmed policy')
    open_both=re.fullmatch(r'(?:[^\d零〇一二两三四五六七八九十百千万]{0,12})?(?:高于|大于|超过)('+NUM+r')(?:元|块钱|块)?[，,、 ]*(?:并且|且)?(?:不超|不超过)('+NUM+r')(?:元|块钱|块)?[。.]?',phrase)
    if open_both:
        a,z=map(lambda n:float(parser(n)),open_both.groups())
        if a>=z:raise ValueError('Empty or contradictory budget limits')
        return 'between',{'type':'range','lower':a,'upper':z,'unit':'CNY','lower_inclusive':False,'upper_inclusive':True},'explicit_open_lower_closed_upper'
    m=re.fullmatch(r'(?:价格|预算|售价)?(?:在)?('+NUM+r')(?:元|块钱|块)?以上[，,、 ]*(?:并且|且)?不超过('+NUM+r')(?:元|块钱|块)?[。.]?',phrase)
    if m:
        a,z=map(lambda n:float(parser(n)),m.groups())
        if a>z:raise ValueError('Contradictory budget limits')
        return 'between',{'type':'range','lower':a,'upper':z,'unit':'CNY','lower_inclusive':True,'upper_inclusive':True},'explicit_both_limits'
    return fallback(phrase)
