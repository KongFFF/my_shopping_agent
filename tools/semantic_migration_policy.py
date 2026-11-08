"""User-confirmed declaration standards, with source/strength gates intact."""
import json
import re
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'experiments/semantic_training_policy_migration_v1/policy.json'


def adjacent_tens_money(phrase):
    if not PATH.exists() or not json.loads(PATH.read_text(encoding='utf-8')).get('adjacent_tens_budget',{}).get('confirmed_by_user'):
        return None
    pattern=r'(?<![零〇一二两三四五六七八九十百千万])([一二两三四五六七八九])百([一二两三四五六七八九])([一二两三四五六七八九])十(?![零〇一二两三四五六七八九十百千万])'
    hits=list(re.finditer(pattern,phrase))
    if len(hits)!=1:return None
    digits={c:i for i,c in enumerate('零一二三四五六七八九')};digits['两']=2
    m=hits[0]
    if digits[m[3]]!=digits[m[2]]+1:return None
    amount=digits[m[1]]*100+digits[m[3]]*10
    return phrase[:m.start()]+str(amount)+phrase[m.end():]


def explicit_money_multiplier(phrase):
    # 4万元/1.3万元/3千元 are explicit multiplication, not an elision rule.
    # 1万3 remains untouched because it omits the final unit.
    scales={'百':100,'千':1000,'万':10000}
    def repl(m):
        from decimal import Decimal
        return format(Decimal(m[1])*scales[m[2]], 'f').rstrip('0').rstrip('.') if '.' in m[1] else str(int(m[1])*scales[m[2]])
    return re.sub(r'(?<![\d.])(\d+(?:\.\d+)?)(百|千|万)(?![\d零〇一二两三四五六七八九十百千万])',repl,phrase)


def round_tens_duo(phrase):
    if not PATH.exists() or not json.loads(PATH.read_text(encoding='utf-8')).get('round_tens_duo_budget',{}).get('confirmed_by_user'):
        return None
    m=re.fullmatch(r'[^\d零〇一二两三四五六七八九十百千万]{0,16}(\d+|[零〇一二两三四五六七八九十百千万]+)(?:元|块钱|块)?多(?:元|块钱|块)?(?:左右|上下)?[。.]?',phrase)
    if not m:return None
    from semantic_contract_v2_r7_final12 import chinese_number
    amount=chinese_number(m[1])
    if amount>=10 and amount%10==0 and amount%100!=0:
        return 'at_most',{'type':'number','value':float(amount+9),'unit':'CNY'},'user_confirmed_round_tens_duo_upper'
    return None


def elided_scale_money(phrase):
    # Same user-confirmed omitted-low-place rule, including numeric spellings.
    # A complete amount (三千五百), a serial number, or two suffix digits is not elision.
    pattern=r'(?<![\d零〇一二两三四五六七八九十百千万])([一二两三四五六七八九]|\d+)(百|千|万)([一二两三四五六七八九]|\d)(?![\d零〇一二两三四五六七八九十百千万])'
    hits=list(re.finditer(pattern,phrase))
    if len(hits)!=1:return None
    m=hits[0];digits={c:i for i,c in enumerate('零一二三四五六七八九')};digits['两']=2
    number=lambda s:int(s) if s.isdigit() else digits[s]
    scale={'百':100,'千':1000,'万':10000}[m[2]]
    amount=number(m[1])*scale+number(m[3])*(scale//10)
    return phrase[:m.start()]+str(amount)+phrase[m.end():]


def latest_declaration(r):
    if not PATH.exists() or not json.loads(PATH.read_text(encoding='utf-8'))['latest_declaration']['confirmed_by_user']:
        return False
    if r['review_status'] != 'approved' or r['importance'] == 'uncertain' or r['expected']['type'] != 'text':
        return False
    value = r['expected']['value']
    if not re.search(r'最新|新款|新出|新上市|新品', value):
        return False
    if re.search(r'今年|近期|当下|最近', value) and not re.search(r'20\d{2}', value):
        return False
    return any(value in e['text'] for e in r['evidence'])
