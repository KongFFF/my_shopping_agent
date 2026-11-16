"""Pure public-demand/state screening rules. No dataset loading or model calls."""
import json
import re
import unicodedata

def norm(text):
    return unicodedata.normalize('NFKC', text).casefold().strip()

def public_state(observation):
    """Parse only rendered Actor-visible lines, never private result state."""
    state = {}
    for line in (observation or '').splitlines():
        if ': ' not in line:
            continue
        key, value = line.split(': ', 1)
        if key in ('asin', 'title', 'price', 'page_type'):
            state[key] = value
        elif key in ('selected_options', 'available_options'):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    state[key] = parsed
            except json.JSONDecodeError:
                pass
    price = state.get('price', '')
    state['numeric_price'] = float(price) if re.fullmatch(r'\d+(?:\.\d+)?', price) else None
    return state

def requirements(contract):
    q = contract['query']
    result = []
    normalized = contract['normalized']
    for field in ('budget', 'specifications'):
        for r in normalized.get(field, []):
            evidence = r.get('evidence') or []
            if r.get('review_status') != 'approved' or r.get('importance') != 'required' or not evidence:
                continue
            if any(q[e['start']:e['end']] != e['text'] for e in evidence):
                raise ValueError('Requirement evidence offsets differ')
            expected = r.get('expected') or {}
            base = {'id': r['id'], 'evidence': evidence, 'operator': r['operator'], 'expected': expected}
            if field == 'budget' and expected.get('type') == 'number' and expected.get('unit') == 'CNY' and r['operator'] in ('at_most', 'less_than'):
                # Avoid approximate policies and deposit/discount/price-basis ambiguity.
                if re.search(r'左右|大概|约|尽量|最好|到手|券|优惠|定金|订金|预定|预订|尾款|全款|分期', q):
                    continue
                if not any(re.search(r'以内|以下|不超过|不得超|别超|不高于|上限|低于|小于|最多|至多|不多于|不能超', e['text']) for e in evidence):
                    continue
                result.append({**base, 'kind': 'explicit_budget'})
            elif field == 'specifications' and r.get('attribute') == 'mass' and expected.get('unit') in ('g', 'kg', 'jin') and r['operator'] == 'eq' and r.get('subject', {}).get('kind') == 'primary_product':
                result.append({**base, 'kind': 'explicit_mass'})
            elif field == 'specifications' and r.get('attribute') == 'color' and expected.get('value') in ('黑色', '白色', '红色', '绿色', '蓝色', '黄色', '紫色', '灰色') and r['operator'] == 'eq' and r.get('subject', {}).get('kind') == 'primary_product':
                result.append({**base, 'kind': 'explicit_color'})
    return result

def compare(state, req):
    expected = req['expected']
    selected = state.get('selected_options')
    if req['kind'] == 'explicit_budget':
        price = state.get('numeric_price')
        if price is None:
            return None
        limit = float(expected['value'])
        valid = price <= limit if req['operator'] == 'at_most' else price < limit
        return {'passed': valid, 'actual': price, 'expected': limit}
    if not isinstance(selected, dict) or not selected:
        return None
    if req['kind'] == 'explicit_mass':
        masses = []
        for value in selected.values():
            value = norm(str(value))
            if re.search(r'[*×]|\d\s*[xX]\s*\d|每|赠|送', value):
                return None
            for number, unit in re.findall(r'(\d+(?:\.\d+)?)\s*(kg|公斤|千克|斤|克|g)(?![a-z])', value):
                masses.append(float(number) * {'kg': 1000, '公斤': 1000, '千克': 1000, '斤': 500, '克': 1, 'g': 1}[unit])
        if len(masses) != 1:
            return None
        target = float(expected['value']) * {'kg': 1000, 'g': 1, 'jin': 500}[expected['unit']]
        return {'passed': abs(masses[0] - target) < 1e-6, 'actual_g': masses[0], 'expected_g': target}
    colors = set()
    for axis, value in selected.items():
        if '色' in axis:
            colors.update(re.findall(r'黑色|白色|红色|绿色|蓝色|黄色|紫色|灰色', str(value)))
    if len(colors) != 1:
        return None
    actual = next(iter(colors))
    return {'passed': actual == expected['value'], 'actual': actual, 'expected': expected['value']}
