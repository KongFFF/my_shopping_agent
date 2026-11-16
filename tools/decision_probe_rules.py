"""Conservative public triggers/atomic diagnostics; never read goal or reward."""
import json
import re
import unicodedata


def public_state(text):
    state = {}
    for line in text.splitlines():
        if ': ' not in line:
            continue
        key, value = line.split(': ', 1)
        if key in ('page_type', 'asin', 'price', 'title'):
            state[key] = value
        elif key in ('available_options', 'selected_options'):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict): state[key] = parsed
            except ValueError:
                pass
    return state


def requirements(query):
    q = unicodedata.normalize('NFKC', query)
    result = []
    if not re.search(r'左右|大概|约|尽量|最好|到手|券|优惠|定金|订金|预定|尾款|全款|分期', q):
        patterns = [r'(?:不超过|不得超过|不能超过|不高于|最多|至多|低于|小于)\s*(\d+(?:\.\d+)?)\s*元',
                    r'(\d+(?:\.\d+)?)\s*元\s*(?:以内|以下)']
        hits = [m for p in patterns for m in re.finditer(p, q)]
        if len(hits) == 1:
            m = hits[0]
            result.append({'kind': 'budget', 'value': float(m.group(1)), 'quote': m.group(),
                           'strict': bool(re.search(r'低于|小于', m.group()))})
    colors = set(re.findall(r'黑色|白色|红色|绿色|蓝色|黄色|紫色|灰色', q))
    if len(colors) == 1:
        # Only a lexical cue until manual review resolves subject/negation.
        result.append({'kind': 'color_cue', 'value': next(iter(colors))})
    return result


def trigger(query, visible, tool, args):
    """Inspect the current projected page and proposed action, before execution."""
    state = public_state(visible)
    if state.get('page_type') != 'product_detail': return None
    axes = {k: v for k, v in state.get('available_options', {}).items()
            if isinstance(v, list) and len(set(map(str, v))) >= 2}
    if not axes: return None
    reqs = requirements(query)
    if tool == 'select_option':
        # A visible value literally present in the query, or a clear budget,
        # identifies a prospective SKU decision without deciding its answer.
        related = any(r['kind'] == 'budget' for r in reqs) or any(
            len(str(v)) >= 2 and str(v) in query for values in axes.values() for v in values)
        if related:
            return {'kind': 'related_spec_before_selection', 'axes': list(axes), 'requirements': reqs}
    if tool == 'buy_now':
        return {'kind': 'prebuy_empty' if not state.get('selected_options') else 'prebuy_selected',
                'axes': list(axes), 'requirements': reqs}
    return None


def atomic_checks(query, visible):
    state = public_state(visible)
    result = []
    for req in requirements(query):
        if req['kind'] != 'budget': continue
        price = str(state.get('price', ''))
        known = bool(re.fullmatch(r'\d+(?:\.\d+)?', price))
        passed = None
        if known:
            passed = float(price) < req['value'] if req['strict'] else float(price) <= req['value']
        result.append({**req, 'actual': float(price) if known else None, 'passed': passed,
                       'scope': 'displayed_price_only_manual_review_required'})
    return result
