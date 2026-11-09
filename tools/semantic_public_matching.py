"""Conservative public-text matching for typed shopping requirements.

Uses only purchased SKU labels and public catalog fields. No task IDs, target
answers, reviews, fuzzy matching or model calls. Missing evidence stays unknown.
"""
from __future__ import annotations

from decimal import Decimal
import re
import unicodedata

PRIMARY = {'kind': 'primary_product', 'name': '主商品'}
COUNT_UNITS = {'片': 'piece', '个': 'piece', '件': 'piece', '支': 'piece', '只': 'piece',
               '罐': 'piece', '桶': 'piece', '瓶': 'bottle', '袋': 'bag', '包': 'bag',
               '盒': 'box', '套': 'set', '双': 'pair', '本': 'book'}
COLORS = ('粉红色', '天蓝色', '浅蓝色', '深蓝色', '浅绿色', '深绿色', '墨绿色',
          '浅灰色', '深灰色', '暗蓝色', '米白色', '象牙白', '象牙色', '自然色',
          '森林绿', '古铜色', '玫瑰金', '香槟色', '橙色', '桔色', '黑色', '白色',
          '红色', '蓝色', '绿色', '黄色', '金色', '银色', '粉色', '灰色', '紫色', '棕色')
NEGATION = re.compile(r'不(?:支持|含|带|赠|适用|具备|是|能|可)?|无|非|未|缺少')


def compact(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(text))).casefold()


def selected_colors(text):
    """Preserve the longest explicit shade; never turn 粉红色 into 红色."""
    hits = []
    for color in sorted(COLORS, key=len, reverse=True):
        for match in re.finditer(re.escape(color), text):
            if any(a <= match.start() and match.end() <= z for a, z, _ in hits):
                continue
            prefix = text[max(0, match.start()-5):match.start()]
            if NEGATION.search(prefix):
                continue
            # Unknown compounds such as 孔雀蓝色 must not become plain 蓝色.
            if len(color) == 2 and prefix and re.search(r'[\u4e00-\u9fff]$', prefix):
                continue
            hits.append((match.start(), match.end(), color))
    return sorted({c for _, _, c in hits})


def ordered_dimensions(text):
    """Explicit metric units, original order, no axis assignment or bare 寸."""
    text = compact(text).replace('×', '*').replace('x', '*')
    parts = text.split('*')
    if len(parts) not in (2, 3):
        return None
    units = {'mm': Decimal(1), '毫米': Decimal(1), 'cm': Decimal(10),
             '厘米': Decimal(10), '公分': Decimal(10), 'm': Decimal(1000), '米': Decimal(1000)}
    parsed = [re.fullmatch(r'(\d+(?:\.\d+)?)(mm|cm|m|毫米|厘米|公分|米)?', part) for part in parts]
    if not all(parsed) or not parsed[-1][2]:
        return None
    # A single trailing unit applies to the ordered tuple (1.5*2米).
    trailing = parsed[-1][2]
    return tuple(Decimal(p[1]) * units[p[2] or trailing] for p in parsed)


def subject_matches(requirement, fact):
    if requirement['subject'] == fact['subject']:
        return True
    # The selected SKU's wearer fit belongs to its intended wearer. This is
    # deliberately not an alias between arbitrary people/components/products.
    return (requirement['subject']['kind'] == 'intended_user'
            and fact['subject'] == PRIMARY
            and requirement['attribute'] in ('wearer_height', 'wearer_weight_limit')
            and fact.get('origin') == 'source_local_sku_rule_v2'
            and all('/customization_options/' in e.get('pointer', '') for e in fact['evidence']))


def claim_terms(requirement):
    from semantic_fact_language_v2 import GROUPS as V2
    from semantic_fact_language_v3 import GROUPS as V3, canonical
    value = requirement['expected'].get('value')
    if not isinstance(value, str):
        return []
    subtype = requirement['subtype']
    terms = {compact(value)}
    # Remove grammatical wrappers, never substantive qualifiers (全天/官方).
    core = re.sub(r'^(?:能够|可以|可)?(?:用于|适用于|带有|具有|具备)', '', compact(value))
    if subtype in ('function', 'material', 'style', 'form_factor'):
        core = re.sub(r'(?:功能|材质|风格|款式)$', '', core)
    if len(core) >= 2:
        terms.add(core)
    target = {canonical(subtype, t) for t in terms}
    for group in list(V2.get(subtype, [])) + list(V3.get(subtype, [])):
        if any(canonical(subtype, word) in target for word in group):
            terms.update(compact(word) for word in group)
    return sorted((t for t in terms if len(t) >= 2), key=len, reverse=True)


def occurrences(text, terms):
    """Literal spans plus local negation; absence is never a negative claim."""
    normalized = unicodedata.normalize('NFKC', text).casefold()
    found = []
    for term in terms:
        for m in re.finditer(re.escape(term), normalized):
            if re.fullmatch(r'[a-z0-9._+-]+', term) and (
                    m.start() and re.match(r'[a-z0-9]', normalized[m.start()-1])
                    or m.end() < len(normalized) and re.match(r'[a-z0-9]', normalized[m.end()])):
                continue
            if any(a <= m.start() and m.end() <= z for a, z, _ in found):
                continue
            left = re.split(r'[，,。；;！!？?\n]', normalized[:m.start()])[-1]
            local = normalized[max(0, m.start()-8):m.end()+8]
            ambiguous = bool(re.search(r'(?:并非不|不是不|不能说不)$', left)
                             or re.search(r'可选|选配|另购|加购', local))
            negative = bool(re.search(r'(?:不(?:支持|含|带|赠|适用(?:于)?|具备|是|能|可)?|无法|不宜(?:用于)?|禁止(?:用于)?|无|非|未|没有|缺少)(?:该|此|本)?$', left))
            found.append((m.start(), m.end(), 'ambiguous' if ambiguous else 'negative' if negative else 'positive'))
    return found


def public_claim(requirement, dimension, product, state):
    """Source-backed positive text fallback, not a return to native rewards."""
    if requirement['subject'] != PRIMARY or requirement['operator'] != 'eq':
        return None
    allowed = {'product_category', 'function', 'purpose', 'material', 'craft', 'style',
               'form_factor', 'condition', 'origin', 'merchant'}
    if requirement['subtype'] not in allowed or requirement['expected']['type'] != 'text':
        return None
    terms = claim_terms(requirement)
    # Therapeutic purpose keywords need an explicit therapeutic claim too;
    # a mere mention of a symptom is not sufficient evidence of treatment.
    symptom = re.fullmatch(r'缓解(.+)', requirement['expected']['value'])
    if requirement['subtype'] == 'purpose' and symptom:
        public_text = ' '.join(str(product.get(k) or '') for k in ('title', 'full_description'))
        if re.search(r'缓解|止痛', public_text):
            terms.append(compact(symptom[1]))
    if not terms:
        return None
    subtype = requirement['subtype']
    if subtype == 'function' and compact(requirement['expected']['value']) in ('配件齐全', '全套配件', '全套附件'):
        # Completeness retains its explicitly approved selected-SKU policy.
        return None
    if subtype == 'condition' and compact(requirement['expected']['value']) in ('网络热卖款', '热卖款', '热卖'):
        # Existing snapshot policy accepts the exact undated declaration;
        # automatic_facts handles it. Dated/negated advertising stays unknown.
        return None
    selected = [(s['pointer'], s['value'], 'sku') for s in state['selected']]
    common = []
    for i, value in enumerate(product.get('attribute') or []):
        if isinstance(value, str):
            common.append((f'/attribute/{i}', value, 'product'))
    for field in ('title', 'full_description', 'small_description', 'shop_name'):
        value = product.get(field)
        if isinstance(value, str) and value:
            if field == 'shop_name' and subtype != 'merchant':
                continue
            common.append(('/' + field, value, 'product'))
    # If a feature is advertised by individual variants, the title cannot
    # lend it to a selected variant which does not make that same claim.
    variant_sensitive = any(occurrences(str(e.get('value', '')), terms)
                            for entries in (product.get('customization_options') or {}).values()
                            for e in entries if isinstance(e, dict))
    candidates = selected if variant_sensitive else selected + common
    if subtype == 'merchant':
        candidates = [c for c in common if c[0] == '/shop_name']
    evidence = []
    for path, text, scope in candidates:
        # Category mentions in accessory/compatibility advertising are not a
        # claim that the accessory is the primary named product.
        for a, z, polarity in occurrences(text, terms):
            if subtype == 'material' and polarity == 'positive':
                before, after = compact(text[:a]), compact(text[z:])
                # A material-looking adjective is not a material claim.
                # 小羊皮 may support 羊皮; 人造羊皮/原木色/金属质感 cannot.
                if (re.search(r'(?:人造|仿(?:制|真)?|合成|pu|pvc|贴面?)$', before)
                        or re.match(r'色|质感|效果|纹理|风格', after)):
                    continue
            if subtype == 'product_category':
                before, after = compact(text[:a]), compact(text[z:])
                if re.search(r'适用|适配|兼容|用于', before[-8:]) or re.match(r'(?:专用)?(?:保护套|收纳盒|收纳袋|配件|支架|屏幕|电池|充电器|贴膜|保护膜|壳|套|盒|袋)', after):
                    continue
            evidence.append({'pointer': path, 'value': text, 'matched_text': text[a:z],
                             'polarity': polarity, 'scope': scope})
    polarities = {e['polarity'] for e in evidence}
    if not evidence:
        return None
    status = ('unknown' if len(polarities) > 1 or 'ambiguous' in polarities else
              'pass' if 'positive' in polarities else 'fail')
    return {'status': status, 'reason': 'public_literal_claim' if len(polarities) == 1 else 'conflicting_public_claims',
            'evidence': evidence}


def automatic_pack_facts(product, state):
    """Explicit purchased pack labels, with conservative invariant fallback."""
    records = []
    count_pattern = re.compile(r'(\d+(?:\.\d+)?)(片|个|件|支|只|罐|桶|瓶|袋|包|盒|套|双|本)(?:装)?(?=$|[)）\]】(（\[【+])')
    mass_pattern = re.compile(r'(\d+(?:\.\d+)?)(kg|g|ml|l|千克|公斤|克|毫升|升)', re.I)
    aliases = {'千克': 'kg', '公斤': 'kg', '克': 'g', '毫升': 'ml', '升': 'l'}

    def emit(st, attr, value, path, label, scope='sku', subject=None):
        records.append({'dimension': 'specifications', 'subtype': st, 'attribute': attr,
                        'subject': subject or PRIMARY, 'value': value,
                        'evidence': [{'pointer': path, 'value': label}], 'scope': scope,
                        'complete': True, 'origin': 'public_pack_rule_v1'})

    def parse(label, path, scope, *, count_only=False):
        text = compact(label)
        if re.search(r'送|赠|随机|不含|无|或|到|至|~|～|\d[-—]\d|可装|能装|容纳|适合|适用|可放|支持', text):
            return
        counts = list(count_pattern.finditer(text))
        masses = list(mass_pattern.finditer(text))
        if len(counts) == 1:
            m = counts[0]
            value = {'type': 'number', 'value': float(m[1]), 'unit': COUNT_UNITS[m[2]]}
            emit('quantity', 'purchase_count', value, path, label, scope)
            emit('packaging', 'pack_count', value, path, label, scope)
        if count_only or len(masses) != 1:
            return
        m = masses[0]
        unit = aliases.get(m[2], m[2].lower())
        if (not counts and mass_pattern.fullmatch(text)) or (len(counts) == 1 and float(counts[0][1]) == 1
                and re.fullmatch(r'\s*[*×]\s*', text[m.end():counts[0].start()])):
            emit('net_content', 'mass' if unit in ('g', 'kg') else 'liquid_volume',
                 {'type': 'number', 'value': float(m[1]), 'unit': unit}, path, label, scope)

    for s in state['selected']:
        parse(s['value'], s['pointer'], 'sku')
    # A title/attribute count can be common only when no SKU offers quantity
    # alternatives. All offered variants are inspected only as a leakage veto.
    varying_pack = any(count_pattern.search(compact(str(e.get('value', ''))))
                       or mass_pattern.search(compact(str(e.get('value', ''))))
                       for entries in (product.get('customization_options') or {}).values()
                       for e in entries if isinstance(e, dict))
    if not varying_pack:
        for i, label in enumerate(product.get('attribute') or []):
            if isinstance(label, str) and count_pattern.fullmatch(compact(label)):
                parse(label, f'/attribute/{i}', 'product', count_only=True)
    return records


def selected_component_claim(requirement, product, state):
    """Unnumbered generic gifts only; no invented contents or quantities."""
    expected = requirement['expected']
    if (requirement['subject'] != PRIMARY or requirement['subtype'] != 'components'
            or requirement['attribute'] != 'gift_presence' or requirement['operator'] != 'contains_all'
            or expected['type'] != 'components'
            or any(i['name'] != '赠品' or i.get('quantity') is not None for i in expected['items'])):
        return None
    found = []
    for s in state['selected']:
        if '赠品' in s['value']:
            if re.search(r'可选|选配|另购|加购', s['value']):
                continue
            polarity = 'negative' if re.search(r'不含赠品|无赠品|不送赠品', s['value']) else 'positive'
            found.append({'pointer': s['pointer'], 'value': s['value'], 'polarity': polarity})
    if not found:
        return None
    polarities = {e['polarity'] for e in found}
    return {'status': 'unknown' if len(polarities) > 1 else 'pass' if 'positive' in polarities else 'fail',
            'reason': 'selected_sku_generic_gift_claim', 'evidence': found}
