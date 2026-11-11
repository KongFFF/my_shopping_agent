"""Pre-model source review for the fixed historical holdout. Not training labels.

Explicit item decisions authored after reading all 47 public SKU worksheets,
before any model judgments. Difficult scopes are unknown or omitted, not guessed.
"""
from purchase_judge_benchmark import DEFAULT, rows, lines, write, digest
from collections import Counter


def build(folder=DEFAULT):
    cases = [c for c in rows(folder / 'cases.jsonl') if c['first_cohort'] == 'holdout_v5']
    assert len(cases) == 47
    authored = []

    def add(index, task, passes=None, unknown=None, fails=None):
        c = cases[index]
        assert c['task_id'] == task
        requirements = {r['id']: r for r in c['request']['requirements']}
        for status, decisions in [('pass', passes or {}), ('unknown', unknown or {}), ('fail', fails or {})]:
            for rid, spec in decisions.items():
                assert rid in requirements
                if isinstance(spec, str):
                    pointers, note = ([] if status == 'unknown' else [spec]), ('公开资料未完整支持该要求。' if status == 'unknown' else '公开文字或当前所选规格明确支持该判断。')
                    if status == 'unknown':
                        note = spec
                else:
                    pointers, note = spec
                authored.append({'case_id': c['case_id'], 'task_id': task, 'product_asin': c['product_asin'],
                    'requirement_id': rid, 'requirement': requirements[rid], 'status': status,
                    'evidence': [{'pointer': p, 'value': c['request']['evidence_table'][p]} for p in pointers],
                    'reason': note, 'reviewed_by': 'assistant', 'reviewed_before_model': True})

    def option(i, axis='颜色分类'):
        pointers = [p for p in cases[i]['request']['evidence_table'] if p.startswith('/customization_options/' + axis + '/')]
        if not pointers and len(cases[i]['selected_options']) == 1:
            pointers = [p for p in cases[i]['request']['evidence_table'] if p.startswith('/customization_options/')]
        assert len(pointers) == 1
        return pointers[0]

    add(0, 2817, {'r002': '/title', 'r004': '/attribute/0', 'migration001': '/attribute/3', 'r003': '/attribute/1', 'r005': '/attribute/2'},
        {'r001': '没有独立订单件数或销售单位证据，不从用户的一把反推。', 'r007': '蓝色没有明确支持浅青色；不凭相近色名推断。'})
    for i in (1, 2):
        common = {'r002': (['/title', option(i)], '魔尺标题与选中多成员组合支持套装类别。'),
                  'r005': '/attribute/2', 'r006': '/title', 'migration001': '/title',
                  'r001': '/attribute/1', 'r004': option(i)}
        unknown = {rid: '没有相同属性的明确公开声明。' for rid in ('r007', 'r008', 'r009', 'migration002', 'migration003')}
        if i == 1:
            add(i, 243, common, unknown, {'r003': ([option(i)], '选中组合只列24、36、48、60四档，不是需求的五档；不借标题72段。')})
        else:
            add(i, 243, {**common, 'r003': option(i)}, unknown)
    add(3, 22999, {'r002': '/category', 'r005': '/attribute/0', 'r009': (['/attribute/1', '/attribute/2'], '幼儿园、小学生均有声明。'),
        'r001': option(3), 'r003': '/attribute/5', 'r004': option(3), 'r006': option(3), 'r007': '/attribute/4', 'r008': '/attribute/3'})
    add(5, 15660, {'r003': '/title', 'r008': '/attribute/2', 'r002': '/attribute/1', 'r004': option(5), 'r005': '/attribute/0'},
        {'r001': '女款未说明适用于妈妈这一年龄/角色。', 'migration001': '自然隐形不完整支持蓬松且无痕。', 'r006': '玉米烫款式未证明采用热烫工艺。'})
    add(6, 14005, {'r003': '/category', 'r005': '/attribute/2', 'r006': option(6), 'r004': option(6),
        'migration001': option(6), 'r002': '/attribute/1', 'r009': option(6)}, {'r001': '天然不能证明无添加。'})
    add(8, 12976, {'r007': '/attribute/3', 'r008': '/attribute/4'},
        {'r002': '仿真发不能证明真发类别，且未明确说明实际纤维材质。', 'r001': '女款未说明中年。',
         'migration001': '自然卷发不等于自然逼真效果的完整声明。', 'r003': '仿真发并未证明真实头发材质。',
         'r004': '深棕色未明确支持棕黑色，不凭近似色判断。', 'r005': '35cm没有加长款基准，不能自行设阈值。'})
    add(9, 12976, {'r002': '/title', 'r001': (['/attribute/0', '/attribute/1'], '中年和女款均明确。'),
        'r007': '/attribute/5', 'r008': '/attribute/4', 'migration001': (['/attribute/3', '/attribute/6'], '自然、逼真均明确。'),
        'r003': '/attribute/2', 'r004': option(9), 'r005': option(9)})
    add(10, 7764, {'r004': '/title', 'r006': '/attribute/5'},
        {rid: '公开资料缺少该属性的明确完整声明，不能从穿着场景或单件上衣推断。' for rid in
         ('r007', 'r008', 'migration001', 'migration002', 'r009', 'r003', 'r001')},
        {'r010': (['/attribute/2'], '明确无袖，与需要袖子长相冲突。'), 'r005': ([option(10, '尺码')], '选中M，不是小码S。')})
    add(11, 7764, {'r004': '/title', 'r010': '/attribute/3', 'r003': option(11), 'r005': option(11, '尺码')},
        {rid: '珠宝店/尊贵/修身/春秋文字没有直接完整支持该具体要求。' for rid in
         ('r007', 'r008', 'migration001', 'migration002', 'r009', 'r006', 'r001')})
    add(12, 8035, {'r001': '/category', 'r004': (['/attribute/1', '/attribute/3', '/attribute/4'], '手账、拼贴、排版均明确。'),
        'migration002': '/attribute/0', 'r003': '/attribute/2', 'r005': option(12), 'r006': option(12), 'migration001': '/attribute/0'},
        {'r002': '猫咪日付标签未完整说明猫咪日常生活图案。'})
    add(14, 12215, {'r003': '/title', 'r002': '/attribute/2', 'r004': (['/attribute/1', '/attribute/3'], '蚊子对象和杀虫用途均声明。'),
        'r001': '/attribute/0', 'r006': option(14)}, {'r005': '30倍稀释与严格1比30混合的定义可能不同，不能自动判等。'})
    add(16, 8492, {'r001': option(16), 'r002': '/title', 'migration001': '/attribute/1', 'r005': option(16)},
        {'r003': '紧致未逐字明确保持张力的完整作用，不扩展功效。'},
        {'r004': ([option(16)], '选中3支，与需要2支明确不符。')})
    add(18, 8492, {'r001': '/title', 'r002': '/title', 'migration001': '/attribute/1'},
        {'r003': '没有保持张力的完整明确声明。', 'r004': '没有选规格，不借其他选项件数。', 'r005': '没有选规格，不借其他选项100g。'})
    add(20, 1905, {'r002': '/title', 'r001': '/attribute/1', 'r005': '/attribute/2', 'r007': '/attribute/5',
        'r004': '/attribute/4', 'r006': '/title', 'r008': option(20)})
    add(21, 1905, {'r002': '/title', 'r007': '/attribute/0'},
        {rid: '所选商品缺少完整明确声明；二合一不证明双筒，彩色外观不证明彩灯。' for rid in ('r001', 'r003', 'r005', 'r004', 'r006')},
        {'r008': ([option(21)], '选中彩色，不是明确要求的黑色款。')})
    add(22, 4748, {'r002': '/title', 'r004': '/title', 'r003': '/attribute/0', 'r006': option(22), 'migration001': option(22)},
        {'r001': '可视不等于附带独立屏幕。'})
    add(24, 1130, {'r003': '/title', 'r002': '/title', 'r004': '/title', 'r005': '/attribute/1', 'r007': option(24), 'r006': option(24)},
        {'r001': '没有独立订单或销售单位数量，不能从用户一台补成实购一台。'})
    for i in (25, 26):
        add(i, 19082, {'r003': '/category', 'r001': option(i, '适用人数'), 'r006': option(i, '适用人数'), 'r002': '/title'},
            {rid: '未明确声明该完整属性；电动不自动证明调节方便，2.92米也不等于近似目标2.9米的恰好值。' for rid in
             ('migration001', 'migration002', 'migration003', 'r004')})
    add(31, 5254, {'r002': '/attribute/1', 'r001': '/attribute/4', 'r003': '/attribute/5', 'r006': option(31)},
        {'r004': '只有老人没有小朋友的适用声明。', 'r005': '定制未指明可同时写身份详情和联系电话。', 'r008': '红绳选项没有钥匙扣证据。'},
        {'r007': ([option(31)], '磨砂饰面与所需抛光饰面冲突。')})
    for i in (32, 33):
        passes = {'r002': '/title', 'r001': '/attribute/0', 'r004': (['/attribute/2', '/attribute/3'], '老人认知障碍和儿童均声明。'),
                  'r003': '/attribute/1', 'r007': option(i), 'r008': option(i)}
        unknown = {'r005': '号码未明确是联系电话，身份信息与联系电话这一完整组合仍不充分。'}
        if i == 32:
            passes['r006'] = option(i)
        else:
            unknown['r006'] = '未明说有挂绳，选项缺省不能证明不存在也不能给分。'
        add(i, 5254, passes, unknown)
    add(36, 23153, {'r001': '/category', 'r006': '/title', 'r007': (['/attribute/7', '/attribute/8'], '两种运动均声明。'),
        'review_gender': '/attribute/2', 'r002': '/attribute/4', 'r003': '/attribute/5', 'r004': option(36),
        'r005': option(36, '尺码'), 'migration001': (['/attribute/0', '/attribute/1'], '夏季与新款同时明确。')})
    add(37, 19514, {'r003': '/category', 'r004': option(37), 'r001': option(37), 'r002': option(37)})
    add(38, 3562, {'r003': '/category', 'r001': '/attribute/0', 'r002': '/attribute/1', 'r006': '/attribute/4', 'r004': option(38)},
        {'r005': '未说明软头。', 'r007': '安全不完整证明无毒声明。', 'r008': '赠图画本但未说明恰好两本。', 'migration001': '没有新款声明。'})
    add(40, 3562, {'r003': '/category', 'r001': '/attribute/0', 'r002': '/attribute/3', 'r006': '/attribute/1',
        'r004': option(40), 'r005': option(40), 'r008': option(40)},
        {'r007': '安全未完整说明无毒。', 'migration001': '没有新款声明。'})
    add(41, 3562, {'r003': '/category', 'r001': '/attribute/0', 'r002': '/attribute/1', 'r006': '/attribute/3',
        'r004': option(41), 'r005': option(41), 'migration001': option(41)},
        {'r007': '只说无毒，未把所有其他安全风险声明齐全。', 'r008': '送2本未指明本的类型，不直接补成画册。'})
    add(44, 6547, {'r004': '/title', 'r001': (['/title', option(44)], '四川乐山冷锅和所选香辣调料共同支持。'),
        'r005': (['/attribute/0', '/attribute/4', '/attribute/5'], '家用和两种菜式均声明。'), 'migration001': '/attribute/6',
        'r003': option(44), 'r002': (['/attribute/1', '/attribute/2'], '四川乐山风味来源文字。'),
        'r007': option(44), 'r008': option(44), 'r009': option(44)},
        {'r006': '208g不是明确目标200g，未定容差，保持未知。'})
    for i in (45, 46):
        add(i, 23022, {'r002': '/title', 'r001': '/attribute/0', 'r003': option(i)})

    identities = [(r['case_id'], r['requirement_id']) for r in authored]
    assert len(identities) == len(set(identities))
    rankings = [
        {'left_case_id': cases[2]['case_id'], 'right_case_id': cases[1]['case_id'], 'relation': 'gt',
         'reason': '243：同样的未知项不凭空得分，五档完整组合必须高于缺72段的四档组合。'},
        {'left_case_id': cases[25]['case_id'], 'right_case_id': cases[26]['case_id'], 'relation': 'eq',
         'reason': '19082：乳胶/海绵不是用户区分条件，其他相关条件相同。'},
        {'left_case_id': cases[45]['case_id'], 'right_case_id': cases[46]['case_id'], 'relation': 'eq',
         'reason': '23022：实际组件一致，是否点击套餐一占位轴不应单独加分。'},
        {'left_case_id': cases[40]['case_id'], 'right_case_id': cases[38]['case_id'], 'relation': 'gt',
         'reason': '3562：明示软头、两本填色本和勾线笔的套餐，应高于这些证据不完整的选项。'},
    ]
    for pair in rankings:
        pair['reviewed_before_model'] = True
    lines(folder / 'reference_atoms.jsonl', authored)
    lines(folder / 'reference_rankings.jsonl', rankings)
    summary = {'reviewed_before_model': True, 'reference_atoms': len(authored),
               'status_counts': dict(Counter(r['status'] for r in authored)),
               'purchase_cases': len({r['case_id'] for r in authored}), 'ranking_pairs': len(rankings),
               'reference_sha256': digest(authored), 'rankings_sha256': digest(rankings),
               'not_train1000_accuracy': True, 'training_assets_modified': False,
               'author': 'assistant', 'user_whole_query_signoff': False,
               'limitations': ['purposive stratified review, not random population estimate',
                              'ambiguous atoms conservatively unknown or excluded',
                              'few clear failures; not sufficient to estimate rare false-positive rates']}
    write(folder / 'reference_summary.json', summary)
    print(summary)


if __name__ == '__main__':
    build()
