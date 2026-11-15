"""Record Step-1 assistant review; not a new reward or training whitelist."""
import json
from collections import Counter
from pathlib import Path
from audit_training_decisions import ROOT, OUT, RUN, load, sha

NOTES = {
390: '白色上衣/M码已选，随后切成黑色裤子；同页有可执行的正确局部选择。',
882: '明确长度、蓝柄和喷头要求，但空规格购买；原候选有相应公开组合。仍未证明所有功能满足。',
906: '空规格；页面主要是抛料头/风送器部件，不能证明整机功率/料箱/管道满足，应同时审查候选。',
1167: '明确别超三百，选择后398元仍购买；当前SKU还不是关公款，不能只修预算就认定整题正确。',
1733: '短query且版本功能差异大；空规格是状态未落实线索，不能仅由空规格断言某版本全满足需求。',
2382: '12元违反明确10元上限；另一8元老式70配置不能据便宜就判适配五羊125。局部预算对照不能直接当偏好。',
2465: '面积整套价/踏板/免费小样混合，且要求按平方米计价；空规格不提供确定配置与价格，保留单位价格口径疑点。',
2572: '空规格候选和一斤买5斤均存在；501g对一斤500g的边界差别不能直接当真实语义错误，另列未知。',
4624: '要求1斤，实际选10斤；同商品有1斤版本，SFT已采到1斤和10斤，GRPO旧复评也有1斤成功。尽量20元不是硬上限。',
5254: '明确30元以下，31元套餐违反局部上限；25元抛光+钥匙圈未独立核实挂脖配件，不把局部预算通过当整题通过。',
5423: '1500元以内却选1980元一体柜；是否能同时满足四锅、煎饼等需另核实，不保证便宜SKU可替代。',
5535: '需求两公斤并附助剂，页面克级/公斤级及是否赠助剂差异明确，模型空规格购买，配置未落实。',
7853: '空规格且候选有标准/旗舰版，签名与二十颗磁力球缺公开核实；不能仅靠补选一轴判成功。',
8818: '需求大号带推车，页面明确区分大小及是否带推车；空规格购买，没有确定配置。',
8963: '要求钛无镀层而当前页面为麦饭石锅，且套装选项空置；材质/候选与配置问题混合，不只研究后缀。',
10181: '预算700以内，799元带底座套餐超额；699元基础手柄仅预算通过，Xbox兼容和物流条款仍需证据。',
10335: '明确190以内，选择后显示199元仍购买；局部预算错误，不验证实体洁牙效果。',
10442: 'query要白色音响却选红色；该页没有明确白色SKU，应换候选。“九十元出头”不使用派生94元硬上限。',
10941: '需要棕色标配，页面有标准/豪华/旗舰及颜色差异；空规格购买，未落实公开配置。',
11005: '两类记录分别为空规格和选1680元配置违反1000上限；容量18L对了不代表预算也满足。',
11348: '220以内、带车载线；同商品219元带线与228元不带线形成明确预算/配件局部对照，另一升级258元也超预算。',
11552: '仅作为排除示例：绳子红色的原子要求不能拿牌子/皮贴的黄色SKU直接比较，主体错误。',
12422: 'query列出大量互斥贴膜/部件名，优先级不清，空规格不能反推出唯一正确配置。',
12502: '同轨迹680元全自动→650元半自动→800元手机控制后购买；明确700上限被违反。两包容量仍未核实。',
12516: '只有儿童/阅读/卧室专用等词，没有指定册数；多册轴空置不自动是需求违背。',
12753: '要求25mm/33根，候选只列16/20规格及35/25根且空规格；应先解决候选，不声称同页选一项就正确。',
12852: '需要5斤装，实际选3斤；当前商品仅2斤/3斤，必须换候选或明确其他订单能力，不把同页分叉当必解。',
13275: '需要2个暖光免打孔，公开候选有相应数量/光色组合，空规格购买未建立所需配置。',
15412: '转速、转子容量和数量组合明显不同，空规格不能确定5000转/400ml配置；不能把初始价格区间下限当最终价。',
15431: '明确80元上限，115元仍购买，且规格25/33根/3米与需求16/25根/3.8米不同，涉及候选恢复。',
22164: '要求黄色，选红色；当前页没有明确黄色值，需要重新找候选，不能只强化该页选项。',
23345: '仅作为排除示例：“四百元的价位”不足以独立推出严格400元上限，418元不直接判硬违规。',
23416: '需求8轮、咖色及约14cm轮子；页面代码没有完整映射，空规格只能记录配置未知。',
}


def main():
    flags = load(OUT / 'screen_flags.json')
    contrasts = load(OUT / 'local_contrasts.json')
    assert {r['task_id'] for r in flags} <= NOTES.keys()
    reviewed = []
    for record in flags:
        r = dict(record)
        kind = r['kind']
        if kind == 'empty_options_candidate':
            status = 'unresolved_configuration_not_semantic_failure_label'
        elif kind == 'explicit_mass' and abs(r['comparison']['actual_g'] - r['comparison']['expected_g']) <= 1:
            status = 'unknown_nominal_mass_boundary'
        else:
            status = 'supported_local_violation_or_regression'
        r.update(assistant_review_status=status, assistant_task_note=NOTES[r['task_id']],
                 limitation='Local evidence only; no independent human truth, whole-task verdict, training reward or whitelist.')
        reviewed.append(r)
    contrast_review = []
    for c in contrasts:
        contrast_review.append({**c, 'assistant_task_note': NOTES[c['task_id']],
                               'whole_preference_ready': False,
                               'reason': 'Separate histories; local pass need not satisfy compatibility, capacity, accessories or all other requirements.'})
    # Store 4 conservative exclusions, including the 501g screening boundary.
    exclusions = []
    for arm, tid, attempt, reason in [('sft', 23345, 7, 'vague_price_not_hard_cap'),
                                      ('grpo', 10442, 7, 'derived_ninety_plus_cap_not_explicit'),
                                      ('sft', 11552, 7, 'component_color_not_primary_color'),
                                      ('grpo', 2572, 6, '501g_nominal_boundary_not_proven_error')]:
        p = RUN / arm / 'attempts' / f'{tid}_{attempt}' / 'try_000/trajectory.json'
        t = load(p)
        buy = next(i for i, s in enumerate(t['steps']) if s['tool_name'] == 'buy_now')
        exclusions.append({'task_id': tid, 'arm': arm, 'attempt': attempt, 'reason': reason,
                           'query': t['initial_result']['instruction'], 'path': p.relative_to(ROOT).as_posix(),
                           'trajectory_sha256': sha(p.read_bytes()), 'buy_turn': buy + 1,
                           'visible_before_buy': t['steps'][buy - 1]['observation']})
    supported = [r for r in reviewed if r['assistant_review_status'] == 'supported_local_violation_or_regression']
    result = {'reviewed_flag_records': len(reviewed), 'reviewed_flag_tasks': len({r['task_id'] for r in reviewed}),
              'review_status_counts': dict(Counter(r['assistant_review_status'] for r in reviewed)),
              'supported_local_violation_tasks': sorted({r['task_id'] for r in supported}),
              'supported_local_violation_episodes': len({r['path'] for r in supported}),
              'separate_history_local_contrasts': len(contrast_review),
              'separate_history_contrast_tasks': len({c['task_id'] for c in contrasts}),
              'training_authorized': False, 'whole_preference_labels_created': 0,
              'limits': 'Assistant screening review, not prevalence/accuracy estimate or checkpoint800 evidence.'}
    for name, obj in [('reviewed_flags.json', reviewed), ('reviewed_local_contrasts.json', contrast_review),
                      ('excluded_or_unknown_examples.json', exclusions), ('review_summary.json', result)]:
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
