"""Archive complete CPU restoration and bounded case review, without new labels for training."""
from collections import Counter
import gzip
import json
from audit_online_windows import OUT, ROOT, inputs, save, sha, dump

NOTES = {
11377: '公开接线/电压/遥控配置不同，空规格未落实；不以区间下限代替最终价。',
1675: '有1W/3W、黄光/其他色温差异，空规格未确定需求配置；不判所有功能满足。',
13411: 'query每袋40米与页面厘米尺寸不相同，存在任务单位疑点；空规格不能解释成唯一明确正确解。',
18889: '1400以内却选1680元偏光镜；变色是否满足另保留未知。',
14635: '部分query乱码，但170元上限清楚；239元购买为局部预算矛盾。',
10181: '把顺丰告示当SKU，显示1999元超过700；基础699元不能据此证明全部兼容性和物流满足。',
373: '要黄色而选天蓝或军绿；当前候选没有黄色，需换候选而非仅同页改选。',
7387: '蓝色需求与红色已选值矛盾，其他品质/来源尚未验证。',
20632: '需求黑色套装/白衬衫，选成白马甲白裤子；颜色上从黑改白，身高规格同时有问题，局部通过不等于整题正确。',
13636: '479g与500g标称包装差异需保留容差/商品事实问题，不据此确认整题错误或制造训练偏好。',
3951: '参考窗口有5kg买20斤的数量错误；后段有白10斤改灰10斤，重量对了但颜色错。',
12852: '要5斤却选3斤，当前候选仅2/3斤，应考虑换商品。',
1231: '预算20，曾见12元配置后改48元联系客服组合；不能由12元证明010-046及全部要求满足。',
22255: '叉车吨位和型号组合可选却空规格；未确定所需联动3-3.5吨，不把所有空规格当语义失败。',
21733: '需求180型黑色，公开有150/180/200及上下片差异，空规格未确定配置。',
12458: '300以内而选中太阳能配置400元；尺寸正确不抵消明确超预算。',
20628: '同商品白色套餐一120元/套餐二125元，明确120上限；配件差异和全部功能仍需独立核实。',
19271: '400以内、粉色M码，选择后419元仍买，且该组被实际消费。',
22164: '黄色需求却买红色；当前商品没有黄色，不能用同页分叉保证修好。',
971: '黑色机身先选黑后改白，购买时还缺套餐/容量定价；明确颜色局部退化，不证明早先配置满足128G。',
2572: '501g对一斤的名义边界留未知；该队列具体SKU为红糖馅，与黑芝麻有独立口味矛盾，不能把它算作重量规则已证明。',
11961: '曾看到35元联系客服项后买43元，并且尺寸9cm/100cm不符；35元项不能充当正确尺寸正例。',
789: '保修版本244/334元跨250预算，可作局部价格比较，不验证实体功效。',
1167: '196元普通款对398元礼盒，后者违反300上限；便宜款缺必需礼盒/关公款，不能生成整体偏好。',
}


def main():
    cache = {}
    names = ('restored_observations.jsonl.gz',) if (OUT / 'restored_observations.jsonl.gz').exists() else ('replays.jsonl.gz', 'replays_parallel.jsonl.gz')
    for name in names:
        p = OUT / name
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            while True:
                try:
                    line = f.readline()
                    if not line: break
                    r = json.loads(line); cache[r['key']] = r
                except (EOFError, json.JSONDecodeError): break
    audit = json.loads((OUT / 'replay_audit.json').read_text())
    assert not audit['partial'] and not audit['mismatches'] and len(cache) == audit['unique_eligible_actions'] == 7395
    target = OUT / 'restored_observations.jsonl.gz'
    with target.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
            for key in sorted(cache): gz.write((dump(cache[key]) + '\n').encode('utf-8'))
    queue = json.loads((OUT / 'review_queue.json').read_text(encoding='utf-8'))
    contrasts = json.loads((OUT / 'local_contrasts.json').read_text(encoding='utf-8'))
    review = []
    for r in queue:
        assert r['task_id'] in NOTES
        status = 'supported_local_contradiction'
        if r['kind'] == 'empty_options_candidate': status = 'unresolved_configuration_only'
        if r['kind'] == 'explicit_mass' and r['task_id'] in (13636, 2572): status = 'unknown_nominal_mass_or_tolerance'
        review.append({**r, 'assistant_review_status': status, 'assistant_note': NOTES[r['task_id']],
                       'training_label_created': False})
    for r in contrasts:
        assert r['task_id'] in NOTES
        review.append({**r, 'assistant_note': NOTES[r['task_id']], 'whole_preference_ready': False,
                       'warning': 'Separate policies/histories; a local pass is not complete task satisfaction.'})
    manifest, occurrences, _, _ = inputs()
    excluded = []
    for o in occurrences:
        r = o['rollout']
        reasons = [key for key in ('guard_rejections', 'error', 'infrastructure_invalid', 'context_compactions') if r.get(key)]
        if reasons:
            excluded.append({'window': o['window'], 'step': o['step'], 'uid': r['uid'], 'rollout_index': r['rollout_index'],
                             'task_id': r['task_id'], 'consumed': o['consumed'], 'reasons': reasons})
    assert len(excluded) == 877
    save('uncertified_index.json', excluded)
    original = {(o['window'], o['rollout']['uid'], o['rollout']['rollout_index']): o['rollout'] for o in occurrences}
    flags = json.loads((OUT / 'screen_flags.json').read_text(encoding='utf-8'))
    cross = {}
    for window in manifest['windows']:
        rs = [r for r in flags if r['window'] == window]
        cross[window] = {'candidate_tasks': len({r['task_id'] for r in rs}),
                        'candidate_episodes': len({(r['uid'], r['rollout_index']) for r in rs}),
                        'types': {kind: dict(Counter(original[(window, r['uid'], r['rollout_index'])]['reward_type'] for r in rs if r['kind'] == kind)) for kind in sorted({r['kind'] for r in rs})}}
    save('reviewed_cases.json', review)
    save('review_summary.json', {'queue_entries': len(queue), 'contrast_groups': len(contrasts),
        'reviewed_distinct_tasks': len({r['task_id'] for r in review}), 'queue_status_counts': dict(Counter(r['assistant_review_status'] for r in review[:len(queue)])),
        'reward_crosscheck': cross, 'restored_cache_sha256': sha(target.read_bytes()), 'restored_unique_actions': len(cache),
        'model_calls': 0, 'legacy_4192_records_used': 0, 'limits': 'Bounded assistant review, not all flags independently adjudicated or an unbiased error rate.'})
    print(dump({'reviewed_tasks': len({r['task_id'] for r in review}), 'cache_unique': len(cache), 'cache_bytes': target.stat().st_size, 'reward_crosscheck': cross}))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8'); main()
