"""Finish CPU-only endpoint diagnosis with explicit assistant review notes.

The notes are case-level development judgments, not independent ground truth.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

from export_endpoint_dev import strict
from analyze_guard_dev50 import variants, evaluate_purchase

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/endpoint_dev_analysis_20261115'
ARCHIVE = ROOT / 'analysis/endpoint_dev_return_20261115/evidence.zip'

# Authored after reading the original queries, visible observations and actions.
# The eight clear-error IDs form a conservative illustrative lower bound only.
NOTES = {
    565: ('mixed_candidate_and_evidence_gap', '选中混搭大低音炮套装，需求为黑色及小低音炮；当前候选没有明确满足这两个条件的完整SKU。不把功能词套装缺失当成唯一错误，也不声称本商品换规格即可满足。'),
    652: ('contract_and_candidate_gap', '750买590元鞋，800换为199元蓝色鞋，均非原标注目标。500左右被原评分设为严格500上限，旧审计目标523元无gold；800仍未明确核实蓝白搭配及全部用途。不能将本题视为纯搜索或纯规格错误。'),
    1220: ('explicit_price_range_and_scoring_gap', '购买250ml规格58元，原句60—70元未被原评分提取上下界；疾病标签大小叶斑病对大斑病/叶斑病的词面差异不足以独立判断完整适用性。存在价格范围核对缺口，不给出实际农药适用建议。'),
    2718: ('clear_visible_decision_error', '打开页面已列出黑色款，却在selected_options={}时直接购买；原奖励因规格依赖价格而不可核验。这是策略未完成可见规格选择，不等同于基础设施故障。'),
    3485: ('clear_visible_decision_error', '页面同时有米白色与米白色常规；需求明确标准款。750选常规成功，800选米白色退化。'),
    4128: ('loop_with_visible_recovery_candidate', '两次打开同一商品，始终有500g-1瓶选项，却反复尝试20/10瓶并换搜索，最终repeat_loop；这是状态恢复/数量预算核对线索，不证明1瓶满足全部实际功能。'),
    5293: ('protocol_placeholder_signal', '500与750/800最终同商品、犬肠道2kg、267元。500额外点击单值0lb而gold；新端点缺此点击而partial。不是已确认的真实肠胃功能或重量理解退化。'),
    7061: ('protocol_and_payment_ambiguity', '最终选先行预定补拍，原标注要求预定请拍并补点说明和注意轴；两者文字均写定金150/全款731，购买字段却270。流程槽位、SKU差别与支付口径混合，不能单纯记为模型不理解预定。'),
    8107: ('clear_visible_decision_error', '需求2寸、5个、送短手柄；公开最终SKU明确4寸10个不带手柄。即使不依赖gold身份，也有尺寸和配件矛盾；该商品没有合适2寸SKU，不能只在同页修后缀。'),
    9316: ('product_fact_gap', '选足金古法轮胎手镯；公开字段不足以确认大于99.9%纯度、宽面和全部用途。隐藏款式=10000也不是原query指定的款式。保留需求证据不足，不将缺词当否定事实。'),
    12066: ('clear_visible_decision_error', '需求250以内；先看到299元，再切换到276元，最终仍买。276价格是Actor选择后的可见反馈，页面另有S10可供继续核验。'),
    12703: ('hidden_requirement_signal', '需求未指定打孔/挂钩；800所选宽3m高2.7m一片可改短与原标注仅挂钩/打孔不同。评分只因打孔要求扣规格，但不能据此确认真实需求违背。颜色与完整遮光另保留未验证。'),
    14156: ('clear_visible_decision_error', '需求宽松鞋型，页面明确有海豚黑（宽版），却选海豚黑普通款；不是缺少可见选项。'),
    14929: ('budget_contract_ambiguity', '原句20块左右，目标唯一规格22元；原评分严格上限20，枚举无法得gold。原指标保留，不事后加回成功，也不能把它当明确的模型违规购买反例。'),
    16252: ('clear_visible_decision_error', '多种公开SKU明确区分分体、容量、自动/半自动、速度和质保，模型直接空规格购买；初始780—1060区间无法代表最终需求SKU价格。'),
    16263: ('product_fact_and_matching_gap', '3斤与1.5kg数量一致，且8元合上限；水溶/水产适用缺独立公开证据。跨商品选项字串不同不等于重量错误，催花/促花词面不匹配不能单独证明能力错误。'),
    16272: ('goal_query_and_fact_gap', 'query要不锈钢，旧奖励却要求铁艺、主卧等词；800选不锈钢边框水波纹，优于750循环的执行完成性，但未独立确认奶油颜色及定制细节，不判整题满足。'),
    17848: ('quantity_verification_gap', 'query要两瓶，最终商品只有柔肤绿/紫选项，轨迹没有建立两瓶配置的证据；不能凭缺数量文字就断言实际只有一瓶，也不能凭颜色对就判整题满足。'),
    19606: ('recovery_and_contract_mix', '750循环；800反复打开单一48寸候选后转到42寸白色33c款并以410元买下。原query未指定31A而评分指定；约340元预算也未被提取。既有恢复改进，也有预算核对风险，不能把全部扣分解释为错误选款。'),
    20134: ('clear_visible_decision_error', 'query明确事事如意，最终选金玉满堂，公开标签已矛盾。当前候选没有事事如意选项，应继续找候选；不能靠同页规格分叉保证修好。'),
    23126: ('clear_visible_decision_error', '已试过灰色决明子/黄荆子/花草填充选项，随后改成明确没有填充物的枕套+内袋并购买，与用户填充偏好相反。不能推断它一定由低价奖励驱动。'),
    2898: ('opaque_option_code_transition', '750/800同商品同价，只从100074690改成100074689。实际看过的Features为空，Description重复标题，没有深蓝与代码的可见映射；不能解释为已学会深蓝选择。'),
    8666: ('mixed_quantity_and_placeholder_transition', '相对500，800从1个改为买10送1，有明确数量改善；相对750，750已选买10送1，800新增单值颜色轴规格才变gold，新增成功主要是流程填槽。'),
    13399: ('supported_candidate_recovery_transition', '500停在48TB候选；750/800继续搜索，选公开标注扩存80TB、脱机克隆+联机读写的迷你标准版699元。支持候选恢复与容量/功能证据改善，不将商家声明当实物验证。'),
    14141: ('supported_candidate_recovery_transition', '750多次搜索并循环；800找到阿里健康大药房候选、军绿色357元，标题可视/家用/医院专用，支持可见候选与需求更一致。不是医疗性能验证。'),
}


def main():
    audit = json.loads((OUT / 'audit.json').read_text(encoding='utf-8'))
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != audit['archive_sha256']:
        raise ValueError('Evidence archive changed')
    models = {}
    with zipfile.ZipFile(ARCHIVE) as handle:
        for step in (750, 800):
            models[step] = {int(t['task_id']): t for t in map(json.loads, handle.read(f'step{step}/trajectories.jsonl').decode('utf-8-sig').splitlines())}
    failed_union = {tid for m in models.values() for tid, t in m.items() if not strict(t)}
    assert failed_union <= NOTES.keys() and len(failed_union) == 24
    cards = []
    for tid, (kind, note) in sorted(NOTES.items()):
        endpoints = {}
        for step, model in models.items():
            t = model[tid]
            # Every executed action and Actor-visible result remains available.
            endpoints[str(step)] = {'success': strict(t), 'query': t['initial_result']['instruction'],
                'actions': [{'turn': i + 1, 'tool': s['tool_name'], 'parameters': s['parameters'], 'observation': s['observation']} for i, s in enumerate(t['steps'])]}
        cards.append({'task_id': tid, 'triage': kind, 'assistant_review': note,
                      'review_scope': 'case-level local evidence, not complete semantic adjudication or independent human truth',
                      'endpoints': endpoints})
    (OUT / 'reviewed_cases.json').write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = ROOT / 'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'
    with gzip.open(archive, 'rt', encoding='utf-8') as handle:
        products = {str(p['asin']): p for p in json.load(handle)}
    checked, oracle = 0, []
    for step, model in models.items():
        for tid, t in model.items():
            terminal = t.get('terminal_result') or {}
            purchase = terminal.get('purchase') or {}
            detail = terminal.get('reward_detail') or {}
            if purchase.get('asin'):
                product = products[str(purchase['asin'])]
                result = evaluate_purchase(product, terminal['goal'], selected_options=purchase.get('options') or {})
                assert result.reward_type == detail['reward_type'] and math.isclose(result.reward, t['final_reward'], abs_tol=1e-6), (step, tid)
                checked += 1
                if step == 800 and not strict(t) and str(purchase['asin']) == str(terminal['goal']['asin']):
                    oracle.append({'task_id': tid, 'original_type': detail['reward_type'], **variants(product, terminal['goal'])})
    clear_ids = sorted(tid for tid, (kind, _) in NOTES.items() if kind == 'clear_visible_decision_error')
    assert len(clear_ids) == 8 and checked == 93 and len(oracle) == 10
    reward_source = ROOT / 'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py'
    report = {'purchase_reward_recomputations': checked, 'mismatches': 0, 'step800_same_target_option_oracle': oracle,
              'reviewed_distinct_tasks': len(cards), 'failed_union_tasks': len(failed_union),
              'step800_clear_local_error_examples': clear_ids,
              'product_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'reward_source_lf_sha256': hashlib.sha256(reward_source.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'limitations': 'Hidden-goal original-reward upper bound only; 8 case judgments are conservative examples, not an independent accuracy estimate.'}
    (OUT / 'reward_replay_and_oracle.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('purchase_reward_recomputations', 'mismatches', 'reviewed_distinct_tasks', 'failed_union_tasks', 'step800_clear_local_error_examples')}, ensure_ascii=False))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    sys.dont_write_bytecode = True
    main()
