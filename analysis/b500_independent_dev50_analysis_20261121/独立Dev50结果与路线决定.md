# 固定B500独立Dev50结果与路线决定

2026-11-21。回传提交8f9dd51fa7fc952bff58625e610918feec1bdbc9，分支codex/b500-dev50-results-20261121T031644Z-izlfsckp。

## 结论

模型导出、服务身份和独立评测执行核验通过。固定B500严格成功28/50，原A500为27/50，净增1题（2个百分点）；未达到冻结的30/50且净增至少3题门槛。因此本次F=0.5/S=1配置不推进800或Final，记录为已完成但未获足够收益支持的消融，不根据当前Dev更换checkpoint或追调权重。

这不是证明所有全失败组降权都无效。当前结果不能支持“本改进训练法比原GRPO更有效”的主要命题，也不能由一个训练seed和50题推出稳定收益。主目标仍为发现关键训练改进，避免转向奖励小修补。

## 证据核验

- 归档SHA256 b796d341f0b239d89114152fdd22afea771e8e5cb30529d738ed0acb46d7e88d与清单一致，ZIP CRC通过。
- 导出源为此前完成运行的global_step_500，LoRA r16/alpha32；568个adapter张量。源Actor云端SHA256 ffb6a3f4994ad0b1fa72b4c0dbf8db8e57adf430ea0d4e7b97975619b2125275，合并权重36b03a911679bc66cfd44122520cb4b826f853151edf30b2512fd00833ea037b。权重未回传，本地核对收据及服务指向，不宣称重新加载了模型。
- vLLM服务模型名、merged目录、上下文24576匹配导出；原评测协议除模型名外与A完全相同，任务顺序一致，只存在smoke3和dev50命令。
- smoke3为2/3，Dev50为28/50；两套轨迹与摘要指纹一致，逐条重算严格gold成功、平均reward和平均步数均与摘要一致。
- Dev50完整50条、无缺题/重复、无非ContextBudgetError执行异常/释放异常。50条均done，上下文溢出0，reward_unverifiable2条仍在分母。流水线exit_code=0，Final未执行。

可复算结果见[result.json](result.json)。

```bash
python -X utf8 tools/analyze_b500_dev50_return.py --archive analysis/b500_independent_dev50_return/20261121T031644Z/evidence.zip --output analysis/b500_independent_dev50_analysis_20261121/result.json
```

## 主要与次要指标

| 指标 | SFT300 | 原A500 | 改进B500 |
|---|---:|---:|---:|
| 原协议独立Dev严格成功 | 24/50 | 27/50 | 28/50 |
| 严格成功率 | 48% | 54% | 56% |
| 平均reward | — | 0.432299 | 0.474589 |
| 平均工具步数 | — | 8.96 | 9.48 |
| repeat_loop | — | 4 | 3 |
| wrong_purchase | — | 3 | 3 |
| reward_unverifiable | — | 2 | 2 |

B相对SFT多4题，但原GRPO本身已多3题，新增方法相对原GRPO只净增1题。平均reward增加约0.04229是次要改善，不能替代主要严格成功指标或事后改推进规则。平均步骤增加0.52，也不能称为效率改进。

## 配对变化

共同成功24题、共同失败19题，B-only4题，A-only3题。exact双侧McNemar p=1.0。按50题配对非参数bootstrap的精确百分位分布，成功率差95%区间约[-8,+12]个百分点；此区间只描述任务样本不确定性，不包含跨训练seed变异。

| task_id | A结果 | B结果 | A/B工具步数 |
|---|---|---|---:|
| 3485 | partial_alternative_purchase | gold_purchase | 7/9 |
| 4128 | repeat_loop | gold_purchase | 18/10 |
| 13399 | partial_alternative_purchase | gold_purchase | 8/16 |
| 14141 | repeat_loop | gold_purchase | 16/13 |
| 2898 | gold_purchase | partial_alternative_purchase | 9/9 |
| 5293 | gold_purchase | partial_alternative_purchase | 10/9 |
| 10077 | gold_purchase | partial_alternative_purchase | 8/8 |

另外有1道原partial变为repeat_loop，故全局循环数只净减少1。当前只能确认结果类型的变化，不能仅据partial标签断言具体是规格、候选、预算或奖励错误，更不能直接归因于F组降权。

## 下一步边界

降级这笔固定配置，暂不追加GPU采样、训练、Final或权重调参。后续若继续主线，优先用已回传的A/B完整轨迹复盘这7道成败变化及共同失败，核查候选比较、规格落实和循环路径如何变化，再判断是否能形成新的机制假设。不把Dev题号、目标商品或答案写入训练规则；新方向仍需低成本资格检查。

单seed历史对照、B150处额外故障恢复及原350预设恢复均保留为解释限制。此前训练内平均reward与本独立harness的指标分开报告；本次补回了训练validation/200—500的原逐题文件，不以它们代替独立Dev结果。
