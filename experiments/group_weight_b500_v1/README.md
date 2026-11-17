# B500协议 v1：SFT起点、全失败组降权

2026-11-17。**研究设计已冻结，尚未实现B训练补丁，尚未执行云端身份预检或GPU训练。** 机器可读的唯一参数来源为[protocol.json](protocol.json)，哈希见[manifest.json](manifest.json)。历史对照核验见[报告](../../analysis/b500_matching_20261117/历史A匹配核验.md)。

## 实验命题及主要变化

检验：同一SFT step300起点、原1000题/奖励/环境下，固定500次有效更新，按组成功情况调整PPO损失贡献能否改善完整任务成功。

原始组优势仍为`A=R−同组均值`，不除标准差，不重写奖励或动态筛选。F组四条均无成功，权重0.5；S组至少一条成功，权重1。训练成功沿用`reward_valid && reward_detail.purchase_success`，包含原生valid_alternative_purchase；不临时改为reward==1或隐藏目标ASIN匹配。独立评测主要指标仍为原严格gold_purchase。

权重应用于**已经进入Actor更新的组**，不能改变采样、有效组筛选、奖励或任务分布。以UID关联同组四条轨迹，随长度平衡/重排保留，不能用rollout_index或当前连续四条推断分组。

## 冻结的归一化

采用整个Actor更新批次的有效token加权平均：

`L_B = Σ(w_group × response_mask × 原clipped PPO逐token损失) / Σ(w_group × response_mask)`。

response_mask只计原来参与Actor训练的生成token，工具回复、padding不计。分母在整个8轨迹更新批次、全部微批/设备统一统计，无梯度；不得逐轨迹或逐微批各归一化。正权重在clip之后乘，等价实现必须证明不改变clip分支与原优势统计。

两组都是F或都是S时，归一化后回到原损失；一F一S时改变相对贡献。该方法不是“每个F更新都减半”，也不是新局部监督或奖励修复。

## 起点、训练及恢复

- **起点固定SFT300**，新建rank16/alpha32 GRPO LoRA与AdamW；不复用A500或A800优化器，不重新SFT。
- 从第1次到第500次有效更新，前15步线性warmup至1e-6，之后恒定。原两题×四采样、8轨迹Actor minibatch、ppo_epochs1及全部资源/采样配方沿用历史A。
- 保存1、10、25以及50的整数倍，最多保留10个正式候选以匹配原配置；校验和安全保存不能删除唯一可恢复checkpoint。
- 在350保存完整model/optimizer/scheduler/extra-RNG/data状态，执行一次预设恢复到351，匹配历史A重启结构；不能新挂adapter或重新warmup。350恢复不重复验证step0，数据恢复行为及首批ID如实留痕。
- 生成服务、环境版本、索引、工具和prompt必须采用历史配方；不能带入购买保护、预算解析补丁、任务调度、组回收或其他旧实验hook。B只在独立overlay中实现组权重及必要日志，不改共享site-packages。
- 原seed设置42；vLLM未显式绑定新seed的字段保持原包/原默认。相同seed不保证异步rollout或不同策略生成的轨迹相同。

## 预算和停止

1. 上限500次有效更新；跳过尝试不算更新，但全部生成均计成本。所有无效、丢弃、回滚也计入**24000条训练生成**上限，下一批开始前预留完整批次数量，不能静默跨限。
2. 整个B阶段GPU占用上限20小时，包含启动、训练、训练内验证、导出与独立Dev。离线无卡预检不计GPU。达到上限保存最近安全状态并停止，标记budget_truncated；未达500不叫B500。
3. 原训练内Dev在0、50、…、500运行，共550回合；独立B500 Dev50一次，固定smoke最多3回合。额外复跑必须记录原因和全部成本，不能补成功轨迹。
4. 非有限loss/梯度、数据/模型/奖励指纹漂移、分组错配、token分母异常、checkpoint不完整、原动态采样连续跳过上限等立即停止并保留证据；不得自动换参数、重启新seed或增加预算。途中普通Dev波动不触发追分或自动调权。
5. 此次不执行Final200，不自动续至800。若以后续训，需要另行确认协议；不基于已查看Final决定。

## 评测与结果解释

**唯一主要端点是固定B500的独立Dev50，与历史A500独立Dev27/50配对比较。** SFT独立Dev24/50作起点参照；训练内A500的29/50不能作独立B的A。保留训练内50步曲线用于行为诊断，但不以B最佳checkpoint替换500追分。

独立Dev沿用相同50题、温度0/top_p1、35步、单回合512 token、上下文24576、关闭压缩及同一投影/guard/runtime。模型服务名与输出路径可改，其余harness需绑定源码和参数；不使用此前独立采样程序代替原评测路径。

报告B-only/A-only成功任务、净成功差、错误购买/循环/无法核验、平均步骤和奖励；配对区间及检验只能描述这50题的不确定性，不能证明跨训练seed显著性。主要结论是相同有效更新数下的结果；全部生成/Actor有效token/真实GPU时间另报，不能假定计算量相同。

预设继续讨论800的筛查条件为B≥30/50、相对A净增≥3题，并审查原成功任务退步、错误购买/循环与生成成本。它不是显著性标准，也不自动授权续训；1—2题变化记不确定，≤A则当前配置未获支持，均不自动追训。

## 运行前必须完成的验收

- 云端只读身份核验通过，原baseline_plan SFT指纹、冻结收据、当前权重三方相等。
- 实现B overlay后完成数值/梯度测试：权重全1复现原损失；FF/SS抵消；不等长FS正确token分母；mask/padding不参与；整批与微批累积等价；UID重排正确；零分母/无效组拒绝；350恢复无重新warmup。
- 保存每组F/S、原优势、实际有效token数、权重/归一化系数、正负优势行为，以及FF/FS/SS计数。累计所有生成和失败尝试成本，不能只记4000条进入学习的轨迹。
- 正式B的第1次更新作为同一条500步训练链的工程验收点，不另起额外无记录试跑；未通过就停，不自动跑剩余499步。

上述实现/测试尚未完成，不能把此目录视为可直接执行训练的发布包。

## 无卡身份预检与Git回传（不训练）

先在AutoDL仓库根目录执行。使用独立克隆，保留现有工作树；以下没有`exit`、`exec`或顶层`set -e`，失败不会主动关闭当前终端。若克隆失败不要运行后续代码；若目标目录已存在，保留它并检查，不删除重建。

```bash
cd /root/autodl-tmp/work/my_shopping_agent
REVIEW_DIR="$PWD/outputs/b500_protocol_review_v1"
git clone --single-branch --branch codex/b500-protocol-v1 "$(git remote get-url origin)" "$REVIEW_DIR"
```

确认克隆成功后，调用既有虚拟环境的Python，仅哈希文件和读取包元数据。哈希模型耗时取决于磁盘，不需要GPU：

```bash
PROJECT=/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon
REVIEW_DIR=/root/autodl-tmp/work/my_shopping_agent/outputs/b500_protocol_review_v1
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RECEIPT="$REVIEW_DIR/analysis/b500_matching_20261117/cloud_preflight/$STAMP.json"
"$PROJECT/.venv/bin/python" "$REVIEW_DIR/tools/check_b500_history.py" \
  --asset-project "$PROJECT" \
  --runtime-project "$PROJECT/outputs/bounded_resume500_to650/runtime" \
  --output "$RECEIPT"
PREFLIGHT_STATUS=$?
printf 'PREFLIGHT_EXIT_CODE=%s\nRECEIPT=%s\n' "$PREFLIGHT_STATUS" "$RECEIPT"
```

若该隔离runtime已不存在或哈希不匹配，只回传失败收据，不随意换当前主工作树继续。预检脚本验证原veRL0.8.0/vLLM0.25.1及绑定的关键源码；未独立封存的依赖/GPU条件仍在历史限制中，包名版本相等不能代替源码身份。

无论通过与否都回传收据，不启动GPU：

```bash
cd /root/autodl-tmp/work/my_shopping_agent/outputs/b500_protocol_review_v1
RETURN_BRANCH="codex/b500-preflight-results-$STAMP"
git switch -c "$RETURN_BRANCH"
git add -- analysis/b500_matching_20261117/cloud_preflight
git commit -m "Return B500 CPU history identity preflight"
git -c http.version=HTTP/1.1 push -u origin "$RETURN_BRANCH"
printf 'RETURN_BRANCH=%s\nRETURN_COMMIT=' "$RETURN_BRANCH"
git rev-parse HEAD
```

只有push成功才表示远端已有结果；网络失败保留原分支/提交并重试push，不重跑模型或删除运行目录。CPU身份通过不代表B实现通过，也不自动开始训练。
