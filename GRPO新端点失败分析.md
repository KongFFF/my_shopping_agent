# GRPO新端点失败分析

2026-11-15。用户要求先完成新端点失败分析，暂不实践有效组保留或启动重训。

## 当前状态

本地已确认缺少原GRPO750/800独立Dev原始轨迹、完整摘要及收据；现有文档中的27/50和29/50来自历史用户回传，不能替代逐题证据。尚未完成真实新端点失败分析，不能输出其错误分类数字或直接沿用500的瓶颈结论。

已准备标准库导出器和离线分析器。7项本地测试通过，包括只读源文件检查、与既定strict定义核对、重复任务、摘要不一致、协议差异、query差异、失败收据以及打包哈希篡改拦截。测试使用临时目录中重标名称的历史500轨迹，仅验证工具，不是750/800实验结果。

## 从AutoDL取回已有结果

用户指定后续优先采用“AutoDL → Git仓库 → 本地”的同步方式。工具已准备通过独立分支`codex/endpoint-dev-analysis`同步；结果使用独立回传分支，不上传大模型或整份outputs。

在AutoDL无卡实例复制执行下列整块命令。先拉取指定分支，只恢复导出脚本，然后生成证据包；回传提交在新建的共享对象检出中完成，不切换原工程分支，也不混入原工程暂存区或其他改动。

```bash
(
set -euo pipefail
cd /root/autodl-tmp/work/my_shopping_agent
git fetch origin codex/endpoint-dev-analysis
CODE_COMMIT=$(git rev-parse FETCH_HEAD)
git restore --source="$CODE_COMMIT" --worktree -- tools/export_endpoint_dev.py

EXPORT_LOG=$(mktemp)
python tools/export_endpoint_dev.py | tee "$EXPORT_LOG"
ARCHIVE=$(sed -n 's/^ARCHIVE: //p' "$EXPORT_LOG")
test -n "$ARCHIVE"
test -f "$ARCHIVE"

RETURN_DIR=$(mktemp -d /root/autodl-tmp/endpoint-dev-return.XXXXXX)
RETURN_BRANCH="codex/endpoint-dev-results-$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_URL=$(git remote get-url origin)
git clone --shared --no-checkout . "$RETURN_DIR"
git -C "$RETURN_DIR" remote set-url origin "$REMOTE_URL"
git -C "$RETURN_DIR" switch -c "$RETURN_BRANCH" "$CODE_COMMIT"
for KEY in user.name user.email; do
    if VALUE=$(git config --get "$KEY"); then
        git -C "$RETURN_DIR" config "$KEY" "$VALUE"
    fi
done
mkdir -p "$RETURN_DIR/analysis/endpoint_dev_return_20261115"
cp "$ARCHIVE" "$RETURN_DIR/analysis/endpoint_dev_return_20261115/evidence.zip"
git -C "$RETURN_DIR" add -- analysis/endpoint_dev_return_20261115/evidence.zip
git -C "$RETURN_DIR" diff --cached --stat
git -C "$RETURN_DIR" commit -m "data: return existing step750 and step800 independent Dev evidence"
git -C "$RETURN_DIR" push -u origin "$RETURN_BRANCH"
printf '\nRETURN_BRANCH=%s\nRETURN_COMMIT=%s\n' "$RETURN_BRANCH" "$(git -C "$RETURN_DIR" rev-parse HEAD)"
)
```

只读来源是父项目下：

```text
upstream/shopping-grpo-longhorizon/outputs/runs/grpo_baseline_20261022_202816/
  evaluation/step750_independent_v1/dev50/
  evaluation/step800_independent_v1/dev50/
```

每份读取`trajectories.jsonl`、`summary.json`、`passed.json`、`exit_status.json`，另读取摘要中绑定的原Dev50任务文件。检查两端点各50个唯一任务、严格成功集合、query、模型名称、协议和完成收据；源文件读取前后内容保持一致。

输出父项目`outputs/endpoint_dev750_800_<时间戳>.zip`，终端打印`ARCHIVE:`与`SHA256:`。随后仅将此zip复制到独立回传检出的`analysis/endpoint_dev_return_20261115/evidence.zip`并提交推送。完成后将`RETURN_BRANCH`和`RETURN_COMMIT`两行告诉助手，由助手拉取并核验。不存在文件或校验失败时保留报错，不重新运行评测pipeline。

回传检出使用本机Git对象共享，不复制模型权重、不初始化子模块，也不完整复制outputs；在成功推送前保留它。若Git身份或推送鉴权失败，保留报错与该目录，不通过反复运行评测解决。脚本本身只读原始证据，新增zip及Git回传属于此轮明确授权操作。

此操作不需要GPU，不启动模型、环境、训练或评测，不访问Final、权重、密钥或API，不覆盖原产物。文件中的收据和模型名称不能替代重新加载权重的验证。

## 本地材料齐全后的分析

```powershell
python tools/analyze_endpoint_dev.py outputs/实际导出文件名.zip
```

分析器核对zip成员和SHA256，将原SFT/500冻结快照与750/800做同口径比较，复算完整摘要和严格成功定义，核对全部task_id、原始Actor query及除模型名外的推理协议。输出到新的`analysis/endpoint_dev_<时间戳>/`，不覆盖已有分析。

输出：

- `audit.json`：来源、指纹、完整性、复算和协议检查。
- `summary.json`：四模型的原奖励类型及失败阶段代理统计。
- `task_pairs.json`、`transitions.json`：逐题成败、持续失败、新修复与退化。
- `failure_review_cards.json`：失败轨迹的公开观察、动作、守卫拒绝、原奖励诊断和待审状态。

自动阶段按互斥规则记录：成功；买到原标注目标但未成功；打开目标后购买其他商品；打开目标但没有完成购买；目标可见但未打开；本轨迹未观察到目标。目标ASIN和原评分细节仅用于离线诊断，不能作为Actor输入，也不等同于真实语义真值。

## 人工式逐例审阅与最终交付

拿到材料后继续由助手逐例阅读，不把自动代理标签当因果结论：

1. 对750/800的失败及500→端点成败变化，定位公开需求、关键可见证据、实际动作和终止原因。
2. 区分候选未发现、候选未利用、明确规格/数量/预算决策错误、循环/恢复/终止错误、信息不足以及任务/评分契约疑点。可以多标签，不能相加当互斥人数。
3. “目标没有出现”仅描述这一条轨迹，不证明检索不可达；“目标打开过”不自动证明信息充分；其他商品也可能满足用户需求。
4. 对原评分的功能/规格扣分核对原需求与可见SKU；占位单值轴、隐藏选项、预算口径冲突等单列，不修改原成功指标，不把未知判失败。
5. 明确哪些500时期的瓶颈已经缓解，哪些在750/800仍有证据；据此更新研究候选，不能预设必须回到分叉、奖励工程或有效组保留。

最终报告应给出四模型对照、配对变化、可追溯案例与研究建议。Dev50已反复使用，审阅结果是开发诊断，不是新的独立泛化确认；本轮不继续逐例研究Final。
