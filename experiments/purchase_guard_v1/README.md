# 缺规格购买保护 v1：原始 1000 题实验

本轮只改购买交互协议，训练仍使用上游原始 1000 题，Dev 仍使用原始 50 题。原始商品库、预算解析、评分函数和奖励数值保持不变；不使用 `task_contract_v2` 的 928 题及其评分修复。928 题冻结包继续保留。

## 行为

1. 原价格解析器确认：商品有一个影响价格的规格轴，而模型尚未选择它。此时拦截 `buy_now`，不生成订单，不结束任务，在结构化 observation 中提示缺少的规格轴名称。
2. 模型可以补选规格后购买，也可以返回搜索。提示不给出正确选项，不读取目标商品、标准答案或隐藏评分标注。
3. 同一条轨迹累计前两次不完整购买均可继续；第三次以有效 `wrong_purchase` 失败结束，使用现有配置的失败奖励（当前为 `-0.85`），`reward_valid=true`、`sampling_invalid=false`。不记录实际购买。
4. 返回搜索、切换商品或执行其他动作不会重置计数；新任务重置计数和历史。原有步数、循环和无进展限制仍生效，可以先于第三次尝试结束任务。
5. 如果选齐后价格仍无法核验，或多轴定价缺少组合价格等商品元数据不足，仍走原来的 `reward_unverifiable` 流程。本版不把元数据问题转成模型失败，也不强制选择所有与价格无关的轴。

重试耗尽属于本版新增的有效失败判定，正常购买的原评分代码未改。它使这类轨迹不再仅因漏选规格而自动令整组失效；如果组内依然同分，或另有无效轨迹，原动态采样过滤仍然适用。修复后能否提高模型成绩，需要下一阶段同一 SFT 检查点的评测和 GRPO 实验检验。

## 文件与隔离

- `tools/purchase_guard_v1.py`：只依据商品规格和价格元数据的检查、提示和失败结果。
- `tools/build_purchase_guard_v1.py`：从原始上游生成独立运行目录，构建原商品库的搜索索引。
- `tools/verify_purchase_guard_v1.py`：检查源代码、原始数据、商品库、索引和运行时清单的一致性。
- `tools/test_purchase_guard_v1.py`：CPU 行为和接口测试。
- 本目录 `enabled/`、`disabled/`：可审查的源码覆盖文件、`runtime.patch` 和构建清单。disabled 对照关闭购买拦截。

覆盖代码涉及服务端购买分支与任务重置、环境结构化 observation、Agent 共用 observation 渲染、GRPO 工具轨迹日志和 agent loop 诊断字段。无需改变工具 schema、模型权重或 GRPO 优化算法。

上游子模块和冻结修复包没有被覆盖。完整运行目录、商品库和 SQLite 索引位于 Git 忽略的 `outputs/` 下，不需要 push。将本目录的小型覆盖文件和上述工具脚本同步到 AutoDL 后，可从原始上游重建。

## 构建与 CPU 验证

从父仓库根目录运行，使用原项目已安装的 Python 环境。首次构建两个目录；构建器拒绝覆盖已有目录，应使用新的输出目录保存新版本。

```bash
python tools/build_purchase_guard_v1.py --materialize outputs/purchase_guard_v1
python tools/build_purchase_guard_v1.py --materialize outputs/purchase_guard_v1_disabled --disabled
python tools/verify_purchase_guard_v1.py outputs/purchase_guard_v1
python tools/verify_purchase_guard_v1.py outputs/purchase_guard_v1_disabled
python -m unittest discover -s tools -p test_purchase_guard_v1.py -v
```

测试默认读取上面两个输出目录。可通过 `PURCHASE_GUARD_BUNDLE` 环境变量更改 enabled 目录；disabled 对照测试读取默认的 `outputs/purchase_guard_v1_disabled`。

本地验证结果（2026-10-30）：20 项新测试通过，包括真实 HTML 渲染、真实规格选择、购买成功、重试耗尽、全局步数限制、跨商品计数、新任务重置、评测工具调用、GRPO 工具调用和动态采样过滤。新旧数据/评分一致性检查通过，两套 bundle 各检查 75 个 Python 文件。已有任务契约的 35 项测试通过，928 题冻结锁校验通过。

这些测试使用小型内存商品夹具，不启动模型或 HTTP 服务；本机缺少 Gym 时仅在测试进程中替代其 Env 基类，商品库初始化和 NLP 模型加载不在本次测试覆盖范围内。尚未执行 AutoDL 的真实服务启动、SFT 重评测或 GRPO 重训。

## 后续运行时的路径约束

构建后的 `outputs/purchase_guard_v1/` 包含：

| 路径 | 用途 |
| --- | --- |
| `agent/shopping_grpo/` | 本实验 Agent 代码；训练与评测都须加载这里的包 |
| `environment/` | 本实验环境；服务应从这里启动，不能连到旧服务 |
| `environment/data/items_eval_train.json` | 原商品库的解压副本 |
| `environment/search_engine/products.sqlite3` | 与该商品库绑定的索引 |
| `data/train.parquet`、`data/train.jsonl` | 原始 1000 题，逐字节复制 |
| `data/validation.parquet`、`data/validation.jsonl` | 原始 Dev-50，逐字节复制 |
| `environment.json` | 本实验运行时哈希与开关清单 |
| `manifest.json` | 原始数据哈希、开关、重试次数、覆盖文件哈希 |

后续启动命令必须同时绑定本实验环境、Agent 包和 `environment.json`，并先检查导入路径和服务身份。不要只更换训练 parquet，也不要把此环境与 `outputs/task_contract_v2_frozen` 的评分运行时混用。
