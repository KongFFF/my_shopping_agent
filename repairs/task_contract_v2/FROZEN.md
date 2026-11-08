# 928题与v2评分契约冻结记录

状态：**FROZEN**。冻结日期：2026-10-29（Asia/Shanghai）。冻结编号：`task-contract-v2-928-20261029`。

本地基准副本：`outputs/task_contract_v2_frozen/`。原`outputs/task_contract_v2/`保留为较早候选，后续使用冻结副本或由同一冻结锁重建的新副本。

## 已固定内容

| 内容 | 固定方式 |
| --- | --- |
| 训练928题、隔离72题 | 精确任务ID、划分文件及审计工件SHA256 |
| 原Dev50 | 名单及parquet保持原样；不按新评分筛题 |
| 原Final任务ID | 仅绑定固定划分，不参与训练、选模或规则拟合 |
| v2预算、下限、区间及标注规则 | 三份运行时替换代码、私有标注修复与manifest SHA256 |
| 比较器、规格定价、品牌别名等依赖 | 运行目录全部相关依赖文件的归一化SHA256 |
| 重建来源 | 原始数据、v1依赖、生成/校验工具及基线环境源码的哈希 |

“出头”以最终确认规则固定：两位金额加4，三位加9，四位加99，五位加999，依此类推；只设上限。普通“左右/上下”仍维持10%，硬上限不放宽。

冻结锁：[freeze.lock.json](freeze.lock.json)。建立锁时的完整验证结果：[freeze_receipt.json](freeze_receipt.json)。

## 冻结后的行为

- 只重建完全相同的工件时，构建器校验通过，不重写冻结包；可以物化到另一个不存在的目录。
- 修改任务划分、标注、规则、相关依赖或重建输入，会导致冻结校验失败；构建器在写入前拒绝覆盖。需要变化时另建契约版本并保留v2。
- 后续采样利用率、GRPO更新与规格决策训练方案使用独立代码/配置，继续共用这928题和v2契约。实验间固定同一评测口径。
- 冻结采用哈希与写入保护，不使用文件系统只读属性；直接手工修改仍会被后续校验识别。

## 核验与重建

在父仓库根目录执行快速检查：

```bash
python tools/freeze_task_contract_v2.py --check
```

在具备PyArrow≥25的环境，对实际副本做完整CPU核验与928题回放：

```bash
python tools/freeze_task_contract_v2.py --check --bundle outputs/task_contract_v2_frozen
```

云端首次重建（目标目录必须不存在）：

```bash
python tools/build_task_contract_v2.py --materialize outputs/task_contract_v2_frozen
python tools/freeze_task_contract_v2.py --check --bundle outputs/task_contract_v2_frozen
```

冻结锁随`repairs/`小工件同步；大数据由固定原始输入重建。同步时也须包含父仓库`.gitattributes`，防止Windows换行转换破坏工件字节。云端已有目标目录时换一个新目录名，并将同一路径传给校验器。

本次完成本地冻结和CPU校验，未部署AutoDL、未启动模型/环境服务、未开始GRPO。后续训练前仍需云端接口冒烟和单独的训练配置。
