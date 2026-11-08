# 已敲定的任务与评分契约 v2

本版本实现用户选择的1A、修改后的2B、3A、缺证直接隔离、5A与6A。没有改变GRPO更新、采样、奖励数值或模型权重。

## 冻结版本

冻结编号：`task-contract-v2-928-20261029`。928题训练、72题隔离、原Dev50和全部v2评分规则以`freeze.lock.json`为准。后续采样/训练算法实验共用这一版本；数据划分、标注或评分契约变化必须另建版本。

冻结锁同时绑定评分入口、比较器、规格定价、品牌别名等依赖。构建器在写入前校验：完全相同的重建可以物化到新目录；产生不同工件时拒绝覆盖冻结包。

## 已固定规则

- 明确价格下限、上限与数值区间在实际购买和候选可接受性判断中都生效。高于250、不超300对应`250 < price <= 300`。
- 两百五=250、一百六=160、一千七=1700、六千二=6200，与显式零区分；明确价格语境中可以省略元。
- **出头采用用户最后确认的规则**：两位整数金额加4；三位加9；四位加99；五位加999，依此类推。公式为位数d≥3时加`10^(d-2)-1`。只生成包含边界的上限，不设最低消费，不叠加左右容差。
- 例：20出头≤24、65出头≤69、100出头≤109、420出头≤429、1000出头≤1099、2000出头≤2099、10000出头≤10999。未定义的个位或非整数出头表达保持待审。
- 多元按金额档位解释，80多的上边界为90且不包含90；预算80多不要求至少花80。九百多按900至1000的开放范围，价格范围与预算意图分别处理。
- 十几采用11至19，四十几采用41至49；三四十采用30至40，五六十采用50至60，三百四五十采用340至350。预算语境只取上限，明确售价范围检查两端。这些口语范围不再叠加10%。
- 普通X元左右/上下沿用原有10%上限容差。明确硬上限优先，不因大概/差不多等词放宽。
- 单价只在批准的1米购买/报价单位可核验时支持；购买时选中的公开规格须提供1米证据。其他单位、工费、定金/尾款等缺证条件保持不可核验并隔离。

## 隔离与标注

35道主类缺证题直接隔离，不改价格、不查实时网页、不补猜测的完整SKU表。已知次要价格/规格缺证也保持隔离，见`manual_holds.json`。总价通过不能代替工费、材质或套餐组成等条件通过。

字符恢复仅改训练任务私有`instruction_options`：完整字符串除批准符号或末尾分号外一致，且同商品只有一个匹配。²还需原文面积依据。模型运行时的规格比较仍为严格匹配，不把型号、颜色、数字设为通配。

7714、21254的需求冲突保留隔离。没有修改用户query或根据目标价格提高预算；没有把无解题改成新的停止购买任务。

## 结果与证据

- `manifest.json`：版本、哈希、划分、恢复和新增隔离ID、用户确认的预算政策。
- `all_train_audit.jsonl`：全部1000题的v1/v2结果、解析依据、实际奖励证据与隔离原因。
- `quarantine.jsonl`：v2隔离任务；隔离保留原任务ID，不删除商品目录。
- `annotation_overrides.json`：私有标注修改前后及来源。
- `validation_audit.json`：原Dev50的静态标注检查，名单不筛选；不代表模型成功率。
- `runtime/constraints.py`、`runtime/reward_features.py`、`runtime/reward.py`及`runtime.patch`：基于固定上游提交的完整替换代码和补丁。
- `changes.md`、`remaining_review.md`：变化索引与剩余隔离题的可读证据。

## 重建与验证

在父仓库根目录运行。仅生成CPU审计、数据和索引，不启动模型或GRPO。

```bash
python tools/build_task_contract_v2.py
python tools/build_task_contract_v2.py --materialize outputs/task_contract_v2_frozen
python tools/freeze_task_contract_v2.py --check --bundle outputs/task_contract_v2_frozen
python -m unittest discover -s tools -p test_task_contract_v2.py -v
python tools/report_task_contract_v2.py
```

物化目录必须不存在；工具拒绝覆盖。若本机已有`outputs/task_contract_v2`，使用新的目录名并把相同路径传给验证器。物化需要PyArrow≥25；不要把本地依赖目录写成云端路径。

快速核验冻结小工件和重建依赖：`python tools/freeze_task_contract_v2.py --check`。含`--bundle`时再核验实际运行目录、数据/索引绑定并执行928题CPU奖励回放。本地正式冻结副本为`outputs/task_contract_v2_frozen/`，较早的`outputs/task_contract_v2/`是历史候选，后续实验使用冻结副本或按同一锁重建的新副本。

`outputs/`内的大文件已被Git忽略。本地使用的隔离PyArrow依赖在`outputs/task_contract_v1_deps`，仅当当前Python缺少可用PyArrow时，按本机实际路径配置`PYTHONPATH`；该路径不属于修复包。

父仓库`.gitattributes`对`repairs/`禁用Git换行转换，并将相关工具源码固定为LF。修复清单绑定文件的精确字节，Windows的`core.autocrlf=true`不能改写它们；同步修复包时应一并包含该属性文件。

物化内容包括训练parquet（只过滤原始行，保留schema与内容）、原样Dev50 parquet、修复版完整商品库、三份修复代码、独立检索索引及绑定清单。原始上游和v1包均保留。

不能只换训练ID/parquet而继续使用旧环境，也不能只替换前两份解析文件遗漏`reward.py`：旧购买门控不会核验下限。启动环境前必须绑定新运行时、商品库、索引与environment清单。云端尚需环境启动/接口冒烟，当前命令不代表已经部署AutoDL。

## 评测口径

Dev/Final名单保持固定，不按新规则筛掉难题。旧模型与新模型应在同一环境和评分口径下比较，单独记录评分器变化；不能将新解析带来的得分变化称作模型能力提升。当前未启动200题Final评测或重新训练。
