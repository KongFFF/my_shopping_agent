# 任务与评分契约修复 v1

本版本处理原训练集90道静态异常，并将通用规则重新应用到全部1000题。它是独立的候选实验版本，不覆盖原始基线，也没有启动训练。

## 处理结果

| 范围 | 修复/保留 | 隔离待审 |
| --- | ---: | ---: |
| 原90道异常 | 28 | 62 |
| 原910道可静态满分任务 | 865 | 45 |
| 新训练清单 | **893** | **107** |

“可静态满分”只证明目标/评分可达，不代表语义标注完全正确。45道新增隔离中，31道依然可满分，但预算含“出头/多元”、下限或单价等当前无法可靠支持的语义；8道在正确提取预算后目标超预算，6道在补上预算后暴露价格不可核验。它们不都等于坏数据，因此单独保留审阅证据。

原90题中，28道恢复涉及15道实际预算变化、6道品牌角色修正、10道规格字符恢复，存在重叠，不能相加。浮点尾数差异不计为预算变化。全1000题共修复13道规格标注，其中3道仍因其他原因隔离。

## 实际修改

1. **预算解析**：统一元/块/块钱，支持中文数值、价格区间、总价优先于工费/单价；“左右/上下”继续采用原协议10%容差，明确“以内/不超过”保持硬上限。不会为了让gold过关加大容差。
2. **品牌语义**：“兼容/适用飞利浦”“类似苹果设备”“苹果专卖店外墙”等辅助对象不再自动解释为商品制造品牌；直接要求品牌的语句仍保留品牌约束。
3. **损坏标注**：只在目标商品存在唯一匹配、差异仅为`?`替代心形/星形/加号等已知字符时，恢复`instruction_options`。不对型号、数字、颜色或任意文字作模糊匹配。
4. **保守隔离**：不猜测多规格轴组合价格，不取最低价，不改用户预算/商品价格，不删除不满足的属性来制造满分。模糊表达保留旧运行时行为，同时从候选训练清单隔离。

原商品顺序、自然语言Query、商品标题、商品属性、可选规格、价格都保持不变。只修改部分训练任务的私有规格标注，并替换两份通用解析代码。全部23421件商品的检索文档指纹保持一致；重新构建索引仅为绑定新商品文件哈希。

## 文件

- `original_90_review.md`：90题可读处理表。
- `original_90_review.jsonl`：90题完整前后差异、Query与奖励证据。
- `all_train_audit.jsonl`：全1000题审计。
- `quarantine.jsonl`：107题隔离原因，原始数据未删除。
- `train.jsonl`：893个保留task_id；`validation.jsonl`保留原50个ID。
- `annotation_overrides.json`：13题精确标注修复及恢复依据。
- `runtime/`、`runtime.patch`：两份修复源代码与相对固定上游的差异。
- `manifest.json`：内容哈希、任务边界及新旧商品哈希。
- `validation_audit.json`：新旧规则对50道Dev目标选购的静态影响，不改验证集名单。

Dev原先48道可静态满分，修正两道“块左右”的解析后，又发现19430此前遗漏“100元左右”的预算，目标超过110元上限；新规则下49道可静态满分。**Dev仍保留50题，不删除19430，也不修改其Query/价格来让它成功。** 将来比较模型时必须使用同一套评分规则，另报这一已知矛盾；本版本没有读取Final Query或结果来调规则。

## 无卡重建与检查

在父仓库执行（本地可用`python`；AutoDL推荐已固定依赖的GRPO虚拟环境Python）：

```bash
upstream/shopping-grpo-longhorizon/.venv/bin/python tools/build_task_contract_v1.py
upstream/shopping-grpo-longhorizon/.venv/bin/python tools/test_task_contract_v1.py -v
```

数据文件生成与已有工作区相互独立；原始子模块应保持原状态。本地已完成8项回归测试，覆盖语义边界、品牌角色、字符唯一性、manifest和划分隔离。

要生成可供下一轮实验使用的parquet、商品库和独立环境源码副本：

```bash
upstream/shopping-grpo-longhorizon/.venv/bin/python tools/build_task_contract_v1.py \
  --materialize outputs/task_contract_v1_final

upstream/shopping-grpo-longhorizon/.venv/bin/python tools/verify_task_contract_v1.py \
  outputs/task_contract_v1_final
```

物化需要PyArrow25+。本机系统Arrow19无法正确读取上游parquet，已在忽略的outputs目录隔离使用Arrow25验证，没有改系统Python或云端依赖。物化目录若存在会拒绝覆盖，重新验证可直接运行verify。

物化产物约149 MiB，全部在被Git忽略的outputs下，包括：

```text
outputs/task_contract_v1_final/
  train.parquet                 # 893题，保留原schema与所选行完整内容
  validation.parquet            # 原50题文件原样复制
  environment.json              # 新数据及运行时指纹
  manifest.json
  materialization_receipt.json
  index_manifest.json
  runtime/
    web_agent_site/engine/      # 修复版解析函数
    shop_env/pack_api.py
    data/items_eval_train.json
    search_engine/products.sqlite3
```

本地从实际物化目录独立导入运行时代码，复算893题全部得到gold；parquet任务集合、schema、代码/商品库哈希、索引绑定均已验证。没有启动HTTP环境、模型或GRPO。

## 下一轮训练前的边界

不能只换`train.parquet`、却继续使用原始环境；修复版任务列表、商品标注、解析代码和环境清单是一个整体。也不要对基线目录直接执行覆盖或把新清单写回原`data/grpo/`。

后续应从物化的`runtime/`启动单独环境，绑定其索引和配置，并显式指定新的训练parquet与环境清单。实际启动前还需要在AutoDL做CPU环境回放/服务冒烟，确认使用的是该副本；本次仅完成本地静态及物化验证，未声称云端已部署。正式GRPO仍从同一SFT step300起点建立新运行目录。

这是一份保守的第一版：原62题和新增45题都可继续凭证据修复后重新纳入，不以修改答案或降低判分标准换取数据量。

## 隔离题初判

全部107题的助手初判见[first_pass_review.md](first_pass_review.md)，完整原文、规格价格和逐题建议见[review_cards.md](review_cards.md)。结构化结论在[first_pass_review.jsonl](first_pass_review.jsonl)，来源哈希与分类计数在[first_pass_summary.json](first_pass_summary.json)。这是独立审阅记录，不改变本包893/107划分或运行时；新增文档不属于v1训练工件。

在仓库根目录执行以下两条命令可重新生成审阅文档，无需模型或PyArrow：

```bash
python tools/review_task_contract_v1.py
python tools/make_task_review_cards.py
```

生成器检查107题覆盖、原文引文、候选规格存在性、v1所有工件哈希；完整SKU价格或统一语义仍缺证的结论不会自动应用。
