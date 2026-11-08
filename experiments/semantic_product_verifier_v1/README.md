# 商品核验首版：离线、来源可追溯

入口是 `tools/verify_semantic_products.py`，核心为 `tools/semantic_product_verifier.py`。输入最新契约、真实商品公共目录、实际所选规格；输出逐项pass/fail/unknown及来源。不会把原标注ASIN或选项当真值，不会把未选SKU当买到的内容。

`catalog_facts.schema.json`固定商品事实格式。事实独立于用户需求，必须绑定商品ASIN和公共字段哈希，说明product或sku作用域；SKU事实还要绑定完整选项字典。证据为商品JSON指针和值，或精确文字区间。仅已审阅事实可进入比较；复杂语义事实不能只靠“原文存在这个字符串”自动获批。

`complete=false`表示非穷尽事实集合，没列出的项目不自动判不存在。多项功能、材料等正向声明不互相排斥；如需确认不具备某能力，用经审阅的negative声明，或完整集合。评论及instructions不允许作为证据。

训练题的整题审批与商品事实审批分别进行。此版本没有为1000份草稿批量填approved，未决需求、未知事实、无效选择及购买未完成不放行奖励。未知返回reward=null，尚不能直接接入要求浮点奖励的训练接口。完整选项不代表已经执行购买；旧轨迹默认只做比较，调用方需由执行事件明确提供purchase_executed=true才可能放行，不能仅根据旧reward_type推断，以免把购买保护耗尽当成成交。

PowerShell示例（在仓库根目录执行，全程CPU，无需密钥）：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\verify_semantic_products.py `
  --contracts .\analysis\semantic_closeout_20261107\validated1000.jsonl `
  --products .\upstream\shopping-grpo-longhorizon\environments\ShopSimulator\shop_env\data\fine_items_eval_train_all.json.gz `
  --purchases .\path\to\purchases.jsonl `
  --output .\outputs\semantic_product_check_v1
```

purchases每行格式：`{"task_id":53,"asin":"实际商品ASIN","selected_options":{"实际轴名":"实际选值"}}`；也可提供包含terminal_result.purchase的既有轨迹。若对应任务没有契约或没有购买快照，会单独报出，不替换成目标商品。输出目录必须未存在。

仅为测试，可把purchases路径替换为 `analysis/semantic_product_verifier_20261107/train20_target_probes.jsonl`；这19份是原目标规格的目录探针，不能当模型评测成绩。

23项单元测试及真实商品冒烟结果见[收尾报告](../../analysis/semantic_closeout_20261107/收尾与商品核验报告.md)。目前事实覆盖很有限，不能据此宣称已完成全量商品语义判定或可以开始正式GRPO。
