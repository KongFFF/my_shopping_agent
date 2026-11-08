# 商品核验v3：确认政策适配

使用final12契约；旧v1/v2保持冻结。旧164条已审商品事实可继续使用。新增的受控事实为composition/member_groups的member_values（实际成员数值列表），以及transaction_terms下sale_quantity、sale_unit_quantity（SKU实际量与单个销售单位对应的量）。

比较器按一对一成员分配核验各档范围，不能用同一个成员满足多个档。数量、实际销售单位与价格均取商品证据，不由原需求或起订量反推。未开刃只有明确商品状态才算满足；金属材质/金色不等于金属质感。

CLI：tools/verify_semantic_products_v3.py。契约：analysis/semantic_final12_20261107/validated1000.jsonl。

商品事实提取：tools/extract_product_facts_v3.py。使用新输出目录，例如：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_product_facts_v3.py --mode 19 --output .\outputs\product_fact_prepare19_v3 --prepare-only
```

实际调用仍沿用LLM_BASE_URL、LLM_MODEL、LLM_API_KEY；学校HTTP需--allow-school-http。LLM输出只能pending，程序检查不是语义审批。本轮仅准备本地输入，未请求API。勿向旧v2输出目录追加v3运行。

[本轮政策及验证](../../analysis/semantic_final12_20261107/12题政策落实报告.md)。本轮关闭固定12题的指定口径，未完成全商品事实覆盖或正式GRPO接入。
