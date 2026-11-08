# r6：数量作用域、量纲与证据处理修复

本版检查原1000题已保存的Flash/r5回答，固定六个语义维度与schema字段枚举，不接入训练、不调用API。旧r5代码、请求、回答和报告保持可复现。

## 实现

- tools/semantic_coverage_v6.py：操作/身体计数不自动当订单数量；补台与入手/推荐等已有明确数量表述的检查；同一主商品两个不同精确订单数不能同时放行。用户确认“推荐＋数量”只提示审阅，不强制补成订单数量；无法确定“配置”是在指主商品还是附件时，仅提示审阅，不强迫模型补成某一角色。
- tools/semantic_repairs_v6.py：只使用原文里已有的唯一上下文定位重复证据，或补同一分句明确出现的主体证据；无法唯一定位仍失败。存储/百分比写count时，仅在数字与原文字面单位对应时改为原文text并加unsupported_unit，不猜GB/Gbit，不补缺失事实。
- 用户确认的“裸寸”政策：原文数字+寸没有英寸说明时保留原文、单位待审。部件名称里的裸寸也记录待审但不扁平化组件结构。“尺寸”一词本身不触发该规则，明确英寸保持不变。
- tools/semantic_scope_v6.py：无充分证据的附件数量等启发式保留审阅提示，不能因别处有尺寸数字就认定附件数量有依据；不靠删除用户明确数量通过检查。
- tools/semantic_contract_v2_r6.py：独立检查/归一化后端。所有证据和表示调整写入audit，所有合同仍pending。

## 离线使用

```powershell
& "D:\shopping_agent\.venv-llm\Scripts\python.exe" "D:\shopping_agent\tools\replay_semantic_v6.py" --source-run "D:\shopping_agent\outputs\semantic_school_train1000_r5_v1_collect" --output "D:\shopping_agent\analysis\semantic_r6_replay_20261106\another_probe"
```

output必须是新的派生目录。已完成的正式回放在analysis/semantic_r6_replay_20261106/confirmed；v1、final和release是修复过程中的探测，不作为最终结论。

重放保留原实际选中回答，只有它未通过且早期已有可用回答时，才在派生结果里另选，并记录selected_changed_ids与行号；原始回答不变。source_attempt_line、audit和代码/输入哈希可供追溯。程序通过数不等于语义准确率，更不是新模型成绩。

复核派生文件：

```powershell
& "D:\shopping_agent\.venv-llm\Scripts\python.exe" "D:\shopping_agent\tools\check_semantic_contract_v6.py" "D:\shopping_agent\analysis\semantic_r6_replay_20261106\confirmed\responses.normalized.jsonl" --mode 1000 --allow-partial
```

此版是离线检查入口，不会让旧r5 API运行静默升级。若后续重新调用LLM，需要独立准备新协议与新运行目录。本轮不需要重跑1000题API。

## 测试

r6专用23项测试：数量角色、矛盾购买数、明确购买数的覆盖、身体部位商品的反例、真实组件数、重复/缺失/跨对象证据、存储与百分比、裸寸/英寸/尺寸一词、方向词与真近似、不修改传入回答。另有25项r5兼容测试已通过。真实API调用为0。
