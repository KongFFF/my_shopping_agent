# 商品事实与核验v2

首批19商品事实、原12题收尾与实际SKU奖励对照见[本轮报告](../../analysis/semantic_product_stage_20261107/收尾与商品事实报告.md)。v1和上一轮报告冻结；v2只供离线诊断，未接入GRPO。

事实包为人工依据公共目录独立整理的数据，不读取需求或正确选项。SKU事实可用selector_mode=contains限定必要的轴；只有该轴确实选中、证据指向同一选项时才适用。公共事实必须对所有SKU成立，不能从列举不同款式的标题借值。

`tools/verify_semantic_products_v2.py`沿用首版CLI参数，使用当前 `analysis/semantic_product_stage_20261107/validated1000.jsonl`，并增加传入 `--catalog-facts analysis/semantic_product_facts_v2_20261107/catalog_facts.jsonl`。旧版契约不能伪造版本直接送进v2。

独立商品事实提取已固定prompt和schema；没有调用API。程序通过的输出仍是pending，须经语义审阅后才可进入已核验事实缓存。先做19商品中的少量请求，勿直接跑完整目录。

PowerShell用法（仓库根目录，沿用已有LLM_BASE_URL、LLM_MODEL、LLM_API_KEY环境变量；不要把密钥写入脚本或聊天）：

```powershell
# 本地准备，不发送任何请求；HTTP地址无需例外开关
.\.venv-llm\Scripts\python.exe .\tools\extract_product_facts.py `
  --mode 19 --output .\outputs\product_fact_prepare19_v2 --prepare-only

# 实际请求，学校HTTP网关须显式开启；先限2个商品
.\.venv-llm\Scripts\python.exe .\tools\extract_product_facts.py `
  --mode 19 --output .\outputs\product_fact_prepare19_v2 `
  --allow-school-http --limit 2
```

如环境变量未保留，可重新设置非密钥部分：

```powershell
$env:LLM_BASE_URL = 'http://aigw.dlut.edu.cn/v1'
$env:LLM_MODEL = 'DeepSeek-V4-Flash-0731-W8A8'
```

API密钥仍放在本机LLM_API_KEY环境变量。默认不发送thinking参数；固定temperature=0。401/403或首个失败会停止，不自动烧完全部请求。原回答记录在attempts.jsonl，程序通过的pending事实输出draft_facts.jsonl。恢复运行会检查输入与配置一致，并重新验证旧回答。

mode=1000是原1000训练题对应商品；mode=catalog是23421个公开候选商品。后者只包含公开商品字段，不包含Final任务文本，但不应在小样本验证前全量调用。若模型买了事实缓存以外的商品，会返回未知，不能因此判错误，也不能退回按ASIN判分。训练接入前需解决实际候选覆盖。

准备文件位于Git忽略的outputs中。catalog范围约有300MB本地请求文件，属于可重建产物，不提交仓库。
