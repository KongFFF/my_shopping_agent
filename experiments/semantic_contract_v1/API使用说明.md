# 本地 API 离线需求提取

脚本：`tools/extract_semantic_requirements.py`。支持 OpenAI 兼容的 Chat Completions HTTP 接口；不依赖 OpenAI SDK。示例使用 DeepSeek 官方 API；其他提供商需提供兼容地址和模型 ID，通常还需加 `--thinking omit`。Anthropic 原生接口、Responses 原生接口不在本脚本范围。

`--mode 50` 使用冻结的 pilot50_v1；`--mode 1000` 使用原始训练集全部1000题。两种模式的 system prompt、user 模板和输出 schema 完全一致，原始 query 不改写。不是928题修复集，也不修改训练环境或实际奖励。

## 1. Windows PowerShell 安装

以下所有命令在 PowerShell 执行，不是在 CMD。无需 AutoDL 或 GPU，也无需激活虚拟环境。

```powershell
Set-Location D:\shopping_agent
python -m venv .venv-llm
.\.venv-llm\Scripts\python.exe -m pip install -r .\experiments\semantic_contract_v1\requirements.txt
$env:PYTHONIOENCODING = "utf-8"
```

若找不到 `python`，安装 Python 3.10 或以上；若已安装 Python Launcher，可将创建环境那条换成 `py -3 -m venv .venv-llm`。

## 2. 填 API 信息

将模型名与地址改成你实际开通的服务。下面是 DeepSeek 官方服务的示例；不要填写网页聊天地址。

```powershell
$env:LLM_BASE_URL = "https://api.deepseek.com"
$env:LLM_MODEL = "deepseek-flash"
$llmSecret = Read-Host "请输入 API Key（输入不回显）" -AsSecureString
$env:LLM_API_KEY = [System.Net.NetworkCredential]::new("", $llmSecret).Password
Remove-Variable llmSecret
```

API Key 在最后的交互提示中输入，不需要写进脚本。不要提交到 Git 或发到聊天中。变量只在当前 PowerShell 及其子进程生效，关闭窗口后要重设。如果已有别的密钥环境变量，可用 `--api-key-env YOUR_VARIABLE_NAME` 指定。

更换 Pro：`$env:LLM_MODEL = "deepseek-v4-pro"`，并为该模型使用新的结果目录，不与 Flash 混用。

其他兼容服务通常使用 `https://服务地址/v1`；脚本在末尾追加 `/chat/completions`。若已经填完整 HTTPS `/chat/completions` 地址不会重复追加。实际调用默认要求 HTTPS，不接受 URL 内嵌用户名、密码、查询参数或片段。

### 大连理工大学校园 HTTP 网关

学校网关请填完整基础路径 `/v1`，模型名填学校实际提供的模型 ID；密钥仍按上面的交互方式设置。

```powershell
$env:LLM_BASE_URL = "http://aigw.dlut.edu.cn/v1"
$env:LLM_MODEL = "替换为学校提供的模型ID"
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --output .\outputs\semantic_school_50_v1 --thinking omit --prepare-only
```

`--prepare-only` 和 `--check-only` 只做本地工作，不因 HTTP 被拒绝、不读取密钥、不发送请求。URL 内嵌凭据、查询参数和片段仍会被拒绝。

实际试跑时显式开启例外：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --output .\outputs\semantic_school_50_v1 --thinking omit --allow-school-http --limit 5
```

去掉 `--limit 5` 即续跑50题。扩展1000题：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 1000 --output .\outputs\semantic_school_1000_v1 --thinking omit --allow-school-http --reuse-from .\outputs\semantic_school_50_v1
```

`--allow-school-http` 只放行 `http://aigw.dlut.edu.cn/v1`（可带末尾斜杠），实际请求路径由脚本追加 `/chat/completions`。不放行其他 HTTP 域名、显式端口或其他路径，且不会跟随重定向。该选项不是生成参数，因此本地准备时不加、实际调用时再加可以使用同一目录。`--thinking omit` 避免把 DeepSeek 专用参数传给校园网关；如果网关不支持 JSON Object，需在新目录准备和调用时都显式加 `--response-format omit`。

脚本版本更新后，旧运行目录的 manifest 会拒绝混用新脚本；请使用新的学校专用目录，保留原目录不改。

## 3. 无费用预检查

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --output .\outputs\semantic_flash_50_v1 --prepare-only
```

只核对原训练集、冻结样本及 prompt 的哈希，生成运行快照，不发送网络请求，不需要密钥，但要设置服务地址和模型名。1000题也可同样加 `--prepare-only` 预检查。

## 4. 先试5题，再跑完50题

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --output .\outputs\semantic_flash_50_v1 --limit 5
```

每题一请求，顺序处理。先查看 `report.json` 的 `validated_tasks` 和 `usage_all_attempts`；有失败查看 `attempts.jsonl` 的 `error`。`--limit 5` 表示本次最多处理5道尚未通过校验的题，不是每次都重复前5题。部分完成退出码2是预期行为。

确认5题输出后，移除 limit：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --output .\outputs\semantic_flash_50_v1
.\.venv-llm\Scripts\python.exe .\tools\check_semantic_pilot50.py .\outputs\semantic_flash_50_v1\responses.jsonl
```

前5题通过校验的结果会跳过。50题完成后建议先做人工语义审阅，再决定是否扩展；结构合格不代表提取完整、语义正确，也不代表可以直接接入奖励。

## 5. 原训练集1000题

50题通过审阅后，用同模型、同参数扩展，并复用已通过程序校验的50题原始输出：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 1000 --output .\outputs\semantic_flash_1000_v1 --reuse-from .\outputs\semantic_flash_50_v1
```

复用不重复调用这些题；同一复用命令再次执行也不会重复写入。复用要求原数据、prompt、schema、模型及API参数一致。若不提供 `--reuse-from`，1000题会独立提取，包括50题样本。

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 1000 --output .\outputs\semantic_flash_1000_v1 --check-only
```

该命令重新核验运行快照、全部原始响应，重建 `responses.jsonl` 和 `report.json`，不调用 API；只有1000题均通过才退出0。复用的50题用量保留在来源目录，1000题报告不重复计算其 token。

## 产物与恢复

### v2证据坐标仍失败：精确定位与离线复用

2026-11-02第二轮六次响应中，证据文字都能在原文唯一定位，但出现end少算1或start偏移1的问题。模型语义仍需另审（3723仍将造型归为颜色）。不要再为纯坐标错误重复请求。

现有六次原始响应可纯离线生成可审阅的派生合同：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\normalize_semantic_evidence.py --source .\outputs\semantic_school_50_v2 --output .\outputs\semantic_school_50_v2_evidence_review
```

该操作不修改来源目录、不读密钥、不发API请求。输出 `responses.normalized.jsonl`、`normalizations.jsonl`、`normalization_source.json` 和 `report.json`；已有样本应显示3题坐标规范化后通过结构核验，原始直接通过仍为0。6次旧调用的用量在报告中保留，不代表本操作发起了请求。

精确定位规则：只修start/end；text必须是未经任何清洗、替换的原文连续片段；若现有坐标已准确则保留；若坐标有误，只有该text在原文唯一出现才定位。无匹配、多个匹配、类型非法、原query/ID被改写时仍拒绝。不改变类型、需求值、偏好、预算或审阅状态。每项变更记录原位置、新位置、原始响应哈希和来源行号。**结构核验通过不等于语义批准。**

后续调用通过 `--evidence-offsets exact-unique` 显式使用该规则（默认strict仍拒绝坐标错误）。先用新目录复用旧响应，再只调用2道新题，凑够5道试点结果：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --output .\outputs\semantic_school_50_v2_offsets --thinking omit --profile guided-v2 --evidence-offsets exact-unique --allow-school-http --reuse-from .\outputs\semantic_school_50_v2 --limit 2
```

模型名、网关、profile及生成参数必须保持与来源v2一致；只有坐标核验政策允许从strict切换为exact-unique。导入时重新读取并校验原始响应，来源的accepted=false和错误保留为source_accepted/source_error，原始响应字节字符串保持不变。复用计数和本轮新增调用用量分开。

如只想导入而不调用API，将上条命令的 `--limit 2` 换成 `--import-only`。部分任务未齐时退出码2是预期结果。再用相同命令去掉import-only或limit即可继续。

此模式结果文件为 `responses.normalized.jsonl`，不能冒充原始LLM输出。1000题模式同样支持该选项及同配置结果复用；先完成50题人工审阅再扩展。

### 校园模型首轮失败后：guided-v2

2026-11-02的首轮校园Flash调用中，模型创造了schema未支持的packaging/quality类型，且多处证据索引错误；同时发现170预算被误放宽成187、“小狗造型”被当作颜色等语义问题。这些输出没有纳入通过结果。

旧版重复请求没有携带校验反馈，要求模型手算字符索引也不可靠。新增独立配置 `--profile guided-v2`：保持50题、1000题原query及输出schema不变，system附加逐维subtype表、类型缺口和预算解释约束，user附加程序计算的完整字符索引表；同一题重试时附上上次输出及具体校验错误。所有实际请求消息保存在attempts.jsonl。没有修改冻结的pilot50_v1，也不替模型修正索引或语义。

配置属于新prompt实验，必须使用新目录，不能与frozen-v1结果合并复用。不支持的礼盒包装、主观品质、造型需求必须按既定schema进入pending_mentions，后续审阅再决定是否扩展类型，不能强制归到颜色等无关维度。

保持你原来设置的学校API地址、模型名和密钥环境变量，先试5题：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 50 --output .\outputs\semantic_school_50_v2 --thinking omit --profile guided-v2 --allow-school-http --limit 5
```

也可先在相同命令末尾加 `--prepare-only` 做本地检查，不会调用API。试跑后需同时检查结构通过率和语义准确性；pending_mentions不是已通过人工审阅，不可直接用于最终奖励。原semantic_school_50_v1目录保留为诊断证据。若guided-v2通过50题审阅，再用相同profile跑1000题：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements.py --mode 1000 --output .\outputs\semantic_school_1000_v2 --thinking omit --profile guided-v2 --allow-school-http --reuse-from .\outputs\semantic_school_50_v2
```

- `run_manifest.json`：原始数据及输入快照哈希、模型、API地址与实际请求参数，不含密钥。
- `tasks.jsonl`、`requests.jsonl`、prompt/schema：本轮精确输入快照。
- `attempts.jsonl`：每次调用的原始完整响应、UTC时间、task_id、程序校验错误和通过标记；失败和重试也保留。
- `responses.jsonl`：通过程序核验的合同，每题一行，审阅状态仍是 pending。
- `report.json`：总数、通过数、缺失ID、复用数、服务端返回的模型名和所有本轮尝试的 token 用量。费用需按服务商价格计算。

用相同命令和目录重新运行即可续跑；只重试尚未通过的题。不自动修复引用位置或模型输出，也不剥离 Markdown 后冒充原始合格输出。默认每题每次启动最多尝试2次，可用 `--attempts 1` 降低重试费用；限流、服务端错误及不合格输出可能触发重试。HTTP 4xx配置/鉴权错误（429除外）立即停止，连续3道题均失败也会停止，避免持续空耗。

默认 temperature=0、max_tokens=8192、DeepSeek thinking=disabled、JSON Object输出加本地完整schema核验。JSON Object模式不保证符合schema。若要换参数或模型，请用新目录；不自动退回另一套请求参数。其他服务不接受 DeepSeek 参数时，从新目录开始加 `--thinking omit`；不支持 JSON Object时可显式加 `--response-format omit`，本地核验仍执行。

Ctrl+C 会保存已落盘响应并移除运行锁；服务端已经处理但客户端未收到的请求可能仍计费。断电/强制结束后，先确认没有该脚本进程，再删除结果目录里的 `.run.lock` 后续跑。若日志末行损坏，脚本会停止而不是静默丢弃证据；请先保留副本再处理。

所有输出默认放在被Git忽略的 `outputs/`。需要同步分析产物时另外整理，勿提交密钥。脚本不会触发训练或购买 API 服务。这里仅用模拟响应和本地准备测试，真实接口需按上面的5题试跑确认。

接口参数依据：[DeepSeek JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)、[思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)。
