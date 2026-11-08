# semantic-v2-r5：数值作用域与非破坏性审阅提示

本版修程序误判，不承诺自动修复所有LLM语义错误。六个顶层语义维度及字段枚举不变，不改原训练1000题、原始用户输入或GRPO奖励。r4及其入口保持原字节，原封存新50题仍可验证；r5使用独立入口，避免旧运行被静默升级。

## 修改内容

- 非预算近似必须绑定数值对象及其证据。方向词“左右自动摇头”“上下移动”不再触发容差归一化；350ml左右、大约200公斤等仍保留原文和待审，不猜容差。不把某项证据中的另一属性数字借给当前文字功能，多条证据的待审引文定位到包含数值的那条。
- 喜欢/希望等整句作用域启发式不再一律形成重试指令。只有与目标字面值直接绑定且没有相反标记时才做强弱硬检查；其余记录review_findings。不会自动把必需项降为偏好，也不把无法证明的偏好升为必需。
- 百分比不能伪装成count，鞋码标签不能用特征计数。此检查在近似转换前执行，防止把量纲错误藏入近似文本。模型应保留真实标签；schema不支持的百分比用原文和unsupported_unit待审，未扩充专用字段。
- “预估/接近”的预算关系没有已确认口径，保留原文待审，不悄悄套用裸金额上限；不由程序决定新的解释政策。
- 审阅提示独立输出并写入审计，检查器也显示。它们不会引发要求模型猜强弱的反馈，但仍不允许据此进入奖励批准。所有提取输出仍pending。

没有为戒指、鱼竿、化妆品等增加专用规则，没有程序补写漏掉的用户需求。

## 入口与输出

- tools/extract_semantic_requirements_r5.py：r5 API提取/本地准备，支持50、1000和regression50；默认profile为semantic-v2-r5。
- tools/collect_semantic_diagnostics_r5.py：与r5准备目录配套的完整诊断收集。不要混用旧入口或旧r4输出目录。
- tools/replay_semantic_v5.py：旧回答的离线回归。只在内存修改schema_version，不改其余模型字段，不发送请求，派生产物必须另存。
- tools/check_semantic_contract_v5.py：r5结果校验，含semantic_review_findings。
- tools/semantic_contract_v2_r5.py、tools/semantic_scope_v5.py：归一化及通用作用域规则。
- review_findings.jsonl：程序不能确定的作用域、偏好和其他审阅提示。report.json增加semantic_review_tasks和semantic_review_findings。结构complete不等于已批准或可用于奖励。

regression50指已经观察过的那组新50题，不再称新盲测。1000模式已保留，但目前不建议整批执行。

## 已完成的无API验证

分析见 analysis/semantic_r5_replay_20261103/修复说明.md。已离线回放原50题73条记录及另一组50题66条记录。离线回归不等于使用r5 prompt重新提取；实际模型收到的反馈已改变，不能用旧回答预测下一轮通过率。

## 可选的小批回归命令（PowerShell，本地CPU，无需GPU）

现在不必整批重跑。若要验证r5实际调用，先准备已经观察过的regression50：

```powershell
Set-Location D:\shopping_agent
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements_r5.py --mode regression50 --output .\outputs\semantic_school_regression50_r5_v1 --profile semantic-v2-r5 --base-url http://aigw.dlut.edu.cn/v1 --model DeepSeek-V4-Flash-0731-W8A8 --thinking omit --prepare-only
```

上述命令不读取密钥、不请求网络。实际调用仍从现有LLM_API_KEY环境变量读取密钥，不写入文件；需明确允许学校HTTP，仅限既定地址。只调用前5题：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\extract_semantic_requirements_r5.py --mode regression50 --output .\outputs\semantic_school_regression50_r5_v1 --profile semantic-v2-r5 --base-url http://aigw.dlut.edu.cn/v1 --model DeepSeek-V4-Flash-0731-W8A8 --thinking omit --allow-school-http --limit 5
```

prepare-only与实际调用须使用相同参数，除本地模式、限额和学校HTTP许可外。若该目录已存在且配置不同，另取新目录，不覆盖旧记录。不从r4运行--reuse-from到r5，版本及prompt不同。

若后续决定收齐本组，使用r5收集器，单独目录：

```powershell
.\.venv-llm\Scripts\python.exe .\tools\collect_semantic_diagnostics_r5.py --source-run .\outputs\semantic_school_regression50_r5_v1 --output .\outputs\semantic_school_regression50_r5_v1_collect --allow-school-http
```

原50题对应--mode 50；原训练1000题对应--mode 1000，均必须另建输出目录。不要把这些命令误当GRPO训练或评测。
