# Ray worker奖励目录传递修正

用户的有卡日志在`Training Progress 0/20`、第一个采样批次创建`AgentLoopWorker`时失败：`load_backend`未找到`SHOPPING_REWARD_TOOLS_DIR`对应的`semantic_training_bundle.py`。这是奖励后端定位/配置传递失败；该片段没有显示显存不足。无卡主进程检查通过，不能据此保证worker具有相同环境。

直接故障已确定是worker得不到正确目录。原配置将`${oc.env:...}`直接放进`ray_init.runtime_env.env_vars`；固定veRL0.8源码的Ray初始化路径使用未显式指定`resolve=True`的`OmegaConf.to_container`，本地真实OmegaConf/Hydra测试可复现把插值字符串原样交给Ray的风险。用户片段没有记录worker原始变量值，因此不把“为空”或“字面插值”冒充为已从云端读取的事实。[固定版本上游初始化代码](https://github.com/volcengine/verl/blob/v0.8.0/verl/trainer/main_ppo.py)

修正包含：

1. 启动器从已经校验的实际启动环境生成明确的Ray runtime值，并以Hydra原生引用器编码；不依赖延迟环境插值，也不把JSON转义误当作Hydra转义。使用白名单，不传播任意API密钥。
2. worker优先从显式序列化的完整trainer配置读取manifest/工具目录；即使两个环境变量都缺失，仍能启用并加载既定奖励，避免静默落回原奖励。配置中的manifest冲突、未展开插值或无效路径仍拒绝。
3. 加载模型前的配置预检检查Ray将接收到的原始环境payload，确认关键路径已是与主进程一致的字面字符串。
4. 新增`probe_evidence_ray_worker.py`：用现有AutoDL训练venv做双层CPU Ray任务，先核对真实环境，再移除两个后端环境变量，检查仍可从明确配置加载冻结后端。它不导入veRL/vLLM、不读取模型权重、不生成轨迹、不执行更新。

本地回归包含真实Hydra解析、原始未解析payload复现、环境变量全清空、不同工作目录的独立Python子进程、配置激活不得静默降级、冲突路径拒绝等。这里没有声称本地已完成真实Ray或GPU试跑：本地未安装Ray，真实Ray自检需在AutoDL执行。

更新了9文件运行快照及奖励冻结清单；原1000题、981/19参与口径、奖励公式、SFT起点和20步上限不变。失败运行的`semantic_evidence_pilot20_v1`保留，重试使用全新的`semantic_evidence_pilot20_v2`，先自检、再dry-run、最后有卡试跑。
