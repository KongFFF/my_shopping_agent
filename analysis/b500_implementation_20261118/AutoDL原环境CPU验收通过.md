# AutoDL原环境CPU验收通过

2026-11-18。回传分支`codex/b500-cpu-results-20261118T034604Z-_uzgmp6c`，提交`d024f124165d7e1fc0a986cd9ed52b93dd736938`。已拉取并核对[原始收据](cloud_cpu_acceptance/20261118T034604Z.json)。

- passed=true、errors为空，13项方法验收全部通过。
- 16项代码检查全部匹配，包括B overlay、验收脚本、冻结PPO参考及当前安装的历史trainer/loss/engine/torch helper、PPO core_algos和padding源码。
- PyTorch为2.11.0+cu130，gpu_initialized=false、model_weights_loaded=false、training_started=false。
- 收据的研究协议SHA256与当前冻结协议一致，仍为390201e5e1a75b4be136ec2e0a89f09f38855b19f3a6c372bb0f64d5c122c2e8。
- 收据绑定的implementation_manifest SHA256为7076dbbadea1064ed4d96d880afe2f989d9b2f8f059aee1f2efbc826867406c1，与本地已发布代码清单一致。

因此，本地2.6与AutoDL2.11之间的这批CPU数学/梯度验收已补齐，不需要重复身份哈希或再次运行相同CPU测试。

通过范围是：组权重、token分母、PPO裁剪数值/梯度、UID/mask/微批等价、hook恢复原优势，以及轻量AdamW/LR/RNG的350→500恢复。它仍不代表真实SFT模型FSDP、环境服务/全部商品资产、GPU预算监控或真实350步恢复已经验收。

下一阶段是准备独立B运行目录、冻结配置落地、运行资产/进程与磁盘检查、24000条全部生成和20小时GPU预算约束、首步检查点验收及350步完整恢复流程。未开始B训练，不执行Final、不自动续800。

implementation_manifest中的状态文字反映代码冻结时刻；最新验收状态以本通过收据及本记录为准，不修改原清单或旧收据以伪装重新验收。
