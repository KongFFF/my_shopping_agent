# B500实现与CPU验收

2026-11-18。方法代码已实现，本地CPU方法测试13项、加流程/路径测试合计26项通过。**未启动GPU，未加载SFT或其他真实模型；AutoDL原环境CPU验收和真实FSDP运行验收尚待完成。**

## 接入方式

`overlay/b500_weighting.py`按UID取得两个完整四轨迹组，从原shopping诊断读取有效性和purchase_success，用原response_mask计算整批8条轨迹的有效Actor token分母。每组同题、标签有效、奖励版本正确、mask有限且二值等条件不满足就停止。

`overlay/b500_hook.py`只包装Driver的`RayPPOTrainer._update_actor`。在原始优势已经计算、动态筛组已经完成后，临时把每条轨迹的优势乘以正系数`w×N/Nw`，调用原Actor，随后在finally恢复原优势。源码guard要求原Ray trainer哈希不变，配置guard限定冻结的单GPU、8轨迹、1 epoch、vanilla/token-mean、无熵/KL/蒸馏。原mask、回报、筛选与Actor实现不修改。

PPO标准裁剪及负优势双裁剪对正的优势缩放齐次，因此该接入与冻结的加权策略损失等价；不是改写原奖励或调用SFT损失。训练日志保存原优势和权重；Actor内部以加权优势计算策略损失，Driver后续原指标仍读取恢复后的原优势。

该hook必须通过**唯一**worker_process_setup_hook启用：`b500_hook.setup_worker`，它先调用原`install_torch_padding_fallback`，再安装包装。保持原隔离runtime的src及overlay在PYTHONPATH，设置绝对`B500_WEIGHT_LOG`；不同时加载其他实验hook，不修改共享site-packages。当前没有可直接执行的GPU launcher，避免绕过后续运行验收。

## 已验证的范围

- 从冻结veRL0.8.0源码提取真实`compute_policy_loss_vanilla`及`agg_loss`，保留双裁剪；参考源码哈希先验证。不是只用简化公式自测。
- 权重全1复现原损失与梯度；FF/SS归一化抵消；FS不等长有效token的损失和梯度与直接加权公式一致，包含正负优势及双裁剪分支。
- tool/padding mask零位不影响损失，梯度为零；UID随机交错重排不影响分组；微批大小1、2、3、8累积均与整批一致，始终使用全局分母。
- 无效标签、跨组/缺UID、不完整组、零/非二值mask、非法权重等拒绝；valid_alternative_purchase沿用原成功定义。
- hook幂等安装，Actor返回或抛异常后都恢复原优势；日志区分started/returned/failed，不能把started当成功更新。
- 轻量CPU参数+AdamW+15步warmup日程+随机生成器在350保存后恢复到500，与不中断路径参数/RNG完全相同。**这不是实际SFT模型、FSDP、CUDA RNG或原DataLoader的350恢复验收**，实际恢复仍按协议在运行中核验。
- 用两窗300次实际更新/600组、2400条轨迹，在原重排顺序下验证F/S标签，得到FF33、FS119、SS148。该标签测试用单token占位mask，不推算历史缺失的真实Actor token分母。

本地PyTorch2.6.0+cu118，测试只在CPU；OpenMP冲突通过进程级MKL_THREADING_LAYER=SEQUENTIAL与OMP_NUM_THREADS=1处理，没有启用KMP_DUPLICATE_LIB_OK。AutoDL原环境PyTorch2.11.0，须独立再验。

## AutoDL无卡CPU验收及Git回传

此前32项身份通过无需重跑。下面只执行CPU合成测试，并核对当前安装的关键引擎、原PPO和padding源码；不会加载模型权重、启动训练或CUDA。和已修复回传流程一样，创建新克隆，不覆盖已有目录，失败即停或回传合法失败收据。

```bash
if (
  cd /root/autodl-tmp/work/my_shopping_agent &&
  git -c http.version=HTTP/1.1 fetch origin codex/b500-weighting-implementation &&
  B500_BOOTSTRAP=$(mktemp /tmp/b500-cpu-return-XXXXXX.py) &&
  git show FETCH_HEAD:tools/run_b500_preflight_return.py > "$B500_BOOTSTRAP" &&
  MKL_THREADING_LAYER=SEQUENTIAL OMP_NUM_THREADS=1 \
    /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/.venv/bin/python \
    "$B500_BOOTSTRAP" --repo /root/autodl-tmp/work/my_shopping_agent --cpu-acceptance
); then
  printf 'CPU验收回传完成；以 CPU_ACCEPTANCE_PASSED 为准。未启动训练。\n'
else
  printf '流程停止，请保留日志和 STATE 路径。未启动训练。\n'
fi
```

结束后回传RETURN_BRANCH与RETURN_COMMIT。若失败回传STOPPED与STATE路径；上传故障使用脚本打印的retry-push，不重跑训练。CPU验收通过后，再准备单独B运行目录、完整资产检查、预算监控、首步保存与350恢复流程，才具备开跑条件。

本地复算：`python tools/freeze_b500_implementation.py`生成代码哈希；在上述MKL/OMP进程配置下执行`python tools/run_b500_cpu_acceptance.py --output outputs/b500_cpu_acceptance.json`。研究protocol.json没有变化，代码清单另记implementation_manifest.json。
