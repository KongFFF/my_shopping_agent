# Fixed SFT/GRPO checkpoint comparison v1

Read `GRPO可学习性与任务调度记录.md` section 3 and `同题复评执行命令.md` in the parent repository.

The frozen list contains 182 early-valid-mixed training tasks, 40 early-all-gold controls and 40 early-no-gold-but-statically-feasible controls. Selection uses early log observations only; no late-result selection, Dev or Final evaluation. Each existing model receives eight fresh episodes per task, 4192 total.

`manifest.json` binds data, controls and parent `tools/` scripts. `runtime_receipt.json` binds the unmodified pinned baseline source and portable logical search index. Physical index and model file hashes are additionally bound per cloud run. Do not refreeze or refresh hashes on AutoDL.

Runtime build uses `git archive a583dba...`, independently of dirty submodule sources. The parent environment wrapper cleans stale same-name episodes on reset for both models; it adds no purchase guard, alternative reward or new observation renderer. Actual state is not copied between rollouts.

CPU preparation does not load a model. The inference pipeline owns only its dedicated model/environment child processes, has no separate billing-clock command, preserves every failed infrastructure attempt, and refuses rerolling completed policy outcomes. Missing attempts do not become zero scores in a complete comparison.
