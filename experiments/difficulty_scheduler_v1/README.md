# Difficulty scheduler v1 pilot

Frozen A/B pilot: original SFT step300, original train1000, original Reward v3,
four rollouts per task, two tasks per generation batch, 4000 generated rollouts
per arm. Only pre-generation task weights differ. Half uniform exploration,
half capped mixed-gold/usable-result priority; 100-update count half-life.

`manifest.json` binds eight scripts, original task rosters, source fixtures,
diagnostic-only risk IDs and the clean baseline runtime receipt. Cloud users
must not refreeze or change these files. Runtime export reads the pinned Git
object, not the dirty upstream worktree. veRL modifications affect only an
isolated copied package under outputs.

The pinned baseline's data/environment.json records one stale source hash.
Export corrects only that metadata entry (448f... -> actual d6db...) and saves
metadata_correction.json. Environment source and scoring are unchanged; both
arms share the correction, also consistent with prior checkpoint-pair source.

`ray_trainer_0_8_0.py` is the official Apache-2.0-licensed veRL v0.8.0 trainer
from https://github.com/volcengine/verl/blob/v0.8.0/verl/trainer/ppo/ray_trainer.py
(original copyright/license header retained). `baseline_dynamic.patch` is the
existing shopping V3 patch from pinned baseline a583dba4. Both are source/test
fixtures, not a second training implementation. Their transformed result is
syntax checked and exercised in CPU integration tests.

Historical Dev50 equals original GRPO validation50 by task ID. It is held out
from optimizer updates, but is a reused development set, not a new confirmation
set. No Final or semantic judge is run. The 90 static no-gold IDs and 18 singleton
axis screen IDs are reporting labels only; they never affect task selection.

See [AutoDL commands](../../难度调度小规模对照执行命令.md) and
[design](../../难度感知任务调度设计.md). New training and model export run only
when the user invokes the documented pipeline on AutoDL.
