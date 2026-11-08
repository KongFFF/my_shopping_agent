# Evidence training runtime snapshot

The upstream submodule remains pinned to its public `a583dba4c149be09622ce58ad6f7a160db77bb41` commit. This main repository owns the eight local source/config changes used by the released evidence-reward pilot. An unpushed submodule commit would not be available on AutoDL, so the reviewed sources are stored under `files/` with canonical LF SHA256 fingerprints in `manifest.json`.

After updating the main repository and initializing the pinned submodule:

```bash
upstream/shopping-grpo-longhorizon/.venv/bin/python tools/install_evidence_training_runtime.py --apply --backup-existing
upstream/shopping-grpo-longhorizon/.venv/bin/python tools/install_evidence_training_runtime.py --check
```

The installer checks all snapshot hashes before writing, backs up every file it changes under `outputs/evidence_runtime_backups/`, and rolls back those changes if installation fails. Existing files with unknown local edits are refused unless `--backup-existing` is explicit. Re-running against an installed snapshot is a no-op.

Only the eight listed source/config files may change. The installer does not write datasets, `data/environment.json`, the catalog, evaluation outputs or model weights. Run the frozen reward manifest check after installation. The submodule will show local modifications after installation; these exact modifications are versioned here and do not need to be pushed to the original author's repository.
