# Evidence training runtime snapshot

The upstream submodule remains pinned to its public `a583dba4c149be09622ce58ad6f7a160db77bb41` commit. This main repository owns the eight local code/config changes plus the previously confirmed environment-metadata correction used by the released evidence-reward pilot. The nine reviewed files are stored under `files/` with canonical LF SHA256 fingerprints in `manifest.json`.

After updating the main repository and initializing the pinned submodule:

```bash
upstream/shopping-grpo-longhorizon/.venv/bin/python tools/install_evidence_training_runtime.py --apply --backup-existing
upstream/shopping-grpo-longhorizon/.venv/bin/python tools/install_evidence_training_runtime.py --check
```

The installer checks all snapshot hashes before writing, backs up every file it changes under `outputs/evidence_runtime_backups/`, and rolls back those changes if installation fails. Existing files with unknown local edits are refused unless `--backup-existing` is explicit. Re-running against an installed snapshot is a no-op.

Only the nine listed source/config files may change. For `data/environment.json`, the installer accepts only the known public baseline or the already-confirmed correction of `web_agent_text_env.py` from `448f5e...` to `d6db15...`; other policy changes are refused even with the backup flag. Equivalent JSON formatting is safely backed up and canonicalized. The installer does not write datasets, catalog, evaluation outputs or model weights. Run the frozen reward manifest check after installation; it now also verifies all five declared environment source hashes against the actual source files. The submodule will show local modifications after installation; these exact modifications are versioned here and do not need to be pushed to the original author's repository.
