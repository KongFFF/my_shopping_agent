"""Offline, audited evidence offset normalization of saved API responses. No API calls."""
import argparse
from pathlib import Path
from extract_semantic_requirements import atomic, digest, dump, journal_read, loads, snapshot, summarize


def normalize_run(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Use a separate output directory outside the original run")
    if (source / ".run.lock").exists():
        raise ValueError("Source run is locked; wait for it to finish")
    manifest_raw = (source / "run_manifest.json").read_bytes()
    manifest = loads(manifest_raw.decode("utf-8"))
    tasks, _, files, _ = snapshot(manifest["dataset"]["mode"], manifest["api"].get("extraction_profile", "frozen-v1"))
    for name, expected in files.items():
        raw = (source / name).read_bytes()
        if raw != expected or digest(raw) != manifest["dataset"]["files_sha256"][name]:
            raise ValueError(f"Source snapshot changed: {name}")
    records = journal_read(source / "attempts.jsonl")
    provenance = {"version": "exact-unique-evidence-offsets-v1", "source": str(source),
                  "source_manifest_sha256": digest(manifest_raw),
                  "source_attempts_sha256": digest((source / "attempts.jsonl").read_bytes()),
                  "scope": "Derived evidence offsets only; not original LLM output or semantic approval."}
    output.mkdir(parents=True, exist_ok=True)
    proof = output / "normalization_source.json"
    if proof.exists():
        if loads(proof.read_text(encoding="utf-8")) != provenance:
            raise ValueError("Source changed; use a new normalization directory")
    elif any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    atomic(proof, dump(provenance) + "\n")
    _, report = summarize(output, tasks, loads(files["output.schema.json"].decode("utf-8")), records, "exact-unique")
    print(dump({k: report[k] for k in ("validated_tasks", "raw_validated_tasks", "normalized_tasks", "total_tasks")}))
    print(f"Derived outputs: {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    normalize_run(args.source, args.output)
