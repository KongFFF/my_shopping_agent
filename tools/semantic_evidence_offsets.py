"""Audited exact-text evidence offsets; never infer or change requirement semantics."""
import copy
from jsonschema import Draft202012Validator
from semantic_contract_v1 import DIMENSIONS


def normalize_evidence(contract, task, schema):
    Draft202012Validator(schema).validate(contract)
    if contract["task_id"] != task["task_id"] or contract["query"] != task["query"]:
        raise ValueError("Original task ID/query must match before evidence normalization")
    result, changes = copy.deepcopy(contract), []
    query = task["query"]
    entries = [(f"/{dimension}/{i}/evidence/{j}", evidence)
               for dimension in DIMENSIONS for i, requirement in enumerate(result[dimension])
               for j, evidence in enumerate(requirement["evidence"])]
    entries += [(f"/pending_mentions/{i}/evidence", mention["evidence"])
                for i, mention in enumerate(result["pending_mentions"])]
    for path, evidence in entries:
        text, start, end = evidence["text"], evidence["start"], evidence["end"]
        if text and 0 <= start < end <= len(query) and query[start:end] == text:
            continue
        matches = [i for i in range(len(query)) if text and query.startswith(text, i)]
        if len(matches) != 1:
            raise ValueError(f"{path}: evidence text must have exactly one original match; found {len(matches)}")
        new_start = matches[0]
        changes.append({"path": path, "text": text, "before": {"start": start, "end": end},
                        "after": {"start": new_start, "end": new_start + len(text)},
                        "method": "unique_exact_original_text"})
        evidence.update(start=new_start, end=new_start + len(text))
    return result, changes
