"""Build auditable development references from data-only edits, never runtime rules.

No API, no human approval, no changes to actor queries or original API journals.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from jsonschema import Draft202012Validator
import semantic_contract_v2_r4 as base
from semantic_relations_v2_r3 import canonicalize
from semantic_policy_v4 import apply_budget_policy

VERSION = "shopping-semantic-reference-v1-draft"
DIMS = copy.deepcopy(base.DIMENSIONS)
if "weight" not in DIMS["specifications"]: DIMS["specifications"].append("weight")
ATTRS = copy.deepcopy(base.ATTRIBUTES)
ATTRS["weight"] = ["mass"]
if "thickness" not in ATTRS["dimensions"]: ATTRS["dimensions"].append("thickness")


def schema():
    s = copy.deepcopy(base.schema())
    s["properties"]["schema_version"]["const"] = VERSION
    types=s["properties"]["specifications"]["items"]["allOf"][1]["properties"]["subtype"]["enum"]
    if "weight" not in types:types.append("weight")
    props = s["$defs"]["requirement"]["properties"]
    props["subtype"]["enum"] = sorted(ATTRS)
    props["attribute"]["enum"] = sorted({a for values in ATTRS.values() for a in values})
    Draft202012Validator.check_schema(s)
    return s


def prepare_base(card):
    raw = copy.deepcopy(card["raw_selected"])
    if raw["query"] != card["query"] or raw["task_id"] != card["task_id"]:
        raise ValueError("Source identity mismatch")
    raw, audit = canonicalize(raw, card, base.DIMENSIONS, base.ATTRIBUTES, base.chinese_number, base.price_basis_from_phrase)
    return raw, audit


def apply_edits(raw, edits):
    result = copy.deepcopy(raw)
    for edit in edits:
        keys = edit["path"].strip("/").split("/")
        if keys[0] not in DIMS and keys[0] != "pending_mentions":
            raise ValueError("Reference edits cannot alter identity, query or metadata")
        target = result
        for key in keys[:-1]:
            target = target[int(key)] if isinstance(target, list) else target[key]
        key = int(keys[-1]) if isinstance(target, list) else keys[-1]
        if target[key] != edit["before"]:
            raise ValueError(f"Patch precondition changed: {edit['path']}")
        target[key] = copy.deepcopy(edit["after"])
    return result


def locate(query, e):
    text = e["text"]
    hits = [i for i in range(len(query)) if query.startswith(text, i)]
    if not text or len(hits) != 1:
        raise ValueError(f"Reference evidence must be unique exact source: {text!r}")
    return {"text":text,"start":hits[0],"end":hits[0]+len(text)}


def normalize(raw):
    c, audit = copy.deepcopy(raw), []
    c["schema_version"] = VERSION
    q = c["query"]
    entries = [(d, r) for d in DIMS for r in c[d]]
    for _, r in entries:
        r["review_status"] = "pending"
        r["evidence"] = [locate(q,e) for e in r["evidence"]]
    for p in c["pending_mentions"]:
        if not isinstance(p["evidence"], dict):
            raise ValueError("Resolve pending evidence shape explicitly in reference data")
        p["evidence"] = locate(q,p["evidence"])
    entries.sort(key=lambda x: (min(e["start"] for e in x[1]["evidence"]), list(DIMS).index(x[0])))
    for i, (_, r) in enumerate(entries,1):
        r["id"] = f"r{i:03d}"
    for r in c["budget"]:
        phrase = r["expected"]["value"]
        clause = base.budget_source_clause(q,phrase)
        # A reference can correct a supplied denominator only via its data patch.
        try:
            basis = base.price_basis_from_phrase(clause)
        except ValueError:
            basis = None
        if r["price_basis"] is not None and r["price_basis"] != basis:
            raise ValueError("Reference denominator needs an explicit source-backed patch")
        if r["subtype"] == "unit_price" and basis is None:
            raise ValueError("Missing unit price denominator")
        if r["subtype"] == "total_price" and basis is not None:
            raise ValueError("Per-unit phrase cannot become total price")
        r["price_basis"] = basis if r["subtype"] != "total_price" else None
        r["evidence"] = [locate(q,{"text":clause})]
        try:
            r["operator"], r["expected"], rule = base.budget_for_role(clause,r["subtype"])
        except ValueError:
            r["importance"] = "uncertain"
            p={"evidence":locate(q,{"text":clause}),"reason":"ambiguous_meaning","importance":"uncertain"}
            if p not in c["pending_mentions"]:c["pending_mentions"].append(p)
        else:
            audit.append({"kind":"budget_policy","id":r["id"],"source":clause,"rule":rule})
            apply_budget_policy(c,r,clause,audit)
    validate(c)
    return c, audit


def validate(c):
    Draft202012Validator(schema()).validate(c)
    ids = set()
    for d in DIMS:
        for r in c[d]:
            if r["id"] in ids:raise ValueError("Duplicate id")
            ids.add(r["id"])
            if r["subtype"] not in DIMS[d] or r["attribute"] not in ATTRS[r["subtype"]]:
                raise ValueError(f"Invalid reference field: {d}/{r['subtype']}/{r['attribute']}")
            if r["review_status"] != "pending":raise ValueError("Reference authoring cannot approve itself")
            for e in r["evidence"]:
                if c["query"][e["start"]:e["end"]] != e["text"]:raise ValueError("Source mismatch")
            if r["subject"]["kind"] not in ("primary_product","order") and r["subject"]["name"] not in "，".join(e["text"] for e in r["evidence"]):
                raise ValueError(f"Unbound reference subject {r['subject']['name']}")
            v=r["expected"];op=r["operator"]
            needed={"at_least":"number","at_most":"number","between":"range","one_of":"text_list","matches_bundle":"bundle"}.get(op)
            if needed and v["type"]!=needed:raise ValueError("Operator type mismatch")
            if v["type"]=="range" and v["lower"] is not None and v["upper"] is not None and v["lower"]>v["upper"]:raise ValueError("Empty range")
            if v["type"]=="bundle" and all(v[k] is not None for k in ("paid_quantity","gift_quantity","received_quantity")) and v["paid_quantity"]+v["gift_quantity"]!=v["received_quantity"]:raise ValueError("Invalid bundle")
            if v["type"] in ("number","range"):
                physical={"mass":["kg","g","jin"],"thickness":["mm","cm","m","inch"],"power":["W"],"voltage":["V"],"energy_capacity":["Wh","kWh"],"electric_charge":["Ah","mAh"]}
                if r["attribute"] in physical and v["unit"] not in physical[r["attribute"]]:raise ValueError("Reference physical unit mismatch")
                if r["subtype"]=="quantity" and v["unit"] not in base.PURCHASE_UNITS:raise ValueError("Reference purchase unit mismatch")
            if r["importance"]=="uncertain" and not any(any(p["evidence"]["start"]<e["end"] and e["start"]<p["evidence"]["end"] for e in r["evidence"]) for p in c["pending_mentions"]):
                raise ValueError(f"Uncertain reference requirement missing pending context: {r['id']}")
    for p in c["pending_mentions"]:
        e=p["evidence"]
        if c["query"][e["start"]:e["end"]]!=e["text"]:raise ValueError("Pending source mismatch")


def build(source, edits_path, output):
    source, edits_path, output = source.resolve(), edits_path.resolve(), output.resolve()
    book=json.loads(edits_path.read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest()!=book["source_sha256"]:raise ValueError("Patch source changed")
    if output.exists() and any(output.iterdir()):raise ValueError("Use an empty reference output")
    cards=[json.loads(l) for l in source.read_text(encoding="utf-8").splitlines() if l.strip()]
    edits={r["task_id"]:r for r in book["tasks"]}
    if len(edits)!=len(cards) or set(edits)!={c["task_id"] for c in cards}:raise ValueError("Reference coverage mismatch")
    results=[];logs=[]
    for card in cards:
        raw,pre=prepare_base(card)
        fixed=apply_edits(raw,edits[card["task_id"]]["edits"])
        c,audit=normalize(fixed)
        if c["query"]!=card["query"]:raise ValueError("Actor query changed")
        results.append(c)
        logs.append({"task_id":c["task_id"],"source_attempt_line":card["selected_attempt_line"],"edits":edits[c["task_id"]]["edits"],"base_audit":pre,"normalization_audit":audit,"approved_for_reward":False})
    output.mkdir(parents=True,exist_ok=True)
    for name, rows in (("reference_candidates.jsonl",results),("reference_audit.jsonl",logs)):
        (output/name).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    (output/"reference.schema.json").write_text(json.dumps(schema(),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report={"source_sha256":book["source_sha256"],"edits_sha256":hashlib.sha256(edits_path.read_bytes()).hexdigest(),"tasks":len(results),"with_pending":sum(bool(c["pending_mentions"]) for c in results),"scope":"Assistant-authored development references, not automated extraction accuracy or human-approved training labels","api_calls":0,"approved_for_reward":0}
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True);parser.add_argument("--edits",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.source,args.edits,args.output),ensure_ascii=False,indent=2))
