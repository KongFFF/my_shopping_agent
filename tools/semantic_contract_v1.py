"""Fixed offline extraction schema. Does not call an LLM or compute rewards."""
import argparse
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/semantic_contract_v1"
VERSION = "shopping-semantic-requirements-v1-draft"
DIMENSIONS = {
    "category": ["product_category"],
    "budget": ["total_price", "unit_price", "deposit", "balance", "processing_fee", "shipping_fee"],
    "functions": ["function", "purpose", "target_user", "use_environment"],
    "specifications": ["quantity", "bundle", "components", "color", "size", "dimensions", "material",
                       "capacity", "net_content", "compatibility", "merchant", "delivery", "warranty"],
    "brand": ["brand"],
    "model": ["product_model", "version"],
}
UNITS = ["CNY", "piece", "bottle", "bag", "box", "pair", "set", "kg", "g", "jin", "ml", "l",
         "mm", "cm", "m", "inch", "W", "V", "day", "month", "year"]


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def tagged(kind, properties):
    return obj({"type": {"const": kind}, **properties})


def schema():
    string = {"type": "string", "minLength": 1}
    number = {"type": "number", "minimum": 0}
    unit = {"enum": UNITS}
    evidence = obj({"start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1}, "text": string})
    expected = {"oneOf": [
        tagged("text", {"value": string}),
        tagged("text_list", {"values": {"type": "array", "items": string, "minItems": 1, "uniqueItems": True}}),
        tagged("number", {"value": number, "unit": unit}),
        tagged("range", {"lower": {"type": ["number", "null"], "minimum": 0},
                         "upper": {"type": ["number", "null"], "minimum": 0}, "unit": unit,
                         "lower_inclusive": {"type": "boolean"}, "upper_inclusive": {"type": "boolean"}}),
        tagged("bundle", {"paid_quantity": {"type": "integer", "minimum": 1},
                          "gift_quantity": {"type": "integer", "minimum": 0}, "unit": unit}),
        tagged("components", {"items": {"type": "array", "minItems": 1,
                                        "items": obj({"name": string, "quantity": number, "unit": unit})}}),
    ]}
    properties = {"schema_version": {"const": VERSION}, "task_id": {"type": "integer", "minimum": 0},
                  "query": string}
    for name, subtypes in DIMENSIONS.items():
        requirement = obj({"id": string, "subtype": {"enum": subtypes},
                           "importance": {"enum": ["required", "preference", "uncertain"]},
                           "operator": {"enum": ["eq", "one_of", "contains_all", "at_least", "at_most", "between", "matches_bundle"]},
                           "expected": expected,
                           "evidence": {"type": "array", "items": evidence, "minItems": 1},
                           "review_status": {"enum": ["pending", "approved"]}})
        properties[name] = {"type": "array", "items": requirement}
    properties["pending_mentions"] = {"type": "array", "items": obj({
        "evidence": evidence, "reason": {"enum": ["ambiguous_meaning", "ambiguous_strength", "unsupported_type", "unsupported_unit", "contradiction"]}})}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **obj(properties)}


def validate(contract, *, require_reviewed=False, source_query=None):
    """Validate structure and evidence references; NOT semantic correctness."""
    Draft202012Validator(schema()).validate(contract)
    ids = set()
    query = contract["query"]
    if source_query is not None and query != source_query:
        raise ValueError("Contract query differs from the original user input")
    if require_reviewed and source_query is None:
        raise ValueError("Original source query is required for reviewed validation")

    def check_evidence(e):
        if not 0 <= e["start"] < e["end"] <= len(query) or query[e["start"]:e["end"]] != e["text"]:
            raise ValueError("Evidence must be an exact character span in the original query")

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite numeric value")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        if isinstance(value, list):
            for item in value:
                finite(item)

    finite(contract)
    for dimension in DIMENSIONS:
        for requirement in contract[dimension]:
            if requirement["id"] in ids:
                raise ValueError("Requirement IDs must be unique across all dimensions")
            ids.add(requirement["id"])
            for e in requirement["evidence"]:
                check_evidence(e)
            value = requirement["expected"]
            if value["type"] == "range":
                low, high = value["lower"], value["upper"]
                if low is None and high is None or low is not None and high is not None and (
                        low > high or low == high and not (value["lower_inclusive"] and value["upper_inclusive"])):
                    raise ValueError("Empty or invalid range")
            required_type = {"matches_bundle": "bundle", "between": "range", "one_of": "text_list",
                             "at_least": "number", "at_most": "number"}.get(requirement["operator"])
            if required_type and value["type"] != required_type:
                raise ValueError("Operator and expected-value type disagree")
            if dimension == "budget" and (value["type"] not in ("range", "number") or value.get("unit") != "CNY"):
                raise ValueError("Budget must use a numeric/range amount in CNY")
            if require_reviewed and (requirement["review_status"] != "approved" or requirement["importance"] == "uncertain"):
                raise ValueError("Contract has an unreviewed/uncertain requirement")
    for mention in contract["pending_mentions"]:
        check_evidence(mention["evidence"])
    if require_reviewed and contract["pending_mentions"]:
        raise ValueError("Resolve pending mentions before scoring")
    return contract


def example():
    query = "需要全钢压脚，买10送1，预算100元以内，最好黑色。"
    contract = {"schema_version": VERSION, "task_id": 0, "query": query,
                **{name: [] for name in DIMENSIONS}, "pending_mentions": []}
    def add(dimension, ident, subtype, importance, operator, expected, quote):
        start = query.index(quote)
        contract[dimension].append({"id": ident, "subtype": subtype, "importance": importance,
                                    "operator": operator, "expected": expected,
                                    "evidence": [{"start": start, "end": start + len(quote), "text": quote}],
                                    "review_status": "pending"})
    add("category", "r1", "product_category", "required", "eq", {"type": "text", "value": "压脚"}, "压脚")
    add("specifications", "r2", "material", "required", "eq", {"type": "text", "value": "全钢"}, "全钢")
    add("specifications", "r3", "bundle", "required", "matches_bundle",
        {"type": "bundle", "paid_quantity": 10, "gift_quantity": 1, "unit": "piece"}, "买10送1")
    add("budget", "r4", "total_price", "required", "at_most", {"type": "number", "value": 100, "unit": "CNY"}, "预算100元以内")
    add("specifications", "r5", "color", "preference", "eq", {"type": "text", "value": "黑色"}, "最好黑色")
    return validate(contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", nargs="?", type=Path)
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--query-file", type=Path, help="Original UTF-8 user query, preserved verbatim")
    parser.add_argument("--write-schema", action="store_true")
    args = parser.parse_args()
    if args.write_schema:
        PACKAGE.mkdir(parents=True, exist_ok=True)
        for name, value in (("requirements.schema.json", schema()), ("example.synthetic.json", example())):
            (PACKAGE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.contract:
        validate(json.loads(args.contract.read_text(encoding="utf-8")), require_reviewed=args.require_reviewed,
                 source_query=args.query_file.read_text(encoding="utf-8") if args.query_file else None)
        print("PASS: structure/evidence validation; semantic correctness still requires review")
