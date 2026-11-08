"""Exact source/predicate review receipts for review-only scope diagnostics.

These never approve a whole contract, alter a reward, or waive structural,
unit, numeric, source-span or execution checks.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/'experiments/semantic_remaining_closeout_v1/scope_reviews.jsonl'
BATCH_PATH=PATH.parents[1]/'semantic_batch_release_v1/scope_reviews.jsonl'
CAPTURE_RAW_DIAGNOSTICS=False
KEYS=('subtype','attribute','subject','importance','operator','expected','price_basis','evidence')

@lru_cache(maxsize=1)
def records():
    return [json.loads(line) for path in (PATH,BATCH_PATH) if path.exists()
            for line in path.read_text(encoding='utf-8').splitlines() if line]

def query_digest(query):return hashlib.sha256(query.encode()).hexdigest()
def contract_digest(contract):return hashlib.sha256(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def reviewed(contract,flag):
    if CAPTURE_RAW_DIAGNOSTICS:return False
    for record in records():
        if record['query_sha256']!=query_digest(contract['query']) or record['code']!=flag['code']:continue
        parts=flag.get('location','').strip('/').split('/')
        if record['kind']=='query_scope_review':
            if record['contract_sha256']==contract_digest(contract) and record['source']==flag.get('source'):return True
        elif len(parts)==2 and parts[0] in contract and parts[1].isdigit():
            r=contract[parts[0]][int(parts[1])]
            if r['review_status']=='approved' and record['predicate']=={key:r[key] for key in KEYS}:return True
    return False
