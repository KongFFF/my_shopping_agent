"""CLI for real purchase snapshots; never substitutes a target product."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from semantic_product_verifier_v2 import verify


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contracts',type=Path,required=True,help='Latest validated1000.jsonl')
    p.add_argument('--products',type=Path,required=True,help='Public product JSON or JSON.gz')
    p.add_argument('--purchases',type=Path,required=True,help='JSONL: task_id, asin, selected_options; or saved trajectories')
    p.add_argument('--catalog-facts',type=Path,help='Optional source-backed reviewed facts JSONL')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Output already exists; choose a new path to keep runs separate')
    opener=gzip.open if args.products.suffix=='.gz' else open
    with opener(args.products,'rt',encoding='utf-8') as handle:products=json.load(handle)
    indexed={str(item['asin']):item for item in products}
    contracts={r['task_id']:r for r in read_rows(args.contracts)}
    facts=list(read_rows(args.catalog_facts)) if args.catalog_facts else []
    results=[];counts=Counter()
    for row in read_rows(args.purchases):
        task=row['task_id'];item=contracts.get(task)
        terminal=row.get('terminal_result') or {}; purchase=terminal.get('purchase') or {}
        asin=str(row.get('asin',purchase.get('asin','')))
        options=row.get('selected_options',purchase.get('options'))
        if item is None:result={'task_id':task,'status':'missing_contract'}
        elif not item.get('program_valid',True):result={'task_id':task,'status':'contract_program_blocked','error':item.get('error')}
        elif not asin or options is None:result={'task_id':task,'status':'no_purchase_snapshot'}
        elif asin not in indexed:result={'task_id':task,'status':'product_missing','asin':asin}
        else:
            try:
                result=verify(item.get('normalized',item),indexed[asin],options,
                    purchase_executed=row.get('purchase_executed') is True,
                    catalog_facts=[f for f in facts if str(f.get('product_asin'))==asin])
                result['status']='verified_diagnostic'
                result['counts']=dict(Counter(result['statuses'].values()))
            except (ValueError,KeyError,TypeError) as error:result={'task_id':task,'status':'input_error','error':str(error)}
        if 'trajectory_id' in row:result['trajectory_id']=row['trajectory_id']
        counts[result['status']]+=1;results.append(result)
    args.output.mkdir(parents=True)
    (args.output/'comparisons.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    summary={'purchases':len(results),'statuses':dict(counts),'reward_ready':sum(r.get('reward_ready',False) for r in results),
             'api_calls':0,'training_runs':0,'note':'Diagnostic comparisons only; no production rewards or automatic contract approval'}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
