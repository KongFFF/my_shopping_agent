"""Independent offline catalog-fact extraction; drafts never approve themselves."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from jsonschema import Draft202012Validator,ValidationError
from urllib.error import URLError
from extract_semantic_requirements import endpoint, post, loads, HTTPFailure
from semantic_product_facts_v3 import schema as fact_schema,validate_fact
from semantic_product_verifier_v3 import fingerprint,pointer
from semantic_public_catalog_v2 import public_product

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'upstream/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'
PROMPT=ROOT/'experiments/semantic_product_verifier_v3/product_fact_system_prompt.txt'


def dump(v):return json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(p):return [loads(x) for x in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
def digest(v):return hashlib.sha256(dump(v).encode()).hexdigest()


def output_schema():
    fact=fact_schema();fact['properties']['review_status']={'const':'pending'}
    defs=fact.pop('$defs');fact.pop('$schema',None)
    result={'$schema':'https://json-schema.org/draft/2020-12/schema','$defs':defs,'type':'object','additionalProperties':False,
        'properties':{'facts':{'type':'array','items':fact},'unresolved':{'type':'array','items':{'type':'object','additionalProperties':False,
        'properties':{'pointer':{'type':'string'},'reason':{'type':'string','minLength':1}},'required':['pointer','reason']}}},'required':['facts','unresolved']}
    Draft202012Validator.check_schema(result);return result


def validate_output(answer,product):
    Draft202012Validator(output_schema()).validate(answer)
    identities=set()
    for f in answer['facts']:
        validate_fact(f)
        if f['review_status']!='pending' or f['product_asin']!=str(product['asin']) or f['product_sha256']!=fingerprint(product):
            raise ValueError('Fact changed identity or approved itself')
        for e in f['evidence']:
            actual=pointer(product,e['pointer'])
            if 'value' not in e or isinstance(actual,(dict,list)) or actual!=e['value']:
                raise ValueError('Use an exact public scalar source value, not offsets or whole objects')
            if e['pointer'].startswith('/customization_options/'):
                tokens=e['pointer'].split('/');axis=tokens[2].replace('~1','/').replace('~0','~')
                entry=product['customization_options'][axis][int(tokens[3])]
                if f['scope']!='sku' or f.get('options',{}).get(axis)!=entry['value']:
                    raise ValueError('Option source is not bound to its SKU axis/value')
        key=dump({k:f.get(k) for k in ('scope','options','dimension','subtype','attribute','subject','value','polarity')})
        if key in identities:raise ValueError('Duplicate fact')
        identities.add(key)
    return answer


def prepare(mode,folder):
    with gzip.open(ARCHIVE,'rt',encoding='utf-8') as handle:original=json.load(handle)
    products={str(p['asin']):p for p in original}
    if mode=='19':
        probes=rows(ROOT/'analysis/semantic_product_verifier_20261107/train20_target_probes.jsonl')
        asins=list(dict.fromkeys(str(r['asin']) for r in probes))
    elif mode=='history':
        asins=loads((ROOT/'analysis/semantic_history_acceptance_20261107/actual_product_asins.json').read_text(encoding='utf-8'))
    elif mode=='catalog':
        # Entire public candidate catalog, without reading task text/answers.
        asins=list(products)
    else:
        # IDs locate product records only. No instruction text, target
        # variants, attributes attached to tasks, or rewards enter requests.
        tasks=[p for p in original for i in p.get('instructions',[]) if i.get('attributes')]
        asins=list(dict.fromkeys(str(tasks[r['task_id']]['asin']) for r in rows(ROOT/'upstream/shopping-grpo-longhorizon/data/grpo/train.jsonl')))
    prompt=PROMPT.read_text(encoding='utf-8');schema=output_schema()
    requests=[];public=[]
    for asin in asins:
        product=public_product(products[asin])
        # Source fingerprint binds the original public catalog record. The
        # request omits images, reviews and non-evidence fields, not answers.
        catalog_hash=fingerprint(products[asin])
        public.append({'product_asin':asin,'catalog_sha256':catalog_hash,'public_product':product})
        user={'product_asin':asin,'product_sha256':catalog_hash,'product':product}
        requests.append({'product_asin':asin,'product_sha256':catalog_hash,'messages':[
            {'role':'system','content':prompt+'\nOUTPUT_SCHEMA\n'+dump(schema)}, {'role':'user','content':dump(user)}]})
    manifest={'mode':mode,'products':len(public),'public_sha256':digest(public),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
              'schema_sha256':digest(schema),'source_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),'queries_sent':0,'automatic_approvals':0}
    existing=folder/'manifest.json'
    if existing.exists() and loads(existing.read_text(encoding='utf-8'))!=manifest:raise ValueError('Prepared inputs changed; use a new output folder')
    folder.mkdir(parents=True,exist_ok=True)
    write(existing,manifest);write(folder/'output.schema.json',schema)
    (folder/'system_prompt.txt').write_text(prompt,encoding='utf-8')
    for name,values in [('public_products.jsonl',public),('requests.jsonl',requests)]:
        (folder/name).write_text(''.join(dump(r)+'\n' for r in values),encoding='utf-8')
    return products,requests,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['19','1000','catalog','history'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--base-url',default=os.getenv('LLM_BASE_URL'))
    p.add_argument('--model',default=os.getenv('LLM_MODEL'))
    p.add_argument('--api-key-env',default='LLM_API_KEY')
    p.add_argument('--allow-school-http',action='store_true')
    p.add_argument('--limit',type=int)
    p.add_argument('--max-tokens',type=int,default=8192)
    p.add_argument('--timeout',type=int,default=180)
    args=p.parse_args()
    if args.limit is not None and args.limit<=0:p.error('--limit must be positive')
    if args.max_tokens<=0 or args.timeout<=0:p.error('Numeric options must be positive')
    if args.base_url:endpoint(args.base_url,allow_school_http=args.allow_school_http,local_only=args.prepare_only)
    args.output.mkdir(parents=True,exist_ok=True)
    lock=args.output/'.run.lock'
    try:handle=lock.open('x',encoding='utf-8')
    except FileExistsError:raise SystemExit('Run locked; check the active process before removing .run.lock')
    try:
        handle.write(str(os.getpid()));handle.close()
        products,requests,manifest=prepare(args.mode,args.output)
        if args.prepare_only:
            print(dump({**manifest,'api_calls':0,'prepared_only':True}));return
        if not args.base_url or not args.model:p.error('Set LLM_BASE_URL and LLM_MODEL')
        url=endpoint(args.base_url,allow_school_http=args.allow_school_http)
        key=os.getenv(args.api_key_env,'').strip()
        if not key:raise SystemExit('Missing '+args.api_key_env+'; no API requests sent')
        config={'model':args.model,'endpoint':url,'max_tokens':args.max_tokens,'temperature':0,'thinking':'omit'}
        config_path=args.output/'api_config.json'
        if config_path.exists() and loads(config_path.read_text(encoding='utf-8'))!=config:raise SystemExit('API configuration changed; use a new output folder')
        write(config_path,config)
        attempts=rows(args.output/'attempts.jsonl');valid={}
        for record in attempts:
            if 'validated' in record:
                try:valid[record['product_asin']]=validate_output(record['validated'],products[record['product_asin']])
                except (ValueError,ValidationError,KeyError,TypeError,IndexError):pass
        completed=0
        for request in requests:
            asin=request['product_asin']
            if asin in valid:continue
            if args.limit is not None and completed>=args.limit:break
            record={'product_asin':asin}
            try:
                _,response=post(url,key,{'model':args.model,'messages':request['messages'],'temperature':0,'stream':False,'max_tokens':args.max_tokens,'response_format':{'type':'json_object'}},args.timeout)
                envelope=loads(response);record['response']=envelope
                raw=loads(envelope['choices'][0]['message']['content'])
                record['validated']=validate_output(raw,products[asin]);valid[asin]=record['validated']
                print(f'product={asin} VALID',flush=True)
            except HTTPFailure as error:
                record['error']=f'HTTP {error.status}'
            except (ValueError,ValidationError,KeyError,TypeError,IndexError,URLError,TimeoutError) as error:record['error']=str(error)[:600]
            with (args.output/'attempts.jsonl').open('a',encoding='utf-8') as f:f.write(dump(record)+'\n')
            completed+=1
            if 'error' in record:
                print(f'product={asin} FAILED '+record['error'],flush=True)
                break
        (args.output/'draft_facts.jsonl').write_text(''.join(dump(f)+'\n' for v in valid.values() for f in v['facts']),encoding='utf-8')
        report={'products':len(requests),'program_valid_products':len(valid),'complete':len(valid)==len(requests),'attempts_this_run':completed,'automatic_approvals':0}
        write(args.output/'report.json',report);print(dump(report))
    finally:
        if not handle.closed:handle.close()
        lock.unlink()

if __name__=='__main__':main()
