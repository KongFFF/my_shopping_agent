"""Build frozen task data and the unmodified pinned baseline runtime, CPU only."""
import argparse
import csv
import gzip
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from collections import defaultdict,Counter
from checkpoint_pair_common import *


def select_tasks():
    source=ROOT/'analysis/grpo_baseline_20261022_202816'
    with (source/'training_diagnosis/groups.csv').open(encoding='utf-8-sig') as f:gs=list(csv.DictReader(f))
    audit={int(r['task_id']):r for r in load(source/'derived/gold_annotation_audit.json') if r['split']=='train'}
    early=defaultdict(list)
    for r in gs:
        if int(r['step'])<=100:early[int(r['task_id'])].append(r)
    mixed={t for t,rs in early.items() if any(r['kept']=='True' and 0<int(r['gold_rollouts'])<int(r['group_size']) for r in rs)}
    all_gold={t for t,rs in early.items() if all(int(r['gold_rollouts'])==int(r['group_size']) for r in rs)}
    no_gold={t for t,rs in early.items() if sum(int(r['gold_rollouts']) for r in rs)==0 and
             (audit[t]['reward_type']=='gold_purchase' or audit[t]['variant_search'].get('gold_found'))}
    if (len(mixed),len(all_gold),len(no_gold))!=(182,416,175):raise ValueError('Source cohort counts changed')
    order=lambda t:digest('checkpoint-pair-20261111:'+str(t))
    result=[]
    for cohort,ids in [('mixed',sorted(mixed,key=order)),('early_all_gold',sorted(all_gold,key=order)[:40]),
                       ('early_no_gold_feasible',sorted(no_gold,key=order)[:40])]:
        for t in ids:result.append({'task_id':t,'query':audit[t]['query'],'query_sha256':digest(audit[t]['query']),'cohort':cohort})
    train={int(r['task_id']) for r in rows(UP/'data/grpo/train.jsonl')}
    forbidden={int(r['task_id']) for name in ('data/grpo/validation.jsonl','data/evaluation/tasks.jsonl') for r in rows(UP/name)}
    if not set(t['task_id'] for t in result)<=train or set(t['task_id'] for t in result)&forbidden:
        raise ValueError('Task split violation')
    provenance={'groups_csv':file_hash(source/'training_diagnosis/groups.csv'),
                'gold_audit':file_hash(source/'derived/gold_annotation_audit.json'),
                'pool_counts':{'mixed':182,'early_all_gold':416,'early_no_gold_feasible':175}}
    return result,provenance


def freeze(development_refresh=False):
    tasks,provenance=select_tasks();PACKAGE.mkdir(parents=True,exist_ok=True)
    raw=''.join(__import__('json').dumps(t,ensure_ascii=False,sort_keys=True)+'\n' for t in tasks).encode()
    path=PACKAGE/'tasks.jsonl'
    if path.exists() and path.read_bytes()!=raw:raise ValueError('Frozen task data differs')
    path.write_bytes(raw)
    files={'tasks.jsonl':file_hash(path)}
    if (PACKAGE/'runtime_receipt.json').exists():files['runtime_receipt.json']=file_hash(PACKAGE/'runtime_receipt.json')
    m={'version':VERSION,'base_commit':BASE,'settings':SETTINGS,'code_sha256':code_receipt(),
       'files_sha256':files,'selection_provenance':provenance,
       'task_counts':dict(Counter(t['cohort'] for t in tasks)),
       'smoke_task_ids':[t['task_id'] for t in tasks[:3]],
       'models':MODEL_PATHS,'runtime':'Unmodified pinned agent and environment; fresh independent episode initialization only'}
    if (PACKAGE/'manifest.json').exists() and load(PACKAGE/'manifest.json')!=m and not development_refresh:
        raise ValueError('Existing protocol differs; do not refreeze after execution')
    save(PACKAGE/'manifest.json',m);print(dump({'frozen':True,'counts':m['task_counts'],'smoke_task_ids':m['smoke_task_ids']}))


def build(bundle):
    check_package();bundle=bundle.resolve()
    if not bundle.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Runtime must be under project outputs')
    lock=PACKAGE/'runtime_receipt.json'
    if bundle.exists():
        if runtime_receipt(bundle)!=load(lock):raise ValueError('Existing runtime changed')
        print('PASS: existing baseline runtime');return
    (ROOT/'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='checkpoint_pair_build_',dir=ROOT/'outputs') as temp:
        temp=Path(temp);base=temp/'base';base.mkdir();archive=temp/'base.tar'
        with archive.open('wb') as f:
            subprocess.run(['git','-C',str(UP),'archive',BASE,'src','environments/ShopSimulator/shop_env'],stdout=f,check=True)
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                if not (base/m.name).resolve().is_relative_to(base.resolve()) or m.issym() or m.islnk():raise ValueError('Unsafe archive member')
            tf.extractall(base,filter='data')
        stage=temp/'runtime';stage.mkdir()
        shutil.copytree(base/'src/shopping_grpo',stage/'agent/shopping_grpo')
        shutil.copytree(base/'environments/ShopSimulator/shop_env',stage/'environment')
        env=stage/'environment'
        with gzip.open(env/'data/fine_items_eval_train_all.json.gz','rb') as f:catalog=f.read()
        (env/'data/items_eval_train.json').write_bytes(catalog)
        sys.path.insert(0,str(env))
        from web_agent_site.engine.search import build_index
        idx=build_index(__import__('json').loads(catalog),env/'search_engine/products.sqlite3',product_data_sha256=digest(catalog))
        save(stage/'index_manifest.json',idx)
        receipt=runtime_receipt(stage)
        if lock.exists() and load(lock)!=receipt:raise ValueError('Rebuilt baseline receipt differs')
        bundle.parent.mkdir(parents=True,exist_ok=True);stage.replace(bundle)
        if not lock.exists():save(lock,receipt)
    print(dump({'runtime_ready':True,'files':len(receipt),'purchase_guard_added':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','build','check'])
    p.add_argument('--bundle',type=Path,default=ROOT/'outputs/checkpoint_pair_v1/runtime')
    p.add_argument('--development-refresh',action='store_true',help='Pre-release only; never after evaluation')
    a=p.parse_args()
    if a.command=='freeze':freeze(a.development_refresh)
    elif a.command=='build':build(a.bundle)
    else:
        check_package()
        if runtime_receipt(a.bundle)!=load(PACKAGE/'runtime_receipt.json'):raise ValueError('Runtime mismatch')
        print('PASS: frozen protocol, tasks, and original baseline runtime')
