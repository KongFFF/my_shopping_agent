"""Build unmodified pinned runtime and freeze 48 train-only tasks, without an actor."""
import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from decision_probe_common import (ROOT, PACKAGE, BASE_COMMIT, VERSION, code_receipt,
    file_hash, load, rows, save, digest, runtime_receipt, check_package)
from decision_probe_rules import requirements

UP = ROOT/'upstream/shopping-grpo-longhorizon'


def build(bundle):
    bundle = bundle.resolve()
    if not bundle.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Runtime must be inside outputs')
    if bundle.exists():
        check_package()
        if runtime_receipt(bundle) != load(PACKAGE/'runtime_receipt.json'):
            raise ValueError('Existing runtime differs')
        print('Existing native runtime verified'); return
    (ROOT/'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'outputs', prefix='decision_native_') as tmp:
        tmp = Path(tmp); archive = tmp/'base.tar'; base = tmp/'base'; base.mkdir()
        with archive.open('wb') as stream:
            subprocess.run(['git', '-c', 'core.autocrlf=false', '-C', str(UP), 'archive', BASE_COMMIT, 'src', 'data/grpo',
                'data/evaluation/tasks.jsonl', 'data/environment.json', 'scripts/serve_model.sh',
                'environments/ShopSimulator/shop_env'], stdout=stream, check=True)
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                if not (base/m.name).resolve().is_relative_to(base) or m.issym() or m.islnk():
                    raise ValueError('Unsafe archive member')
            tf.extractall(base, filter='data')
        stage = tmp/'runtime'; stage.mkdir()
        shop = base/'environments/ShopSimulator/shop_env'
        ignore = shutil.ignore_patterns('data', 'search_engine', '__pycache__', '.venv*', '.git', 'outputs')
        shutil.copytree(shop, stage/'environment', ignore=ignore)
        shutil.copytree(base/'src/shopping_grpo', stage/'agent/shopping_grpo', ignore=ignore)
        shutil.copytree(base/'data/grpo', stage/'data')
        shutil.copy2(base/'scripts/serve_model.sh', stage/'serve_model.sh')
        # serve_model.sh is recorded as reference; launch uses its exact CLI below.
        # Pinned data/environment.json has one stale source hash. The returned
        # original-training manifest corrects that hash; executable files remain
        # byte-for-byte pinned Git sources and must match every declared binding.
        shutil.copy2(PACKAGE/'environment_metadata.json', stage/'environment.json')
        raw = gzip.decompress((shop/'data/fine_items_eval_train_all.json.gz').read_bytes())
        catalog = json.loads(raw)
        (stage/'environment/data').mkdir(); (stage/'environment/data/items_eval_train.json').write_bytes(raw)
        envmeta = load(stage/'environment.json')
        if digest(raw) != envmeta['product_data_sha256']: raise ValueError('Catalog differs')
        for name, expected in envmeta['runtime_files_sha256'].items():
            if not any(file_hash(p) == expected for p in (stage/'environment').rglob(name)):
                raise ValueError('Native source binding differs: '+name)
        sys.path.insert(0, str(stage/'environment'))
        from web_agent_site.engine.search import build_index
        index = build_index(catalog, stage/'environment/search_engine/products.sqlite3', product_data_sha256=digest(raw))
        save(stage/'index_manifest.json', index)
        save(stage/'manifest.json', {'variant': VERSION, 'native_unmodified': True, 'base_commit': BASE_COMMIT})
        # Same instruction ordering as pinned get_existed_goals; no goal SKU is exported.
        queries = [p['instruction'] for item in catalog if 'instructions' in item
                   for p in item['instructions'] if p.get('attributes')]
        ids = [int(r['task_id']) for r in rows(stage/'data/train.jsonl')]
        forbidden = {int(r['task_id']) for p in (stage/'data/validation.jsonl', base/'data/evaluation/tasks.jsonl') for r in rows(p)}
        assert len(ids) == len(set(ids)) == 1000 and not set(ids) & forbidden
        ordered = sorted(ids, key=lambda t: digest('decision800-v1:uniform:'+str(t)))
        uniform = ordered[:32]
        enriched = sorted([t for t in ids if t not in uniform and requirements(queries[t])],
                          key=lambda t: digest('decision800-v1:enriched:'+str(t)))[:16]
        if len(enriched) != 16: raise ValueError('Not enough public requirement tasks')
        # Interleave strata; smoke comprises first four, never adaptive task selection.
        selected = [t for i in range(16) for t in (uniform[2*i], uniform[2*i+1], enriched[i])]
        tasks = [{'task_id': t, 'query': queries[t], 'query_sha256': digest(queries[t]),
                  'stratum': 'uniform32' if t in uniform else 'public_requirement16'} for t in selected]
        PACKAGE.mkdir(parents=True, exist_ok=True)
        rawtasks = ''.join(json.dumps(t, ensure_ascii=False, sort_keys=True)+'\n' for t in tasks).encode()
        receipt = runtime_receipt(stage)
        manifest = {'version': VERSION, 'base_commit': BASE_COMMIT, 'code_sha256': code_receipt(),
            'source_sha256': {'train': file_hash(stage/'data/train.jsonl'), 'environment': file_hash(stage/'environment.json')},
            'selection': '32 smallest sha256(decision800-v1:uniform:ID), then 16 requirement-cue tasks by enriched hash, interleaved 2:1',
            'settings': {'tasks': 48, 'group_size': 4, 'smoke_tasks': 4, 'gpu_seconds': 7200,
                         'max_model_calls': 6000, 'max_completion_tokens': 250000,
                         'temperature': .7, 'top_p': .9, 'max_tokens': 512, 'max_steps': 35,
                         'context_window': 24576, 'remaining_steps_at_trigger_min': 3},
            'task_ids': selected}
        frozen = {'environment_metadata.json': (PACKAGE/'environment_metadata.json').read_bytes(),
                  'tasks.jsonl': rawtasks, 'runtime_receipt.json': (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()}
        for name, data in frozen.items():
            p = PACKAGE/name
            if p.exists() and p.read_bytes() != data: raise ValueError('Frozen input differs: '+name)
            p.write_bytes(data)
        manifest['files_sha256'] = {name: file_hash(PACKAGE/name) for name in frozen}
        if (PACKAGE/'manifest.json').exists() and load(PACKAGE/'manifest.json') != manifest:
            raise ValueError('Protocol differs; do not overwrite a frozen experiment')
        save(PACKAGE/'manifest.json', manifest)
        bundle.parent.mkdir(parents=True, exist_ok=True); stage.replace(bundle)
    print('PASS: native runtime and 48 tasks frozen; no model/GPU used')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, default=ROOT/'outputs/decision_probe_v1/runtime')
    a = p.parse_args(); build(a.bundle)
