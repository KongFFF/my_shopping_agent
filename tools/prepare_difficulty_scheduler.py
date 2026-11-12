"""Freeze inputs, export the pinned runtime, and stage an isolated veRL overlay."""
import argparse
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from difficulty_scheduler import VERSION, canonical, read, sha, write
from difficulty_scheduler_patch import ORIGINAL, DYNAMIC, build_trainer

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / 'upstream/shopping-grpo-longhorizon'
PACKAGE = ROOT / 'experiments/difficulty_scheduler_v1'
BASE = 'a583dba4c149be09622ce58ad6f7a160db77bb41'
SCRIPTS = ['difficulty_scheduler.py', 'difficulty_scheduler_hook.py', 'difficulty_scheduler_patch.py',
           'prepare_difficulty_scheduler.py', 'difficulty_scheduler_environment.py',
           'run_difficulty_scheduler.py', 'report_difficulty_scheduler.py',
           'test_difficulty_scheduler.py']
SETTINGS = {'budget': 4000, 'seed': 20261112, 'group_size': 4, 'tasks_per_batch': 2,
            'uniform_fraction': .5, 'cap_multiple': 4, 'half_life_updates': 100,
            'save_every_updates': 50, 'warmup_updates': 15,
            'arms': ['uniform', 'adaptive'], 'environment_port': 5723, 'model_port': 8023}
OLD_ENV_FILE_HASH = '448f5ec31fe5a71d1e43376d42cd84787a0e48fe96a8032fc0383f65e5928ea2'
PINNED_ENV_FILE_HASH = 'd6db15f282849847a364d8d36af0eed113c1c97184a1ded98eb1fd29dab577d1'


def git_bytes(relative):
    return subprocess.check_output(['git', '-C', str(UP), 'show', BASE + ':' + relative])


def text_hash(path):
    return sha(Path(path).read_text(encoding='utf-8-sig').replace('\r\n', '\n'))


def file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def check_package():
    m = read(PACKAGE / 'manifest.json')
    if m['version'] != VERSION or m['settings'] != SETTINGS or m['base'] != BASE:
        raise ValueError('Frozen experiment settings changed')
    if m['code'] != {n: text_hash(ROOT / 'tools' / n) for n in SCRIPTS}:
        raise ValueError('Frozen experiment code changed')
    for name, expected in m['files'].items():
        if text_hash(PACKAGE / name) != expected:
            raise ValueError('Frozen input changed: ' + name)
    train, dev, validation, final = (lines(PACKAGE / n) for n in
        ('train_tasks.jsonl', 'dev50_tasks.jsonl', 'validation_tasks.jsonl', 'old_final_tasks.jsonl'))
    ids = lambda rows: {int(r['task_id']) for r in rows}
    if len(train) != 1000 or len(ids(train)) != 1000 or len(dev) != 50 or len(ids(dev)) != 50:
        raise ValueError('Train1000/Dev50 roster changed')
    if ids(train) & (ids(dev) | ids(validation) | ids(final)) or ids(dev) & ids(final):
        raise ValueError('Task split overlap')
    if ids(dev) != ids(validation):
        raise ValueError('Historical Dev50 and original validation roster unexpectedly differ')
    return m


def freeze():
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for source, target in [('data/grpo/train.jsonl', 'train_tasks.jsonl'),
                           ('data/grpo/validation.jsonl', 'validation_tasks.jsonl'),
                           ('data/evaluation/tasks.jsonl', 'old_final_tasks.jsonl')]:
        (PACKAGE / target).write_bytes(git_bytes(source))
    history = ROOT / 'analysis/grpo_baseline_20261022_202816/sft_dev50/trajectories.jsonl'
    dev = sorted({int(r['task_id']) for r in lines(history)})
    if len(dev) != 50:
        raise ValueError('Historical Dev50 source not complete')
    (PACKAGE / 'dev50_tasks.jsonl').write_text(''.join(canonical({'task_id': t}) + '\n' for t in dev), encoding='utf-8', newline='\n')
    audit = ROOT / 'analysis/grpo_baseline_20261022_202816/derived/gold_annotation_audit.json'
    screened = ROOT / 'analysis/checkpoint_pair_result_20261112/redundant_axis_screen.json'
    no_gold = sorted(r['task_id'] for r in read(audit) if r['split'] == 'train' and not
                     (r['reward_type'] == 'gold_purchase' or r.get('variant_search', {}).get('gold_found')))
    risk = sorted({r['task_id'] for r in read(screened)})
    if len(no_gold) != 90 or len(risk) != 18:
        raise ValueError('Known diagnostic cohorts changed')
    write(PACKAGE / 'diagnostic_labels.json', {'static_no_gold_ids': no_gold,
          'singleton_axis_screen_ids': risk, 'label_sources': {'static': file_hash(audit), 'screen': file_hash(screened)},
          'use': 'Offline reporting only; not scheduler inputs, not exclusions or semantic truth'})
    names = ['train_tasks.jsonl', 'dev50_tasks.jsonl', 'validation_tasks.jsonl', 'old_final_tasks.jsonl',
             'ray_trainer_0_8_0.py', 'baseline_dynamic.patch', 'diagnostic_labels.json']
    if (PACKAGE / 'runtime_receipt.json').exists():
        names.append('runtime_receipt.json')
    write(PACKAGE / 'manifest.json', {'version': VERSION, 'base': BASE, 'settings': SETTINGS,
          'code': {n: text_hash(ROOT / 'tools' / n) for n in SCRIPTS},
          'files': {n: text_hash(PACKAGE / n) for n in names},
          'dev_source_sha256': file_hash(history), 'training_executed': False})
    check_package()
    print('PASS: frozen train1000, historical Dev50 (= validation50) and experiment code')


def receipt(bundle):
    from checkpoint_pair_common import index_hash
    project = Path(bundle) / 'project'
    result = {}
    for path in sorted(project.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc', '.log'):
            continue
        if path.name == 'index_manifest.json':
            continue  # Physical, machine-local SQLite binding is checked separately.
        name = path.relative_to(project).as_posix()
        result[name] = index_hash(path) if path.suffix == '.sqlite3' else file_hash(path)
    return result


def check_runtime(bundle, overlay=False):
    m = check_package()
    bundle = Path(bundle).resolve()
    if receipt(bundle) != read(PACKAGE / 'runtime_receipt.json'):
        raise ValueError('Isolated baseline runtime differs')
    if read(bundle / 'metadata_correction.json') != {
        'file': 'project/data/environment.json', 'field': 'runtime_files_sha256.web_agent_text_env.py',
        'before': OLD_ENV_FILE_HASH, 'after': PINNED_ENV_FILE_HASH,
        'source_modified': False, 'reward_modified': False}:
        raise ValueError('Environment metadata correction receipt differs')
    env = bundle / 'project/environments/ShopSimulator/shop_env'
    if file_hash(env / 'search_engine/products.sqlite3') != read(bundle / 'index_manifest.json')['index_sha256']:
        raise ValueError('Physical search index changed')
    if overlay:
        installed = read(bundle / 'overlay_receipt.json')
        actual = {p.relative_to(bundle / 'overlay').as_posix(): file_hash(p)
                  for p in sorted((bundle / 'overlay').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
        if installed['files'] != actual:
            raise ValueError('Isolated veRL overlay changed')
        trainer = bundle / 'overlay/verl/trainer/ppo/ray_trainer.py'
        expected = build_trainer((PACKAGE / 'ray_trainer_0_8_0.py').read_text(encoding='utf-8'),
                                 (PACKAGE / 'baseline_dynamic.patch').read_text(encoding='utf-8'))
        if trainer.read_text(encoding='utf-8') != expected:
            raise ValueError('Scheduler trainer differs from frozen transform')
    return m


def build(bundle):
    check_package()
    bundle = Path(bundle).resolve()
    if not bundle.is_relative_to((ROOT / 'outputs').resolve()):
        raise ValueError('Build target must be inside project outputs')
    if bundle.exists():
        check_runtime(bundle)
        print('PASS: existing isolated baseline runtime')
        return
    (ROOT / 'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='schedule_build_', dir=ROOT / 'outputs') as tmp:
        tmp = Path(tmp)
        archive, stage = tmp / 'source.tar', tmp / 'runtime'
        project = stage / 'project'
        project.mkdir(parents=True)
        with archive.open('wb') as f:
            subprocess.run(['git', '-C', str(UP), 'archive', BASE, 'src', 'configs', 'scripts',
                            'data/grpo', 'data/environment.json', 'environments/ShopSimulator/shop_env'], stdout=f, check=True)
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.issym() or member.islnk() or not (project / member.name).resolve().is_relative_to(project):
                    raise ValueError('Unsafe Git archive')
            tf.extractall(project, filter='data')
        env = project / 'environments/ShopSimulator/shop_env'
        # Pinned Git source and the previous checkpoint-pair receipt agree on
        # d6db..., but data/environment.json still records the obsolete 448f....
        # Correct only the stale metadata in this exported copy; never source.
        manifest_path = project / 'data/environment.json'
        manifest = read(manifest_path)
        actual = file_hash(env / 'web_agent_site/envs/web_agent_text_env.py')
        recorded = manifest['runtime_files_sha256']['web_agent_text_env.py']
        if actual != PINNED_ENV_FILE_HASH or recorded != OLD_ENV_FILE_HASH:
            raise ValueError('Unexpected pinned environment metadata/source: refuse implicit correction')
        manifest['runtime_files_sha256']['web_agent_text_env.py'] = actual
        write(manifest_path, manifest)
        write(stage / 'metadata_correction.json', {'file': 'project/data/environment.json',
              'field': 'runtime_files_sha256.web_agent_text_env.py', 'before': recorded, 'after': actual,
              'source_modified': False, 'reward_modified': False})
        import gzip
        with gzip.open(env / 'data/fine_items_eval_train_all.json.gz', 'rb') as f:
            raw = f.read()
        (env / 'data/items_eval_train.json').write_bytes(raw)
        sys.path.insert(0, str(env))
        from web_agent_site.engine.search import build_index
        index = build_index(json.loads(raw), env / 'search_engine/products.sqlite3', product_data_sha256=sha(raw))
        write(stage / 'index_manifest.json', index)
        observed = receipt(stage)
        lock = PACKAGE / 'runtime_receipt.json'
        if lock.exists() and read(lock) != observed:
            raise ValueError('Rebuilt baseline receipt differs')
        if not lock.exists():
            write(lock, observed)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(bundle)
    print('PASS: pinned original source/data exported; original worktree untouched')


def overlay(bundle):
    check_runtime(bundle)
    if importlib.metadata.version('verl') != '0.8.0':
        raise ValueError('Requires verl==0.8.0')
    source = Path(importlib.util.find_spec('verl').origin).parent
    trainer = source / 'trainer/ppo/ray_trainer.py'
    if text_hash(trainer) not in (ORIGINAL, DYNAMIC):
        raise ValueError('Installed veRL trainer is unknown; refusing to copy')
    destination = Path(bundle) / 'overlay'
    if destination.exists():
        check_runtime(bundle, overlay=True)
        print('PASS: existing isolated veRL overlay')
        return
    destination.mkdir()
    shutil.copytree(source, destination / 'verl', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    patched = build_trainer((PACKAGE / 'ray_trainer_0_8_0.py').read_text(encoding='utf-8'),
                            (PACKAGE / 'baseline_dynamic.patch').read_text(encoding='utf-8'))
    (destination / 'verl/trainer/ppo/ray_trainer.py').write_text(patched, encoding='utf-8', newline='\n')
    write(Path(bundle) / 'overlay_receipt.json', {'source': str(source), 'files': {
        p.relative_to(destination).as_posix(): file_hash(p) for p in sorted(destination.rglob('*')) if p.is_file()}})
    check_runtime(bundle, overlay=True)
    print('PASS: isolated veRL copy patched; site-packages unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'build', 'overlay', 'check'])
    parser.add_argument('--bundle', type=Path, default=ROOT / 'outputs/difficulty_scheduler_v1/runtime')
    parser.add_argument('--with-overlay', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'build':
        build(args.bundle)
    elif args.command == 'overlay':
        overlay(args.bundle)
    else:
        check_runtime(args.bundle, args.with_overlay)
        print('PASS: frozen scheduler protocol, task boundaries and isolated runtime')


if __name__ == '__main__':
    main()
