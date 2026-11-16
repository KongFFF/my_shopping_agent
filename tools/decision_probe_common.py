"""Shared, standard-library-only contracts for the frozen local sampling probe."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'experiments/decision_probe_v1'
BASE_COMMIT = 'a583dba4c149be09622ce58ad6f7a160db77bb41'
VERSION = 'decision-probe-v1'


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'


def digest(value):
    return hashlib.sha256(value.encode('utf-8') if isinstance(value, str) else value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_hash(path):
    return digest(Path(path).read_text(encoding='utf-8-sig'))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(dump(value).encode('utf-8'))
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n')
        f.flush()


def code_receipt():
    names = ['decision_probe_common.py', 'prepare_decision_probe.py',
             'run_decision_probe_environment.py', 'run_decision_probe.py',
             'report_decision_probe.py', 'decision_probe_rules.py', 'export_decision_probe.py']
    return {name: code_hash(ROOT / 'tools' / name) for name in names}


def check_package():
    m = load(PACKAGE / 'manifest.json')
    if m['version'] != VERSION or m['code_sha256'] != code_receipt():
        raise ValueError('Probe code changed; freeze a new protocol before sampling')
    for name, h in m['files_sha256'].items():
        if file_hash(PACKAGE / name) != h:
            raise ValueError('Frozen input changed: ' + name)
    return m


def runtime_receipt(bundle):
    bundle = Path(bundle)
    # Bind every runtime Python file, configuration, catalog and search index.
    paths = [p for base in (bundle/'environment', bundle/'agent') for p in base.rglob('*')
             if p.is_file() and p.suffix in ('.py', '.json', '.html', '.jinja', '.sql')]
    paths += [bundle / x for x in ('environment.json', 'manifest.json', 'serve_model.sh',
              'environment/data/items_eval_train.json', 'environment/search_engine/products.sqlite3')]
    index = bundle/'environment/search_engine/products.sqlite3'
    if file_hash(index) != load(bundle/'index_manifest.json')['index_sha256']:
        raise ValueError('Search index bytes differ from its local build receipt')
    return {p.relative_to(bundle).as_posix(): (index_content_hash(p) if p.suffix == '.sqlite3'
            else code_hash(p) if p.suffix == '.py' else file_hash(p))
            for p in sorted(set(paths))}


def index_content_hash(path):
    # SQLite embeds Python/SQLite versions in the manifest table, so identical
    # indexes rebuilt on Windows/Linux cannot share their binary file digest.
    # Bind indexed documents and semantic settings globally; bind actual bytes
    # separately per build/run. Reject corruption before making this comparison.
    connection = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    try:
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Search index integrity check failed')
        manifest = json.loads(connection.execute('SELECT payload FROM manifest').fetchone()[0])
        for key in ('python_version', 'sqlite_version'): manifest.pop(key, None)
        h = hashlib.sha256(dump(manifest).encode('utf-8'))
        for row in connection.execute('SELECT * FROM products ORDER BY rowid'):
            h.update(dump(row).encode('utf-8'))
        return h.hexdigest()
    finally:
        connection.close()


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, started_at, seconds=7200):
        self.deadline = float(started_at) + seconds

    def remaining(self):
        return self.deadline - time.time()

    def check(self):
        if self.remaining() <= 0:
            raise BudgetExceeded('Two-hour GPU wall-clock budget exhausted')

    def timeout(self, maximum=60):
        self.check()
        return max(0.01, min(maximum, self.remaining()))
