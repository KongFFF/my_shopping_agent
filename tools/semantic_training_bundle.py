"""Frozen source-label registry and CPU reward backend for the real GRPO loop.

No LLM, model loading, target-ASIN lookup, or actor prompt changes. Source
release needs exact whole-query proof or explicit batch acceptance; program
validity alone never grants permission or falls back to Reward v3.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from semantic_contract_v2_r7_final12 import DIMENSIONS, validate
from semantic_evidence_reward import VERSION, score_terminal_failure, score_training_verified
from semantic_history_adapter import canonicalize_purchase
from semantic_product_verifier_v4 import verify

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream/shopping-grpo-longhorizon'
DEFAULT_OUTPUT = ROOT / 'analysis/semantic_training_signoff_20261107'


def sha256(path):
    path = Path(path)
    if path.suffix in ('.py', '.json', '.jsonl', '.yaml', '.toml'):
        # Git may check text out with CRLF on Windows and LF on AutoDL.
        # Only newline encoding is canonicalized; content must remain identical.
        return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build_bundle(output=DEFAULT_OUTPUT, *, approvals_path=None, contracts_path=None, batch_release_path=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = json.loads((ROOT / 'analysis/semantic_latest.json').read_text(encoding='utf-8'))
    contracts_path = Path(contracts_path).resolve() if contracts_path else ROOT / latest['contracts']
    contracts = rows(contracts_path)
    batch = None
    batch_paths = {}
    batch_released = set()
    batch_held = set()
    if batch_release_path:
        from semantic_batch_release import verify_release
        batch, batch_paths, batch_released, batch_held = verify_release(
            batch_release_path, root=ROOT, contracts=contracts_path, sha256=sha256, rows=rows)
    from build_semantic_pilot50 import source_tasks
    original_queries, _ = source_tasks()  # Retains only task ID/query; never target answers.
    original_queries = {r['task_id']: r['query'] for r in original_queries}
    train_path = UPSTREAM / 'data/grpo/train.jsonl'
    val_path = UPSTREAM / 'data/grpo/validation.jsonl'
    eval_path = UPSTREAM / 'data/evaluation/tasks.jsonl'
    train_ids = [int(r['task_id']) for r in rows(train_path)]
    validation_ids = [int(r['task_id']) for r in rows(val_path)]
    eval_ids = [int(r['task_id']) for r in rows(eval_path)]
    if len(train_ids) != 1000 or len(set(train_ids)) != 1000:
        raise ValueError('Expected the unchanged original 1000 unique training tasks')
    if set(train_ids) & (set(validation_ids) | set(eval_ids)):
        raise ValueError('Training/validation/evaluation overlap')
    if len(contracts) != 1000 or {r['task_id'] for r in contracts} != set(train_ids):
        raise ValueError('Contracts must cover exactly the original 1000-task pool')
    # The prior seven approvals are whole-query reviews, not program-valid flags.
    approvals_path = Path(approvals_path) if approvals_path else ROOT / 'analysis/semantic_train_smoke7_20261107/reviewed_contracts7.jsonl'
    approvals_path = approvals_path.resolve()
    approvals = {r['task_id']: r for r in rows(approvals_path)}
    review_paths = [ROOT / 'analysis' / rel for rel in (
        'semantic_review_four_steps_20261106/step1_review20.jsonl',
        'semantic_review_four_steps_20261106/step3_representative_review42.jsonl',
        'semantic_review_four_steps_20261106/step4_review30.jsonl',
        'semantic_acceptance_20261107/review40.jsonl',
        'semantic_targeted_review106_20261107/review106.jsonl')]
    prior_review = {}
    for path in review_paths:
        for record in rows(path):
            task = record.get('task_id')
            if task in original_queries and (not record.get('query') or record['query'] == original_queries[task]):
                prior_review.setdefault(task, []).append(path.relative_to(ROOT).as_posix())
    registry = []
    for row in contracts:
        c = row['normalized']
        if c['task_id'] != row['task_id'] or c['query'] != row['query'] or row['query'] != original_queries[row['task_id']]:
            raise ValueError('Source query/task mismatch')
        validate(c, source_query=row['query'])
        approved = row.get('whole_contract_approved') is True and row.get('reward_approved') is True
        if approved:
            proof = approvals.get(row['task_id'])
            if not proof or proof['normalized'] != c or not proof['whole_contract_approved']:
                raise ValueError(f"Missing exact whole-query approval: {row['task_id']}")
            validate(c, source_query=row['query'], require_reviewed=True)
        pending = sum(r['review_status'] != 'approved' for d in DIMENSIONS for r in c[d])
        registry.append({'task_id': row['task_id'], 'query_sha256': hashlib.sha256(row['query'].encode()).hexdigest(),
                         'contract_sha256': hashlib.sha256(json.dumps(c, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':')).encode()).hexdigest(),
                         'program_valid': True, 'whole_source_approved': approved,
                         'status': 'signed' if approved else 'held_source_review',
                         'pending_requirement_labels': pending, 'pending_mentions': len(c['pending_mentions']),
                         'pending_reason_codes': sorted({p['reason'] for p in c['pending_mentions']}),
                         'has_uncertain_importance': any(r['importance'] == 'uncertain' for d in DIMENSIONS for r in c[d]),
                         'prior_targeted_or_sample_reviews': sorted(set(prior_review.get(row['task_id'], []))),
                         'hold_kind': None if approved else ('source_or_policy_pending' if c['pending_mentions']
                                                            else 'whole_query_approval_missing'),
                         'reason': None if approved else 'No exact whole-query signoff; targeted prior reviews are not whole-query approval'})
        entry = registry[-1]
        entry['training_released'] = approved or row['task_id'] in batch_released
        entry['approval_basis'] = 'exact_whole_query_review' if approved else (
            'batch_acceptance' if row['task_id'] in batch_released else None)
        if batch:
            entry['status'] = 'released_for_pilot' if entry['training_released'] else 'held_source_limitation'
            entry['hold_kind'] = None if entry['training_released'] else 'frozen_source_limitation'
            entry['reason'] = None if entry['training_released'] else 'User-confirmed whole-group invalid, no old-reward fallback'
    registry_path = output / 'registry1000.jsonl'
    registry_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in registry), encoding='utf-8')
    query_path = output / 'original_queries1000.jsonl'
    query_path.write_text(''.join(json.dumps({'task_id': t, 'query': original_queries[t]}, ensure_ascii=False) + '\n'
                                 for t in train_ids), encoding='utf-8')
    assets = {
        'contracts': contracts_path, 'registry': registry_path, 'whole_source_approvals': approvals_path,
        'original_queries': query_path,
        'catalog': UPSTREAM / 'environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz',
        'catalog_facts': ROOT / latest['reviewed_pilot_fact_pack'],
        'policy': ROOT / latest['evidence_reward_policy'],
        'train_ids': train_path, 'validation_ids': val_path, 'evaluation_ids': eval_path,
        'train_parquet': UPSTREAM / 'data/grpo/train.parquet',
        'validation_parquet': UPSTREAM / 'data/grpo/validation.parquet',
        'environment': UPSTREAM / 'data/environment.json',
        'prior_batch_policy_replay': ROOT / 'analysis/semantic_clear_error_revision_20261106/confirmed_policy_review.jsonl',
    }
    assets.update({f'prior_review_{i}': path for i, path in enumerate(review_paths)})
    if batch:
        assets['batch_release'] = Path(batch_release_path).resolve()
        assets.update({'batch_'+name:path for name,path in batch_paths.items()})
    # Include transitive local code/schema/policy dependencies, not just the top-level scorer.
    dependencies = list((ROOT / 'tools').glob('semantic_*.py'))
    dependencies += [p for p in (ROOT / 'experiments').rglob('*.json') if 'semantic_' in str(p.relative_to(ROOT))]
    dependencies += [p for p in (UPSTREAM / 'src/shopping_grpo/training/grpo/adapter').glob('*.py')]
    dependencies += [UPSTREAM / 'scripts/train_grpo.py', UPSTREAM / 'configs/agent_loop.yaml', UPSTREAM / 'configs/grpo.yaml']
    dependencies += [ROOT / 'tools/build_semantic_pilot50.py']
    # Runtime validation reads these exact source-scope receipts. Freeze them
    # alongside code, otherwise a label's readiness could change after hashing.
    scope_receipts = ROOT / 'experiments/semantic_remaining_closeout_v1/scope_reviews.jsonl'
    if scope_receipts.exists():dependencies.append(scope_receipts)
    batch_receipts = ROOT / 'experiments/semantic_batch_release_v1/scope_reviews.jsonl'
    if batch_receipts.exists():dependencies.append(batch_receipts)
    dependencies += [UPSTREAM / 'pyproject.toml']
    dependencies += [UPSTREAM / 'scripts/check_grpo_runtime.py']
    dependencies += list((UPSTREAM / 'environments/ShopSimulator/shop_env/web_agent_site/engine').glob('*.py'))
    manifest = {'version': 'semantic-training-signoff-v1', 'reward_version': VERSION,
                'hash_policy': 'sha256_text_lf_for_py_json_jsonl_yaml_toml; binary_bytes_otherwise',
                'objective': 'verified_public_evidence_satisfaction', 'original_training_tasks': 1000,
                'signed_tasks': sum(r['whole_source_approved'] for r in registry),
                'training_released_tasks': sum(r['training_released'] for r in registry),
                'pilot_training_ready': bool(batch),
                'release_scope': 'bounded_pilot' if batch else 'legacy_whole_query_signoff',
                'tasks_with_pending_mentions': sum(bool(r['pending_mentions']) for r in registry),
                'tasks_with_uncertain_importance': sum(r['has_uncertain_importance'] for r in registry),
                'prior_review_provenance_tasks': sum(bool(r['prior_targeted_or_sample_reviews']) for r in registry),
                'held_without_pending_mentions': sum(r['hold_kind'] == 'whole_query_approval_missing' for r in registry),
                'held_tasks': [r['task_id'] for r in registry if not r['training_released']],
                'formal_training_ready': all(r['whole_source_approved'] for r in registry),
                'validation_policy': 'unchanged_native_v3_separate_from_evidence_training_objective',
                'actor_queries_changed': 0, 'tasks_removed': 0,
                'assets': {k: {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha256(p)} for k, p in assets.items()},
                'dependencies': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(set(dependencies))}}
    dump(output / 'manifest.json', manifest)
    dump(output / 'summary.json', {k: v for k, v in manifest.items() if k not in ('assets', 'dependencies', 'held_tasks')})
    return manifest


class TrainingEvidenceBackend:
    """One read-only instance per worker; all trajectory state stays in ContextVars."""
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('version') != 'semantic-training-signoff-v1' or self.manifest.get('reward_version') != VERSION:
            raise ValueError('Unsupported evidence training manifest')
        if self.manifest.get('objective') != 'verified_public_evidence_satisfaction':
            raise ValueError('Unexpected training objective')
        if self.manifest.get('hash_policy') != 'sha256_text_lf_for_py_json_jsonl_yaml_toml; binary_bytes_otherwise':
            raise ValueError('Unsupported frozen hash policy')
        self.paths = {}
        for name, asset in self.manifest['assets'].items():
            path = (ROOT / asset['path']).resolve()
            if not path.is_relative_to(ROOT) or sha256(path) != asset['sha256']:
                raise ValueError(f'Frozen evidence asset mismatch: {name}')
            self.paths[name] = path
        for rel, expected in self.manifest['dependencies'].items():
            path = (ROOT / rel).resolve()
            if not path.is_relative_to(ROOT) or sha256(path) != expected:
                raise ValueError(f'Frozen reward dependency mismatch: {rel}')
        # A manifest file can match its frozen hash while declaring stale
        # hashes for the real environment. Reject that before GPU preflight.
        from semantic_environment_preflight import check_environment
        self.environment_check=check_environment(self.paths['environment'],UPSTREAM,sha256)
        self.contracts = {r['task_id']: r for r in rows(self.paths['contracts'])}
        self.registry = {r['task_id']: r for r in rows(self.paths['registry'])}
        self.train_ids = {int(r['task_id']) for r in rows(self.paths['train_ids'])}
        self.validation_ids = {int(r['task_id']) for r in rows(self.paths['validation_ids'])}
        eval_ids = {int(r['task_id']) for r in rows(self.paths['evaluation_ids'])}
        if len(self.train_ids) != 1000 or len(self.contracts) != 1000 or len(self.registry) != 1000 or set(self.contracts) != self.train_ids or set(self.registry) != self.train_ids:
            raise ValueError('Evidence manifest no longer covers original train1000')
        if self.train_ids & (self.validation_ids | eval_ids):
            raise ValueError('Evidence training/validation/evaluation overlap')
        self.approved_ids = set()
        source_queries = {r['task_id']: r['query'] for r in rows(self.paths['original_queries'])}
        approvals = {r['task_id']: r for r in rows(self.paths['whole_source_approvals'])}
        for task, row in self.contracts.items():
            c = row['normalized']; entry = self.registry[task]
            validate(c, source_query=row['query'])
            if c['task_id'] != task or c['query'] != row['query'] or row['query'] != source_queries.get(task):
                raise ValueError('Source identity mismatch')
            query_hash = hashlib.sha256(row['query'].encode()).hexdigest()
            contract_hash = hashlib.sha256(json.dumps(c, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if entry['query_sha256'] != query_hash or entry['contract_sha256'] != contract_hash:
                raise ValueError('Registry source/contract fingerprint mismatch')
            if entry['whole_source_approved']:
                if not row.get('whole_contract_approved') or not row.get('reward_approved'):
                    raise ValueError('Registry cannot override source approval')
                if task not in approvals or approvals[task]['normalized'] != c or not approvals[task]['whole_contract_approved']:
                    raise ValueError('Missing exact reviewed source approval')
                validate(c, source_query=row['query'], require_reviewed=True)
                self.approved_ids.add(task)
        if self.manifest['signed_tasks'] != len(self.approved_ids):
            raise ValueError('Signoff count mismatch')
        self.released_ids = set(self.approved_ids)
        self.held_ids = set()
        self.batch_release = None
        if 'batch_release' in self.paths:
            from semantic_batch_release import verify_release
            self.batch_release, _, released, self.held_ids = verify_release(
                self.paths['batch_release'], root=ROOT, contracts=self.paths['contracts'], sha256=sha256, rows=rows)
            self.released_ids.update(released)
            if self.released_ids | self.held_ids != self.train_ids or self.released_ids & self.held_ids:
                raise ValueError('Every original task must be released or explicitly held')
            if self.manifest.get('pilot_training_ready') is not True or self.manifest.get('release_scope') != 'bounded_pilot':
                raise ValueError('Missing bounded-pilot release declaration')
        elif self.manifest.get('pilot_training_ready'):
            raise ValueError('Pilot release missing acceptance certificate')
        if self.manifest.get('training_released_tasks',len(self.approved_ids)) != len(self.released_ids):
            raise ValueError('Training release count mismatch')
        if any(bool(self.registry[t].get('training_released',t in self.approved_ids)) != (t in self.released_ids) for t in self.train_ids):
            raise ValueError('Registry cannot override batch release roster')
        self._products = None
        self.facts = {}
        for fact in rows(self.paths['catalog_facts']):
            self.facts.setdefault(str(fact['product_asin']), []).append(fact)

    def require_ready(self, *, train_parquet=None, validation_parquet=None):
        if not self.batch_release and (self.approved_ids != self.train_ids or not self.manifest['formal_training_ready']):
            raise ValueError(f'Original train1000 NOT signed for training: {len(self.approved_ids)}/1000 approved, '
                             f'{1000-len(self.approved_ids)} held. See registry1000.jsonl; no silent task filtering/fallback.')
        for name, actual in [('train_parquet', train_parquet), ('validation_parquet', validation_parquet)]:
            if actual is not None and sha256(actual) != sha256(self.paths[name]):
                raise ValueError(f'{name} differs from unchanged frozen actor input')

    def require_run_config(self, config):
        from semantic_batch_release import require_pilot_config
        require_pilot_config(self.batch_release, config)

    def require_run_overrides(self, overrides):
        """Cheap launch check; runtime/worker validate the resolved Hydra config."""
        if not self.batch_release:return
        config={}
        keys=('trainer.total_training_steps','trainer.resume_mode','data.train_batch_size',
              'actor_rollout_ref.rollout.n','shopping_dynamic_sampling.enable',
              'shopping_dynamic_sampling.max_num_gen_batches','shopping_dynamic_sampling.max_consecutive_skipped_updates')
        for token in overrides:
            key,sep,value=token.partition('=')
            if not sep or key not in keys:continue
            try:value=json.loads(value)
            except json.JSONDecodeError:pass
            cursor=config;parts=key.split('.')
            for p in parts[:-1]:cursor=cursor.setdefault(p,{})
            cursor[parts[-1]]=value
        self.require_run_config(config)

    def is_validation(self, task_id):
        if task_id in self.validation_ids:
            return True
        if task_id not in self.train_ids:
            raise ValueError(f'Task {task_id} outside frozen train/validation split')
        return False

    @property
    def products(self):
        if self._products is None:
            with gzip.open(self.paths['catalog'], 'rt', encoding='utf-8') as handle:
                self._products = {str(p['asin']): {k: v for k, v in p.items() if k not in ('instructions', 'reviews')}
                                  for p in json.load(handle)}
        return self._products

    def score(self, task_id, state):
        invalid = {'reward': None, 'reward_valid': False, 'evidence_success': None,
                   'semantic_success': None, 'version': VERSION, 'objective': self.manifest['objective']}
        if task_id in self.held_ids:
            return {**invalid, 'type': 'source_contract_held', 'basis': 'user_confirmed_whole_group_invalid'}
        if task_id not in self.released_ids:
            return {**invalid, 'type': 'source_contract_unsigned'}
        if state.get('infrastructure_invalid') or state.get('error'):
            return {**invalid, 'type': 'execution_invalid'}
        terminal = state.get('_evidence_terminal') or {}
        if terminal.get('done') is not True or terminal.get('over') is not True:
            return {**invalid, 'type': 'execution_unproven'}
        purchase = terminal.get('purchase')
        if not purchase:
            return score_terminal_failure({'terminal_result': terminal, 'error': state.get('error'),
                'steps': [{'tool_name': state['steps'][-1]['tool'], 'result': terminal}] if state.get('steps') else []})
        if terminal.get('termination_reason') not in ('gold_purchase', 'valid_alternative_purchase',
                'partial_alternative_purchase', 'wrong_purchase', 'reward_unverifiable'):
            return {**invalid, 'type': 'purchase_terminal_mismatch'}
        p = self.products.get(str(purchase.get('asin')))
        if p is None or not isinstance(purchase.get('options'), dict):
            return {**invalid, 'type': 'catalog_purchase_unproven'}
        mapping = canonicalize_purchase(p, purchase['options'])
        if mapping['errors']:
            return {**invalid, 'type': 'purchase_options_unproven', 'mapping_errors': mapping['errors']}
        row = self.contracts[task_id]
        verification = verify(row['normalized'], p, mapping['selected_options'],
                              catalog_facts=self.facts.get(str(p['asin']), []), purchase_executed=True)
        result = score_training_verified(row['normalized'], verification, source_query=row['query'])
        if result['reward_valid'] and not math.isfinite(result['reward']):
            raise ValueError('Non-finite training reward')
        return {**result, 'verification': verification, 'purchase_asin': str(p['asin'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--approvals', type=Path, help='Exact whole-source approval ledger; never inferred from program_valid')
    parser.add_argument('--contracts', type=Path)
    parser.add_argument('--batch-release', type=Path)
    parser.add_argument('--check', type=Path, help='Check existing bundle without API/GPU calls')
    args = parser.parse_args()
    if args.check:
        backend = TrainingEvidenceBackend(args.check)
        print(json.dumps({'frozen_assets_valid': True, 'original_tasks': len(backend.train_ids),
                          'signed_tasks': len(backend.approved_ids), 'training_released_tasks': len(backend.released_ids),
                          'held_tasks': len(backend.train_ids-backend.released_ids),
                          'release_scope': backend.manifest.get('release_scope'),
                          **backend.environment_check,
                          'pilot_training_ready': bool(backend.batch_release),
                          'max_optimizer_steps': backend.batch_release['max_optimizer_steps'] if backend.batch_release else None,
                          'formal_training_ready': backend.approved_ids == backend.train_ids}))
    else:
        manifest = build_bundle(args.output, approvals_path=args.approvals, contracts_path=args.contracts,
                                batch_release_path=args.batch_release)
        print(json.dumps({k: v for k, v in manifest.items() if k not in ('assets', 'dependencies', 'held_tasks')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
