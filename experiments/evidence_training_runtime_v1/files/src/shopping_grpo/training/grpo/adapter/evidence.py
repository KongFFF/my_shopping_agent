"""Optional evidence objective; keeps native benchmark metrics separate."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path


def configured_runtime_binding(config, name):
    value = config or {}
    for key in ('ray_kwargs', 'ray_init', 'runtime_env', 'env_vars'):
        value = value.get(key, {})
    binding = value.get(name, '')
    if not isinstance(binding, str) or '${' in binding:
        raise ValueError('Unresolved Ray runtime binding: ' + name)
    return binding


def validate_literal_ray_bindings(bindings, expected):
    for name, value in expected.items():
        actual = bindings.get(name)
        if not isinstance(actual, str) or '${' in actual or actual != value:
            raise ValueError('Ray worker requires a literal launch binding for ' + name)


def load_backend(manifest_path, trainer_config=None):
    # The full trainer config is serialized to AgentLoopWorker explicitly.
    # Prefer its literal path even if Ray replaces/omits the job environment.
    configured_manifest = configured_runtime_binding(trainer_config, 'SHOPPING_EVIDENCE_MANIFEST')
    if configured_manifest and Path(configured_manifest).resolve() != Path(manifest_path).resolve():
        raise ValueError('Worker manifest differs from explicit trainer config')
    binding = configured_runtime_binding(trainer_config, 'SHOPPING_REWARD_TOOLS_DIR')
    tools_dir = Path(binding or os.environ.get('SHOPPING_REWARD_TOOLS_DIR', '')).resolve()
    if not (tools_dir / 'semantic_training_bundle.py').is_file():
        raise ValueError('Evidence reward requires SHOPPING_REWARD_TOOLS_DIR with semantic_training_bundle.py')
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from semantic_training_bundle import TrainingEvidenceBackend
    backend = TrainingEvidenceBackend(manifest_path)
    backend.require_ready()  # Direct Hydra launches must obey the same release gate.
    backend.require_run_config(trainer_config)
    return backend


def capture_terminal(result):
    """Retain execution evidence internally, excluding task/target annotations."""
    purchase = result.get('purchase')
    detail = result.get('reward_detail') or {}
    guard = (detail.get('evidence') or {}).get('purchase_guard')
    return {'done': result.get('done') is True, 'over': result.get('over') is True,
            'termination_reason': result.get('termination_reason'),
            'purchase': {'asin': purchase.get('asin'), 'options': copy.deepcopy(purchase.get('options'))}
                        if isinstance(purchase, dict) and purchase else None,
            'reward_detail': {'evidence': {'purchase_guard': copy.deepcopy(guard)}} if guard else {}}


def apply_evidence_reward(backend, task_id, state, breakdown):
    """Runs before AgentLoopOutput.reward_score; invalid results cannot train."""
    baseline = {'version': state.get('reward_version'), 'type': state.get('reward_type'),
                'valid': state.get('reward_valid'), 'reward': dict(breakdown)}
    if backend.is_validation(task_id):
        return breakdown, {'branch': 'validation_native_v3', 'baseline': baseline}
    score = backend.score(task_id, state)
    valid = score['reward_valid'] is True
    value = float(score['reward']) if valid else 0.0
    shaped = dict(breakdown)
    # Diagnostic native/strict fields retain baseline meaning; all optimization
    # and group-validity fields use the new objective consistently.
    shaped.update(total=value, terminal_utility=value, sampling_invalid=not valid,
                  reward_unverifiable=not valid, purchase_success=float(score.get('evidence_success') is True),
                  semantic=float(score.get('evidence_success') is True), evidence_coverage=0.0,
                  efficiency=0.0, penalty_overlong=0.0, penalty_repeat=0.0, penalty_unfinished=0.0)
    statuses = (score.get('verification') or {}).get('statuses') or {}
    if statuses:
        shaped['evidence_coverage'] = sum(s != 'unknown' for s in statuses.values()) / len(statuses)
    state.update(reward_version=score['version'], reward_type=score['type'],
                 reward_valid=valid, reward_unverifiable=not valid)
    return shaped, {'branch': 'training_evidence', 'baseline': baseline, 'score': score}
