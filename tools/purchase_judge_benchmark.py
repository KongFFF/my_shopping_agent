"""Isolated CPU/API purchase-judge benchmark. Never modifies training assets.

prepare: chronological train-only purchases, rule baseline and cache simulation.
run: automatic model judge, exact-source validation, bounded concurrent batches.
report: independent reference accuracy, hybrid rewards, timings and decision gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from urllib.error import URLError

from jsonschema import Draft202012Validator
from extract_semantic_requirements import endpoint, post, loads, HTTPFailure
from semantic_training_bundle import TrainingEvidenceBackend, sha256
from semantic_product_verifier_v4 import verify, snapshot, contract_fingerprint
from semantic_contract_v2_r7_final12 import DIMENSIONS
from semantic_evidence_reward import score_training_verified

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/purchase_judge_v1'
DEFAULT = ROOT / 'analysis/purchase_judge_benchmark_20261110/dataset'
MANIFEST = ROOT / 'analysis/semantic_pilot30_evidence_20261109/training_bundle/manifest.json'
SOURCES = [
    ('warm_v3', 'analysis/semantic_evidence_pilot20_v3_20261109/training_diagnostics.jsonl'),
    ('warm_v4', 'analysis/semantic_evidence_pilot3_v4_results/training_diagnostics.jsonl'),
    ('holdout_v5', 'analysis/semantic_evidence_pilot20_v5_seed20261110_results/training_diagnostics.jsonl'),
]
VERSION = 'purchase-judge-benchmark-v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def rows(path):
    return [loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()] if Path(path).exists() else []


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def lines(path, values):
    Path(path).write_text(''.join(canonical(v) + '\n' for v in values), encoding='utf-8')


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(canonical(value) + '\n')


def output_schema():
    evidence = {'type': 'object', 'additionalProperties': False,
                'properties': {'pointer': {'type': 'string'}, 'value': {'type': ['string', 'number', 'boolean', 'null']}},
                'required': ['pointer', 'value']}
    decision = {'type': 'object', 'additionalProperties': False, 'properties': {
        'id': {'type': 'string'}, 'status': {'enum': ['pass', 'fail', 'unknown']},
        'reason_code': {'enum': ['supported', 'contradicted', 'not_stated', 'ambiguous_scope', 'conflicting_evidence']},
        'reason': {'type': 'string', 'minLength': 1},
        'evidence': {'type': 'array', 'items': evidence}},
        'required': ['id', 'status', 'reason_code', 'reason', 'evidence']}
    return {'type': 'object', 'additionalProperties': False,
            'properties': {'decisions': {'type': 'array', 'items': decision}}, 'required': ['decisions']}


def validate_answer(answer, request):
    Draft202012Validator(output_schema()).validate(answer)
    expected = {r['id'] for r in request['requirements']}
    actual = [d['id'] for d in answer['decisions']]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Exactly one decision for every requested requirement is required')
    table = request['evidence_table']
    for d in answer['decisions']:
        allowed = {'pass': {'supported'}, 'fail': {'contradicted'},
                   'unknown': {'not_stated', 'ambiguous_scope', 'conflicting_evidence'}}[d['status']]
        if d['reason_code'] not in allowed:
            raise ValueError('Status and reason_code disagree')
        if d['status'] != 'unknown' and not d['evidence']:
            raise ValueError('Pass/fail requires exact public evidence')
        for e in d['evidence']:
            if e['pointer'] not in table or canonical(e['value']) != canonical(table[e['pointer']]):
                raise ValueError('Evidence must exactly match an allowed selected-SKU public scalar')
    return answer


def evidence_table(product, selected):
    """Whitelist common claims and ONLY selected option entries; no answers/reviews."""
    result = {}
    for field in ('title', 'sub_title', 'shop_name', 'category', 'full_description', 'brand', 'model'):
        if isinstance(product.get(field), str) and product[field]:
            result['/' + field] = product[field]
    for i, value in enumerate(product.get('attribute') or []):
        if isinstance(value, str):
            result[f'/attribute/{i}'] = value
    state = snapshot(product, selected)
    if state['selection_errors']:
        raise ValueError('Cannot prepare invalid purchased selection')
    for option in state['selected']:
        # No other option and no original per-option is_selected flag enters a request.
        result[option['pointer']] = option['value']
    return result


def request_for(contract, product, selected):
    return {'query': contract['query'], 'selected_options': selected,
            'requirements': [{'dimension': dim, **r} for dim in DIMENSIONS if dim != 'budget' for r in contract[dim]],
            'evidence_table': evidence_table(product, selected),
            'output_schema': output_schema()}


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    backend = TrainingEvidenceBackend(MANIFEST)
    prompt = (EXPERIMENT / 'system_prompt.txt').read_text(encoding='utf-8')
    policy = {'version': VERSION, 'prompt_sha256': digest(prompt), 'schema_sha256': digest(output_schema()),
              'reward_objective': backend.manifest['objective'],
              'hybrid_policy': 'retain_all_deterministic_pass_fail; replace_only_unknown_nonbudget_with_valid_judge',
              'budget': 'deterministic_only', 'human_in_training_loop': False,
              'asset_manifest_sha256': sha256(MANIFEST), 'scorer_code_sha256': sha256(Path(__file__))}
    cases, events, skipped = {}, [], Counter()
    baseline_seconds = []
    for cohort, relative in SOURCES:
        for n, event in enumerate(rows(ROOT / relative)):
            if event['event'] != 'generation_batch':
                continue
            members = []
            for r in event['rollouts']:
                old = r.get('evidence_reward', {}).get('score', {})
                old_v = old.get('verification', {})
                task = r['task_id']
                member = {'uid': r['uid'], 'rollout_index': r['rollout_index'], 'task_id': task,
                          'original_reward': old.get('reward'), 'original_valid': old.get('reward_valid'),
                          'case_id': None}
                if not old.get('purchase_asin'):
                    skipped[old.get('type', 'no_purchase')] += 1
                elif task not in backend.released_ids or old.get('reward_valid') is not True:
                    skipped['held_or_invalid_purchase'] += 1
                else:
                    asin = str(old['purchase_asin'])
                    product = backend.products[asin]
                    selected = old_v['snapshot']['selected_options']
                    if snapshot(product, selected)['product_sha256'] != old_v['snapshot']['product_sha256']:
                        raise ValueError('Historical product fingerprint mismatch')
                    contract = backend.contracts[task]['normalized']
                    identity = {'contract': contract_fingerprint(contract), 'product': old_v['snapshot']['product_sha256'],
                                'selected_options': selected, 'policy': policy}
                    case_id = digest(identity)
                    member['case_id'] = case_id
                    if case_id not in cases:
                        start = time.perf_counter()
                        v = verify(contract, product, selected, catalog_facts=(), purchase_executed=True)
                        reward = score_training_verified(contract, v, source_query=contract['query'])
                        baseline_seconds.append(time.perf_counter() - start)
                        cases[case_id] = {'case_id': case_id, 'first_cohort': cohort, 'task_id': task,
                            'product_asin': asin, 'selected_options': selected, 'contract': contract,
                            'product_sha256': old_v['snapshot']['product_sha256'],
                            'request': request_for(contract, product, selected),
                            'rule_verification': v, 'rule_reward': reward, 'occurrences': 0}
                    cases[case_id]['occurrences'] += 1
                members.append(member)
            events.append({'cohort': cohort, 'event_id': f'{cohort}:{n}', 'global_step': event['global_step'],
                           'generation_batch': event['generation_batch'], 'members': members})
    # Sequential simulation is optimistic: concurrent in-flight coalescing is NOT a ready-cache hit.
    warm = {c['case_id'] for c in cases.values() if c['first_cohort'].startswith('warm')}
    warm_products = {c['product_asin'] for c in cases.values() if c['first_cohort'].startswith('warm')}
    holdout = [m['case_id'] for e in events if e['cohort'] == 'holdout_v5' for m in e['members'] if m['case_id']]
    seen = set(warm)
    stream_hits = 0
    for key in holdout:
        stream_hits += key in seen
        seen.add(key)
    profile = {}
    for cohort, _ in SOURCES:
        subset = [e for e in events if e['cohort'] == cohort]
        ms = [m for e in subset for m in e['members']]
        ids = [m['case_id'] for m in ms if m['case_id']]
        profile[cohort] = {'groups': len({(e['event_id'], m['uid']) for e in subset for m in e['members']}),
                           'rollouts': len(ms), 'purchase_occurrences': len(ids), 'unique_requests': len(set(ids)),
                           'tasks': len({m['task_id'] for m in ms}),
                           'products': len({cases[k]['product_asin'] for k in ids})}
    summary = {'version': VERSION, 'api_calls': 0, 'train_only': True, 'dev_final_clean_used': False,
               'cohorts': profile, 'unique_requests': len(cases), 'warm_unique_requests': len(warm),
               'holdout_purchase_occurrences': len(holdout), 'holdout_unique_requests': len(set(holdout)),
               'holdout_prefill_exact_hits': sum(k in warm for k in holdout),
               'holdout_products_seen_in_warm': len({cases[k]['product_asin'] for k in holdout} & warm_products),
               'holdout_optimistic_sequential_hits_including_repeats': stream_hits,
               'holdout_unique_misses_after_dedup': len(set(holdout) - warm),
               'within_batch_duplicates_require_waiting_for_shared_request': True,
               'excluded_from_judge': dict(skipped), 'rule_only_seconds': describe(baseline_seconds),
               'rule_atom_counts': dict(Counter(s for c in cases.values() for s in c['rule_verification']['statuses'].values())),
               'accuracy_and_api_latency_measured': False}
    manifest = {'policy': policy, 'sources': {p: sha256(ROOT / p) for _, p in SOURCES},
                'cases_sha256': digest(list(cases.values())), 'events_sha256': digest(events)}
    existing = folder / 'dataset_manifest.json'
    if existing.exists() and loads(existing.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Benchmark inputs/code changed; use a new dataset directory')
    write(existing, manifest)
    lines(folder / 'cases.jsonl', cases.values())
    lines(folder / 'events.jsonl', events)
    write(folder / 'preparation_summary.json', summary)
    (folder / 'system_prompt.txt').write_text(prompt, encoding='utf-8')
    write(folder / 'output.schema.json', output_schema())
    # Blind review worksheet has no native labels and no model output.
    lines(folder / 'review_inputs_holdout.jsonl', ({'case_id': c['case_id'], 'task_id': c['task_id'],
          'product_asin': c['product_asin'], **c['request']} for c in cases.values() if c['first_cohort'] == 'holdout_v5'))
    print(canonical(summary), flush=True)
    return summary


def describe(values):
    if not values:
        return {'n': 0, 'mean': None, 'p50': None, 'p95': None, 'max': None}
    xs = sorted(values)
    return {'n': len(xs), 'mean': statistics.mean(xs), 'p50': statistics.median(xs),
            'p95': xs[max(0, math.ceil(.95 * len(xs)) - 1)], 'max': max(xs)}


def env_config(path):
    config = {k: os.getenv(k, '') for k in ('LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL')}
    if path:
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, sep, value = line.partition('=')
            if not sep or name.strip() not in config:
                raise ValueError('Configuration allows only LLM_API_KEY, LLM_BASE_URL, LLM_MODEL assignments')
            config[name.strip()] = value.strip().strip('\"\'')
    return config


def load_dataset(folder):
    manifest = loads((folder / 'dataset_manifest.json').read_text(encoding='utf-8'))
    cases = rows(folder / 'cases.jsonl')
    events = rows(folder / 'events.jsonl')
    if digest(cases) != manifest['cases_sha256'] or digest(events) != manifest['events_sha256']:
        raise ValueError('Frozen benchmark data changed')
    prompt = (folder / 'system_prompt.txt').read_text(encoding='utf-8')
    if digest(prompt) != manifest['policy']['prompt_sha256'] or sha256(Path(__file__)) != manifest['policy']['scorer_code_sha256']:
        raise ValueError('Prompt or benchmark code changed since preparation')
    return {c['case_id']: c for c in cases}, events, manifest, prompt


def run(folder, output, config, *, concurrency=4, attempts=2, timeout=60, limit=None,
        max_tokens=4096, thinking='disabled', allow_school_http=False, transport=post):
    cases, events, manifest, prompt = load_dataset(folder)
    url = endpoint(config['LLM_BASE_URL'], allow_school_http=allow_school_http)
    if not config['LLM_API_KEY'] or not config['LLM_MODEL']:
        raise ValueError('Missing LLM_API_KEY or LLM_MODEL; no requests sent')
    output.mkdir(parents=True, exist_ok=True)
    settings = {'dataset_sha256': digest(manifest), 'endpoint': url, 'model': config['LLM_MODEL'],
                'concurrency': concurrency, 'attempts': attempts, 'timeout': timeout,
                'max_tokens': max_tokens, 'thinking': thinking, 'temperature': 0,
                'reference_sha256': digest(rows(folder / 'reference_atoms.jsonl')),
                'ranking_reference_sha256': digest(rows(folder / 'reference_rankings.jsonl'))}
    path = output / 'config.json'
    if path.exists() and loads(path.read_text(encoding='utf-8')) != settings:
        raise ValueError('API configuration changed; use a new run output directory')
    lock = output / '.run.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        write(path, settings)
        namespace = digest(settings)
        valid = {}
        for r in rows(output / 'answers.jsonl'):
            if r['namespace'] != namespace or r['case_id'] not in cases:
                raise ValueError('Answer cache belongs to another configuration')
            validate_answer(r['answer'], cases[r['case_id']]['request'])
            valid[r['case_id']] = r
        stop = threading.Event()
        write_lock = threading.Lock()
        attempted = set()

        def judge(key):
            case = cases[key]
            start = time.perf_counter()
            for attempt in range(1, attempts + 1):
                if stop.is_set():
                    return None
                body = {'model': settings['model'], 'messages': [{'role': 'system', 'content': prompt},
                        {'role': 'user', 'content': canonical(case['request'])}],
                        'stream': False, 'temperature': 0, 'max_tokens': max_tokens,
                        'response_format': {'type': 'json_object'}}
                if thinking != 'omit':
                    body['thinking'] = {'type': thinking}
                t = time.perf_counter()
                log = {'case_id': key, 'task_id': case['task_id'], 'attempt': attempt, 'namespace': namespace}
                result = None
                try:
                    _, raw = transport(url, config['LLM_API_KEY'], body, timeout)
                    envelope = loads(raw)
                    log['usage'] = envelope.get('usage', {})
                    choice = envelope['choices'][0]
                    if choice.get('finish_reason') != 'stop':
                        raise ValueError('Model did not finish normally')
                    answer = validate_answer(loads(choice['message']['content']), case['request'])
                    result = {'case_id': key, 'namespace': namespace, 'answer': answer,
                              'model_reported': envelope.get('model'), 'usage': envelope.get('usage', {}),
                              'total_seconds': time.perf_counter() - start, 'attempts': attempt}
                    log.update(valid=True, usage=envelope.get('usage', {}))
                except HTTPFailure as error:
                    # Never persist gateway response bodies or Authorization headers.
                    log.update(valid=False, error=f'HTTP {error.status}')
                    if error.status in (401, 403, 402, 404):
                        stop.set()
                except (ValueError, KeyError, TypeError, IndexError, URLError, TimeoutError, OSError) as error:
                    # Schema errors can embed source text; logs stay bounded and never include secrets.
                    log.update(valid=False, error=type(error).__name__ + ': ' + str(error)[:250])
                except Exception as error:
                    # Includes JSON-schema ValidationError without serializing a whole response.
                    log.update(valid=False, error=type(error).__name__)
                log['seconds'] = time.perf_counter() - t
                if log.get('error'):
                    log['error'] = log['error'].replace(config['LLM_API_KEY'], '[redacted]')
                with write_lock:
                    append(output / 'attempts.jsonl', log)
                    if result:
                        append(output / 'answers.jsonl', result)
                    print(f"task={case['task_id']} attempt={attempt} " + ('VALID' if result else log['error']), flush=True)
                if result:
                    return result
                if attempt < attempts and not stop.is_set():
                    time.sleep(.5 * attempt)
            return None

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for event in events:
                keys = list(dict.fromkeys(m['case_id'] for m in event['members'] if m['case_id']))
                ready = sum(m['case_id'] in valid for m in event['members'] if m['case_id'])
                pending = [k for k in keys if k not in valid and k not in attempted]
                needed = list(pending)
                if limit is not None:
                    pending = pending[:max(0, limit - len(attempted))]
                if stop.is_set() or (limit is not None and len(attempted) >= limit and pending == []):
                    break
                start = time.perf_counter()
                futures = {pool.submit(judge, k): k for k in pending}
                attempted.update(pending)
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        valid[result['case_id']] = result
                append(output / 'batches.jsonl', {'event_id': event['event_id'], 'cohort': event['cohort'],
                    'generation_batch': event['generation_batch'], 'global_step': event['global_step'],
                    'seconds': time.perf_counter() - start, 'ready_cache_hit_occurrences': ready,
                    'submitted_unique_requests': len(pending), 'requested_unique_keys': keys,
                    'missing_after_batch': [k for k in keys if k not in valid],
                    'replay_mode': 'barrier_per_saved_generation_batch; no_rollout_compute_overlap',
                    'partial_due_to_limit': pending != needed})
        return report(folder, output)
    finally:
        lock.unlink()


def hybrid_score(case, answer):
    verification = copy.deepcopy(case['rule_verification'])
    if answer:
        for decision in answer['decisions']:
            rid = decision['id']
            if verification['statuses'][rid] == 'unknown':
                verification['statuses'][rid] = decision['status']
                verification['comparisons'][rid] = {**verification['comparisons'][rid], **decision,
                                                     'reason': 'automatic_judge:' + decision['reason_code']}
    return score_training_verified(case['contract'], verification, source_query=case['contract']['query']), verification


def report(folder, output):
    cases, events, manifest, _ = load_dataset(folder)
    settings = loads((output / 'config.json').read_text(encoding='utf-8')) if (output / 'config.json').exists() else None
    answers = {}
    for r in rows(output / 'answers.jsonl'):
        if settings is None or r['namespace'] != digest(settings) or r['case_id'] not in cases:
            raise ValueError('Unbound answer in report')
        validate_answer(r['answer'], cases[r['case_id']]['request'])
        answers[r['case_id']] = r
    references = rows(folder / 'reference_atoms.jsonl')
    rankings = rows(folder / 'reference_rankings.jsonl')
    if settings and (digest(references) != settings['reference_sha256'] or digest(rankings) != settings['ranking_reference_sha256']):
        raise ValueError('Independent reference changed after model run began')
    reference_ids = set()
    audit = []
    for r in references:
        identity = (r['case_id'], r['requirement_id'])
        if identity in reference_ids:
            raise ValueError('Duplicate independent reference atom')
        reference_ids.add(identity)
        c = cases[r['case_id']]
        if r['status'] not in ('pass', 'fail', 'unknown') or r.get('reviewed_before_model') is not True:
            raise ValueError('Reference must be a pre-model reviewed judgment')
        if r['requirement_id'] not in c['rule_verification']['statuses']:
            raise ValueError('Unknown reference requirement')
        # Source quotation existence does not assert semantic entailment.
        for e in r['evidence']:
            if canonical(c['request']['evidence_table'].get(e['pointer'])) != canonical(e['value']):
                raise ValueError('Reference evidence is not an exact allowed scalar')
        if r['status'] != 'unknown' and not r['evidence']:
            raise ValueError('Reference pass/fail needs public evidence')
        predicted = None
        hybrid = c['rule_verification']['statuses'][r['requirement_id']]
        if c['case_id'] in answers:
            ds = {d['id']: d['status'] for d in answers[c['case_id']]['answer']['decisions']}
            predicted = ds.get(r['requirement_id'])
            if hybrid == 'unknown' and predicted is not None:
                hybrid = predicted
        audit.append({**r, 'judge': predicted, 'hybrid': hybrid,
                      'rule': c['rule_verification']['statuses'][r['requirement_id']]})
    evaluated = [r for r in audit if r['judge'] is not None]
    negatives = [r for r in evaluated if r['status'] != 'pass']
    accuracy = {'reference_atoms': len(audit), 'judged_reference_atoms': len(evaluated),
        'reference_label_counts': dict(Counter(r['status'] for r in audit)),
        'rule_accuracy': sum(r['rule'] == r['status'] for r in audit) / len(audit) if audit else None,
        'judge_accuracy': sum(r['judge'] == r['status'] for r in evaluated) / len(evaluated) if evaluated else None,
        'hybrid_accuracy': sum(r['hybrid'] == r['status'] for r in evaluated) / len(evaluated) if evaluated else None,
        'false_pass_count': sum(r['judge'] == 'pass' for r in negatives),
        'false_pass_denominator_nonpass': len(negatives),
        'confusion': dict(Counter(r['status'] + '->' + str(r['judge']) for r in evaluated)),
        'reference_author': 'assistant_source_review_not_user_whole_query_signoff',
        'scope': 'stratified_train_purchase_atoms; not_train1000_accuracy; not_reward_formula_accuracy'}
    attempts = rows(output / 'attempts.jsonl')
    batches = rows(output / 'batches.jsonl')
    if settings and any(r['namespace'] != digest(settings) for r in attempts):
        raise ValueError('Mixed API configurations in attempt journal')
    scored, group_rows = {}, []
    for key, c in cases.items():
        score, v = hybrid_score(c, answers[key]['answer']) if key in answers else (None, None)
        scored[key] = {'case_id': key, 'task_id': c['task_id'], 'product_asin': c['product_asin'],
                       'rule_reward': c['rule_reward']['reward'], 'judge_available': key in answers,
                       'hybrid_reward': score['reward'] if score else None,
                       'hybrid_statuses': v['statuses'] if v else None}
    for event in events:
        for uid in dict.fromkeys(m['uid'] for m in event['members']):
            ms = [m for m in event['members'] if m['uid'] == uid]
            before = [cases[m['case_id']]['rule_reward']['reward'] if m['case_id'] else m['original_reward'] for m in ms]
            after = [scored[m['case_id']]['hybrid_reward'] if m['case_id'] else m['original_reward'] for m in ms]
            def state(values):
                if any(not m['original_valid'] for m in ms):
                    return 'original_execution_invalid'
                if any(v is None for v in values):
                    return 'judge_unavailable'
                return 'tie' if max(values) - min(values) <= 1e-8 else 'eligible'
            group_rows.append({'cohort': event['cohort'], 'uid': uid, 'task_id': ms[0]['task_id'],
                               'before': before, 'after': after, 'before_state': state(before), 'after_state': state(after)})
    usage = Counter()
    for attempt in attempts:
        for name, value in attempt.get('usage', {}).items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                usage[name] += value
    # Only first-exposure batches with all required requests submitted are included.
    waits = [b['seconds'] for b in batches if b['submitted_unique_requests'] and not b['missing_after_batch'] and not b['partial_due_to_limit']]
    ranking_checks = []
    for pair in rankings:
        a, b = scored[pair['left_case_id']], scored[pair['right_case_id']]
        if pair.get('reviewed_before_model') is not True:
            raise ValueError('Ranking must be frozen before model output')
        if a['task_id'] != b['task_id']:
            raise ValueError('Ranking comparisons must be within the same task')
        if pair['relation'] not in ('gt', 'eq'):
            raise ValueError('Unsupported ranking relation')
        def compare(x, y):
            if x is None or y is None:
                return None
            return x > y + 1e-8 if pair['relation'] == 'gt' else abs(x - y) <= 1e-8
        ranking_checks.append({**pair, 'rule_pass': compare(a['rule_reward'], b['rule_reward']),
                               'hybrid_pass': compare(a['hybrid_reward'], b['hybrid_reward'])})
    gates = {'responses_complete': len(answers) == len(cases),
             'reference_minimum_100_atoms': len(evaluated) >= 100,
             'reference_has_at_least_10_fail_20_unknown': sum(r['status'] == 'fail' for r in evaluated) >= 10 and sum(r['status'] == 'unknown' for r in evaluated) >= 20,
             'hybrid_accuracy_at_least_95pct': accuracy['hybrid_accuracy'] is not None and accuracy['hybrid_accuracy'] >= .95,
             'no_false_pass_in_audited_nonpass': bool(negatives) and accuracy['false_pass_count'] == 0,
             'reviewed_rankings_all_passed': bool(ranking_checks) and all(r['hybrid_pass'] is True for r in ranking_checks),
             'real_grpo_overlap_and_wait_budget_measured': False}
    decision = ('do_not_integrate_online_judge; accuracy_or_coverage_gate_failed' if len(answers) == len(cases) and not all(v for k, v in gates.items() if k != 'real_grpo_overlap_and_wait_budget_measured') else
                'offline_quality_candidate_only; actual_gpu_wait_budget_not_measured' if len(answers) == len(cases) else
                'incomplete_benchmark; do_not_start_gpu_training')
    summary = {'version': VERSION, 'api_calls': len(attempts), 'valid_requests': len(answers), 'total_requests': len(cases),
               'complete': len(answers) == len(cases), 'accuracy': accuracy,
               'api_attempt_seconds': describe([a['seconds'] for a in attempts]),
               'valid_request_seconds_including_retries': describe([a['total_seconds'] for a in answers.values()]),
               'complete_cold_batch_wait_seconds': describe(waits),
               'batch_timing_scope': 'CPU-client barrier replay; not_measured_GPU_idle_time; no_overlap_with_actual_rollout',
               'http_or_validation_errors': dict(Counter(a['error'] for a in attempts if not a['valid'])),
               'usage_all_attempts_with_reported_usage': dict(usage), 'currency_cost': None,
               'ranking_checks': ranking_checks,
               'group_counts': dict(Counter(g['before_state'] + '->' + g['after_state'] for g in group_rows)),
               'holdout_group_counts': dict(Counter(g['before_state'] + '->' + g['after_state'] for g in group_rows if g['cohort'] == 'holdout_v5')),
               'provisional_engineering_gates_not_reward_policy': gates, 'recommendation': decision,
               'training_assets_modified': False, 'human_wait_in_runtime': False,
               'native_reward_used_as_truth': False, 'new_policy_training_performed': False}
    output.mkdir(parents=True, exist_ok=True)
    lines(output / 'reference_comparison.jsonl', audit)
    lines(output / 'purchase_rewards.jsonl', scored.values())
    lines(output / 'group_comparison.jsonl', group_rows)
    lines(output / 'ranking_comparison.jsonl', ranking_checks)
    write(output / 'report.json', summary)
    print(canonical(summary), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'report'])
    p.add_argument('--dataset', type=Path, default=DEFAULT)
    p.add_argument('--output', type=Path)
    p.add_argument('--env-file', type=Path)
    p.add_argument('--base-url')
    p.add_argument('--model')
    p.add_argument('--allow-school-http', action='store_true')
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--attempts', type=int, default=2)
    p.add_argument('--timeout', type=int, default=60)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--thinking', choices=['disabled', 'enabled', 'omit'], default='disabled')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if min(args.concurrency, args.attempts, args.timeout, args.max_tokens) < 1 or (args.limit is not None and args.limit < 1):
        p.error('Numeric options must be positive')
    if args.action == 'prepare':
        if args.base_url:
            endpoint(args.base_url, local_only=True)
        prepare(args.dataset)
    elif args.action == 'report':
        report(args.dataset, args.output or args.dataset / 'deepseek_flash')
    else:
        config = env_config(args.env_file)
        config['LLM_BASE_URL'] = args.base_url or config['LLM_BASE_URL']
        config['LLM_MODEL'] = args.model or config['LLM_MODEL']
        run(args.dataset, args.output or args.dataset / 'deepseek_flash', config,
            concurrency=args.concurrency, attempts=args.attempts, timeout=args.timeout, limit=args.limit,
            max_tokens=args.max_tokens, thinking=args.thinking, allow_school_http=args.allow_school_http)


if __name__ == '__main__':
    main()

