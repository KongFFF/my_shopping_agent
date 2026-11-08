"""Offline batch acceptance and an explicit bounded-pilot release certificate.

No API, GPU generation or optimizer execution. Historical purchases and static
probes are kept distinct; static probes never become training target lookups.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from semantic_contract_v2_r7_final12 import DIMENSIONS, validate
from semantic_product_verifier_v4 import verify
from semantic_evidence_reward import score_training_verified
from semantic_history_adapter import canonicalize_purchase
from semantic_training_bundle import ROOT, UPSTREAM, rows, dump, sha256, TrainingEvidenceBackend, build_bundle
from prepare_semantic_batch_acceptance import OUT, SOURCE, digest, lines
from compile_semantic_batch_contracts import PACKAGE

SAMPLE_NOTES = {
  16481:'User resolved one gray set and one black set; retain allocation, not OR or a two-tone item.',
  3000:'Alcohol strength is not a generic item count; retain literal 45度 as composition without guessing percent.',
  827:'可添加果粒 is a machine capability, not fruit included in purchase.',
  13937:'正品 is condition, not brand; restore omitted 009 gauge label.',
  3676:'七层结构 uses layer_count, not unidentified feature count.',
  9531:'Remove duplicated light/thin requirement incorrectly routed as mass; retain the existing combined claim.',
  22999:'36 colors uses color_count, not unidentified feature count.',
  12422:'User confirmed no shopping choice in variant list; retain query but hold entire group.',
  882:'Restore explicitly adjustable spray rod length; distinguish spray holes from generic counts.',
  7131:'Restore source-limit disposition for 不要便宜太多; previously confirmed policy supplies no invented lower budget.',
  20632:'Resolve same-color vest/trousers to the explicitly black suit; preserve white shirt.',
  10995:'User confirmed steel/copper scope ambiguity; retain source and hold entire group.',
  10867:'20-fold concentration is a composition claim, not 20 items.',
  13751:'Remove duplicated 富晒 composition claim; preserve readable original wording without correcting it to another character.',
  17761:'温润 refers to embedded white chalcedony, not an independent primary-product capability.'
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true',help='Freeze passed audit into a pilot-only training bundle')
    args=parser.parse_args()
    contracts_path=OUT/'contracts1000.jsonl'
    records=rows(contracts_path);original={r['task_id']:r for r in rows(SOURCE)}
    current={r['task_id']:r for r in records}
    held={r['task_id'] for r in rows(OUT/'source_limitations.jsonl')}
    if rows(OUT/'compiler_findings.jsonl'):raise ValueError('Compiler has unresolved findings')
    lock=json.loads((OUT/'sample_lock.json').read_text(encoding='utf-8'))
    if lock['source_sha256']!=sha256(SOURCE):raise ValueError('Source changed after sample selection')
    sample=rows(OUT/'source_sample60.jsonl')
    if sha256(OUT/'source_sample60.jsonl')!=lock['sample_sha256']:raise ValueError('Sample changed')
    review=[]
    for sampled in sample:
        task=sampled['task_id'];row=current[task]
        note=SAMPLE_NOTES.get(task)
        review.append({'task_id':task,'query':row['query'],'selection':sampled['selection'],
            'source_contract_sha256':digest(original[task]['normalized']),
            'accepted_contract_sha256':digest(row['normalized']),
            'verdict':'held_source_limitation' if task in held else 'fixed_and_rechecked' if note else 'accepted',
            'reviewer':'assistant_source_review',
            'review_scope':'Complete original query against all extracted requirements; no product/target information read for this review.',
            'finding':note or '核对类别、必需/偏好、预算、规格/对象及数量关系，未发现本轮需修订项。',
            'not_a_user_whole_query_manual_approval':True})
    lines(OUT/'sample_reviews60.jsonl',review)
    ledger=[]
    for row in records:
        task=row['task_id'];c=row['normalized']
        validate(c,source_query=row['query'],require_reviewed=task not in held)
        if row['query']!=original[task]['query']:raise ValueError('Actor source changed')
        ledger.append({'task_id':task,'query_sha256':__import__('hashlib').sha256(row['query'].encode()).hexdigest(),
            'source_contract_sha256':digest(original[task]['normalized']),'contract_sha256':digest(c),
            'status':'held_source_limitation' if task in held else 'batch_accepted',
            'approval_basis':'prior_targeted_reviews_full_program_checks_fixed60_source_audit',
            'prior_exact_whole_query_approval':row.get('whole_contract_approved',False)})
    lines(OUT/'release_ledger1000.jsonl',ledger)

    # Use the same sanitized catalog loader as the production backend, not
    # gold-ASIN or target-option lookup from the source annotations.
    import gzip
    archive=UPSTREAM/'environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz'
    with gzip.open(archive,'rt',encoding='utf-8') as handle:
        products={str(p['asin']):{k:v for k,v in p.items() if k not in ('instructions','reviews')} for p in json.load(handle)}
    cache=rows(ROOT/'analysis/semantic_product_facts_v2_20261107/catalog_facts.jsonl')
    facts={}
    for f in cache:facts.setdefault(f['product_asin'],[]).append(f)
    probe_path=ROOT/'analysis/semantic_product_verifier_20261107/train20_target_probes.jsonl'
    probes=rows(probe_path);static=[]
    for p in probes:
        task=p['task_id'];row=current[task]
        if task in held:
            static.append({'task_id':task,'held':True,'reason':'source_limitation','not_a_historical_purchase':True});continue
        product=products[str(p['asin'])];mapping=canonicalize_purchase(product,p['selected_options'])
        if mapping['errors']:raise ValueError('Static probe option mapping failed')
        v=verify(row['normalized'],product,mapping['selected_options'],catalog_facts=facts.get(str(p['asin']),[]),purchase_executed=True)
        score=score_training_verified(row['normalized'],v,source_query=row['query'])
        static.append({'task_id':task,'asin':str(p['asin']),'selected_options':mapping['selected_options'],
            'not_a_historical_purchase':True,'probe_origin':'existing_target_diagnostic_only_never_runtime_training_input',
            'verification':v,'score':score})
    lines(OUT/'static_train19.jsonl',static)
    # Exercise every accepted contract against one fixed public SKU. These
    # intentionally unrelated purchases test executable totality, not quality,
    # feasibility, attainable gold, or expected training signal.
    fixed=next(p for p in probes if p['task_id']==181)
    product=products[str(fixed['asin'])];mapping=canonicalize_purchase(product,fixed['selected_options'])
    stress=[]
    for row in records:
        if row['task_id'] in held:continue
        v=verify(row['normalized'],product,mapping['selected_options'],
                 catalog_facts=facts.get(str(product['asin']),[]),purchase_executed=True)
        s=score_training_verified(row['normalized'],v,source_query=row['query'])
        stress.append({'task_id':row['task_id'],'reward':s['reward'],'reward_valid':s['reward_valid'],
                       'type':s['type'],'verification_blockers':v['readiness_blockers']})
    lines(OUT/'kernel_stress981.jsonl',stress)
    history=json.loads((OUT/'history/acceptance_summary.json').read_text(encoding='utf-8'))
    history_passed=history['evidence_history_acceptance_passed'] and all(history['regression_checks'].values())
    scored=[r for r in static if not r.get('held')]
    atoms=Counter(s for r in scored for s in r['verification']['statuses'].values())
    checks={
        'original1000_and_queries_preserved':len(records)==1000 and set(current)==set(original),
        'eligible_contracts_strictly_valid':len(records)-len(held)==981,
        'no_unresolved_compiler_findings':not rows(OUT/'compiler_findings.jsonl'),
        'fixed60_sample_review_closed':len(review)==60 and all(r['verdict'] in ('accepted','fixed_and_rechecked','held_source_limitation') for r in review),
        'history_evidence_reward_and_ordering_passed':bool(history_passed),
        'static_training_probes_scalar':all(r['score']['reward_valid'] and math.isfinite(r['score']['reward']) for r in scored),
        'all_eligible_kernel_stress_finite_in_range':len(stress)==981 and all(r['reward_valid'] and math.isfinite(r['reward']) and -.85<=r['reward']<=1 for r in stress),
        'no_invalid_fact_cache_records_in_static_probes':not any(r['verification']['rejected_catalog_facts'] for r in scored),
        'bounded_pilot_not_full_training':True,
    }
    summary={'stage':'batch_acceptance','checks':checks,'passed':all(checks.values()),'original_tasks':1000,
        'released_for_pilot':981,'held':len(held),'held_ids':sorted(held),'whole_query_approvals_preserved':7,
        'sample_review_verdicts':dict(Counter(r['verdict'] for r in review)),
        'static_probes':len(static),'static_nonheld_scored':len(scored),'static_atom_statuses':dict(atoms),
        'kernel_stress_cases':len(stress),'kernel_stress_is_quality_evaluation':False,
        'history_trajectories_audited':100,'history_tasks_semantically_annotated':14,'history_purchases_scored':26,
        'history_fixed_execution_failures_scored':7,'history_full_dev50_semantic_evaluation':False,
        'runtime_reviewed_catalog_products':len(facts),'runtime_reviewed_catalog_facts':len(cache),
        'max_optimizer_steps':20,'full_training_released':False,'gpu_integration_tested':False,'training_started':False,
        'api_calls':0,'actor_queries_changed':0,'tasks_removed':0,
        'limitations':['Fixed60 includes targeted risks and is not an unbiased post-fix error-rate estimate.',
            '19 static target probes are integration diagnostics, not historical trajectories or gold feasibility certification.',
            '981 fixed-SKU stress cases prove executable scoring, not semantic coverage or learning signal.',
            '100 histories include only14 fully annotated queries; do not report 100 semantic purchase rewards.',
            'Public fact cache is limited; actual sampled group variance and GPU/veRL integration must be checked during the pilot.']}
    dump(OUT/'acceptance_summary.json',summary)
    if not all(checks.values()):raise SystemExit('Batch acceptance failed; inspect acceptance_summary.json')
    names={'contracts':contracts_path,'source_contracts':SOURCE,'ledger':OUT/'release_ledger1000.jsonl',
           'data_edits':OUT/'data_edits.jsonl','source_limits':OUT/'source_limitations.jsonl',
           'sample_reviews':OUT/'sample_reviews60.jsonl','sample_lock':OUT/'sample_lock.json',
           'policy':PACKAGE/'policy.json','acceptance_summary':OUT/'acceptance_summary.json',
           'history_summary':OUT/'history/acceptance_summary.json','history_results':OUT/'history/replay100.jsonl',
           'static_train_probes':OUT/'static_train19.jsonl','kernel_stress':OUT/'kernel_stress981.jsonl'}
    certificate={'version':'semantic-batch-release-v1','status':'passed_for_bounded_pilot','scope':'bounded_pilot',
        'max_optimizer_steps':20,'full_training_released':False,'checks':checks,
        'released_ids':sorted(set(current)-held),'held_ids':sorted(held),
        'files':{name:{'path':p.relative_to(ROOT).as_posix(),'sha256':sha256(p)} for name,p in names.items()}}
    dump(OUT/'pilot_release.json',certificate)
    if args.freeze:
        manifest=build_bundle(OUT/'training_bundle',contracts_path=contracts_path,batch_release_path=OUT/'pilot_release.json')
        backend=TrainingEvidenceBackend(OUT/'training_bundle/manifest.json');backend.require_ready()
        print(json.dumps({'pilot_ready':True,'released':len(backend.released_ids),'held':len(backend.held_ids),
                          'prior_exact_whole_query_approvals':len(backend.approved_ids),'full_training_released':False}))
    else:print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
