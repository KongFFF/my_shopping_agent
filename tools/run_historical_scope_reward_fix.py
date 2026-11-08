"""Reproduce the scope/reward repair against the same locked histories.

Writes a separate acceptance directory. Does not rewrite train1000 labels,
original histories, or the first acceptance report; no GPU/network calls.
"""
import argparse
import json
from pathlib import Path
from freeze_historical_reward_audit import ROOT, OUT as BASELINE, digest, write
import build_history_acceptance_contracts as contracts
import build_history_product_facts as facts
import replay_historical_semantic_reward as replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'analysis/semantic_history_representation_20261107')
    args=parser.parse_args();out=args.output.resolve()
    if out==BASELINE.resolve():parser.error('Use a separate output directory to preserve the first acceptance')
    out.mkdir(parents=True,exist_ok=True)
    train=ROOT/'analysis/semantic_final12_20261107/validated1000.jsonl';before=digest(train)
    for name in ('sample_lock.json','dev50_queries.jsonl','trajectory_inventory100.jsonl','actual_purchases.jsonl','actual_product_asins.json','inventory_summary.json'):
        source=BASELINE/name;target=out/name
        if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('Frozen input differs: '+name)
        target.write_bytes(source.read_bytes())
    contracts.main(out);facts.main(out);replay.main(out)
    assert digest(train)==before,'Training labels unexpectedly changed'
    write(out/'repair_manifest.json',{'baseline':str(BASELINE.relative_to(ROOT)),
        'output':str(out.relative_to(ROOT)),'train1000_unchanged_sha256':before,'api_calls':0,
        'confirmed_elided_money_policy':'2026-11-07 user confirmed: 四百五=450, 三千五=3500, monetary context only; existing approximation rule retained',
        'confirmed_qualitative_standard':'2026-11-07 user confirmed: explicit public assertion of the same qualitative property is acceptable; missing claims stay unknown; hope/approximation/damaged text remain pending',
        'confirmed_bundle_introduction':'2026-11-07 user confirmed: generic primary-item selection introduction is not a one-piece order requirement when explicit same-item bundle quantities are provided; explicit only-one and independent order counts remain enforced',
        'confirmed_evidence_training_policy':'2026-11-07 user chose A: verified satisfaction earns credit; unknown remains in fixed denominator, does not establish real failure; complete semantic truth is reported separately',
        'confirmed_exact_approximation_center':'2026-11-07 user confirmed: exact 500g satisfies 500g-around; off-center stays unknown, no numeric tolerance invented',
        'confirmed_component_sets':'2026-11-07 user confirmed: six cup/saucer/spoon groups means six of each; one composite reward requirement',
        'confirmed_readable_source':'2026-11-07 user confirmed: retain original damaged query and score readable 喜气; do not invent missing words',
        'confirmed_sample_closeout':'2026-11-07 user confirmed: both specific hope requirements are required; Dev12703 hot-selling requirement is required and uses explicit frozen-snapshot declarations',
        'source_hashes':{name:digest(ROOT/'tools'/name) for name in ('semantic_source_scope_proofs.py',
           'semantic_contract_v2_r7_final12.py','semantic_fact_language_v3.py','semantic_product_verifier_v4.py',
           'semantic_reward_history.py','build_history_acceptance_contracts.py','build_history_product_facts.py',
           'replay_historical_semantic_reward.py','run_historical_scope_reward_fix.py',
           'semantic_evidence_reward.py','semantic_product_facts_v3.py','semantic_acceptance_representation.py',
           'semantic_component_group_proofs.py','semantic_component_comparison.py','semantic_review_decisions.py')},
        'source_bound_review_decisions_sha256':digest(ROOT/'experiments/semantic_history_review_20261107/user_decisions.json')})


if __name__=='__main__':main()
