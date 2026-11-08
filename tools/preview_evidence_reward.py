"""Compare strict truth scores with the user-confirmed evidence objective.

The proposed objective counts verified support, not latent real-world truth.
Unknowns remain unknown in verification and complete success metrics. The
denominator is the fixed contract, preventing unknown fields being dropped.
Only fully reviewed contracts or existing decisive failures get a score. This
command writes a comparison report; it does not start any training.
"""
import argparse
import json
from pathlib import Path
from semantic_evidence_reward import score_training_verified


def preview(contract,verification,strict_score):
    score=score_training_verified(contract,verification,source_query=contract['query'])
    return {'active_for_training':False,'policy_confirmed_by_user':True,'objective':score['objective'],
            'reward':score['reward'],'truth_statuses_unchanged':True,'complete_semantic_success':score['semantic_success'],
            'basis':score.get('basis',score['type'])}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    out=args.directory
    contracts={r['task_id']:r['normalized'] for r in map(json.loads,(out/'dev14_contracts.jsonl').read_text(encoding='utf-8').splitlines()) if r['program_valid']}
    records=[]
    for r in map(json.loads,(out/'replay100.jsonl').read_text(encoding='utf-8').splitlines()):
        if r.get('verification') is None:continue
        records.append({'task_id':r['task_id'],'run':r['run'],'strict_reward':r['score']['reward'],
                        'strict_semantic_success':r['score']['semantic_success'],**preview(contracts[r['task_id']],r['verification'],r['score'])})
    (out/'evidence_reward_preview.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    report={'active_for_training':False,'strict_scalar_rewards':sum(r['strict_reward'] is not None for r in records),
            'proposed_evidence_scores':sum(r['reward'] is not None for r in records),'records':len(records),
            'policy_confirmed_by_user':True,'training_started':False}
    (out/'evidence_reward_preview_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
