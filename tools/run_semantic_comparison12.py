"""Run one model on the fixed 12-task r5 diagnostic sample.

Uses the immutable r5 extractor and collector, without changing their sources.
Credentials are read only for actual requests. Resume skips saved model replies,
including validation failures; a diagnostic failure must not trigger rebilling.
"""
import argparse
import os
from pathlib import Path
import sys
from extract_semantic_requirements_r5 import ROOT, atomic, digest, dump, endpoint, journal_read, loads, snapshot
from collect_semantic_diagnostics_r5 import aggregate, collect

SAMPLE = ROOT / 'analysis/semantic_r5_actual_first5_20261103/comparison12'


def inputs(sample=SAMPLE):
    sample=sample.resolve()
    raw=(sample/'manifest.json').read_bytes();lock=loads(raw.decode('utf-8'))
    tasks,requests,files,provenance=snapshot('regression50','semantic-v2-r5')
    signature={k:v for k,v in provenance['files_sha256'].items() if k not in ('tasks.jsonl','requests.jsonl')}
    if signature!=lock['protocol_files_sha256']:
        raise ValueError('Comparison protocol changed; do not silently upgrade the fixed sample')
    for name,h in lock['files_sha256'].items():
        if digest((sample/name).read_bytes())!=h:raise ValueError('Fixed comparison file changed: '+name)
    selected=[loads(l) for l in (sample/'tasks.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    rows=[loads(l) for l in (sample/'requests.template.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    wanted=set(lock['task_ids'])
    if len(wanted)!=12 or len(selected)!=12 or selected!=[t for t in tasks if t['task_id'] in wanted]:
        raise ValueError('Fixed 12 tasks differ from original training queries')
    if rows!=[r for r in requests if r['task_id'] in wanted]:
        raise ValueError('Fixed model messages changed')
    files=dict(files)
    files['tasks.jsonl']=''.join(dump(t)+'\n' for t in selected).encode('utf-8')
    files['requests.jsonl']=''.join(dump(r)+'\n' for r in rows).encode('utf-8')
    provenance=dict(provenance,mode='comparison12',count=12,
                    comparison_manifest_sha256=digest(raw),
                    files_sha256={n:digest(b) for n,b in files.items()})
    return selected,rows,files,provenance


def donor_records(donor,tasks,requests,files,config,provenance):
    if donor is None:return [],None
    donor=donor.resolve();m=loads((donor/'run_manifest.json').read_text(encoding='utf-8'))
    if m['api']!=config or m['dataset']['source_hashes']!=provenance['source_hashes']:
        raise ValueError('Reuse requires identical model, API controls and original source')
    for name,data in files.items():
        if name in ('tasks.jsonl','requests.jsonl'):continue
        raw=(donor/name).read_bytes()
        if digest(raw)!=m['dataset']['files_sha256'][name] or raw!=data:
            raise ValueError('Reuse protocol differs: '+name)
    from extract_semantic_requirements_r5 import contract_from_response
    by={t['task_id']:t for t in tasks};rows={r['task_id']:r for r in requests}
    path=donor/'attempts.jsonl';raw=path.read_bytes();records=[];seen=set()
    for r in journal_read(path):
        tid=r['task_id']
        if tid not in by or tid in seen or not r.get('accepted') or r.get('http_status')!=200:continue
        # Reuse only actual first-pass messages matching this experiment.
        if r.get('request_messages')!=rows[tid]['messages']:continue
        contract_from_response(r['raw_response'],by[tid],loads(files['output.schema.json'].decode('utf-8')))
        records.append(dict(r,imported_from=str(donor),source_accepted=True));seen.add(tid)
    return records,{'source':str(donor),'manifest_sha256':digest((donor/'run_manifest.json').read_bytes()),
                    'journal_sha256':digest(raw),'reused_task_ids':sorted(seen)}


def prepare(output,config,*,sample=SAMPLE,reuse_from=None,attempts=2,timeout=180):
    tasks,requests,files,provenance=inputs(sample)
    output=output.resolve();sample=sample.resolve()
    protected=[sample,ROOT/'experiments/semantic_contract_v2_r5',ROOT/'experiments/semantic_holdout50_v1']
    if reuse_from is not None:protected.append(reuse_from.resolve())
    if any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Use an independent output directory outside the sample/protocol/reuse source')
    seeds,donor=donor_records(reuse_from,tasks,requests,files,config,provenance)
    sources=[Path(__file__),ROOT/'tools/extract_semantic_requirements_r5.py',ROOT/'tools/collect_semantic_diagnostics_r5.py']
    manifest={'version':'semantic-comparison12-run-v1','dataset':provenance,'api':config,'seed_attempts':len(seeds),
              'reuse':donor,'code_sha256':{str(p.relative_to(ROOT)):digest(p.read_text(encoding='utf-8').encode('utf-8')) for p in sources},
              'controls':{'attempts_per_new_task':attempts,'timeout':timeout,'skip_saved_model_failures':True,
                          'comparison':'Same fixed messages and API controls except model; gateway default reasoning may differ'}}
    output.mkdir(parents=True,exist_ok=True)
    path=output/'run_manifest.json'
    if path.exists():
        if loads(path.read_text(encoding='utf-8'))!=manifest:
            raise ValueError('Run config/source changed; use a NEW output directory')
        for name,data in files.items():
            if (output/name).read_bytes()!=data:raise ValueError('Run snapshot changed: '+name)
        if journal_read(output/'seed_attempts.jsonl')!=seeds:raise ValueError('Saved reuse evidence changed')
    else:
        if any(p.name!='.run.lock' for p in output.iterdir()):raise ValueError('New output directory must be empty')
        for name,data in files.items():(output/name).write_bytes(data)
        atomic(output/'seed_attempts.jsonl',''.join(dump(r)+'\n' for r in seeds))
        atomic(path,dump(manifest)+'\n')
    return tasks,requests,loads(files['output.schema.json'].decode('utf-8')),manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--base-url',default=os.getenv('LLM_BASE_URL'))
    parser.add_argument('--api-key-env',default='LLM_API_KEY')
    parser.add_argument('--allow-school-http',action='store_true')
    parser.add_argument('--reuse-from',type=Path)
    parser.add_argument('--attempts',type=int,default=2)
    parser.add_argument('--timeout',type=int,default=180)
    parser.add_argument('--max-tokens',type=int,default=8192)
    parser.add_argument('--limit',type=int)
    local=parser.add_mutually_exclusive_group()
    local.add_argument('--prepare-only',action='store_true')
    local.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not args.base_url:parser.error('Set --base-url or LLM_BASE_URL')
    if min(args.attempts,args.timeout,args.max_tokens)<=0 or args.limit is not None and args.limit<=0:
        parser.error('Numeric options must be positive')
    config={'endpoint':endpoint(args.base_url,allow_school_http=args.allow_school_http,local_only=args.prepare_only or args.check_only),
            'model':args.model,'thinking':'omit','extraction_profile':'semantic-v2-r5','evidence_offsets':'strict',
            'response_format':'json_object','max_tokens':args.max_tokens,'temperature':0}
    # Validate protected paths before creating the destination or lock.
    output=args.output.resolve()
    protected=[SAMPLE.resolve(),(ROOT/'experiments').resolve()]
    if args.reuse_from:protected.append(args.reuse_from.resolve())
    if any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Use an independent output directory')
    output.mkdir(parents=True,exist_ok=True);lock=output/'.run.lock'
    try:handle=lock.open('x',encoding='utf-8')
    except FileExistsError:raise ValueError('Run locked; check running process before removing lock') from None
    try:
        with handle:handle.write(str(os.getpid()))
        tasks,requests,schema,manifest=prepare(output,config,reuse_from=args.reuse_from,attempts=args.attempts,timeout=args.timeout)
        if args.prepare_only or args.check_only:
            _,report=aggregate(output,tasks,schema,manifest)
            print(dump({k:report[k] for k in ('total_tasks','collected_tasks','validated_tasks','reused_tasks')}))
            print('Local preparation/check only; no API requests.')
            print('Report: '+str(output/'report.json'))
            return 0 if args.prepare_only or report['collection_complete'] else 2
        key=os.getenv(args.api_key_env,'').strip()
        if not key:raise ValueError('Missing '+args.api_key_env+'; no API requests made')
        return collect(output,tasks,requests,schema,manifest,key,args.limit)
    finally:lock.unlink()


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted; rerun the same command to resume.',file=sys.stderr);raise SystemExit(130)
    except Exception as exc:
        print('ERROR: '+str(exc),file=sys.stderr);raise SystemExit(1)
