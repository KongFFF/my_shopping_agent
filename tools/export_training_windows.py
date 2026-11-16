"""Export existing online GRPO windows for offline audit; never load models.

Default: local archive only. --git-upload explicitly publishes a result branch.
Uses standard library; excludes legacy fixed-checkpoint reevaluation trajectories.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

RUN = Path('outputs/runs/grpo_baseline_20261022_202816')
WINDOWS = {'reference351_500': ('formal_resume350_v1', 351, 500),
           'late651_800': ('formal_resume650_to800_v1', 651, 800)}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def inventory(folder):
    """Names/sizes only. Do not descend into weights, Dev, Final or services."""
    found = []
    for current, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith('global_step_') and d not in
                   ('validation', 'evaluation', 'export', '.git', 'wandb', 'swanlog', 'checkpoints')]
        for name in files:
            path = Path(current) / name
            if path.suffix in ('.json', '.jsonl', '.yaml', '.yml'):
                found.append({'path': str(path), 'bytes': path.stat().st_size,
                              'possible_rollout': any(word in str(path.relative_to(folder)).lower() for word in ('rollout', 'trajectory', 'trace'))})
    return sorted(found, key=lambda r: r['path'])


def inspect_log(raw, start, end):
    events, selected, updates, generated_uids = Counter(), [], [], set()
    rollout_count = 0
    has_actions = has_observations = 0
    task_ids, committed_uids, consumed_tasks = set(), set(), set()
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        step = int(event['global_step'])
        if not start <= step <= end:
            raise ValueError(f'Unexpected log step {step} outside {start}..{end} at line {number}')
        events[event['event']] += 1
        selected.append(event)
        if event['event'] == 'optimizer_step':
            if event['metrics'].get('training/optimizer_updated') != 1:
                raise ValueError('Optimizer event lacks confirmed update')
            updates.append(step)
            rs = event.get('rollouts', [])
            grouped = Counter(str(r['uid']) for r in rs)
            if len(rs) != 8 or len(grouped) != 2 or any(n != 4 for n in grouped.values()):
                raise ValueError('Each optimizer update must consume 2 groups of 4 rollouts')
            if committed_uids & grouped.keys():
                raise ValueError('Generated UID consumed by multiple updates')
            committed_uids.update(grouped)
            consumed_tasks.update(int(r['task_id']) for r in rs)
        if event['event'] == 'generation_batch':
            group_ids = {str(g['uid']) for g in event.get('groups', [])}
            if generated_uids & group_ids:
                raise ValueError('Repeated generated group UID')
            generated_uids.update(group_ids)
            rs = event.get('rollouts', [])
            grouped = Counter(str(r['uid']) for r in rs)
            if set(grouped) != group_ids or any(n != 4 for n in grouped.values()):
                raise ValueError('Generation groups must each have 4 diagnostic rollouts')
            for r in rs:
                rollout_count += 1
                task_ids.add(int(r['task_id']))
                has_actions += isinstance(r.get('actions'), list)
                has_observations += any(key in r for key in ('observations', 'trajectory', 'messages'))
    if updates != list(range(start, end + 1)):
        raise ValueError(f'Incomplete or repeated optimizer chain for {start}..{end}')
    if not committed_uids <= generated_uids:
        raise ValueError('Optimizer consumed UID not present in generation records')
    return {'start': start, 'end': end, 'optimizer_updates': len(updates),
            'events': dict(events), 'generated_groups': len(generated_uids),
            'generated_rollouts': rollout_count, 'generated_unique_tasks': len(task_ids),
            'generated_task_ids': sorted(task_ids), 'consumed_unique_tasks': len(consumed_tasks),
            'rollouts_with_actions': has_actions, 'rollouts_with_observation_like_field': has_observations,
            'observation_field_note': 'Field presence only, not proof of complete Actor-visible observations.'}


def collect(parent):
    project = parent / 'upstream/shopping-grpo-longhorizon'
    run = project / RUN
    blobs, sources, reports, files = {}, {}, {}, {}

    def add(path, name, transform=None):
        raw = path.read_bytes()
        sources[name] = {'path': str(path), 'original_sha256': digest(raw), 'original_bytes': len(raw)}
        payload = transform(raw) if transform else raw
        blobs[name] = payload
        sources[name].update(archive_sha256=digest(payload), archive_bytes=len(payload))
        files[path] = digest(raw)
        return raw

    for label, (folder_name, start, end) in WINDOWS.items():
        folder = run / folder_name
        prep = run / 'preparation' / folder_name
        raw = add(folder / 'training_diagnostics.jsonl', f'{label}/training_diagnostics.jsonl')
        reports[label] = inspect_log(raw, start, end)
        files_list = inventory(folder)
        blobs[f'{label}/file_inventory.json'] = json_bytes(files_list)
        spec_path = prep / 'launch_spec.json'
        spec_raw = spec_path.read_bytes()
        spec = json.loads(spec_raw.decode('utf-8-sig'))
        # Export provenance without environment variables or launch command secrets.
        filtered = {k: spec[k] for k in ('project', 'asset_project', 'output', 'resume_step', 'total_updates', 'source_sha256') if k in spec}
        add(spec_path, f'{label}/launch_binding.json', lambda _: json_bytes(filtered))
        status = prep / 'exit_status.json'
        if status.exists():
            status_obj = json.loads(add(status, f'{label}/exit_status.json').decode('utf-8-sig'))
            if status_obj.get('exit_code') != 0:
                raise ValueError(f'{label}: training exit receipt is not successful')
        else:
            reports[label]['exit_receipt_missing'] = True
        # Save bound source code/config, including mismatches for diagnosis;
        # actual historical identity is assessed after return, not silently fixed.
        source_checks = []
        for index, (source, expected) in enumerate(sorted(spec.get('source_sha256', {}).items())):
            p = Path(source)
            check = {'path': source, 'expected_sha256': expected, 'exists': p.is_file()}
            if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.json', '.patch') and p.stat().st_size <= 4 * 1024 * 1024:
                content = p.read_bytes()
                # Source bindings should not contain credentials. Refuse accidental
                # secret assignments; never inspect .env or credential files.
                if re.search(rb'(?im)^\s*[\w.-]*(?:api_key|password|access_token|secret_key)\s*[:=]\s*["\']?[^\s"\'{}$][^\r\n]+', content):
                    check['not_copied'] = 'possible credential assignment'
                else:
                    name = f'{label}/bound_sources/{index:03d}_{p.name}'
                    add(p, name)
                    check.update(actual_sha256=digest(content), matches_expected=digest(content) == expected, archive_path=name)
            source_checks.append(check)
        blobs[f'{label}/source_checks.json'] = json_bytes(source_checks)
        composed = prep / 'composed_config.yaml'
        if composed.exists():
            # Resolved configs can contain env values: retain only the sections
            # needed for reconstruction, stripping runtime env and logging blocks.
            raw = composed.read_bytes()
            config_lines, top = [], ''
            for line in raw.decode('utf-8-sig').splitlines():
                if line and not line[0].isspace() and not line.startswith('#'):
                    top = line.split(':', 1)[0]
                if top in ('ray_kwargs', 'trainer', 'global_profiler') or re.search(r'(?i)api_key|password|access_token|secret_key', line):
                    continue
                config_lines.append(line)
            add(composed, f'{label}/recipe_excerpt.yaml', lambda _: ('\n'.join(config_lines) + '\n').encode('utf-8'))
    train_raw = add(project / 'data/grpo/train.jsonl', 'inputs/train.jsonl')
    train_ids = {int(json.loads(line)['task_id']) for line in train_raw.decode('utf-8-sig').splitlines() if line.strip()}
    if len(train_ids) != 1000 or any(not set(report['generated_task_ids']) <= train_ids for report in reports.values()):
        raise ValueError('Expected original 1000 training tasks and no generated tasks outside that set')
    add(project / 'configs/agent_loop.yaml', 'inputs/current_project_agent_loop.yaml')
    environment_meta = project / 'data/environment.json'
    if environment_meta.exists():
        add(environment_meta, 'inputs/environment.json')
    baseline_plan = run / 'preparation/baseline_plan.json'
    if baseline_plan.exists():
        add(baseline_plan, 'inputs/baseline_data_binding.json', lambda raw: json_bytes({
            k: v for k, v in json.loads(raw.decode('utf-8-sig')).items()
            if k in ('data_sha256', 'upstream_head', 'base', 'project')}))
    for path, expected in files.items():
        if digest(path.read_bytes()) != expected:
            raise ValueError(f'Source changed while exporting: {path}')
    manifest = {'schema': 'online-training-window-export-v1', 'sources': sources, 'windows': reports,
                'scope': 'All generation and optimizer events in the retained 351..500 and 651..800 windows.',
                'limits': ['No legacy 4192 reevaluation traces, weights, Dev or Final included.',
                           'Source/config mismatch is reported, not assumed replay-ready.',
                           'These are changing training policies, not fixed checkpoint500/800 evaluations.']}
    for name, raw in blobs.items():
        if name not in sources:
            manifest.setdefault('generated_files', {})[name] = {'sha256': digest(raw), 'bytes': len(raw)}
    return blobs, manifest


def publish(parent, archive, code_ref):
    def git(*args, cwd=parent):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()
    commit = git('rev-parse', '--verify', code_ref + '^{commit}')
    remote = git('remote', 'get-url', 'origin')
    clone = Path(tempfile.mkdtemp(prefix='training-window-return-', dir=parent / 'outputs'))
    git('clone', '--shared', '--no-checkout', str(parent), str(clone))
    git('remote', 'set-url', 'origin', remote, cwd=clone)
    branch = 'codex/training-window-results-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    git('switch', '-c', branch, commit, cwd=clone)
    for key in ('user.name', 'user.email'):
        value = subprocess.run(['git', 'config', '--get', key], cwd=parent, capture_output=True, text=True)
        if value.returncode == 0:
            git('config', key, value.stdout.strip(), cwd=clone)
    target = clone / 'analysis/online_training_window_return_20261116/evidence.zip'
    target.parent.mkdir(parents=True)
    target.write_bytes(archive.read_bytes())
    git('add', '--', 'analysis/online_training_window_return_20261116/evidence.zip', cwd=clone)
    git('commit', '-m', 'data: return retained online GRPO training windows', cwd=clone)
    git('push', '-u', 'origin', branch, cwd=clone)
    print('RETURN_BRANCH=' + branch)
    print('RETURN_COMMIT=' + git('rev-parse', 'HEAD', cwd=clone))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path.cwd())
    parser.add_argument('--git-upload', action='store_true')
    parser.add_argument('--code-ref', default='HEAD')
    args = parser.parse_args()
    parent = args.parent.resolve()
    blobs, manifest = collect(parent)
    output = parent / 'outputs'
    output.mkdir(exist_ok=True)
    archive = output / ('online_training_windows_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as handle:
        for name, raw in blobs.items():
            handle.writestr(name, raw)
        handle.writestr('manifest.json', json_bytes(manifest))
    print(json.dumps(manifest['windows'], ensure_ascii=False, indent=2))
    print('ARCHIVE: ' + str(archive))
    print('SHA256: ' + digest(archive.read_bytes()))
    if args.git_upload:
        if archive.stat().st_size >= 90 * 1024 * 1024:
            raise ValueError('Archive exceeds Git return size limit; retain local archive and inspect scope')
        publish(parent, archive, args.code_ref)


if __name__ == '__main__':
    main()
