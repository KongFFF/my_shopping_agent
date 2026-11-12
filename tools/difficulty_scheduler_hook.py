"""Driver-only hooks for the pinned veRL trainer; all generated groups observed."""
import os
from pathlib import Path
from difficulty_scheduler import TaskScheduler, append, canonical, classify, read, sha, write


def setup_worker():
    """Preserve original compatibility hook, seed each new Ray worker consistently."""
    import random
    import numpy
    import torch
    from shopping_grpo.training.grpo.compat import install_torch_padding_fallback
    install_torch_padding_fallback()
    spec = read(os.environ['SHOPPING_SCHEDULER_SPEC'])
    random.seed(spec['seed'])
    numpy.random.seed(spec['seed'])
    torch.manual_seed(spec['seed'])


class DriverSchedule:
    def __init__(self, trainer):
        self.spec = read(os.environ['SHOPPING_SCHEDULER_SPEC'])
        self.output = Path(self.spec['output'])
        self.path = self.output / 'scheduler_state.json'
        self.events = self.output / 'scheduler_events.jsonl'
        self.binding = sha(canonical(self.spec))
        self.runtime = {'skipped_updates': 0, 'consecutive_skips': 0}
        self.last_checkpoint_generated = None
        self.dataset = trainer.train_dataset
        self.collate = trainer.train_dataloader.collate_fn
        self.mapping = {}
        for index, row in enumerate(self.dataset.dataframe):
            extra = row.get('extra_info') or {}
            task = extra.get('task_id')
            if task is None:
                # Original parquet stores tool task IDs in tools_kwargs; no guessed row index.
                tools = extra.get('tools_kwargs') or {}
                task = tools.get('task_id')
            if task is None:
                raise ValueError('Cannot bind original parquet task_id: ' + repr(extra))
            task = int(task)
            if task in self.mapping:
                raise ValueError('Duplicate parquet task')
            self.mapping[task] = index
        if sorted(self.mapping) != sorted(self.spec['task_ids']):
            raise ValueError('Dataset filtered/changed the original train1000 roster')
        if self.path.exists():
            stored = read(self.path)
            if stored['binding'] != self.binding:
                raise ValueError('Resume spec changed')
            self.scheduler = TaskScheduler.from_state(stored['scheduler'])
            self.runtime = stored['runtime']
            checkpoint = self.output / f'global_step_{trainer.global_steps}' / 'scheduler_state.json'
            # A fault tail is never silently rerolled or restored with stale weights.
            untouched = (self.scheduler.generated == 0 and trainer.global_steps == 0
                         and not self.events.exists() and not self.scheduler.pending)
            if not untouched and (not checkpoint.exists() or read(checkpoint) != stored or stored['scheduler']['pending']):
                raise RuntimeError('Uncommitted generation/update tail: retain artifacts and investigate; no automatic reroll')
            if self.scheduler.updates != trainer.global_steps:
                raise ValueError('Model/optimizer and scheduler checkpoint mismatch')
            if not untouched:
                self.last_checkpoint_generated = self.scheduler.generated
        else:
            if trainer.global_steps or self.events.exists():
                raise ValueError('Missing scheduler state for existing training')
            self.scheduler = TaskScheduler(self.spec['task_ids'], self.spec['arm'], self.spec['seed'], self.spec['budget'])
            self.persist()
        self.ready_uids = None

    def persist(self):
        write(self.path, {'binding': self.binding, 'scheduler': self.scheduler.state_dict(), 'runtime': self.runtime})

    @property
    def exhausted(self):
        return self.scheduler.exhausted

    def batches(self, trainer):
        while not self.scheduler.exhausted:
            pending = self.scheduler.reserve()
            self.persist()
            append(self.events, {'event': 'selection', **pending, 'generated': self.scheduler.generated})
            yield self.collate([self.dataset[self.mapping[t['task_id']]] for t in pending['tasks']])

    def observe(self, batch, group_stats):
        infos = batch.non_tensor_batch['shopping'].tolist()
        try:
            pending = self.scheduler.observe(infos)
        except BaseException as exc:
            append(self.events, {'event': 'generation_error', 'pending': self.scheduler.pending, 'error': repr(exc)})
            raise
        # Token counts include rejected/all-equal groups, not just trained groups.
        response = batch.batch['response_mask'].sum().item()
        attention = batch.batch['attention_mask'].sum().item()
        self.persist()
        append(self.events, {'event': 'generation', **pending, 'generated': self.scheduler.generated,
                            'response_tokens': int(response), 'input_plus_response_tokens': int(attention),
                            'groups': group_stats['groups'], 'rollouts': infos,
                            'rollout_uids': batch.non_tensor_batch['uid'].tolist()})

    def committed(self, step):
        # The actual aligned batch, including any cross-generation concatenation,
        # is recorded independently of which groups merely passed the filter.
        self.scheduler.commit_update(step)
        self.runtime['consecutive_skips'] = 0
        self.persist()
        append(self.events, {'event': 'update', 'update': step, 'generated': self.scheduler.generated,
                             'consumed_uids': self.ready_uids})
        self.ready_uids = None

    def ready(self, batch):
        uids = batch.non_tensor_batch['uid'].tolist()
        if len(uids) != 8 or len(set(uids)) != 2 or any(uids.count(u) != 4 for u in set(uids)):
            raise ValueError('Actual optimizer batch must contain two aligned four-rollout UID groups')
        self.ready_uids = list(dict.fromkeys(uids))
        append(self.events, {'event': 'ready', 'uids': self.ready_uids,
                            'response_tokens': int(batch.batch['response_mask'].sum().item()),
                            'input_plus_response_tokens': int(batch.batch['attention_mask'].sum().item())})

    def skipped(self, total, consecutive):
        self.runtime = {'skipped_updates': total, 'consecutive_skips': consecutive}
        self.persist()
        append(self.events, {'event': 'skip', **self.runtime, 'generated': self.scheduler.generated})

    def checkpoint(self, trainer):
        if self.scheduler.pending or trainer.global_steps != self.scheduler.updates:
            raise ValueError('Checkpoint attempted with incomplete scheduler transaction')
        self.persist()
        target = self.output / f'global_step_{trainer.global_steps}' / 'scheduler_state.json'
        write(target, read(self.path))
        self.last_checkpoint_generated = self.scheduler.generated

    def boundary_stop(self, trainer, residual_groups=0):
        """Called only after an update or after the original buffer has been cleared."""
        if not self.exhausted and not (self.output / 'STOP').exists():
            return False
        if self.last_checkpoint_generated != self.scheduler.generated:
            trainer._save_checkpoint()
            self.checkpoint(trainer)
        write(self.output / 'training_complete.json', {
            'complete': self.exhausted, 'stopped': not self.exhausted,
            'arm': self.scheduler.arm, 'generated_rollouts': self.scheduler.generated,
            'optimizer_updates': self.scheduler.updates,
            'consumed_groups': self.scheduler.consumed_groups,
            'residual_groups_discarded': residual_groups,
            'spec_binding': self.binding})
        print('SCHEDULER_TRAINING_COMPLETE' if self.exhausted else 'SCHEDULER_SAFE_STOP', flush=True)
        return True
