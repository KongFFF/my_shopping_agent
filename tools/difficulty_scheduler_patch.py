"""Exact, fail-closed source transform for official veRL 0.8.0 + baseline V3."""
import ast
import re
from difficulty_scheduler import sha

ORIGINAL = 'de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66'
DYNAMIC = '684b491e20ba9d41e91d5010186d4d08b01a01fc67f8a77d17c086b0381e00a3'
MARKER = '# SHOPPING_DIFFICULTY_SCHEDULER_V1'


def unified_patch(source, patch):
    lines, output, cursor = source.splitlines(keepends=True), [], 0
    parts = patch.splitlines(keepends=True)
    i = 0
    while i < len(parts):
        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', parts[i])
        if not match:
            i += 1
            continue
        start, count = int(match[1]), int(match[2] or 1)
        index = start if count == 0 else start - 1
        if index < cursor:
            raise ValueError('Overlapping patch hunk')
        output.extend(lines[cursor:index])
        cursor, i = index, i + 1
        while i < len(parts) and not parts[i].startswith('@@'):
            line = parts[i]
            i += 1
            if line.startswith(('-', ' ')):
                if lines[cursor].rstrip('\r\n') != line[1:].rstrip('\r\n'):
                    raise ValueError('Baseline patch context mismatch')
                cursor += 1
            if line.startswith(('+', ' ')):
                output.append(line[1:])
    output.extend(lines[cursor:])
    return ''.join(output)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Expected one pinned trainer anchor: ' + repr(before[:110]))
    return source.replace(before, after, 1)


def build_trainer(source, baseline_patch):
    source = source.replace('\r\n', '\n')
    if sha(source) == ORIGINAL:
        source = unified_patch(source, baseline_patch.replace('\r\n', '\n'))
    if sha(source) != DYNAMIC:
        raise ValueError('Unknown veRL trainer source; refusing to patch')
    edits = [
        ('        current_epoch = self.global_steps // len(self.train_dataloader)',
         '        from difficulty_scheduler_hook import DriverSchedule\n'
         '        schedule = DriverSchedule(self)\n'
         '        dynamic_skipped_updates = schedule.runtime["skipped_updates"]\n'
         '        dynamic_consecutive_skips = schedule.runtime["consecutive_skips"]\n'
         '        current_epoch = 0\n'
         '        if schedule.exhausted:\n'
         '            schedule.boundary_stop(self)\n'
         '            self._shutdown_dump_executor()\n'
         '            return'),
        ('            for batch_dict in self.train_dataloader:',
         '            for batch_dict in schedule.batches(self):'),
        ('                is_last_step = current_step >= self.total_training_steps',
         '                is_last_step = schedule.exhausted'),
        ('                        dynamic_num_gen_batches += 1',
         '                        schedule.observe(batch, group_stats)\n'
         '                        dynamic_num_gen_batches += 1'),
        ('                            if dynamic_num_gen_batches >= dynamic_max_gen_batches:',
         '                            if (dynamic_num_gen_batches >= dynamic_max_gen_batches\n'
         '                                    or schedule.exhausted):'),
        ('                                dynamic_accepted_batches = []',
         '                                residual_groups = dynamic_accepted_prompts\n'
         '                                dynamic_accepted_batches = []'),
        ('                                if (\n                                    dynamic_consecutive_skips',
         '                                if schedule.boundary_stop(self, residual_groups):\n'
         '                                    self._shutdown_dump_executor()\n'
         '                                    progress_bar.close()\n'
         '                                    return\n'
         '                                if (\n                                    dynamic_consecutive_skips'),
        ('                                if schedule.boundary_stop(self, residual_groups):',
         '                                schedule.skipped(dynamic_skipped_updates, dynamic_consecutive_skips)\n'
         '                                if schedule.boundary_stop(self, residual_groups):'),
        ('                            self.global_steps = current_step',
         '                            self.global_steps = current_step\n'
         '                        schedule.committed(current_step)'),
        ('                        self.checkpoint_manager.sleep_replicas()\n                        dynamic_consecutive_skips = 0',
         '                        schedule.ready(batch)\n'
         '                        self.checkpoint_manager.sleep_replicas()\n                        dynamic_consecutive_skips = 0'),
        ('                if is_last_step:\n                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):',
         '                if schedule.boundary_stop(self):\n'
         '                    self._shutdown_dump_executor()\n'
         '                    progress_bar.close()\n'
         '                    return\n'
         '                if is_last_step:\n                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):'),
        ('        torch.save(dataloader_state_dict, dataloader_local_path)',
         '        torch.save(dataloader_state_dict, dataloader_local_path)\n'
         '        if hasattr(self, "difficulty_schedule"):\n'
         '            self.difficulty_schedule.checkpoint(self)'),
        ('        schedule = DriverSchedule(self)',
         '        schedule = DriverSchedule(self)\n        self.difficulty_schedule = schedule'),
        ('        # Ensure dump executor is shut down when training loop ends without reaching is_last_step',
         '        if not schedule.exhausted:\n'
         '            raise RuntimeError("Scheduler stopped without reaching the fixed rollout budget")\n'
         '        # Ensure dump executor is shut down when training loop ends without reaching is_last_step'),
    ]
    for before, after in edits:
        source = replace_once(source, before, after)
    source = MARKER + '\n' + source
    ast.parse(source)
    return source
