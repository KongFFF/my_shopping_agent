"""CPU-only online task selection; no model, reward or GRPO dependencies."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

VERSION = 'difficulty-scheduler-v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as f:
        f.write(canonical(value) + '\n')
        f.flush()
        import os
        os.fsync(f.fileno())
    temp.replace(path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as f:
        f.write(canonical(value) + '\n')
        f.flush()
        import os
        os.fsync(f.fileno())


def bounded_probabilities(weights):
    """Half uniform / half weighted, capped by 4/K, with water filling."""
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError('Weights must be finite, nonnegative and nonempty')
    k = len(weights)
    floor, ceiling = .5 / k, min(4 / k, 1.)
    p, mass, active = [floor] * k, .5, list(range(k))
    while active and mass > 1e-15:
        total = sum(weights[i] for i in active)
        portions = {i: mass * (weights[i] / total if total else 1 / len(active)) for i in active}
        saturated = [i for i in active if p[i] + portions[i] > ceiling]
        if not saturated:
            for i in active:
                p[i] += portions[i]
            break
        for i in saturated:
            mass -= ceiling - p[i]
            p[i] = ceiling
        active = [i for i in active if i not in saturated]
    if not math.isclose(sum(p), 1, abs_tol=1e-10):
        raise AssertionError('Probability mass lost')
    return p


def classify(info):
    """Use explicit original AgentLoop fields; unknown never means failure."""
    required = ('task_id', 'reward', 'done', 'reward_valid', 'reward_type',
                'infrastructure_invalid', 'reward_unverifiable', 'overlong')
    if any(key not in info for key in required):
        raise ValueError('Incomplete original shopping diagnostic')
    reward = info['reward']
    if not isinstance(reward, dict) or 'sampling_invalid' not in reward:
        raise ValueError('Missing original reward sampling_invalid')
    for key in ('terminal_utility', 'total'):
        if not isinstance(reward.get(key), (float, int)) or not math.isfinite(reward[key]):
            raise ValueError('Missing finite reward: ' + key)
    policy_context = info.get('termination_reason') in (
        'context_hard_limit_exceeded', 'context_budget_exhausted',
        'context_compaction_unsupported_routed_experts')
    if info['infrastructure_invalid'] and not policy_context:
        raise RuntimeError('Infrastructure-invalid rollout: ' + str(info.get('error')))
    reasons = [key for key in ('reward_unverifiable', 'overlong') if info[key]]
    if policy_context:
        reasons.append('policy_context_limit')
    if not info['reward_valid']:
        reasons.append('reward_invalid')
    if reward['sampling_invalid'] and not reasons:
        reasons.append('reward_sampling_invalid')
    valid = not reasons
    gold = valid and info['done'] is True and info['reward_type'] == 'gold_purchase'
    return bool(gold), bool(valid), reasons


class TaskScheduler:
    def __init__(self, task_ids, arm, seed, budget=4000):
        self.ids = sorted(int(t) for t in task_ids)
        if len(set(self.ids)) != len(self.ids) or len(self.ids) < 2:
            raise ValueError('Task IDs must be unique and at least two')
        if arm not in ('uniform', 'adaptive') or budget <= 0 or budget % 8:
            raise ValueError('Bad arm or rollout budget (must be a positive multiple of 8)')
        self.arm, self.seed, self.budget = arm, int(seed), int(budget)
        self.rng = random.Random(seed)
        self.counts = {t: [0., 0., 0., 0] for t in self.ids}  # s, v, m, optimizer age
        self.generated = self.batches = self.updates = 0
        self.pending = None
        self.consumed_groups = self.consumed_rollouts = 0

    def _aged(self, task):
        s, v, m, age = self.counts[task]
        if age > self.updates:
            raise ValueError('Counts are from a future optimizer version')
        decay = 2 ** (-(self.updates - age) / 100)
        return s * decay, v * decay, m * decay

    def estimates(self, task):
        s, v, m = self._aged(task)
        p, q = (s + .5) / (v + 1), (v + 1) / (m + 1)
        return p, q, q ** 4 * (1 - p ** 4 - (1 - p) ** 4)

    @property
    def exhausted(self):
        return self.generated >= self.budget

    def reserve(self):
        if self.pending is not None or self.exhausted:
            raise ValueError('Unresolved generation or exhausted budget')
        available = self.ids.copy()
        chosen = []
        for _ in range(2):
            estimates = [self.estimates(t) for t in available]
            weights = [x[2] for x in estimates]
            probs = ([1 / len(available)] * len(available) if self.arm == 'uniform'
                     else bounded_probabilities(weights))
            position = self.rng.choices(range(len(available)), weights=probs)[0]
            task = available[position]
            chosen.append({'task_id': task, 'probability': probs[position],
                           'candidate_count': len(available), 'p': estimates[position][0],
                           'q': estimates[position][1], 'weight': weights[position],
                           'at_cap': math.isclose(probs[position], 4 / len(available))})
            available.pop(position)
        self.batches += 1
        # Reserve before dispatch: incomplete/faulted work is never silently free.
        self.generated += 8
        self.pending = {'batch': self.batches, 'policy_update': self.updates, 'tasks': chosen}
        return copy.deepcopy(self.pending)

    def observe(self, infos):
        if self.pending is None or len(infos) != 8:
            raise ValueError('Expected exactly the pending eight rollouts')
        selected = [r['task_id'] for r in self.pending['tasks']]
        grouped = {t: [] for t in selected}
        # Validate entire batch before mutating statistics.
        for info in infos:
            t = int(info['task_id'])
            if t not in grouped:
                raise ValueError('Generated an unselected task')
            grouped[t].append(classify(info))
        if any(len(rs) != 4 for rs in grouped.values()):
            raise ValueError('Expected four rollouts per selected task')
        for t, rs in grouped.items():
            s, v, m = self._aged(t)
            self.counts[t] = [s + sum(r[0] for r in rs), v + sum(r[1] for r in rs), m + 4, self.updates]
        pending = self.pending
        self.pending = None
        return pending

    def commit_update(self, step):
        if self.pending or step != self.updates + 1 or self.consumed_rollouts + 8 > self.generated:
            raise ValueError('Optimizer commit must be consecutive and after observation')
        self.updates = step
        self.consumed_groups += 2
        self.consumed_rollouts += 8

    def state_dict(self):
        return {'version': VERSION, 'ids': self.ids, 'arm': self.arm, 'seed': self.seed,
                'budget': self.budget, 'counts': self.counts, 'generated': self.generated,
                'batches': self.batches, 'updates': self.updates, 'pending': self.pending,
                'consumed_groups': self.consumed_groups, 'consumed_rollouts': self.consumed_rollouts,
                'rng': self.rng.getstate()}

    @classmethod
    def from_state(cls, state):
        if state['version'] != VERSION:
            raise ValueError('Scheduler version changed')
        obj = cls(state['ids'], state['arm'], state['seed'], state['budget'])
        obj.counts = {int(k): list(v) for k, v in state['counts'].items()}
        for key in ('generated', 'batches', 'updates', 'pending', 'consumed_groups', 'consumed_rollouts'):
            setattr(obj, key, copy.deepcopy(state[key]))
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value
        obj.rng.setstate(tuples(state['rng']))
        if obj.generated != obj.batches * 8 or obj.generated > obj.budget:
            raise ValueError('Inconsistent generated budget')
        if obj.consumed_rollouts != obj.updates * 8 or obj.consumed_groups != obj.updates * 2 or obj.consumed_rollouts > obj.generated:
            raise ValueError('Inconsistent optimizer consumption')
        if set(obj.counts) != set(obj.ids):
            raise ValueError('Count roster changed')
        for s, v, m, age in obj.counts.values():
            if not all(math.isfinite(x) for x in (s, v, m)) or not 0 <= s <= v <= m or not 0 <= age <= obj.updates:
                raise ValueError('Invalid scheduler counts')
        return obj
