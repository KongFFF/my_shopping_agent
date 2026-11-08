# GRPO 训练后：Dev-50 汇总与 checkpoint 选择

## 为什么有多份结果

这是同一次训练在不同时间的模型快照。正式配方每 50 次优化器更新保存一次 checkpoint，并运行一次 Dev-50。保存文件记录模型及恢复训练所需的优化器等状态；验证文件记录当时模型完成这 50 道题的表现。二者用途不同。

| 训练步数 | 模型含义 | 验证用途 |
| --- | --- | --- |
| 0 | GRPO 开始前的 SFT step 300 起点 | 训练内的起点参照，不需要另有 `global_step_0` 文件 |
| 50、100、…、500 | 已完成相应次数 GRPO 更新的模型 | 共 10 个正式选择候选 |
| 1、10、25 | 为防意外中断增加的早期 checkpoint | 没有对应完整周期验证，不进入本轮选优 |

11 轮验证反复使用**同一套 50 个任务**。应按 checkpoint 分别统计，得到一条学习曲线；不能把它们相加当作 550 道独立题，也不能将十份权重平均称为“汇总”。训练更久不保证成功率更高，最终 step 500 的意义是训练终点，不是默认最优。

## 在看成绩前固定规则

1. 正式候选为 50/100/…/500，step 0 只作起点参照。
2. 优先选择训练内 Dev-50 中 **Reward v3 满分（score=1）任务数最多**的 checkpoint。
3. 满分数相同时，选择**平均终局奖励更高**的版本；仍相同则选择**较早 step**。不按 Final-200 Clean 成绩改选。

此规则沿用任务成功优先、奖励辅助的思路；这里的具体并列规则是本次查看 GRPO 汇总成绩前确定的，不声称训练前已经写死。

训练框架导出的 JSONL 可能只有输入、输出和标量奖励，没有完整的环境终局证据。当前固定配方中，`gold_purchase=1.0`，其他终局奖励均低于 1；无正常可核验终局、基础设施无效的轨迹在 adapter 中记 0，并关闭了额外长度 shaping。因此，在校验配方后，可以从 score=1 统计训练内 gold 成功数。**这是一项基于固定奖励定义推得的训练内指标，不冒充独立评测报告里的完整严格成功率验收。** 不解析模型输出中的“购买成功”等文字来判题。

选出的 checkpoint 是后续导出与独立评测的候选。step 0 同样使用训练内 agent 路径，其分数不必等于之前独立评测脚本的 SFT Dev-50 48%；两条评测路径最终仍需固定协议核对。

## 一段完成汇总与候选选择（无卡模式可执行）

以下代码在 AutoDL Bash 终端执行。不启动环境、Ray、Torch 或推理服务，不读取 Final 数据或结果，不加载权重。它只校验原配方与验证文件，生成小体积 CSV/JSON。会保留全部原始结果与 checkpoint。

它读取旧目录的 0/50/…/350 和续跑目录的 400/450/500。任何轮次少题、重复文件、非有限奖励或配方不符都会停止，不会丢弃失败题后重新计算成功率。比较各轮初始输入的多重集合来检查验证题面一致；这不代替逐任务 ID 审计。

```bash
set +e
set +u
if /root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon/.venv/bin/python - <<'PY'
import ast, collections, csv, hashlib, json, math
from pathlib import Path
from omegaconf import OmegaConf

root = Path('/root/autodl-tmp/work/my_shopping_agent/upstream/shopping-grpo-longhorizon')
run = root / 'outputs/runs/grpo_baseline_20261022_202816'
old_prep = run / 'preparation/formal_memfix_v1'
new_prep = run / 'preparation/formal_resume350_v1'
dest = run / 'preparation/dev50_selection_v1'
specs = [json.loads((p/'launch_spec.json').read_text()) for p in (old_prep,new_prep)]
outputs = [Path(s['output']) for s in specs]
assert outputs == [run/'formal_memfix_v1',run/'formal_resume350_v1']

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()

for spec in specs:
    for name, expected in spec['source_sha256'].items():
        assert digest(name) == expected, f'封存源码已改变：{name}'
    assert spec['env']['SHOPPING_LENGTH_SHAPING_ENABLE'].lower() == 'false'

baseline = json.loads((run/'preparation/baseline_plan.json').read_text())
for rel in ('data/grpo/validation.jsonl','data/grpo/validation.parquet'):
    assert digest(root/rel) == baseline['data_sha256'][rel], rel
ids = [int(json.loads(line)['task_id']) for line in
       (root/'data/grpo/validation.jsonl').read_text().splitlines() if line.strip()]
assert len(ids)==len(set(ids))==50
assert specs[0]['env']['GRPO_VAL_FILE'] == specs[1]['env']['GRPO_VAL_FILE']
assert Path(specs[0]['env']['GRPO_VAL_FILE']) == root/'data/grpo/validation.parquet'

protocol_keys = [
    'actor_rollout_ref.rollout.val_kwargs.n',
    'actor_rollout_ref.rollout.val_kwargs.do_sample',
    'actor_rollout_ref.rollout.val_kwargs.temperature',
    'actor_rollout_ref.rollout.val_kwargs.top_p',
    'actor_rollout_ref.rollout.max_model_len',
    'actor_rollout_ref.rollout.multi_turn.max_tool_response_length',
]
configs = [OmegaConf.load(p/'composed_config.yaml') for p in (old_prep,new_prep)]
protocols = [{k:OmegaConf.select(cfg,k) for k in protocol_keys} for cfg in configs]
assert protocols[0]==protocols[1], '两段训练的验证参数不一致。'
assert protocols[0][protocol_keys[0]]==1
assert protocols[0][protocol_keys[1]] is False
assert protocols[0][protocol_keys[2]]==0
agent = OmegaConf.load(root/'configs/agent_loop.yaml')[0]
assert agent.reward_mode=='constraint_aware'

manifest_path = root/'data/environment.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['reward']['version']=='shopsimulator-reward-v3'
reward_path = root/'environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py'
assert digest(reward_path)==manifest['runtime_files_sha256']['reward.py']
tree = ast.parse(reward_path.read_text())
reward_node = next(n for n in tree.body if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='DEFAULT_REWARDS' for t in n.targets))
reward_values = ast.literal_eval(reward_node.value)
assert {k:manifest['reward'][k] for k in reward_values}==reward_values
assert reward_values['gold_purchase']==1.0
assert max(v for k,v in reward_values.items() if k!='gold_purchase') < 1.0

policy = {
    'schema':'shopping-grpo-dev50-selection-v1',
    'candidate_steps':list(range(50,501,50)), 'baseline_step':0,
    'ranking':['gold_reward_1_count descending','mean_reward descending','step ascending'],
    'metric_source':'Reward v3 score=1 inferred gold success; independent strict evaluation pending',
    'gold_score_abs_tolerance':1e-6, 'expected_tasks_per_step':50,
    'validation_jsonl_sha256':digest(root/'data/grpo/validation.jsonl'),
    'validation_parquet_sha256':digest(root/'data/grpo/validation.parquet'),
    'validation_protocol':protocols[0],
}
# 先封存规则，再读取和排序成绩。重复运行只接受同一规则。
dest.mkdir(parents=True,exist_ok=True)
policy_path = dest/'selection_policy.json'
if policy_path.exists():
    assert json.loads(policy_path.read_text())==policy, '已有另一份选优规则，不能覆盖。'
else:
    with policy_path.open('x') as f:
        json.dump(policy,f,ensure_ascii=False,indent=2)

groups = collections.defaultdict(list)
input_fingerprints = collections.defaultdict(collections.Counter)
evidence = []
schema_keys = set()
for folder, allowed in ((outputs[0],set(range(0,351,50))),(outputs[1],{400,450,500})):
    files = sorted((folder/'validation').glob('*.jsonl'))
    assert files, f'没有验证文件：{folder}'
    for path in files:
        evidence.append({'path':str(path),'sha256':digest(path)})
        with path.open() as f:
            for line_no,line in enumerate(f,1):
                if not line.strip(): continue
                row = json.loads(line)
                step = int(row['step'])
                assert step in allowed, (path,line_no,step)
                assert not isinstance(row['score'],bool)
                score = float(row['score'])
                assert math.isfinite(score) and -0.850001<=score<=1.000001, (path,line_no,score)
                assert isinstance(row['input'],str) and row['input'], (path,line_no,'missing input')
                groups[step].append(score)
                fingerprint = hashlib.sha256(row['input'].encode()).hexdigest()
                input_fingerprints[step][fingerprint]+=1
                schema_keys.update(row)
assert set(groups)==set(range(0,501,50)), sorted(groups)
for step in sorted(groups):
    assert len(groups[step])==50, f'step {step} 记录数为 {len(groups[step])}，预期 50。'
    assert input_fingerprints[step]==input_fingerprints[0], f'step {step} 初始输入与 step 0 不一致。'

rows = []
for step in sorted(groups):
    scores = groups[step]
    gold = sum(math.isclose(s,1.0,rel_tol=0,abs_tol=policy['gold_score_abs_tolerance']) for s in scores)
    folder = outputs[0] if step<=350 else outputs[1]
    checkpoint = folder/f'global_step_{step}' if step else None
    if checkpoint is not None:
        for name in ('actor/model_world_size_1_rank_0.pt','actor/optim_world_size_1_rank_0.pt',
                     'actor/extra_state_world_size_1_rank_0.pt','data.pt'):
            path = checkpoint/name
            assert path.is_file() and path.stat().st_size>0, path
    rows.append({'step':step,'tasks':50,'gold_reward_1_count':gold,
        'gold_reward_1_rate':gold/50,'mean_reward':math.fsum(scores)/50,
        'min_reward':min(scores),'max_reward':max(scores),
        'checkpoint':str(checkpoint) if checkpoint else '',
        'role':'SFT starting policy' if step==0 else 'GRPO candidate'})

ranking = sorted([r for r in rows if r['step']!=0],
    key=lambda r:(-r['gold_reward_1_count'],-r['mean_reward'],r['step']))
chosen = ranking[0]
report = {'policy':policy,'rows':rows,'ranked_candidate_steps':[r['step'] for r in ranking],
    'selected':chosen,'source_validation_files':evidence,'validation_row_keys':sorted(schema_keys),
    'input_multisets_match':True,'independent_strict_evaluation_pending':True}
with (dest/'dev50_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(dest/'dev50_selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

print('验证文件字段：',sorted(schema_keys))
print('\nstep | 训练内 gold 满分数 | 满分比例 | 平均奖励 | 相对 step 0 满分数变化')
for row in rows:
    delta = row['gold_reward_1_count']-rows[0]['gold_reward_1_count']
    print(f"{row['step']:>4} | {row['gold_reward_1_count']:>2}/50 | "
          f"{row['gold_reward_1_rate']:>6.1%} | {row['mean_reward']:+.6f} | {delta:+d}")
print('\n按已固定规则排名：',[r['step'] for r in ranking])
print('选出的 GRPO 候选：step',chosen['step'])
print('checkpoint：',chosen['checkpoint'])
print('actor 目录：',str(Path(chosen['checkpoint'])/'actor'))
print('汇总文件：',dest/'dev50_summary.csv')
print('选择依据与证据：',dest/'dev50_selection.json')
print('下一步导出该候选，使用固定独立评测协议核验；当前未导出、未启动任何推理。')
PY
then
  echo '训练内 Dev-50 汇总与候选选择完成。'
else
  echo '汇总未通过；保留上方具体错误，不跳过校验或读取 Final 来改选。'
fi
```

## 如何解读输出

- 比较 step 0 与各训练步数，观察训练内成功数及奖励是否提升、是否中途见顶回落。50 题中多成功 1 题就是 2 个百分点，轻微差距不宜解释为稳定提升。
- 采用脚本打印的规则赢家进入后续导出流程。即使 step 0 更好，也保留这一事实：此时只能说选出了 GRPO 候选，不能宣称 GRPO 优于 SFT。
- 最优恰好是 step 500 时，无需为了“必须有对照”重复评测同一个 checkpoint；SFT 始终是主要对照。最优在中间时，step 500 仍可作为训练后期的诊断参照，但不据此反复查看 Final 挑模型。
- 所有轮次分母保持 50；不能把上下文超限、奖励无法核验或零分任务删掉来提高指标。只有 score 的日志无法区分所有零分原因，也无法据此生成完整的 done/guard/error 分布；这些留给独立轨迹评测。

训练框架 JSONL 的默认输出字段可参考 [veRL trainer 的验证导出实现](https://github.com/verl-project/verl/blob/v0.8.0/verl/trainer/ppo/ray_trainer.py)。本项目满分含义以固定的本机 Reward v3、adapter 和封存环境为准，命令会检查对应源码与配方。
