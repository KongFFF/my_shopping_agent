"""CPU boundaries: frozen cohorts, strict scoring, resume, repairs and missing data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext
from urllib.error import URLError
import checkpoint_pair_common as c
import run_checkpoint_pair as runner
import report_checkpoint_pair as reporter
import run_checkpoint_pair_pipeline as pipeline_runner
from prepare_checkpoint_pair import select_tasks
from checkpoint_pair_environment import fresh_reset

ASIN='123456789012'


def call(name,**args):
    return {'role':'assistant','content':None,'tool_calls':[{'id':'call-'+name,'type':'function',
        'function':{'name':name,'arguments':json.dumps(args)}}]}


class FakeEnv:
    instances=[]
    wrong_query=False
    def __init__(self,**kwargs):
        self.released=False;self.page='search_home';self.options={};self.instances.append(self)
    def public(self):
        z={'observation_version':'shopping-observation-v2','page_type':self.page,
           'actions':['search'],'search_available':self.page in ('search_home','search_results')}
        if self.page=='search_results':z.update(products=[{'asin':ASIN,'rank':1}],actions=['search',ASIN])
        if self.page=='product_detail':z.update(product={'asin':ASIN},actions=['red','blue','buy now'],
            available_options={'color':['red','blue']},selected_options=self.options.copy())
        if self.page=='terminal':z.update(actions=[],search_available=False)
        return z
    def reset(self,tid):
        task=next(t for t in c.rows(c.PACKAGE/'tasks.jsonl') if t['task_id']==tid)
        return {'instruction':'Instruction: '+('wrong' if self.wrong_query else task['query']),
                'environment_version':'shopsimulator-environment-v2.1','observation_state':self.public()}
    def step(self,action):
        action=action.casefold()
        if action.startswith('search['):self.page='search_results'
        elif action==f'click[{ASIN}]':self.page='product_detail'
        elif action in ('click[red]','click[blue]'):self.options['color']=action[6:-1]
        elif action=='click[buy now]':self.page='terminal'
        done=self.page=='terminal';good=self.options.get('color')=='red';kind='gold_purchase' if good else 'partial_alternative_purchase'
        detail={'reward_version':'shopsimulator-reward-v3','reward_type':kind,'reward_valid':True,
                'purchase_success':good,'termination_reason':kind}
        return {'observation_state':self.public(),'done':done,'over':done,'reward':1. if done and good else .1,
                'reward_valid':True,'reward_detail':detail if done else {},'termination_reason':kind}
    def release(self):self.released=True


class FakeClient:
    def __init__(self,color='red'):
        self.actions=[call('search_products',query='x'),call('open_product',asin=ASIN),call('select_option',value=color),call('buy_now')]
    def complete(self,*_):return self.actions.pop(0)
    def project_observation(self,name,raw,params):return raw,None


class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=c.ROOT/'outputs',prefix='pair_test_')
        self.run=Path(self.tmp.name);self.bundle=c.ROOT/'outputs/checkpoint_pair_v1/runtime'
        self.task=c.rows(c.PACKAGE/'tasks.jsonl')[0];FakeEnv.instances.clear();FakeEnv.wrong_query=False
        self.cfg={'bundle':str(self.bundle),'protocol_sha256':'test-protocol'}
        c.save(self.run/'run.json',self.cfg)
        c.load_agent(self.bundle)
    def tearDown(self):self.tmp.cleanup()
    def execute(self,client=None,repair=False,attempt=0):
        with patch.object(runner,'checked',return_value=self.cfg):
            return runner.evaluate_slot(self.run,'sft',self.task,attempt,repair,
                                       client_factory=client or FakeClient,env_factory=FakeEnv)

    def test_frozen_task_selection_counts_disjoint_and_repeatable(self):
        ts,_=select_tasks();self.assertEqual(ts,c.rows(c.PACKAGE/'tasks.jsonl'))
        self.assertEqual(len({t['task_id'] for t in ts}),262)
        self.assertEqual([t['task_id'] for t in ts[:3]],[21607,13310,9199])

    def test_seed_shared_across_models_but_varies_by_task_attempt_turn(self):
        self.assertEqual(c.seed_for(53,0,1),c.seed_for(53,0,1))
        self.assertEqual(len({c.seed_for(53,a,t) for a in range(8) for t in range(4)}),32)

    def test_gold_strict_score_and_resume_skip(self):
        v=self.execute();self.assertTrue(v['strict_success']);self.assertTrue(v['training_gold_proxy'])
        before=len(FakeEnv.instances);self.assertEqual(self.execute(),v)
        self.assertEqual(len(FakeEnv.instances),before)
        self.assertTrue(all(e.released for e in FakeEnv.instances))

    def test_wrong_policy_outcome_is_completed_and_not_repaired(self):
        v=self.execute(lambda:FakeClient('blue'));self.assertFalse(v['strict_success'])
        self.assertEqual(v['classification'],'completed')
        with self.assertRaisesRegex(ValueError,'never be rerolled'):self.execute(repair=True)

    def test_context_limit_is_policy_failure(self):
        from shopping_grpo.environment.context import ContextBudgetError
        class ContextClient:
            def complete(self,*_):raise ContextBudgetError('test limit')
        v=self.execute(ContextClient);self.assertEqual(v['classification'],'completed')
        self.assertFalse(v['strict_success']);self.assertTrue(FakeEnv.instances[-1].released)

    def test_guard_limit_is_policy_failure(self):
        class GuardClient:
            def complete(self,*_):return call('select_option',value='red')
        v=self.execute(GuardClient);self.assertEqual(v['status'],'invalid_action_limit')
        self.assertEqual(v['classification'],'completed');self.assertFalse(v['strict_success'])

    def test_network_failure_requires_explicit_same_slot_repair(self):
        class NetworkClient:
            def complete(self,*_):raise URLError('offline')
        with self.assertRaisesRegex(RuntimeError,'Infrastructure error'):self.execute(NetworkClient)
        with self.assertRaisesRegex(ValueError,'explicit repair'):self.execute()
        v=self.execute(repair=True);self.assertTrue(v['strict_success']);self.assertEqual(v['try_index'],1)
        self.assertTrue((runner.slot_dir(self.run,'sft',self.task['task_id'],0)/'try_000/trajectory.json').exists())

    def test_interrupted_slot_cannot_silently_rerun(self):
        path=runner.slot_dir(self.run,'sft',self.task['task_id'],0)/'try_000';path.mkdir(parents=True)
        c.save(path/'start.json',{'at':1})
        with self.assertRaisesRegex(ValueError,'explicit repair'):self.execute()

    def test_query_mismatch_stops_and_releases(self):
        FakeEnv.wrong_query=True
        with self.assertRaisesRegex(RuntimeError,'Infrastructure error'):self.execute()
        self.assertTrue(FakeEnv.instances[-1].released)

    def test_strict_score_does_not_accept_incomplete_gold_terminal(self):
        self.execute()
        p=runner.latest_try(runner.slot_dir(self.run,'sft',self.task['task_id'],0))
        trace=c.load(p/'trajectory.json');trace['terminal_result']['over']=False
        v=c.outcome(trace);self.assertFalse(v['strict_success']);self.assertEqual(v['classification'],'infrastructure_error')

    def test_fresh_reset_removes_stale_terminal_state(self):
        env=SimpleNamespace(session_prefix='slot-0',server=SimpleNamespace(user_sessions={'slot-0-53':{'done':True}}))
        api=SimpleNamespace(shop_agent=lambda e,*_:copy.deepcopy(e.server.user_sessions));fresh_reset(api)
        self.assertEqual(api.shop_agent(env,0,'reset',53),{})

    def test_partial_report_does_not_fill_missing_as_failure(self):
        self.execute()
        with patch.object(reporter,'checked',return_value=self.cfg):reporter.report(self.run,self.run/'report')
        summary=c.load(self.run/'report/summary.json');self.assertFalse(summary['complete'])
        self.assertEqual(summary['coverage']['sft:completed'],1)
        self.assertIsNone(summary['cohorts']['mixed']['main_statistics'])

    def test_report_detects_modified_trajectory(self):
        self.execute();p=runner.latest_try(runner.slot_dir(self.run,'sft',self.task['task_id'],0))/'trajectory.json'
        p.write_text('{}',encoding='utf-8')
        with patch.object(reporter,'checked',return_value=self.cfg):
            with self.assertRaisesRegex(ValueError,'modified'):reporter.report(self.run,self.run/'report')

    def test_paired_statistic_uses_task_mean_not_independent_rollout_test(self):
        values=[{'sft_p':0,'grpo_p':1,'sft_successes':0,'grpo_successes':8},
                {'sft_p':1,'grpo_p':0,'sft_successes':8,'grpo_successes':0}]
        z=reporter.paired_stats(values);self.assertEqual(z['delta_mean'],0)
        self.assertEqual(z['observed_improved'],1);self.assertEqual(z['observed_regressed'],1)

    def test_planned_full_reuses_smoke_slots(self):
        full={(t['task_id'],a) for t,a in runner.planned('full')}
        smoke={(t['task_id'],a) for t,a in runner.planned('smoke')}
        self.assertEqual(len(full),2096);self.assertEqual(len(smoke),3);self.assertTrue(smoke<=full)

    def test_full_cannot_start_before_both_smokes(self):
        with patch.object(runner,'checked',return_value=self.cfg),patch.object(runner,'services') as service:
            with self.assertRaisesRegex(ValueError,'Both smoke'):runner.evaluate(self.run,'sft','full')
            service.assert_not_called()
        self.assertFalse((self.run/'driver.lock').exists())

    def test_smoke_pipeline_passes_and_completed_phase_does_not_reload_models(self):
        original=runner.evaluate_slot
        def fake(run,arm,task,attempt,repair=False):
            return original(run,arm,task,attempt,repair,client_factory=FakeClient,env_factory=FakeEnv)
        with patch.object(runner,'checked',return_value=self.cfg),patch.object(runner,'services',return_value=nullcontext()) as service,patch.object(runner,'evaluate_slot',side_effect=fake):
            runner.evaluate(self.run,'sft','smoke')
            before=len(FakeEnv.instances)
            runner.evaluate(self.run,'sft','smoke')
            self.assertEqual(len(FakeEnv.instances),before);self.assertEqual(service.call_count,1)
        self.assertTrue(c.load(self.run/'smoke_sft.json')['passed'])

    def test_pipeline_order_and_existing_final_report(self):
        c.save(self.run/'report_final/summary.json',{'complete':True})
        calls=[]
        class Process:
            def __init__(self,cmd,**kwargs):calls.append(cmd)
            def wait(self,**kwargs):return 0
            def poll(self):return 0
        with patch.object(pipeline_runner,'checked',return_value=self.cfg),patch.object(pipeline_runner.subprocess,'Popen',Process):
            self.assertEqual(pipeline_runner.pipeline(self.run),0)
        self.assertEqual([(cmd[cmd.index('--phase')+1],cmd[cmd.index('--model')+1]) for cmd in calls],
                         [('smoke','sft'),('smoke','grpo'),('full','sft'),('full','grpo')])
        self.assertFalse((self.run/'pipeline.lock').exists())

    def test_pipeline_stops_when_a_smoke_fails(self):
        calls=[]
        class Process:
            def __init__(self,cmd,**kwargs):calls.append(cmd);self.code=0 if len(calls)==1 else 1
            def wait(self,**kwargs):return self.code
            def poll(self):return self.code
        with patch.object(pipeline_runner,'checked',return_value=self.cfg),patch.object(pipeline_runner.subprocess,'Popen',Process):
            self.assertEqual(pipeline_runner.pipeline(self.run),1)
        self.assertEqual(len(calls),2)
        self.assertEqual(c.load(self.run/'pipeline_exit.json')['exit_code'],1)


if __name__=='__main__':unittest.main()
