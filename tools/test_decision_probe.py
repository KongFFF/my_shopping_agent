"""Replay/guard/cost boundaries for the new protocol; no model calls."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch
import run_decision_probe as runner
from decision_probe_common import ROOT, Budget, digest, dump, save, append, load
from decision_probe_rules import trigger, atomic_checks, requirements
from report_decision_probe import report, candidate


def assistant(tool, **args):
    return {'role':'assistant','content':None,'tool_calls':[{'id':'c','function':{'name':tool,'arguments':json.dumps(args)}}]}


class Env:
    instances = []
    def __init__(self, **kwargs):
        self.env_idx=len(self.instances); self.instances.append(self); self.options={}; self.count=0; self.page='search_home'; self.released=False
    def result(self):
        state={'page_type':self.page,'asin':'123456789012','price':'100',
               'available_options':{'颜色':['黑色','白色']} if self.page=='product_detail' else {},'selected_options':dict(self.options)}
        done=self.page=='terminal'
        return {'observation_state':state,'env_idx':self.env_idx,'idx':1,'done':done,'reward':1. if self.options.get('颜色')=='黑色' else -.85,
                'reward_valid':True,'reward_detail':{'reward_type':'gold_purchase' if self.options.get('颜色')=='黑色' else 'wrong_purchase'}}
    def reset(self, tid):
        return {**self.result(),'instruction':'Instruction: 需要黑色，100元以内','environment_version':'shopsimulator-environment-v2.1'}
    def release(self): self.released=True


def render(state):
    return '\n'.join(k+': '+(json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else str(v)) for k,v in state.items())


def parse(call):
    return call['function']['name'],json.loads(call['function']['arguments'])


def execute(env, call, index):
    name,args=parse(call);env.count+=1
    if name=='search_products':env.page='search_results'
    elif name=='open_product':env.page='product_detail'
    elif name=='select_option':env.options['颜色']=args['value']
    elif name=='buy_now':env.page='terminal'
    result=env.result()
    return {'tool_name':name,'parameters':args,'tool_call':call,'step_index':index,'result':result,'done':result['done'],'observation':render(result['observation_state'])}


class Client:
    def __init__(self,key):
        self.calls=iter([assistant('select_option',value='白色' if key.endswith('/1') else '黑色'),assistant('buy_now')] if not key.endswith('prefix') else
                        [assistant('search_products',query='衣服'),assistant('open_product',asin='123456789012'),assistant('invalid'),assistant('select_option',value='黑色')])
    def complete(self,*_):return next(self.calls)
    def project_observation(self,tool,obs,args):return obs,{'truncated':False}


class ProbeTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'outputs').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='decision_test_');self.run=Path(self.tmp.name);Env.instances=[]
        self.task={'task_id':1,'query':'需要黑色，100元以内','stratum':'uniform32'}
    def tearDown(self):self.tmp.cleanup()
    def engine(self):
        engine=runner.Engine.__new__(runner.Engine);engine.run=self.run;engine.budget=Budget(time.time());engine.env_factory=Env
        engine.r=SimpleNamespace(_initial_messages=lambda t,i:[{'role':'user','content':i['instruction']}],render_structured_observation=render,
            _enforce_serial_tool_call=lambda a:(a,[]),SHOP_TOOL_SCHEMAS=[],_tool_call_name_args=parse,
            action_reject_reason=lambda n,a,o:'invalid' if n=='invalid' else None,MAX_BLOCKED_TOOL_CALLS=3,
            action_guard_tool_message=lambda c,r,o:{'role':'tool','content':r},_execute_tool_call=execute,
            _tool_message=lambda c,s:{'role':'tool','content':s['observation']},ToolExecutionError=type('ToolExecutionError',(Exception,),{}))
        engine.client=lambda key:Client(key)
        engine.state=lambda env:{'state_sha256':digest(dump([env.count,env.page,env.options]))}
        return engine
    def test_trigger_before_related_choice_not_first_detail(self):
        e=self.engine();p=e.execute(self.task,'1/B/prefix',stop_at_branch=True)
        self.assertTrue(p['branch']);self.assertEqual(len(p['steps']),2);self.assertEqual(p['consecutive_blocked_calls'],1)
        self.assertEqual(p['trigger']['kind'],'related_spec_before_selection');self.assertFalse(p['trigger']['executed'])
        self.assertEqual(Env.instances[0].options,{})
    def test_restore_independent_state_and_full_actor_history(self):
        e=self.engine();p=e.execute(self.task,'1/B/prefix',stop_at_branch=True)
        ts=[e.execute(self.task,f'1/B/{j}',prefix=p) for j in range(2)]
        self.assertTrue(all(t['done'] for t in ts));self.assertEqual(len({x.env_idx for x in Env.instances}),3)
        self.assertEqual(ts[0]['prefix_message_sha256'],p['branch_message_sha256'])
        self.assertNotEqual(ts[0]['terminal_result']['reward'],ts[1]['terminal_result']['reward'])
        self.assertTrue(all(x.released for x in Env.instances))
    def test_guard_counter_restored_not_reset(self):
        e=self.engine();p=e.execute(self.task,'1/B/prefix',stop_at_branch=True)
        c=Client('1/B/0');c.calls=iter([assistant('invalid'),assistant('invalid')]);e.client=lambda key:c
        t=e.execute(self.task,'1/B/0',prefix=p)
        self.assertEqual(t['status'],'invalid_action_limit');self.assertEqual(t['blocked_tool_calls'][-1]['consecutive_count'],3)
        self.assertEqual(len(t['steps']),2)
    def test_mismatch_fails_and_releases(self):
        e=self.engine();p=e.execute(self.task,'1/B/prefix',stop_at_branch=True);p['branch_state']={}
        with self.assertRaisesRegex(RuntimeError,'state mismatch'):e.execute(self.task,'1/B/0',prefix=p)
        self.assertTrue(Env.instances[-1].released)
    def test_actor_history_mismatch_fails(self):
        e=self.engine();p=e.execute(self.task,'1/B/prefix',stop_at_branch=True);p['messages'].append({'role':'user','content':'tamper'})
        with self.assertRaisesRegex(RuntimeError,'history mismatch'):e.execute(self.task,'1/B/0',prefix=p)
    def test_trigger_ignores_hidden_state(self):
        visible='page_type: product_detail\navailable_options: {"尺寸": ["S", "M"]}\nselected_options: {}'
        self.assertIsNone(trigger('衣服',visible,'open_product',{}))
        self.assertIsNone(trigger('衣服',visible,'select_option',{'value':'S'}))
        self.assertEqual(trigger('衣服',visible,'buy_now',{})['kind'],'prebuy_empty')
    def test_budget_unknown_and_strict_boundary(self):
        self.assertEqual(atomic_checks('低于100元','price: 100')[0]['passed'],False)
        self.assertIsNone(atomic_checks('100元以内','price: 80-120')[0]['passed'])
        self.assertFalse(requirements('100元左右'))
    def test_budget_clock_cannot_be_reset(self):
        save(self.run/'budget.json', {'started_at':1})
        with patch.object(runner,'checked_run',return_value={}):
            with self.assertRaisesRegex(ValueError,'cannot be reset'):runner.clock(self.run)
    def test_expired_budget_stops_before_environment_lease(self):
        e=self.engine();e.budget=Budget(time.time()-7201)
        with self.assertRaises(runner.BudgetExceeded):e.execute(self.task,'1/A/0')
        self.assertEqual(Env.instances,[])
    def test_mid_trajectory_budget_preserves_trace_and_stops(self):
        e=self.engine()
        def complete(*_):raise runner.BudgetExceeded('token cap reached')
        e.client=lambda key:SimpleNamespace(complete=complete)
        with self.assertRaises(runner.BudgetExceeded):e.execute(self.task,'1/A/0')
        ts=__import__('decision_probe_common').rows(self.run/'trajectories.jsonl')
        self.assertEqual(ts[0]['status'],'budget_exhausted');self.assertEqual(Env.instances[0].count,0)
        self.assertTrue(Env.instances[0].released)
    def test_report_keeps_48_denominator_and_prefix_cost(self):
        e=self.engine();e.execute(self.task,'1/B/prefix',stop_at_branch=True)
        append(self.run/'usage.jsonl',{'path':'1/B/prefix','usage':{'prompt_tokens':123,'completion_tokens':17}})
        planned=[self.task]+[{'task_id':i,'query':'衣服','stratum':'uniform32'} for i in range(2,49)]
        with patch('report_decision_probe.rows',side_effect=lambda p:planned if p.name=='tasks.jsonl' else __import__('decision_probe_common').rows(p)):
            report(self.run,self.run/'report')
        s=load(self.run/'report/summary.json');self.assertEqual(s['planned_tasks'],48);self.assertEqual(s['costs']['B']['completion_tokens'],17)
        self.assertEqual(s['trigger_counts']['not_attempted'],47);self.assertEqual(s['reviewed_pairs'],0)
    def test_public_review_excludes_private_terminal(self):
        e=self.engine();p=e.execute(self.task,'1/B/prefix',stop_at_branch=True);t=e.execute(self.task,'1/B/0',prefix=p)
        t['terminal_result']['goal']={'asin':'SECRET'}
        c=candidate(t,self.task);self.assertNotIn('SECRET',dump(c));self.assertNotIn('reward_detail',dump(c))


if __name__=='__main__':unittest.main()
