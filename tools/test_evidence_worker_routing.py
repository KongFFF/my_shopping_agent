"""Real Hydra serialization and isolated-process reward routing regressions."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_semantic_training_integration as production_test
from test_semantic_batch_release import pilot_config
from semantic_training_bundle import ROOT,UPSTREAM
from omegaconf import OmegaConf
from hydra import compose,initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

OUT=ROOT/'analysis/semantic_batch_acceptance_20261108'
MANIFEST=OUT/'training_bundle/manifest.json'


def launcher():
    spec=importlib.util.spec_from_file_location('worker_route_launcher',UPSTREAM/'scripts/train_grpo.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def bindings():
    return {'SHOPPING_EVIDENCE_MANIFEST':str(MANIFEST),'SHOPPING_REWARD_TOOLS_DIR':str(ROOT/'tools'),
            'PYTHONPATH':os.pathsep.join((str(UPSTREAM/'src'),str(ROOT/'tools')))}


def config():
    result=pilot_config();result['ray_kwargs']={'ray_init':{'runtime_env':{'env_vars':bindings()}}};return result


class WorkerRoutingTests(unittest.TestCase):
    def test_original_lazy_environment_payload_reproduces_bad_ray_values(self):
        from shopping_grpo.training.grpo.adapter.evidence import validate_literal_ray_bindings
        cfg=OmegaConf.create({key:'${oc.env:'+key+',\'\'}' for key in bindings()})
        with patch.dict(os.environ,bindings()):
            payload=OmegaConf.to_container(cfg)  # same unresolved serialization boundary as veRL0.8
            self.assertIn('${',payload['SHOPPING_REWARD_TOOLS_DIR'])
            with self.assertRaisesRegex(ValueError,'literal launch binding'):
                validate_literal_ray_bindings(payload,bindings())

    def test_hydra_parses_launch_bindings_into_real_literal_worker_payload(self):
        from shopping_grpo.training.grpo.adapter.evidence import validate_literal_ray_bindings
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as temp:
            path=Path(temp).resolve();assert path.is_relative_to((ROOT/'outputs').resolve())
            (path/'config.yaml').write_text('ray_kwargs:\n  ray_init:\n    runtime_env:\n      worker_process_setup_hook: keep_this_hook\n      env_vars: {}\n',encoding='utf-8')
            GlobalHydra.instance().clear()
            with initialize_config_dir(version_base=None,config_dir=str(path)):
                cfg=compose(config_name='config',overrides=launcher().ray_runtime_overrides(bindings()))
            payload=OmegaConf.to_container(cfg,resolve=False)['ray_kwargs']['ray_init']['runtime_env']
            validate_literal_ray_bindings(payload['env_vars'],bindings())
            self.assertEqual(payload['worker_process_setup_hook'],'keep_this_hook')

    def test_launcher_whitelist_avoids_forwarding_credentials_or_lazy_values(self):
        module=launcher();env={**bindings(),'UNRELATED_API_KEY':'private-not-forwarded'}
        result=module.ray_runtime_overrides(env)
        self.assertFalse(any('UNRELATED_API_KEY' in value for value in result))
        with self.assertRaisesRegex(ValueError,'literal string'):
            module.ray_runtime_overrides({'SHOPPING_REWARD_TOOLS_DIR':'${oc.env:ANYTHING}'})

    def test_worker_loads_from_explicit_config_with_no_backend_environment(self):
        from shopping_grpo.training.grpo.adapter.evidence import load_backend
        with patch.dict(os.environ,{},clear=True):
            backend=load_backend(MANIFEST,trainer_config=config())
        self.assertEqual(len(backend.released_ids),981)

    def test_constructor_cannot_silently_fall_back_when_environment_and_yaml_flag_missing(self):
        from shopping_grpo.training.grpo.adapter.agent_loop import ShoppingToolAgentLoop
        def base_init(instance,*args,trainer_config=None,**kwargs):instance.config=trainer_config.config
        with patch.object(production_test.ToolAgentLoop,'__init__',base_init),patch.dict(os.environ,{},clear=True):
            loop=ShoppingToolAgentLoop(trainer_config=types.SimpleNamespace(config=config()),reward_mode='constraint_aware')
        self.assertIsNotNone(loop.evidence_backend)
        self.assertEqual(len(loop.evidence_backend.released_ids),981)

    def test_fresh_process_reads_serialized_config_without_driver_sys_path_or_environment(self):
        code='''import json,sys
sys.path.insert(0,sys.argv[1])
from shopping_grpo.training.grpo.adapter.evidence import load_backend
b=load_backend(sys.argv[2],trainer_config=json.loads(sys.argv[3]))
print(json.dumps({'released':len(b.released_ids),'held':len(b.held_ids)}))
'''
        env={k:v for k,v in os.environ.items() if k not in (*bindings(),'PYTHONPATH')}
        env['PYTHONIOENCODING']='utf-8'
        r=subprocess.run([sys.executable,'-c',code,str(UPSTREAM/'src'),str(MANIFEST),json.dumps(config())],
            env=env,cwd=ROOT/'outputs',capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads(r.stdout),{'released':981,'held':19})

    def test_unresolved_or_conflicting_worker_config_is_rejected(self):
        from shopping_grpo.training.grpo.adapter.evidence import load_backend
        cfg=config();cfg['ray_kwargs']['ray_init']['runtime_env']['env_vars']['SHOPPING_REWARD_TOOLS_DIR']='${oc.env:MISSING}'
        with self.assertRaisesRegex(ValueError,'Unresolved Ray runtime binding'):load_backend(MANIFEST,trainer_config=cfg)
        cfg=config();cfg['ray_kwargs']['ray_init']['runtime_env']['env_vars']['SHOPPING_EVIDENCE_MANIFEST']='another-manifest.json'
        with self.assertRaisesRegex(ValueError,'differs from explicit trainer config'):load_backend(MANIFEST,trainer_config=cfg)


if __name__=='__main__':unittest.main()
