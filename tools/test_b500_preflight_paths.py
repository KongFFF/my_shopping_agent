from pathlib import Path
import unittest
from check_b500_history import resolve_source_path


class SourcePathTests(unittest.TestCase):
    def test_shared_lock_is_checked_in_asset_project(self):
        self.assertEqual(resolve_source_path(Path('/old/uv.lock'),Path('/old'),Path('/asset'),Path('/runtime')),Path('/asset/uv.lock'))

    def test_shared_venv_and_data_remain_shared(self):
        for relative in ['.venv/lib/python/site-packages/a.py','data/environment.json','environments/ShopSimulator/a.py']:
            self.assertEqual(resolve_source_path(Path('/old')/relative,Path('/old'),Path('/asset'),Path('/runtime')),Path('/asset')/relative)

    def test_isolated_code_and_project_config_remain_isolated(self):
        for relative in ['configs/agent_loop.yaml','pyproject.toml','scripts/train_grpo.py']:
            self.assertEqual(resolve_source_path(Path('/old')/relative,Path('/old'),Path('/asset'),Path('/runtime')),Path('/runtime')/relative)

    def test_outside_original_project_rejected(self):
        with self.assertRaises(ValueError):resolve_source_path(Path('/elsewhere/uv.lock'),Path('/old'),Path('/asset'),Path('/runtime'))


if __name__=='__main__':unittest.main()
