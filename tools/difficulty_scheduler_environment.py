"""Serve only the frozen baseline environment; independent reset, no reward edits."""
import argparse
import importlib
import os
import sys
from prepare_difficulty_scheduler import check_runtime, file_hash, receipt
from difficulty_scheduler import VERSION, canonical, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    args = parser.parse_args()
    from pathlib import Path
    bundle = Path(args.bundle).resolve()
    check_runtime(bundle)
    environment = bundle / 'project/environments/ShopSimulator/shop_env'
    os.environ.update(SHOP_ENV_CONFIG=str(environment / 'configs/environment.json'),
                      SHOP_SEARCH_INDEX=str(environment / 'search_engine/products.sqlite3'),
                      SHOP_MAX_STEPS='35', SHOPSIM_ENV_SLOTS='8',
                      SHOP_ENVIRONMENT_VERSION='shopsimulator-environment-v2.1')
    os.chdir(environment / 'shop_env')
    sys.path[:0] = [str(environment / 'shop_env'), str(environment)]
    api = importlib.import_module('pack_api')
    original = api.shop_agent
    def fresh(env, env_idx, action, idx=None, response=None):
        if action == 'reset':
            if env.session_prefix is None:
                raise ValueError('Expected independent slot-scoped sessions')
            env.server.user_sessions.pop(f'{env.session_prefix}-{idx}', None)
        return original(env, env_idx, action, idx, response)
    api.shop_agent = fresh
    from flask import jsonify
    identity = {'version': VERSION, 'runtime': sha(canonical(receipt(bundle))),
                'index': file_hash(environment / 'search_engine/products.sqlite3'), 'slots': 8}
    @api.app.route('/api/difficulty_scheduler_identity', methods=['GET'])
    def identify():
        return jsonify(identity)
    api.initialize_environments()
    api.app.run(host='127.0.0.1', port=5723, threaded=False)


if __name__ == '__main__':
    main()
