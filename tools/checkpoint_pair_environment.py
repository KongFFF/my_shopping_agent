"""Serve the pinned original environment with fresh reset and read-only identity."""
import argparse
import importlib
import os
import sys
from pathlib import Path
from checkpoint_pair_common import *


def fresh_reset(api):
    original=api.shop_agent
    def independent(env,env_idx,action,idx=None,response=None):
        if action=='reset':
            if env.session_prefix is None:raise ValueError('Expected slot-scoped sessions')
            env.server.user_sessions.pop(f'{env.session_prefix}-{idx}',None)
        return original(env,env_idx,action,idx,response)
    api.shop_agent=independent


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
    check_package();bundle=a.bundle.resolve();receipt=runtime_receipt(bundle)
    if receipt!=load(PACKAGE/'runtime_receipt.json'):raise ValueError('Baseline runtime differs')
    env=bundle/'environment'
    os.environ.update(SHOP_ENV_CONFIG=str(env/'configs/environment.json'),SHOP_SEARCH_INDEX=str(env/'search_engine/products.sqlite3'),
                      SHOP_MAX_STEPS='35',SHOPSIM_ENV_SLOTS='8',SHOP_ENVIRONMENT_VERSION='shopsimulator-environment-v2.1')
    os.chdir(env/'shop_env');sys.path[:0]=[str(env/'shop_env'),str(env)]
    api=importlib.import_module('pack_api');fresh_reset(api)
    from flask import jsonify
    @api.app.route('/api/checkpoint_pair_identity',methods=['GET'])
    def identity():
        return jsonify({'version':VERSION,'runtime_sha256':digest(dump(receipt)),'code_sha256':code_receipt(),
                        'index_sha256':file_hash(env/'search_engine/products.sqlite3'),'max_steps':35,'slots':8})
    api.initialize_environments();api.app.run(host='127.0.0.1',port=5721,threaded=False)


if __name__=='__main__':main()
