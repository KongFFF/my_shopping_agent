"""Launch an isolated original environment plus read-only identity/state digests."""
import argparse
import dataclasses
import importlib
import os
import sys
from pathlib import Path
from decision_probe_common import PACKAGE, check_package, load, runtime_receipt, digest, dump, code_receipt, file_hash


def plain_state(value, session):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value.replace(session, '<session>')
    if isinstance(value, dict):
        return {str(k): plain_state(v, session) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [plain_state(v, session) for v in value]
    if isinstance(value, set):
        return sorted((plain_state(v, session) for v in value), key=dump)
    if dataclasses.is_dataclass(value):
        return plain_state(dataclasses.asdict(value), session)
    if hasattr(value, '__dict__'):
        return plain_state(vars(value), session)
    raise TypeError('Unserializable replay state: ' + type(value).__name__)


def state_receipt(env):
    # Include complete session: progress, candidate history, purchase guard, goal,
    # and browser state. Only the digest leaves this diagnostic endpoint.
    snapshot = {'session': env.server.user_sessions[env.session],
                'url': env.browser.current_url, 'html': env.browser.page_source,
                'history': env.history, 'prev_actions': env.prev_actions,
                'prev_obs': env.prev_obs,
                'episode_fields': {k: getattr(env, k, None) for k in
                                   ('instruction_text','instruction_simple','goal_options','user_persona','reason_key')},
                'max_steps': getattr(env.server, 'max_steps', None)}
    normalized = plain_state(snapshot, env.session)
    return {'state_sha256': digest(dump(normalized)), 'scope': 'session+browser+environment-history+episode-fields-v2'}


def install_fresh_reset(api):
    original = api.shop_agent
    def fresh_reset(env, env_idx, action, idx=None, response=None):
        if action == 'reset':
            # The upstream reset updates fields but leaves old done/reward/
            # verbose_info keys. A new independent episode must not inherit them.
            if env.session_prefix is None:
                raise ValueError('Probe requires slot-scoped session names')
            env.server.user_sessions.pop(f'{env.session_prefix}-{idx}', None)
            # Native reset only assigns these on tasks with a persona, otherwise
            # leaking unrelated previous-task metadata into the reset response.
            # _initial_messages uses instruction only; neither is Actor input.
            for name in ('user_persona', 'reason_key'):
                if hasattr(env, name): delattr(env, name)
        return original(env, env_idx, action, idx, response)
    api.shop_agent = fresh_reset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--port', type=int, default=5714)
    a = p.parse_args()
    check_package()
    bundle = a.bundle.resolve()
    receipt = runtime_receipt(bundle)
    if receipt != load(PACKAGE/'runtime_receipt.json'):
        raise ValueError('Environment bundle differs from frozen runtime')
    envroot = bundle/'environment'
    os.environ.update(SHOP_ENV_CONFIG=str(envroot/'configs/environment.json'),
                      SHOP_SEARCH_INDEX=str(envroot/'search_engine/products.sqlite3'),
                      SHOP_MAX_STEPS='35', SHOPSIM_ENV_SLOTS='8',
                      SHOP_ENVIRONMENT_VERSION='shopsimulator-environment-v2.1')
    os.chdir(envroot/'shop_env')
    sys.path[:0] = [str(envroot/'shop_env'), str(envroot)]
    api = importlib.import_module('pack_api')
    install_fresh_reset(api)
    from flask import jsonify, request

    @api.app.route('/api/local_probe_identity', methods=['GET'])
    def identity():
        return jsonify({'version': 'decision-probe-v1', 'runtime_sha256': digest(dump(receipt)),
                        'code_sha256': code_receipt(), 'slots': 8, 'max_steps': 35,
                        'index_physical_sha256': file_hash(envroot/'search_engine/products.sqlite3')})

    @api.app.route('/api/local_probe_state', methods=['POST'])
    def state():
        idx = request.json.get('env_idx')
        if not isinstance(idx, int) or not 0 <= idx < len(api.envs):
            return jsonify({'error': 'invalid env_idx'}), 400
        return jsonify(state_receipt(api.envs[idx]))

    api.initialize_environments()
    api.app.run(host='127.0.0.1', port=a.port, threaded=False)


if __name__ == '__main__': main()
