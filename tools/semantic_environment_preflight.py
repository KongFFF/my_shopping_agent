"""Cross-check frozen Environment v2.1 metadata against real source files.

Uses the reward bundle's declared LF text hash policy. The Linux/GPU runtime
retains its separate exact-byte preflight before model loading.
"""
import json

RUNTIME_FILES={
 'observation.py':'environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py',
 'pack_api.py':'environments/ShopSimulator/shop_env/shop_env/pack_api.py',
 'reward.py':'environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py',
 'slot_lease_pool.py':'environments/ShopSimulator/shop_env/shop_env/slot_lease_pool.py',
 'web_agent_text_env.py':'environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py'}


def check_environment(manifest_path,upstream,sha256):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('environment_version')!='shopsimulator-environment-v2.1' or manifest.get('lease_contract')!='explicit-client-release-v1':
        raise ValueError('Frozen environment metadata is not the required Environment v2.1 contract')
    expected=manifest.get('runtime_files_sha256')
    if not isinstance(expected,dict) or set(expected)!=set(RUNTIME_FILES):
        raise ValueError('Frozen environment runtime hash roster is missing or incomplete')
    mismatches={}
    for name,rel in RUNTIME_FILES.items():
        path=upstream/rel
        actual=sha256(path) if path.is_file() else None
        if actual!=expected[name]:mismatches[name]={'expected':expected[name],'actual':actual,'path':rel}
    if mismatches:
        raise ValueError('Frozen environment metadata disagrees with actual runtime sources: '+json.dumps(mismatches,sort_keys=True))
    return {'runtime_files_verified':len(RUNTIME_FILES),'environment_metadata_matches_runtime':True}
