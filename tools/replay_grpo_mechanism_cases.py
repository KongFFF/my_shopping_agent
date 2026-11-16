"""Reproduce five known online cases and inspect native hard gates; no model."""
import inspect
import json
from pathlib import Path
from audit_online_windows import ROOT, inputs, runtime


def main():
    _,occ,meta,_=inputs()
    flags=json.loads((ROOT/'analysis/online_window_audit_20261116/screen_flags.json').read_text(encoding='utf-8'))
    replay,hashes=runtime(meta)
    # Reuse the exact bound replay factory, including its dependency isolation,
    # rather than instantiate a second environment with different defaults.
    env=inspect.getclosurevars(replay).nonlocals['env']
    from web_agent_site.engine.reward import evaluate_purchase
    cases=[]
    for tid,step in [(22652,755),(19271,675),(15497,659),(5423,711),(373,775)]:
        f=next(f for f in flags if f['window']=='late651_800'and f['step']==step and f['task_id']==tid)
        o=next(o for o in occ if o['window']==f['window']and o['key']==f['key']and o['step']==f['step'])
        r=o['rollout'];got=replay(r);session=env.server.user_sessions[env.session]
        detail=session.get('verbose_info',{})
        assert got['reward_type']==r['reward_type']and abs(got['reward']-r['reward']['terminal_utility'])<1e-7
        row={'task_id':tid,'step':step,'uid':r['uid'],'generation_rollout_index':r['rollout_index'],
            'consumed':o['consumed'],'query':got['query'],'original_reward':r['reward']['terminal_utility'],
            'replayed_reward':got['reward'],'reward_type':got['reward_type'],
            'goal_price_upper':session['goal'].get('price_upper'),'price_resolution':session.get('price_resolution'),
            'selected_options':session.get('options'),'hard_gates':detail.get('hard_gates'),'requirement':f.get('requirement')}
        if f['kind']=='explicit_budget':
            goal={**session['goal'],'price_upper':f['requirement']['expected']['value']}
            changed=evaluate_purchase(env.server.product_item_dict[session['asin']],goal,
                selected_options=session['options'],price_resolution=session.get('price_resolution'),
                rewards=env.server.environment_config['reward'])
            row['terminal_only_counterfactual']={'reward':changed.reward,'reward_type':changed.reward_type,
                'hard_gates':changed.hard_gates,'limit':'Only a copied goal upper bound changed; no prefix/dynamics/training counterfactual.'}
        cases.append(row)
    out=ROOT/'analysis/grpo_mechanism_audit_20261116/native_case_replays.json'
    out.write_text(json.dumps({'runtime_hashes':hashes,'cases':cases},ensure_ascii=False,indent=2),encoding='utf-8')
    from web_agent_site.engine.variant_price import candidate_options_for_evaluation,resolve_variant_price
    missing=json.loads((ROOT/'analysis/grpo_mechanism_audit_20261116/budget_parser_candidates.json').read_text(encoding='utf-8'))
    targets=[]
    for item in missing:
        goal=env.server.goals[item['task_id']]
        assert goal['instruction_text']==item['query']and goal['price_upper']is None
        product=env.server.product_item_dict[goal['asin']]
        selected,resolution=candidate_options_for_evaluation(product,goal['required_options_by_key'])
        price=resolve_variant_price(product,selected)
        limit=item['requirement']['expected']['value']
        targets.append({'task_id':item['task_id'],'query':item['query'],'explicit_upper':limit,
            'target_asin':goal['asin'],'target_title':product['title'],'selected_for_audit_only':selected,
            'selection_resolution':resolution,'price_resolution':price,
            'listed_required_variant_over_budget':price.get('status')=='pass'and price['price']>limit,
            'limits':'Native target requirements plus deterministic completion, not a proposed Actor action or whole-task feasibility proof.'})
    (out.parent/'budget_target_consistency.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: five original outcomes reproduced; four explicit-budget terminal-only checks saved.')


if __name__=='__main__':main()
