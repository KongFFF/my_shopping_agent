"""Unique environment-normalized purchase mapping, not fuzzy SKU repair."""
from web_agent_site.engine.reward_features import canonicalize_option_axis,normalize_option_text


def canonicalize_purchase(product,recorded):
    axes=product.get('customization_options') or {};mapped={};proof=[];errors=[]
    for old_axis,old_value in recorded.items():
        candidates=[a for a in axes if a==old_axis]
        if not candidates:candidates=[a for a in axes if canonicalize_option_axis(a)==canonicalize_option_axis(old_axis)]
        if len(candidates)!=1:errors.append({'reason':'axis_mapping_not_unique','axis':old_axis});continue
        axis=candidates[0]
        if axis in mapped:errors.append({'reason':'axis_collision','axis':old_axis});continue
        exact=[e for e in axes[axis] if e.get('value')==old_value]
        choices=exact or [e for e in axes[axis] if normalize_option_text(e.get('value'))==normalize_option_text(old_value)]
        if len(choices)!=1:errors.append({'reason':'value_mapping_not_unique','axis':old_axis,'value':old_value});continue
        value=choices[0]['value'];mapped[axis]=value
        proof.append({'recorded_axis':old_axis,'recorded_value':old_value,'catalog_axis':axis,'catalog_value':value,
                      'method':'exact' if old_axis==axis and old_value==value else 'unique_environment_normalization'})
    return {'selected_options':mapped,'mapping_evidence':proof,'errors':errors}
