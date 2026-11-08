"""r6 review-only extensions without changing frozen r5 behavior."""
from semantic_scope_v5 import approximation_evidence, direct_hope_scopes, unit_source_flags
from semantic_scope_v5 import split_flags as base_split


def split_flags(c, flags):
    uncertain = [f for f in flags if f['code'] in ('component_count_source_unproven','recommendation_quantity_scope') or (f['code'] in ('purchase_quantity_coverage','quantity_scope_pending') and f.get('source',{}).get('source','').startswith('推荐')) or (f['code'] == 'component_quantity_coverage' and f.get('source',{}).get('source','').startswith('配置'))]
    blocking, review = base_split(c, [f for f in flags if f not in uncertain])
    review.extend(dict(f, severity='review', action='Review source binding; do not delete a stated count or invent a count merely to pass') for f in uncertain)
    return blocking, review
