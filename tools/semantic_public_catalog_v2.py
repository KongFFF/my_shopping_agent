"""Remove task/answer/private fields before preparing product extraction."""
import copy

FIELDS=('asin','title','sub_title','shop_name','category','full_description','attribute','customization_options',
        'pricing','variant_combinations','brand','model')
OPTION_FIELDS=('value','price','price_string','is_available')


def public_product(product):
    result={k:copy.deepcopy(product[k]) for k in FIELDS if k in product}
    if 'customization_options' in result:
        result['customization_options']={axis:[{k:v for k,v in e.items() if k in OPTION_FIELDS} for e in entries]
              for axis,entries in result['customization_options'].items()}
    return result
