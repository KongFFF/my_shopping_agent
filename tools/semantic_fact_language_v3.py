"""Small typed vocabulary extension; no fuzzy matching or inferred quality."""
from semantic_fact_language_v2 import canonical as prior, base

GROUPS={
 'brand_positioning':[('潮流品牌','时髦品牌','潮牌')],
 'style':[('复古','复古风')],
 'customization':[('可定制','可定做','可以定制','可以定做')],
 'craft':[('古法','古法工艺')],
 'condition':[('网络热卖款','热卖款','热卖')],
 'form_factor':[('圆筒','圆柱形','圆柱')],
 'function':[('全遮光','100%遮光'),('质量有保障','质量要有保障','品质保证','质量保证'),('轻','轻量')],
}


def implies(subtype,actual,expected):
    # Directional taxonomy: 古法 is a traditional craft, not conversely.
    return subtype=='craft' and canonical(subtype,actual)=='古法' and canonical(subtype,expected)=='传统工艺'


def canonical(subtype,text):
    value=prior(subtype,text)
    for group in GROUPS.get(subtype,[]):
        if value in {base(x) for x in group}:return base(group[0])
    return value
