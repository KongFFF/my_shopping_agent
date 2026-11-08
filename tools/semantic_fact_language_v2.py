"""Finite reviewed vocabulary. No embedding, substring entailment or LLM judge."""
import re
import unicodedata

# Equalities are restricted to a subtype, and used symmetrically. Stronger
# propositions (耐浸泡, 全天, 官方认证, 铜芯) are deliberately not simplified.
GROUPS={
 'product_category': [('电吉他弦','电吉他的琴弦'),('床头板','床头靠背'),('露天柱体灯','户外柱头灯'),
                      ('气垫','气垫CC霜'),('中阮','中阮乐器')],
 'function': [('速干','干得快','干的快','干得很快','干得比较快','速干材质'),
              ('防锈','不容易生锈','不易生锈','不易锈'),('雾化','雾化功能'),
              ('大屏','大屏显示','大屏幕显示'),('耐摔','不怕摔'),('耐折','不怕折'),
              ('可调','可以调节'),('自动','自行运作'),('360度旋转','全方位转动'),
              ('补充DHA','能补充DHA')],
 'size':[('XXL','2XL'),('XXXL','3XL'),('XXXXL','4XL'),('XXXXXL','5XL')],
 'style':[('紧身','版型紧身'),('宽松','版型宽松'),('现代简约','现代简约风')],
 'material':[('合金','合金材质'),('鳕鱼','鳕鱼为原料')],
 'color':[('橙色','桔色','橙色调'),('暗蓝色','暗蓝色系'),('森林绿','类似森林绿的色调'),('古铜色','老铜色')],
 'craft':[('梭织','梭织工艺')],
 'form_factor':[('胶囊','胶囊形状'),('斜挎','斜挎款式')],
 'use_environment':[('春夏','春夏季节')],
 'purpose':[('跑步','跑步穿')],
}


def base(text):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(text))).casefold()


def canonical(subtype,text):
    value=base(text)
    if subtype=='size':value=re.sub(r'码$','',value)
    for group in GROUPS.get(subtype,[]):
        if value in {base(x) for x in group}:return base(group[0])
    return value
