"""Versioned source-backed catalog-fact interchange schema."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from semantic_contract_v2_r7_final12 import schema as contract_schema, DIMENSIONS, ATTRIBUTES

VERSION='semantic-product-facts-v3-draft'


def schema():
    defs=contract_schema()['$defs'];value=copy.deepcopy(defs['expected'])
    value['oneOf']=[v for v in value['oneOf'] if v['properties']['type']['const']!='member_ranges']
    value['oneOf'].append({'type':'object','additionalProperties':False,
        'properties':{'type':{'const':'member_values'},'member_attribute':{'const':'segment_count'},'unit':{'const':'count'},
                      'values':{'type':'array','minItems':1,'items':{'type':'integer','minimum':1}}},
        'required':['type','member_attribute','unit','values']})
    for variant in value['oneOf']:
        if variant['properties']['type']['const']=='bundle':
            variant['properties']['same_item']={'type':['boolean','null']}
            variant['properties']['unit']={'enum':variant['properties']['unit']['enum']+[None]}
    evidence={'oneOf':[
        {'type':'object','properties':{'pointer':{'type':'string','pattern':'^/'},'value':{}},'required':['pointer','value'],'additionalProperties':False},
        {'type':'object','properties':{'pointer':{'type':'string','pattern':'^/'},'text':{'type':'string','minLength':1},
          'start':{'type':'integer','minimum':0},'end':{'type':'integer','minimum':1}},'required':['pointer','text','start','end'],'additionalProperties':False}]}
    result={'$schema':'https://json-schema.org/draft/2020-12/schema','type':'object','additionalProperties':False,
      '$defs':{'value':value,'subject':defs['subject'],'evidence':evidence},'properties':{
        'product_asin':{'type':'string','minLength':1},'product_sha256':{'type':'string','pattern':'^[a-f0-9]{64}$'},
        'scope':{'enum':['product','sku']},'options':{'type':['object','null'],'additionalProperties':{'type':'string'}},
        'selector_mode':{'enum':['exact','contains']},
        'review_note':{'type':'string','minLength':1},
        'dimension':{'enum':list(DIMENSIONS)},'subtype':{'enum':list(ATTRIBUTES)},'attribute':{'enum':sorted({a for v in ATTRIBUTES.values() for a in v})},
        'subject':{'$ref':'#/$defs/subject'},'value':{'$ref':'#/$defs/value'},'polarity':{'enum':['positive','negative']},
        'complete':{'type':'boolean'},'review_status':{'enum':['pending','approved']},
        'evidence':{'type':'array','minItems':1,'items':{'$ref':'#/$defs/evidence'}}},
      'required':['product_asin','product_sha256','scope','dimension','subtype','attribute','subject','value','complete','review_status','evidence']}
    Draft202012Validator.check_schema(result)
    return result


VALIDATOR=Draft202012Validator(schema())


def validate_fact(record):
    VALIDATOR.validate(record)
    if record['subtype'] not in DIMENSIONS[record['dimension']] or record['attribute'] not in ATTRIBUTES[record['subtype']]:
        raise ValueError('Fact dimension/subtype/attribute mismatch')
    if record['scope']=='sku' and not isinstance(record.get('options'),dict):raise ValueError('SKU fact requires options')
    if record.get('selector_mode')=='contains' and (record['scope']!='sku' or not record.get('options')):
        raise ValueError('Partial selector requires at least one SKU axis')
    v=record['value']
    if record['attribute']=='percentage' or v.get('unit')=='percent':
        if record['subtype']!='composition' or record['attribute']!='percentage' or v.get('unit')!='percent':raise ValueError('Percent fact needs composition/percentage')
        numbers=[v['value']] if v['type']=='number' else [v['lower'],v['upper']] if v['type']=='range' else []
        if not numbers or any(n is not None and not 0<=n<=100 for n in numbers):raise ValueError('Invalid percentage fact')


if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'experiments/semantic_product_verifier_v3/catalog_facts.schema.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(schema(),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path)
