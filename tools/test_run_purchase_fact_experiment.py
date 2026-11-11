from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import run_purchase_fact_experiment as experiment


class ProtocolTests(unittest.TestCase):
    def test_model_inputs_are_query_independent_and_selected_only(self):
        _,requests,_=experiment.load(experiment.OUTPUT)
        self.assertEqual(len(requests),46)
        for r in requests.values():
            payload=r['public']
            self.assertEqual(set(payload),{'selected_options','evidence_table','output_schema'})
            self.assertNotIn('requirements',json.dumps(payload))
            for pointer in payload['evidence_table']:
                self.assertFalse(pointer.startswith(('/instructions','/reviews')))

    def test_stored_raw_proposal_cannot_approve_itself(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in ('protocol.json','requests.jsonl','system_prompt.txt','reference_original.jsonl','reference_additional.jsonl'):
                shutil.copyfile(experiment.OUTPUT/name,folder/name)
            calls=[]
            def fake(url,key,body,timeout):
                calls.append(body)
                content={'facts':[{'kind':'claim','role':'declaration','subject':None,
                    'pointer':'/title','quote':'a fabricated span','actual':{'text':'a fabricated span'}}]}
                return 200,json.dumps({'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}]})
            with patch.object(experiment.old,'post',side_effect=fake),patch.object(experiment,'report',return_value={}):
                experiment.run(folder,{'LLM_BASE_URL':'https://example.com','LLM_MODEL':'test','LLM_API_KEY':'secret'},limit=1)
            self.assertEqual(len(calls),1)
            result=experiment.old.rows(folder/'answers.jsonl')[0]
            self.assertEqual(result['compiled']['facts'],[])
            self.assertEqual(len(result['compiled']['rejected']),1)
            self.assertNotIn('secret',(folder/'api_config.json').read_text())
            user=json.loads(calls[0]['messages'][1]['content'])
            self.assertNotIn('query',user)
            self.assertNotIn('_checks',user)


if __name__=='__main__':unittest.main()
