"""Public deployment boundaries and end-to-end prediction with an isolated database."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
import json
import os
import sys
import tempfile
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '준비패키지'))
from fastapi.testclient import TestClient
from app_kamp import app
import quality_service as service


class PublicDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ict-public-test-')
        self.env = patch.dict(os.environ, {'ICT_PUBLIC_DEMO': '1', 'ICT_DB_PATH': str(Path(self.temp.name) / 'history.sqlite3')})
        self.env.start()
        service.model_info.cache_clear()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.env.stop()
        service.model_info.cache_clear()
        self.temp.cleanup()

    def test_public_metadata_and_sample_do_not_use_local_private_files(self):
        schema = self.client.get('/schema').json()
        self.assertTrue(schema['public_demo'])
        self.assertFalse(schema['sample_availability']['real'])
        self.assertEqual((schema['max_rows'], schema['history_limit']), (500, 20))
        self.assertEqual(self.client.get('/sample?kind=real').status_code, 404)
        self.assertEqual(self.client.get('/model').json(), json.loads((service.RELEASE / 'benchmark_public.json').read_text(encoding='utf-8')))
        self.assertEqual(self.client.get('/sample?kind=synthetic').content, (service.ROOT / 'share' / 'synthetic_demo.csv').read_bytes())

    def test_public_example_prediction_and_export(self):
        self.assertEqual(self.client.get('/runs').json()['total'], 0)
        raw = self.client.get('/sample?kind=synthetic').content
        response = self.client.post('/predict', files={'file': ('synthetic_demo.csv', raw)})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['predicted_rows'], 120)
        self.assertEqual(result['data_scope'], 'synthetic_kamp60')
        self.assertEqual(self.client.get('/runs/' + result['run_id']).json(), result)
        exported = self.client.get('/runs/' + result['run_id'] + '/export')
        self.assertEqual(len(pd.read_csv(BytesIO(exported.content))), 120)

    def test_free_tier_row_limit_for_csv_and_xlsx(self):
        df = pd.DataFrame({'example': range(501)})
        for name in ('too_many.csv', 'too_many.xlsx'):
            if name.endswith('csv'): raw = df.to_csv(index=False).encode()
            else:
                buffer = BytesIO(); df.to_excel(buffer, index=False); raw = buffer.getvalue()
            response = self.client.post('/preview', files={'file': (name, raw)})
            self.assertEqual(response.status_code, 422)
            self.assertIn('500', response.json()['detail'])

    def test_history_keeps_latest_twenty_only(self):
        for i in range(22):
            service.save_run({'run_id': str(i), 'created_at': f'2026-11-16T00:00:{i:02d}+00:00',
                              'filename': 'synthetic.csv', 'data_scope': 'synthetic_kamp60', 'model_version': 'test', 'predicted_rows': 1})
        history = self.client.get('/runs').json()
        self.assertEqual(history['total'], 20)
        self.assertEqual(history['items'][0]['run_id'], '21')
        self.assertEqual(self.client.get('/runs/0').status_code, 404)
        self.assertEqual(self.client.get('/runs/2').status_code, 200)


if __name__ == '__main__': unittest.main()
