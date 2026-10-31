"""Integration and regression checks, always using an isolated SQLite database."""
from pathlib import Path
from io import BytesIO
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '준비패키지'))
from fastapi.testclient import TestClient
from app_kamp import app
import quality_service as service


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_db = os.environ.get('ICT_DB_PATH')
        cls.temp = tempfile.TemporaryDirectory(prefix='ict-tests-')
        os.environ['ICT_DB_PATH'] = str(Path(cls.temp.name) / 'test.sqlite3')
        cls.client = TestClient(app)
        cls.df = pd.read_csv(service.ROOT / 'share' / 'test_upload.csv', encoding='utf-8-sig', dtype={'batch_id': str})
        cls.metrics = {}

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        if cls.original_db is None: os.environ.pop('ICT_DB_PATH', None)
        else: os.environ['ICT_DB_PATH'] = cls.original_db
        cls.temp.cleanup()
        dest = Path(os.environ.get('ICT_EVIDENCE_DIR', str(ROOT / 'tmp' / 'test-evidence')))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'api_measurements.json').write_text(json.dumps(cls.metrics, indent=2), encoding='utf-8')

    def post(self, endpoint, df=None, mapping=None, units=None, name='test.csv', raw=None):
        if raw is None: raw = (self.df if df is None else df).to_csv(index=False).encode('utf-8-sig')
        return self.client.post(endpoint, files={'file': (name, raw)},
                                data={'mapping': json.dumps(mapping or {}, ensure_ascii=False), 'units': json.dumps(units or {}, ensure_ascii=False)})

    def test_01_full_upload_validation_prediction_history_export(self):
        p = self.post('/preview').json()
        self.assertEqual(p['rows'], 20)
        self.assertEqual(len(p['file_sha256']), 64)
        v = self.post('/validate').json()
        self.assertEqual((v['valid_rows'], v['error_rows']), (20, 0))
        start = time.perf_counter(); response = self.post('/predict'); elapsed = time.perf_counter() - start
        self.assertEqual(response.status_code, 200, response.text)
        r = response.json(); self.metrics['predict_20_rows_seconds'] = elapsed
        self.assertLess(elapsed, 5)
        self.assertEqual(r['predicted_rows'], 20)
        expected = service.bundle()['pipeline'].predict_proba(self.df[service.FEATURES])[:, 1]
        by_id = {row['batch_id']: row['risk_score'] for row in r['results']}
        np.testing.assert_allclose([by_id[k] for k in self.df.batch_id], expected, atol=1e-8)
        self.assertEqual(self.client.get('/runs/'+r['run_id']).json(), r)
        # New client models an API process restart: no in-memory history cache is used.
        with TestClient(app) as restarted:
            self.assertEqual(restarted.get('/runs/'+r['run_id']).json(), r)
        self.assertGreaterEqual(self.client.get('/runs').json()['total'], 1)
        exported = self.client.get('/runs/'+r['run_id']+'/export')
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(len(pd.read_csv(BytesIO(exported.content))), 20)

    def test_02_xlsx_and_cp949_and_ids(self):
        df = self.df.head(2).copy(); df['batch_id'] = ['0001', '0002']
        output = BytesIO(); df.to_excel(output, index=False)
        r = self.post('/predict', raw=output.getvalue(), name='input.xlsx')
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual({x['batch_id'] for x in r.json()['results']}, {'0001','0002'})
        r = self.post('/preview', raw=df.to_csv(index=False).encode('cp949'))
        self.assertEqual(r.json()['encoding'], 'CP949')

    def test_03_alias_mapping_and_label_leakage(self):
        feature = service.FEATURES[0]
        d = self.df.rename(columns={'batch_id': '작업번호', feature: '원본출력'})
        self.assertEqual(self.post('/predict', d, mapping={'작업번호': 'batch_id', '원본출력': feature}).status_code, 200)
        d = self.df.rename(columns={feature: '불량수량'})
        self.assertEqual(self.post('/predict', d, mapping={'불량수량': feature}).status_code, 422)
        self.assertEqual(self.post('/validate', mapping={service.FEATURES[0]:service.FEATURES[1]}).status_code, 422)

    def test_04_duplicates_missing_invalid_and_partial_success(self):
        d = self.df.head(9).copy().astype(object)
        d.loc[0, 'batch_id'] = d.loc[1, 'batch_id']
        d.loc[2, service.FEATURES[:3]] = ''
        d.loc[3, service.FEATURES[:4]] = ''
        d.loc[4, service.FEATURES[0]] = 'abc'
        d.loc[5, service.FEATURES[1]] = -1
        d.loc[6, service.FEATURES[0]] = np.inf
        d.loc[7, 'process_end_at'] = '2000-01-01'
        r = self.post('/predict', d)
        self.assertEqual(r.status_code, 200, r.text)
        p = r.json(); self.assertEqual((p['predicted_rows'], p['error_rows']), (2, 7))
        self.assertEqual(p['missing_cells'], 3)
        imputed = next(x for x in p['results'] if x['imputed_columns'])
        self.assertEqual(imputed['features'][service.FEATURES[0]], service.bundle()['reference']['median'][service.FEATURES[0]])
        self.assertEqual(len(self.client.get('/runs/'+p['run_id']+'/export?kind=errors').content)>0, True)

    def test_05_units_conversion_means_and_standard_deviations(self):
        d = self.df.head(3).copy()
        units = {}
        for f in service.FEATURES:
            if service.unit_for(f) == '%': d[f] /= 100; units[f] = 'fraction'
            else:
                d[f] = d[f] * 9 / 5 + (0 if f.endswith('_Std') else 32); units[f] = '°F'
        raw = self.post('/predict', self.df.head(3)).json()
        conv = self.post('/predict', d, units=units).json()
        np.testing.assert_allclose([r['risk_score'] for r in raw['results']], [r['risk_score'] for r in conv['results']], atol=1e-6)
        d = self.df.head(1).copy(); f = next(x for x in service.FEATURES if '온도' in x and x.endswith('_Avg'))
        d[f] += 273.15
        p = self.post('/predict', d, units={f:'K'}).json()
        self.assertAlmostEqual(p['results'][0]['features'][f], float(self.df.iloc[0][f]))
        self.assertEqual(self.post('/validate', units={f:'unknown'}).status_code, 422)

    def test_06_quality_comparison_is_excluded_from_inference(self):
        d = self.df.head(2).copy(); d['total_count']=10000; d['defect_count']=[1,9999]; d['good_count']=[9999,1]
        a = self.post('/predict', d).json(); b = self.post('/predict', self.df.head(2)).json()
        self.assertEqual([r['risk_score'] for r in a['results']], [r['risk_score'] for r in b['results']])
        self.assertEqual(sorted(r['actual_defect_rate_percent'] for r in a['results']), [.01,99.99])
        d.loc[0, 'good_count']=10000
        self.assertEqual(self.post('/validate', d).json()['error_rows'], 1)

    def test_07_synthetic_is_separate(self):
        raw = (service.ROOT / 'share' / 'synthetic_demo.csv').read_bytes()
        r = self.post('/predict', raw=raw).json()
        self.assertEqual((r['data_scope'], r['predicted_rows']), ('synthetic_kamp60',120))
        d = self.df.head(2).copy(); d.loc[0, 'data_source']='KAMP_heat_quality_60'
        self.assertEqual(self.post('/predict', d).status_code, 422)
        d['data_source']='another_process'
        self.assertEqual(self.post('/predict', d).status_code, 422)

    def test_08_file_and_schema_boundaries(self):
        for name,raw,expected in [('x.txt',b'x',415),('x.xlsx',b'broken',422),('x.csv',b'',422),('x.csv',b'a,a\n1,2',422),('x.csv',b'x'*(service.MAX_BYTES+1),413)]:
            self.assertEqual(self.post('/preview',name=name,raw=raw).status_code, expected)
        self.assertEqual(self.post('/preview',pd.DataFrame({'a':range(10001)})).status_code,422)
        self.assertEqual(self.post('/preview',pd.DataFrame([[1]*101],columns=[str(i) for i in range(101)])).status_code,422)
        self.assertEqual(self.post('/validate',self.df.drop(columns=service.FEATURES[0])).status_code,422)
        r = self.client.post('/validate',files={'file':('x.csv',self.df.to_csv(index=False).encode())},data={'mapping':'[]'})
        self.assertEqual(r.status_code,422)
        self.assertEqual(self.client.get('/runs/not-found').status_code,404)
        self.assertEqual(self.client.get('/runs?limit=0').status_code,422)
        extreme=self.df.head(1).copy(); extreme[service.FEATURES[1]]=1e100
        self.assertEqual(self.post('/predict',extreme).status_code,422)

    def test_09_legacy_migration_preserves_payload(self):
        current = os.environ['ICT_DB_PATH']; legacy = str(Path(self.temp.name)/'legacy.sqlite3')
        payload = {'run_id':'old','predicted_rows':1,'data_scope':'KAMP_heat_quality_60','model_version':'old-version','results':[]}
        db = sqlite3.connect(legacy); db.execute('CREATE TABLE runs(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        db.execute('INSERT INTO runs VALUES (?,?)',('old',json.dumps(payload))); db.commit(); db.close()
        os.environ['ICT_DB_PATH']=legacy
        try:
            self.assertEqual(self.client.get('/runs/old').json(),payload)
            item=self.client.get('/runs').json()['items'][0]
            self.assertIsNone(item['created_at']); self.assertEqual(item['model_version'],'old-version')
        finally: os.environ['ICT_DB_PATH']=current

    def test_10_export_formula_safety(self):
        d=self.df.head(1).copy(); d['batch_id']='=HYPERLINK("bad")'
        r=self.post('/predict',d).json()
        exported=self.client.get('/runs/'+r['run_id']+'/export').content
        value=pd.read_csv(BytesIO(exported)).iloc[0]['작업_ID']
        self.assertTrue(value.startswith("'="))

    def test_11_split_and_artifact_integrity(self):
        model=service.model_info(); self.assertEqual(len(model['validation_comparison']),3)
        groups=model['split']
        self.assertEqual([groups[k]['batches'] for k in ['train','validation','test']],[78,22,36])
        self.assertLess(groups['train']['end'],groups['validation']['start'])
        self.assertLess(groups['validation']['end'],groups['test']['start'])
        if (service.ROOT/'split_manifest.csv').exists():
            manifest=pd.read_csv(service.ROOT/'split_manifest.csv')
            self.assertTrue(manifest.batch_id.is_unique)
        self.assertEqual(model['test_status'],'previously_seen_holdout_development_reference')
        import hashlib
        self.assertEqual(hashlib.sha256((service.RELEASE/'model.joblib').read_bytes()).hexdigest(),model['model_sha256'])
        self.assertEqual(set(service.bundle()['features']),set(service.FEATURES))

    def test_12_distribution_assets_and_sample_availability(self):
        public=json.loads((service.RELEASE/'benchmark_public.json').read_text(encoding='utf-8'))
        self.assertEqual(public['quality_relation_mode'],'group_means_min5')
        for feature in public['feature_analysis']:
            self.assertLessEqual(len(feature['quality_relation']),8)
            self.assertTrue(all(p['count']>=5 for p in feature['quality_relation']))
        schema=self.client.get('/schema').json()
        self.assertEqual(schema['sample_availability']['real'],service.sample_path('real').exists())
        self.assertEqual(self.client.get('/sample?kind=real').status_code,200 if schema['sample_availability']['real'] else 404)
        response=self.client.get('/sample?kind=synthetic');self.assertEqual(response.status_code,200)
        self.assertEqual(set(pd.read_csv(BytesIO(response.content)).data_source),{'synthetic_kamp60'})

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SystemTests))
    evidence = Path(os.environ.get('ICT_EVIDENCE_DIR', str(ROOT/'tmp/test-evidence')))
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence/'api_results.json').write_text(json.dumps({
        'tests_run':result.testsRun, 'failures':len(result.failures), 'errors':len(result.errors),
        'status':'passed' if result.wasSuccessful() else 'failed',
        'tests':[name for name in dir(SystemTests) if name.startswith('test_')]
    }, indent=2), encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
