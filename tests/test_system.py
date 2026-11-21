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
from app import app
import quality_service as service


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_db = os.environ.get('ICT_DB_PATH')
        cls.temp = tempfile.TemporaryDirectory(prefix='ict-tests-')
        os.environ['ICT_DB_PATH'] = str(Path(cls.temp.name) / 'test.sqlite3')
        cls.client = TestClient(app)
        full = pd.read_csv(service.SHARE / 'test_upload.csv', encoding='utf-8-sig', dtype={'part_id': str})
        cls.upload = full                                     # 오류 행 포함
        cls.df = full[full.part_id.str.startswith('UPL-')].iloc[1:].reset_index(drop=True)   # 정상 행만
        cls.F = service.features(); cls.REQ = service.required_features()
        cls.OPT = [f for f in cls.F if f not in cls.REQ]
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

    def post(self, endpoint, df=None, mapping=None, name='test.csv', raw=None):
        if raw is None: raw = (self.df if df is None else df).to_csv(index=False).encode('utf-8-sig')
        return self.client.post(endpoint, files={'file': (name, raw)},
                                data={'mapping': json.dumps(mapping or {}, ensure_ascii=False)})

    def test_01_full_upload_validation_prediction_history_export(self):
        n = len(self.df)
        p = self.post('/preview').json()
        self.assertEqual(p['rows'], n); self.assertEqual(len(p['file_sha256']), 64)
        v = self.post('/validate').json()
        self.assertEqual((v['valid_rows'], v['error_rows']), (n, 0))
        start = time.perf_counter(); response = self.post('/predict'); elapsed = time.perf_counter() - start
        self.assertEqual(response.status_code, 200, response.text)
        r = response.json(); self.metrics['predict_rows_seconds'] = {'rows': n, 'seconds': elapsed}
        self.assertLess(elapsed, 5)
        self.assertEqual(r['predicted_rows'], n)
        X = self.df[self.F].apply(pd.to_numeric)
        expected = service.bundle()['pipeline'].predict_proba(X)[:, 1]
        by_id = {row['part_id']: row['risk_score'] for row in r['results']}
        np.testing.assert_allclose([by_id[k] for k in self.df.part_id], expected, atol=1e-8)
        thr = service.bundle()['alert_threshold']
        for row in r['results']:
            self.assertEqual(row['review_priority'], '우선 검토' if row['risk_score'] >= thr else '통상 검토')
        self.assertEqual(self.client.get('/runs/'+r['run_id']).json(), r)
        with TestClient(app) as restarted:   # API 프로세스 재시작: 메모리 캐시 없이 DB에서 읽음
            self.assertEqual(restarted.get('/runs/'+r['run_id']).json(), r)
        self.assertGreaterEqual(self.client.get('/runs').json()['total'], 1)
        exported = self.client.get('/runs/'+r['run_id']+'/export')
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(len(pd.read_csv(BytesIO(exported.content))), n)

    def test_02_xlsx_and_cp949_and_ids(self):
        df = self.df.head(2).copy(); df['part_id'] = ['0001', '0002']
        output = BytesIO(); df.to_excel(output, index=False)
        r = self.post('/predict', raw=output.getvalue(), name='input.xlsx')
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual({x['part_id'] for x in r.json()['results']}, {'0001', '0002'})
        df['material'] = '폴리프로필렌'   # 비ASCII 문자가 있어야 인코딩 판별 가능
        r = self.post('/preview', raw=df.to_csv(index=False).encode('cp949'))
        self.assertEqual(r.json()['encoding'], 'CP949')

    def test_03_alias_mapping_and_leakage_guard(self):
        f = self.REQ[0]
        d = self.df.rename(columns={'part_id': '부품번호', f: '원본보압'})
        self.assertEqual(self.post('/predict', d, mapping={'부품번호': 'part_id', '원본보압': f}).status_code, 200)
        for leaky in ['LBL_NOK', 'IR_Area', 'SCA_Weight', '불량여부']:
            d = self.df.rename(columns={f: leaky})
            self.assertEqual(self.post('/predict', d, mapping={leaky: f}).status_code, 422, leaky)
        self.assertEqual(self.post('/validate', mapping={'part_id': 'unknown'}).status_code, 422)
        # 원본 컬럼명(MET_MachineCycleID 등)은 제안 매핑으로 자동 연결
        raw = self.df.rename(columns={'part_id': 'MET_MachineCycleID', 'cycle_start_at': 'MET_TimeCycleStart'})
        sug = self.post('/preview', raw).json()['suggested_mapping']
        self.assertEqual((sug['MET_MachineCycleID'], sug['MET_TimeCycleStart']), ('part_id', 'cycle_start_at'))

    def test_04_bundled_error_rows_and_partial_success(self):
        r = self.post('/predict', self.upload)
        self.assertEqual(r.status_code, 200, r.text)
        p = r.json()
        self.assertEqual(p['input_rows'], len(self.upload))
        self.assertEqual(p['predicted_rows'] + p['error_rows'], len(self.upload))
        reasons = ' '.join(x for e in p['errors'] for x in e['reasons'])
        for text in ['중복', '허용 범위', '필수 특징 누락', '날짜 형식', 'actual_label']:
            self.assertIn(text, reasons)
        self.assertGreater(len(self.client.get('/runs/'+p['run_id']+'/export?kind=errors').content), 0)

    def test_05_missing_policy(self):
        d = self.df.head(4).copy().astype(object)
        for f in self.OPT: d.loc[0, f] = ''
        d.loc[1, self.REQ[0]] = ''
        d.loc[2, self.REQ[0]] = 'abc'
        d.loc[3, self.REQ[0]] = np.inf
        p = self.post('/predict', d).json()
        self.assertEqual((p['predicted_rows'], p['error_rows']), (1, 3))
        self.assertEqual(p['missing_cells'], len(self.OPT))
        row = p['results'][0]
        self.assertEqual(sorted(row['imputed_columns']), sorted(self.OPT))
        for f in self.OPT:
            self.assertEqual(row['features'][f], service.bundle()['reference']['median'][f])
            self.assertIsNone(row['original_numeric_features'][f])
        if self.OPT:   # 선택 특징 열이 통째로 없어도 예측
            r = self.post('/predict', self.df.drop(columns=self.OPT))
            self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.post('/validate', self.df.drop(columns=self.REQ[0])).status_code, 422)

    def test_06_no_unit_conversion_and_range_check(self):
        schema = self.client.get('/schema').json()
        spec = {x['name']: x for x in schema['fields'] if x['kind'] == 'feature'}
        self.assertEqual(set(spec), set(self.F))
        self.assertTrue(all(spec[f]['unit'] for f in self.F))
        f = self.REQ[0]; d = self.df.head(1).copy()
        d[f] = spec[f]['max'] + 1
        self.assertEqual(self.post('/validate', d).json()['error_rows'], 1)
        r = self.post('/predict', self.df.head(1)).json()
        self.assertEqual(r['units'], {x: spec[x]['unit'] for x in self.F})

    def test_07_label_is_excluded_from_inference(self):
        d = self.df.head(4).copy(); d['actual_label'] = [1, 1, 0, 0]
        a = self.post('/predict', d).json(); b = self.post('/predict', d.drop(columns='actual_label')).json()
        self.assertEqual([r['risk_score'] for r in a['results']], [r['risk_score'] for r in b['results']])
        cmp = a['summary']['label_comparison']
        self.assertEqual((cmp['count'], cmp['positive_count']), (4, 2))
        self.assertEqual(cmp['tp'] + cmp['fn'] + cmp['fp'] + cmp['tn'], 4)
        self.assertIsNone(b['summary']['label_comparison'])

    def test_08_sources_are_separate(self):
        raw = (service.SHARE / 'synthetic_demo.csv').read_bytes()
        r = self.post('/predict', raw=raw).json()
        self.assertEqual((r['data_scope'], r['predicted_rows']), ('synthetic_im', 120))
        d = self.df.head(2).copy(); d.loc[0, 'data_source'] = 'ProBayes_IM_v2'
        self.assertEqual(self.post('/predict', d).status_code, 422)
        d['data_source'] = 'another_process'
        self.assertEqual(self.post('/predict', d).json()['detail']['valid_rows'], 0)

    def test_09_file_and_schema_boundaries(self):
        for name, raw, expected in [('x.txt', b'x', 415), ('x.xlsx', b'broken', 422), ('x.csv', b'', 422),
                                    ('x.csv', b'a,a\n1,2', 422), ('x.csv', b'x'*(service.MAX_BYTES+1), 413)]:
            self.assertEqual(self.post('/preview', name=name, raw=raw).status_code, expected)
        self.assertEqual(self.post('/preview', pd.DataFrame({'a': range(10001)})).status_code, 422)
        self.assertEqual(self.post('/preview', pd.DataFrame([[1]*101], columns=[str(i) for i in range(101)])).status_code, 422)
        r = self.client.post('/validate', files={'file': ('x.csv', self.df.to_csv(index=False).encode())}, data={'mapping': '[]'})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(self.client.get('/runs/not-found').status_code, 404)
        self.assertEqual(self.client.get('/runs?limit=0').status_code, 422)
        d = self.df.head(1).copy(); d['cycle_end_at'] = '2000-01-01 00:00:00'
        self.assertEqual(self.post('/predict', d).status_code, 422)

    def test_10_legacy_history_and_export_safety(self):
        current = os.environ['ICT_DB_PATH']; legacy = str(Path(self.temp.name)/'legacy.sqlite3')
        payload = {'run_id': 'old', 'predicted_rows': 1, 'data_scope': 'KAMP_heat_quality_60', 'model_version': 'old-version',
                   'errors': [], 'results': [{'batch_id': 'B1', 'risk_score': .5, 'review_priority': '통상 검토',
                                              'outside_training_reference': []}]}
        db = sqlite3.connect(legacy); db.execute('CREATE TABLE runs(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        db.execute('INSERT INTO runs VALUES (?,?)', ('old', json.dumps(payload))); db.commit(); db.close()
        os.environ['ICT_DB_PATH'] = legacy
        try:
            self.assertEqual(self.client.get('/runs/old').json(), payload)
            item = self.client.get('/runs').json()['items'][0]
            self.assertIsNone(item['created_at']); self.assertEqual(item['model_version'], 'old-version')
            self.assertEqual(self.client.get('/runs/old/export').status_code, 200)   # 이전 형식 이력도 내보내기 가능
        finally: os.environ['ICT_DB_PATH'] = current
        d = self.df.head(1).copy(); d['part_id'] = '=HYPERLINK("bad")'
        r = self.post('/predict', d).json()
        value = pd.read_csv(BytesIO(self.client.get('/runs/'+r['run_id']+'/export').content)).iloc[0]['부품_ID']
        self.assertTrue(value.startswith("'="))

    def test_11_release_integrity(self):
        import hashlib
        model = service.model_info(); b = service.bundle()
        self.assertEqual(hashlib.sha256((service.RELEASE/'model.joblib').read_bytes()).hexdigest(), model['model_sha256'])
        self.assertEqual(model['test_status'], 'previously_seen_holdout_development_reference')
        self.assertEqual(model['selected_model'], b['selected_model'])
        self.assertEqual(model['alert_threshold'], b['alert_threshold'])
        self.assertEqual(sum(r['selected'] for r in model['validation_comparison']), 1)
        self.assertTrue(json.loads((service.RELEASE/'release_check.json').read_text(encoding='utf-8'))['passed'])
        for f in b['features']:
            self.assertFalse(service.is_leaky(f), f)

    def test_12_distribution_assets_and_sample_availability(self):
        public = json.loads((service.RELEASE/'benchmark_public.json').read_text(encoding='utf-8'))
        self.assertEqual(public['quality_relation_mode'], 'group_means_min5')
        for feature in public['feature_analysis']:
            self.assertLessEqual(len(feature['quality_relation']), 10)
            self.assertTrue(all(p['count'] >= 5 for p in feature['quality_relation']))
        schema = self.client.get('/schema').json()
        self.assertEqual(schema['sample_availability']['real'], service.sample_path('real').exists())
        self.assertEqual(self.client.get('/sample?kind=real').status_code, 200 if schema['sample_availability']['real'] else 404)
        for kind in ['synthetic', 'test']:
            response = self.client.get('/sample?kind='+kind); self.assertEqual(response.status_code, 200)
            self.assertEqual(set(pd.read_csv(BytesIO(response.content)).data_source), {'synthetic_im'})

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
