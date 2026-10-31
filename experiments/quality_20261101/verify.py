"""Check preservation, split isolation and saved candidate inference reproducibility."""
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from common import HERE,ROOT,KAMP,sha,save_json,compact_features,metrics

class ExperimentChecks(unittest.TestCase):
    def test_original_data_and_deployed_model_unchanged(self):
        audit=json.loads((HERE/'results/data_audit.json').read_text(encoding='utf-8'))
        self.assertEqual(sha(KAMP/'release/model.joblib'),audit['model_sha256'])
        self.assertEqual(sha(KAMP/'joined_batches.csv'),audit['joined_sha256'])
        source=ROOT/'참고자료/KAMP_열처리_품질보증/원본/data'
        for name,record in audit['original_files'].items(): self.assertEqual(sha(source/name),record['sha256'])
    def test_original_release_and_history_preserved(self):
        snapshot=HERE/'baseline_snapshot'
        manifest=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))['sha256']
        for name,digest in manifest.items():
            self.assertEqual(sha(snapshot/name),digest)
            if 'kamp' in name and ('release' in name or name.endswith('.sqlite3')):
                self.assertEqual(sha(ROOT/name),digest,name)
    def test_public_download_hashes(self):
        source=HERE/'sources/uci_secom'
        provenance=json.loads((source/'provenance.json').read_text())
        for name,record in provenance['files'].items(): self.assertEqual(sha(source/name),record['sha256'])
    def test_split_isolation(self):
        for name,id_col,time_col in [('kamp_v1','batch_id','process_end_at'),('secom_v1','entity_id','test_time')]:
            run=HERE/'runs'/name;m=pd.read_csv(run/'split_manifest.csv')
            self.assertTrue(m[id_col].is_unique)
            pieces=[m[m.split==s] for s in ['train','validation','test']]
            for a,b in zip(pieces,pieces[1:]): self.assertLess(a[time_col].max(),b[time_col].min())
            if name=='secom_v1':
                days=pd.to_datetime(m.test_time).dt.date
                self.assertEqual(m.groupby(days).split.nunique().max(),1)
                self.assertEqual(sha(run/'split_manifest.csv'),json.loads((run/'split_lock.json').read_text())['sha256'])
    def test_model_locks_and_training_only_imputation(self):
        for name in ['kamp_v1','secom_v1']:
            run=HERE/'runs'/name
            lock=json.loads((run/'selection_lock.json').read_text())
            model=run/'models'/f"{lock['winner']}.joblib"
            self.assertEqual(sha(model),lock['model_sha256'])
            bundle=joblib.load(model)
            self.assertEqual(bundle['threshold'],lock['threshold'])
            self.assertFalse(any(c in bundle['features'] for c in ['defect_count','risk_label','test_time','entity_id','batch_id']))
            cv=json.loads((run/'cv_results.json').read_text(encoding='utf-8'))
            winner=[c for c in cv if c['id']==lock['winner']][0]
            self.assertEqual(len(winner['retained_features']),8 if name=='kamp_v1' else 40)
            self.assertEqual(winner['cv_mean_ap'],max(c['cv_mean_ap'] for c in cv if not c['baseline']))
    def test_kamp_inference_reproduction(self):
        run=HERE/'runs/kamp_v1';result=json.loads((run/'final_results.json').read_text())
        b=joblib.load(run/'models'/f"{result['winner']}.joblib")
        d=pd.read_csv(KAMP/'joined_batches.csv',dtype={'batch_id':str}).sort_values(['process_end_at','batch_id']).reset_index(drop=True)
        expected=pd.read_csv(run/'test_predictions.csv',dtype={'id':str})
        X=compact_features(d.set_index('batch_id').loc[expected.id])
        actual=b['pipeline'].predict_proba(X)[:,1]
        np.testing.assert_allclose(actual,expected.score,atol=1e-14)
        self.assertEqual(result['decision']['action'],'retain_deployed_RF')
        train=pd.read_csv(run/'split_manifest.csv',dtype={'batch_id':str})
        Xtrain=compact_features(d.set_index('batch_id').loc[train.loc[train.split=='train','batch_id']])
        retained=b['pipeline'].named_steps['prune'].transform(Xtrain)
        np.testing.assert_allclose(np.nanmedian(retained,axis=0),b['pipeline'].named_steps['impute'].statistics_)
    def test_secom_inference_reproduction(self):
        run=HERE/'runs/secom_v1';result=json.loads((run/'final_results.json').read_text())
        b=joblib.load(run/'models'/f"{result['winner']}.joblib")
        raw=pd.read_csv(HERE/'sources/uci_secom/secom.data',sep=r'\s+',header=None)
        expected=pd.read_csv(run/'test_predictions.csv')
        X=raw.iloc[expected.id.to_numpy()-1].copy();X.columns=b['features']
        np.testing.assert_allclose(b['pipeline'].predict_proba(X)[:,1],expected.score,atol=1e-14)
        m=pd.read_csv(run/'split_manifest.csv')
        Xtrain=raw.iloc[m.loc[m.split=='train','entity_id'].to_numpy()-1].copy();Xtrain.columns=b['features']
        retained=b['pipeline'].named_steps['prune'].transform(Xtrain)
        np.testing.assert_allclose(np.nanmedian(retained,axis=0),b['pipeline'].named_steps['impute'].statistics_)
    def test_temporal_features_source_and_finiteness(self):
        m=json.loads((HERE/'results/temporal_manifest.json').read_text(encoding='utf-8'))
        f=HERE/'results/temporal_features.csv';d=pd.read_csv(f)
        self.assertEqual(sha(f),m['output_sha256']);self.assertEqual(d.shape,(136,36))
        self.assertTrue(np.isfinite(d.drop(columns='batch_id').to_numpy()).all())
        for group in m['groups']:
            self.assertTrue((d[group+'_max']>=d[group+'_min']).all())
            np.testing.assert_allclose(d[group+'_late_mean']-d[group+'_early_mean'],d[group+'_late_minus_early'],atol=1e-10)

if __name__=='__main__':
    import sys
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExperimentChecks))
    save_json(HERE/'verification/experiment_checks.json',dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),
        status='passed' if r.wasSuccessful() else 'failed',tests=[n for n in dir(ExperimentChecks) if n.startswith('test_')]))
    sys.exit(0 if r.wasSuccessful() else 1)
