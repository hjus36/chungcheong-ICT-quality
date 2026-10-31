"""Reproducible release experiment. Never select candidates on the old test set."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from train_kamp import metrics

ROOT = Path(__file__).resolve().parent / 'kamp'
OUT = ROOT / 'release'
SEED = 20261101


def score(y, p, t):
    result = metrics(y, p, t)
    result['accuracy'] = float(np.mean((np.asarray(p) >= t) == np.asarray(y)))
    result['alert_rate'] = float(np.mean(np.asarray(p) >= t))
    return result


def make_pipeline(name, value, y):
    if name == 'LogisticRegression':
        estimator = LogisticRegression(C=value, class_weight='balanced', max_iter=3000, random_state=SEED)
    elif name == 'RandomForest':
        estimator = RandomForestClassifier(n_estimators=200, max_depth=value, min_samples_leaf=6,
                                            class_weight='balanced', random_state=SEED, n_jobs=2)
    else:
        estimator = XGBClassifier(n_estimators=80, max_depth=value, learning_rate=.05, min_child_weight=3,
                                  reg_lambda=5, scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
                                  random_state=SEED, n_jobs=2, eval_metric='logloss')
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scale', StandardScaler() if name == 'LogisticRegression' else 'passthrough'),
                     ('model', estimator)])


def main():
    OUT.mkdir(exist_ok=True)
    audit = json.loads((ROOT / 'audit.json').read_text(encoding='utf-8'))
    features = audit['features']
    data = pd.read_csv(ROOT / 'joined_batches.csv', encoding='utf-8-sig', dtype={'batch_id': str})
    data = data.sort_values(['process_end_at', 'batch_id']).reset_index(drop=True)
    manifest = pd.read_csv(ROOT / 'split_manifest.csv', encoding='utf-8-sig', dtype={'batch_id': str})
    data = data.merge(manifest[['batch_id', 'split']], on='batch_id', validate='one_to_one')
    train, val, test = [data[data.split == s].reset_index(drop=True) for s in ['train', 'validation', 'test']]
    label_threshold = float(train.defect_rate_percent.quantile(.75))
    y = (train.defect_rate_percent >= label_threshold).astype(int)
    yv = (val.defect_rate_percent >= label_threshold).astype(int)
    yt = (test.defect_rate_percent >= label_threshold).astype(int)
    cv_results, comparison, fitted = [], [], {}
    # Within each time fold use that fold's training-only Q3 for labels as well as preprocessing.
    for name, grid in {'LogisticRegression': [.01, .1, 1.], 'RandomForest': [2, 3, 5], 'XGBoost': [1, 2, 3]}.items():
        candidates = []
        for value in grid:
            folds = []
            for tr, va in TimeSeriesSplit(n_splits=3).split(train):
                # Purge batches whose completion overlaps the next fold's first start.
                tr = np.array([i for i in tr if train.loc[i, 'process_end_at'] < train.loc[va, 'process_start_at'].min()])
                boundary = float(train.loc[tr, 'defect_rate_percent'].quantile(.75))
                a = (train.loc[tr, 'defect_rate_percent'] >= boundary).astype(int)
                b = (train.loc[va, 'defect_rate_percent'] >= boundary).astype(int)
                if a.nunique() < 2 or b.nunique() < 2:
                    continue
                p = make_pipeline(name, value, a)
                p.fit(train.loc[tr, features], a)
                folds.append({'train_count': len(tr), 'validation_count': len(va), 'label_threshold': boundary,
                              'average_precision': score(b, p.predict_proba(train.loc[va, features])[:, 1], .5)['average_precision']})
            mean_ap = float(np.mean([f['average_precision'] for f in folds])) if folds else -1.
            candidate = {'model': name, 'parameter': 'C' if name == 'LogisticRegression' else 'max_depth',
                         'value': value, 'mean_cv_average_precision': mean_ap, 'folds': folds}
            cv_results.append(candidate)
            candidates.append(candidate)
        selected = max(candidates, key=lambda c: c['mean_cv_average_precision'])
        pipe = make_pipeline(name, selected['value'], y)
        pipe.fit(train[features], y)
        probabilities = pipe.predict_proba(val[features])[:, 1]
        choices = [(round(float(t), 2), score(yv, probabilities, t)) for t in np.arange(.1, .91, .05)]
        threshold, result = max(choices, key=lambda s: (s[1]['f2'], s[1]['precision'], s[0]))
        comparison.append({'model': name, 'parameter': selected['parameter'], 'value': selected['value'],
                           'alert_threshold': threshold, **result})
        fitted[name] = pipe
    winner = max(comparison, key=lambda c: (c['f2'], c['precision']))
    pipe = fitted[winner['model']]
    threshold = winner['alert_threshold']
    reference = {k: train[features].quantile(q).to_dict() for k, q in [('q05', .05), ('median', .5), ('q95', .95)]}
    reference['mean'] = train[features].mean().to_dict()
    reference['std'] = train[features].std().to_dict()
    bundle = {'pipeline': pipe, 'features': features, 'alert_threshold': threshold,
              'defect_rate_threshold_percent': label_threshold, 'version': 'kamp60-release-20261101',
              'data_source': 'KAMP_heat_quality_60', 'reference': reference, 'selected_model': winner['model']}
    joblib.dump(bundle, OUT / 'model.joblib')
    # Selection is complete. This previously inspected holdout is only a development reference.
    ptest = pipe.predict_proba(test[features])[:, 1]
    evaluation = score(yt, ptest, threshold)
    perm = permutation_importance(pipe, val[features], yv, scoring='average_precision', n_repeats=20,
                                  random_state=SEED, n_jobs=1)
    estimator = pipe.named_steps['model']
    weight = estimator.coef_[0] if hasattr(estimator, 'coef_') else estimator.feature_importances_
    analytics = []
    for i, feature in enumerate(features):
        hist, edges = np.histogram(train[feature].dropna(), bins=8)
        corr = train[feature].corr(train.defect_rate_percent, method='spearman')
        analytics.append({'feature': feature, 'q05': reference['q05'][feature], 'median': reference['median'][feature],
                          'q95': reference['q95'][feature], 'model_weight': float(weight[i]),
                          'permutation_ap_drop': float(perm.importances_mean[i]),
                          'permutation_ap_std': float(perm.importances_std[i]),
                          'spearman': None if pd.isna(corr) else float(corr),
                          'histogram': [{'low': float(edges[j]), 'high': float(edges[j+1]), 'count': int(n)} for j, n in enumerate(hist)],
                          'quality_relation': [{'value': float(row[feature]), 'defect_rate_percent': float(row.defect_rate_percent)}
                                               for _, row in train.iterrows()]})
    report = {'version': bundle['version'], 'selected_model': winner['model'], 'alert_threshold': threshold,
              'model_sha256': hashlib.sha256((OUT / 'model.joblib').read_bytes()).hexdigest(),
              'data_sha256': hashlib.sha256((ROOT / 'joined_batches.csv').read_bytes()).hexdigest(),
              'seed': SEED, 'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__},
              'selection_rule': 'Train time-CV AP selects parameters per family; May validation F2 then precision selects threshold/model.',
              'defect_rate_threshold_percent': label_threshold, 'defect_rate_threshold_ppm': label_threshold * 10000,
              'split': json.loads((ROOT / 'benchmark.json').read_text(encoding='utf-8'))['split'],
              'cv_comparison': cv_results, 'validation_comparison': comparison, 'test': evaluation,
              'test_status': 'previously_seen_holdout_development_reference',
              'test_all_normal': score(yt, np.zeros(len(yt)), .5), 'test_all_risk': score(yt, np.ones(len(yt)), .5),
              'validation_all_risk': score(yv, np.ones(len(yv)), .5),
              'feature_analysis': analytics, 'feature_count': len(features), 'source_url': audit['source_url'],
              'limitations': ['독립 품질 라벨은 136작업입니다.', '공정 완료 후 집계값이며 실시간 조기예측이 아닙니다.',
                              '위험 점수는 보정된 불량 확률·예상 불량률이 아닙니다.', '5~95% 구간은 학습 통계이며 기업 규격이 아닙니다.',
                              '보류자료는 이전에 확인한 개발 참고 평가입니다. 신규 외부 검증은 없습니다.',
                              '제품·재질·검사 시각과 기업 규격이 없어 현장 일반화를 확인하지 못했습니다.']}
    (OUT / 'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(comparison).to_csv(OUT / 'validation_comparison.csv', index=False, encoding='utf-8-sig')
    predictions = test[['batch_id', 'defect_rate_percent']].copy()
    predictions['risk_label'] = yt
    predictions['risk_score'] = ptest
    predictions['predicted_risk'] = (ptest >= threshold).astype(int)
    predictions.to_csv(OUT / 'holdout_predictions.csv', index=False, encoding='utf-8-sig')
    # Synthetic rows test the SAME application contract but never train/evaluate the model.
    rng = np.random.default_rng(SEED)
    synthetic = train[features].sample(120, replace=True, random_state=SEED).reset_index(drop=True)
    for feature in features:
        synthetic[feature] += rng.normal(0, max(float(train[feature].std()), .001) * .12, len(synthetic))
        if feature.endswith('_Std'):
            synthetic[feature] = synthetic[feature].clip(lower=0)
        elif 'OP' in feature or 'CP' in feature:
            synthetic[feature] = synthetic[feature].clip(lower=0, upper=100)
    synthetic.insert(0, 'batch_id', [f'SYN-{i+1:04}' for i in range(len(synthetic))])
    synthetic['process_start_at'] = pd.date_range('2026-10-01', periods=len(synthetic), freq='4h').astype(str)
    synthetic['process_end_at'] = (pd.to_datetime(synthetic.process_start_at) + pd.Timedelta(hours=3)).astype(str)
    synthetic['data_source'] = 'synthetic_kamp60'
    z1 = (synthetic[features[0]] - reference['mean'][features[0]]) / max(reference['std'][features[0]], .001)
    z2 = (synthetic[features[2]] - reference['mean'][features[2]]) / max(reference['std'][features[2]], .001)
    rate = np.clip(.0002 + .0002 * z1**2 + .00015 * np.maximum(z1*z2, 0) + rng.uniform(0, .0003, len(synthetic)), 0, .02)
    synthetic['total_count'] = 10000
    synthetic['defect_count'] = rng.binomial(10000, rate)
    synthetic['good_count'] = synthetic.total_count - synthetic.defect_count
    synthetic.to_csv(OUT / 'synthetic_demo.csv', index=False, encoding='utf-8-sig')
    (OUT / 'synthetic_definition.json').write_text(json.dumps({'seed': SEED, 'rows': len(synthetic),
        'source': 'synthetic_kamp60', 'purpose': 'Functional tests only; excluded from all model training/evaluation.',
        'method': 'Resample training feature vectors; independent 0.12 std jitter; clip negative std and OP/CP to physical bounds.',
        'quality_formula': 'p=clip(.0002+.0002*z1^2+.00015*max(z1*z2,0)+U(0,.0003),0,.02); defects=Binomial(10000,p)',
        'z_features': [features[0], features[2]]}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'selected_model': winner['model'], 'threshold': threshold, 'test': evaluation}, ensure_ascii=False))


if __name__ == '__main__':
    main()
