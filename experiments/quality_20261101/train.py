"""Bounded experiments. Selection locks precede validation/test evaluation.

Reruns require a NEW --run directory. No deployed file is written by this script.
"""
import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from common import (ROOT,HERE,KAMP,SEED,sha,save_json,metrics,choose_threshold,
                    calibration,StableColumns,TopFeatures,compact_features)

def pipeline(family, value, y, k, baseline=False):
    if family=='LogisticRegression':
        estimator=LogisticRegression(C=value,class_weight='balanced',max_iter=3000,random_state=SEED)
    elif family=='RandomForest':
        estimator=RandomForestClassifier(n_estimators=200,max_depth=int(value),min_samples_leaf=6,
            class_weight='balanced',random_state=SEED,n_jobs=2)
    else:
        estimator=XGBClassifier(n_estimators=80,max_depth=int(value),learning_rate=.05,
            min_child_weight=3,reg_lambda=5,scale_pos_weight=float((y==0).sum()/(y==1).sum()),
            random_state=SEED,n_jobs=2,eval_metric='logloss')
    steps=[] if baseline else [('prune',StableColumns())]
    steps.append(('impute',SimpleImputer(strategy='median')))
    if not baseline: steps.append(('select',TopFeatures(k)))
    steps.extend([('scale',StandardScaler() if family=='LogisticRegression' else 'passthrough'),('model',estimator)])
    return Pipeline(steps)

def paired_day_interval(y, candidate, baseline, days, repetitions=2000):
    y=np.asarray(y); candidate=np.asarray(candidate); baseline=np.asarray(baseline);days=np.asarray(days)
    unique=np.unique(days);rng=np.random.default_rng(SEED);values=[]
    for _ in range(repetitions):
        selected=rng.choice(unique,len(unique),replace=True)
        idx=np.concatenate([np.flatnonzero(days==d) for d in selected])
        if len(np.unique(y[idx]))<2: continue
        values.append(average_precision_score(y[idx],candidate[idx])-average_precision_score(y[idx],baseline[idx]))
    return dict(method='paired calendar-day block bootstrap AP difference; descriptive, not independent external evidence',
        repetitions=repetitions,valid=len(values),lower=float(np.quantile(values,.025)),upper=float(np.quantile(values,.975)))

def prepare_kamp(out):
    d=pd.read_csv(KAMP/'joined_batches.csv',dtype={'batch_id':str}).sort_values(['process_end_at','batch_id']).reset_index(drop=True)
    manifest=pd.read_csv(KAMP/'split_manifest.csv',dtype={'batch_id':str})[['batch_id','split']]
    d=d.merge(manifest,on='batch_id',validate='one_to_one')
    audit=json.loads((KAMP/'audit.json').read_text(encoding='utf-8'))
    compact=compact_features(d)
    temporal=pd.read_csv(HERE/'results/temporal_features.csv',dtype={'batch_id':str}).set_index('batch_id').loc[d.batch_id].reset_index(drop=True)
    features={'baseline36':d[audit['features']], 'compact16':compact,
              'compact16_plus_temporal35':pd.concat([compact,temporal],axis=1)}
    train_idx=np.flatnonzero(d.split=='train')
    assert np.array_equal(train_idx,np.arange(78))
    q=float(d.loc[train_idx,'defect_rate_percent'].quantile(.75))
    y=(d.defect_rate_percent>=q).astype(int)
    folds=[]
    for i,(tr,va) in enumerate(TimeSeriesSplit(n_splits=3).split(train_idx)):
        tr=tr[(d.loc[tr,'process_end_at']<d.loc[va,'process_start_at'].min()).to_numpy()]
        boundary=float(d.loc[tr,'defect_rate_percent'].quantile(.75))
        a=(d.loc[tr,'defect_rate_percent']>=boundary).astype(int)
        b=(d.loc[va,'defect_rate_percent']>=boundary).astype(int)
        folds.append((tr,va,a,b,boundary))
    ids=d.batch_id; days=pd.to_datetime(d.process_end_at).dt.strftime('%Y-%m-%d')
    d[['batch_id','process_start_at','process_end_at','split']].to_csv(out/'split_manifest.csv',index=False)
    save_json(out/'data_manifest.json',dict(source='KAMP_heat_quality_60',rows=136,train=78,validation=22,test=36,
        joined_sha256=sha(KAMP/'joined_batches.csv'),temporal_sha256=sha(HERE/'results/temporal_features.csv'),
        label_threshold_percent=q,test_status='Previously inspected development reference',
        feature_sets={k:list(v.columns) for k,v in features.items()}))
    return d.split,features,y,folds,ids,days,q

def prepare_secom(out):
    source=HERE/'sources/uci_secom'
    # Read dates and row IDs first; freeze split independently of labels and features.
    lines=(source/'secom_labels.data').read_text().splitlines()
    timestamps=pd.to_datetime([s.split(maxsplit=1)[1].strip().strip('"') for s in lines],format='%d/%m/%Y %H:%M:%S')
    meta=pd.DataFrame(dict(entity_id=np.arange(1,len(lines)+1),test_time=timestamps))
    meta=meta.sort_values(['test_time','entity_id']).reset_index(drop=True)
    days=meta.test_time.dt.strftime('%Y-%m-%d'); unique=np.sort(days.unique())
    a,b=int(len(unique)*.6),int(len(unique)*.8)
    split=pd.Series(np.where(days.isin(unique[:a]),'train',np.where(days.isin(unique[a:b]),'validation','test')))
    meta['split']=split
    meta.to_csv(out/'split_manifest.csv',index=False)
    save_json(out/'split_lock.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        method='Chronological 60/20/20 percent of unique days, floor boundaries; frozen before feature/label audit.',
        sha256=sha(out/'split_manifest.csv'),cutoff_validation=unique[a],cutoff_test=unique[b]))
    raw=pd.read_csv(source/'secom.data',sep=r'\s+',header=None)
    assert len(raw)==len(lines) and raw.shape[1]==590
    raw.columns=[f'signal_{i:03}' for i in range(1,591)]
    order=meta.entity_id.to_numpy()-1
    X=raw.iloc[order].reset_index(drop=True)
    labels=np.array([int(s.split(maxsplit=1)[0]) for s in lines])
    assert set(labels)=={-1,1}
    y=pd.Series((labels[order]==1).astype(int))
    train_idx=np.flatnonzero(split=='train'); train_days=np.sort(days.iloc[train_idx].unique())
    folds=[]
    for trday,vaday in TimeSeriesSplit(n_splits=3).split(train_days):
        tr=np.flatnonzero(days.isin(train_days[trday])); va=np.flatnonzero(days.isin(train_days[vaday]))
        folds.append((tr,va,y.iloc[tr],y.iloc[va],None))
    # Do not compute test metrics or class statistics until the trained selection lock exists.
    trainX=X.iloc[train_idx]
    save_json(out/'data_manifest.json',dict(source='UCI_SECOM_179',source_files=json.loads((source/'provenance.json').read_text())['files'],
        rows=len(X),actual_signal_columns=X.shape[1],website_reported_features=591,
        discrepancy='Downloaded numeric matrix has590 columns; preserve raw data, do not invent signal591.',
        split_rows=split.value_counts().to_dict(),training_missing_cells=int(trainX.isna().sum().sum()),
        training_all_missing_columns=int(trainX.isna().all().sum()),training_constant_columns=int((trainX.nunique(dropna=True)<=1).sum()),
        train_positives=int(y.iloc[train_idx].sum()),
        duplicate_feature_rows=int(X.duplicated().sum()),
        sample_unit='Production entity; factory/lot identifiers absent, independence of entities unproven.',
        timing='Timestamp is in-house test time. Sensor-to-test ordering is not documented; no early-warning claim.',
        semantics='Anonymized signals with unspecified units; not suitable for physical explanations or KAMP input.',
        label='Original line-test pass(-1)->0 and fail(+1)->1; no synthetic or proxy labels.'))
    return split,{'secom590':X},y,folds,meta.entity_id,days,None

def run(dataset, name):
    out=HERE/'runs'/name
    out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir()
    (out/'code').mkdir()
    for source in HERE.glob('*.py'):
        shutil.copy2(source,out/'code'/source.name)
    protocol=json.loads((HERE/'protocol.json').read_text())
    save_json(out/'protocol.json',protocol)
    save_json(out/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        sklearn=sklearn.__version__,xgboost=xgboost.__version__,seed=SEED,
        source_code_sha256={p.name:sha(p) for p in HERE.glob('*.py')},created_utc=datetime.now(timezone.utc).isoformat()))
    split,features,y,folds,ids,days,q=(prepare_kamp(out) if dataset=='kamp' else prepare_secom(out))
    train_idx=np.flatnonzero(split=='train'); val_idx=np.flatnonzero(split=='validation');test_idx=np.flatnonzero(split=='test')
    configs=[]
    for fs in features:
        for family,values in [('LogisticRegression',[.01,.1]),('RandomForest',[2,3]),('XGBoost',[1,2])]:
            for v in values: configs.append(dict(id=f'{fs}_{family}_{v}',features=fs,family=family,value=v,baseline=False))
    if dataset=='kamp': configs.append(dict(id='deployed_RF_protocol_control',features='baseline36',family='RandomForest',value=2,baseline=True))
    save_json(out/'candidate_plan.json',configs)
    rows=[];models={};oofrecords=[]
    for c in configs:
        X=features[c['features']]; foldstats=[];yy=[];pp=[]
        for i,(tr,va,a,b,boundary) in enumerate(folds):
            if len(np.unique(a))<2 or len(np.unique(b))<2:
                raise ValueError('Single-class CV fold; protocol revision required before training')
            pipe=pipeline(c['family'],c['value'],a,40 if dataset=='secom' else 8,c['baseline'])
            pipe.fit(X.iloc[tr],a);p=pipe.predict_proba(X.iloc[va])[:,1]
            yy.extend(b);pp.extend(p)
            foldstats.append(dict(fold=i,train=len(tr),validation=len(va),train_positive=int(a.sum()),
                validation_positive=int(b.sum()),label_boundary=boundary,
                average_precision=average_precision_score(b,p),
                no_skill_ap=float(b.mean()),train_end=str(days.iloc[tr].max()),validation_start=str(days.iloc[va].min())))
            oofrecords.extend(dict(candidate=c['id'],fold=i,id=str(ids.iloc[ix]),label=int(label),score=float(score))
                for ix,label,score in zip(va,b,p))
        threshold=choose_threshold(yy,pp)
        fitted=pipeline(c['family'],c['value'],y.iloc[train_idx],40 if dataset=='secom' else 8,c['baseline'])
        fitted.fit(X.iloc[train_idx],y.iloc[train_idx])
        model_path=out/'models'/f"{c['id']}.joblib"
        joblib.dump(dict(pipeline=fitted,features=list(X.columns),threshold=threshold,dataset=dataset,
            selected_config=c,protocol_sha256=sha(out/'protocol.json'),label_threshold_percent=q),model_path)
        selected_indices=np.arange(X.shape[1])
        if 'prune' in fitted.named_steps: selected_indices=selected_indices[fitted.named_steps['prune'].indices_]
        if 'select' in fitted.named_steps: selected_indices=selected_indices[fitted.named_steps['select'].indices_]
        result={**c,'input_features':X.shape[1],'retained_features':list(X.columns[selected_indices]),
            'cv_mean_ap':float(np.mean([f['average_precision'] for f in foldstats])),
            'cv_mean_ap_lift':float(np.mean([f['average_precision']-f['no_skill_ap'] for f in foldstats])),
            'folds':foldstats,'oof':metrics(yy,pp,threshold),'threshold':threshold,'model_sha256':sha(model_path)}
        rows.append(result);models[c['id']]=fitted
        print(f"{dataset}: {c['id']} CV_AP={result['cv_mean_ap']:.4f} threshold={threshold:.3f}",flush=True)
    # Baseline is a reference, not an eligible new candidate.
    candidates=[r for r in rows if not r['baseline']]
    winner=sorted(candidates,key=lambda r:(-r['cv_mean_ap'],r['input_features'],r['id']))[0]
    save_json(out/'cv_results.json',rows)
    pd.DataFrame(oofrecords).to_csv(out/'oof_predictions.csv',index=False)
    lock=dict(winner=winner['id'],model_sha256=winner['model_sha256'],threshold=winner['threshold'],
        rule=protocol[dataset]['selection'],created_utc=datetime.now(timezone.utc).isoformat(),
        validation_metrics_not_yet_computed=True,test_metrics_not_yet_computed=True)
    save_json(out/'selection_lock.json',lock)
    validation=[];predictions=[]
    for r in rows:
        p=models[r['id']].predict_proba(features[r['features']].iloc[val_idx])[:,1]
        validation.append(dict(id=r['id'],**metrics(y.iloc[val_idx],p,r['threshold']),calibration=calibration(y.iloc[val_idx],p)))
        predictions.extend(dict(candidate=r['id'],id=str(ids.iloc[ix]),label=int(label),score=float(score))
            for ix,label,score in zip(val_idx,y.iloc[val_idx],p))
    yp=y.iloc[val_idx];pw=models[winner['id']].predict_proba(features[winner['features']].iloc[val_idx])[:,1]
    winmetrics=metrics(yp,pw,winner['threshold'])
    baseval={name:metrics(yp,np.full(len(yp),score),.5) for name,score in
        [('all_normal',0),('all_risk',1),('train_prevalence',float(y.iloc[train_idx].mean()))]}
    if dataset=='kamp':
        deployed=joblib.load(KAMP/'release/model.joblib')
        pb=deployed['pipeline'].predict_proba(features['baseline36'].iloc[val_idx])[:,1]
        baseval['deployed']=metrics(yp,pb,deployed['alert_threshold'])
        interval=paired_day_interval(yp,pw,pb,days.iloc[val_idx])
        bm=baseval['deployed'];wm=winmetrics
        gates=dict(ap=wm['average_precision']>=bm['average_precision']+.05,f2=wm['f2']>=bm['f2'],
            recall=wm['recall']>=.75,precision=wm['precision']>=bm['precision']+.1,
            workload=wm['alert_rate']<=.5,paired_ap_interval=interval['lower']>0,
            original_input_contract=winner['features'] in ['baseline36','compact16'])
        decision=dict(action='eligible_for_integration_review' if all(gates.values()) else 'retain_deployed_RF',
            gates=gates,validation_ap_difference_ci=interval,
            active_model_sha256=sha(KAMP/'release/model.joblib'),winner=winner['id'])
    else:
        decision=dict(action='separate_research_model_only',winner=winner['id'],
            reason='Different process and anonymized inputs; cannot replace KAMP web model.')
    # Decision is persisted before any new candidate test score is computed.
    save_json(out/'deployment_decision.json',decision)
    save_json(out/'validation.json',dict(candidates=validation,baselines=baseval))
    pd.DataFrame(predictions).to_csv(out/'validation_predictions.csv',index=False)
    testp=models[winner['id']].predict_proba(features[winner['features']].iloc[test_idx])[:,1]
    testy=y.iloc[test_idx]
    testmetrics=metrics(testy,testp,winner['threshold'])
    testbase={name:metrics(testy,np.full(len(testy),score),.5) for name,score in
        [('all_normal',0),('all_risk',1),('train_prevalence',float(y.iloc[train_idx].mean()))]}
    if dataset=='kamp':
        pb=deployed['pipeline'].predict_proba(features['baseline36'].iloc[test_idx])[:,1]
        testbase['deployed']=metrics(testy,pb,deployed['alert_threshold'])
    report=dict(dataset=dataset,winner=winner['id'],threshold=winner['threshold'],validation=winmetrics,
        test=testmetrics,test_baselines=testbase,test_calibration=calibration(testy,testp),decision=decision,
        test_status='Previously seen development reference' if dataset=='kamp' else 'First evaluation of chronological same-dataset holdout; not external-factory validation',
        selection_lock_sha256=sha(out/'selection_lock.json'),split_manifest_sha256=sha(out/'split_manifest.csv'),
        deployment_decision_sha256=sha(out/'deployment_decision.json'))
    save_json(out/'final_results.json',report)
    pd.DataFrame(dict(id=ids.iloc[test_idx].to_numpy(),day=days.iloc[test_idx].to_numpy(),label=testy.to_numpy(),
        score=testp,predicted=(testp>=winner['threshold']).astype(int))).to_csv(out/'test_predictions.csv',index=False)
    print(json.dumps(dict(dataset=dataset,winner=winner['id'],validation=winmetrics,test=testmetrics,decision=decision),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['kamp','secom'],required=True)
    p.add_argument('--run',required=True);args=p.parse_args();run(args.dataset,args.run)
