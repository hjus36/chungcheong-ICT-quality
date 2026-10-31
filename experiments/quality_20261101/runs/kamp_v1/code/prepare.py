"""Snapshot baseline, audit original labels, and extract label-free temporal features."""
import argparse
import shutil
import json
import sys
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from common import ROOT,HERE,KAMP,sha,save_json,metrics,compact_features

def snapshot():
    dest=HERE/'baseline_snapshot'
    if dest.exists(): return
    dest.mkdir(parents=True)
    paths=[ROOT/'README.md', ROOT/'docs', KAMP/'release', KAMP/'pilot_model.joblib',
           KAMP/'benchmark.json', KAMP/'holdout_predictions.csv', KAMP/'runs.sqlite3']
    hashes={}
    for p in paths:
        if not p.exists(): continue
        target=dest/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if p.is_dir(): shutil.copytree(p,target)
        else: shutil.copy2(p,target)
        for f in p.rglob('*') if p.is_dir() else [p]:
            if f.is_file(): hashes[str(f.relative_to(ROOT))]=sha(f)
    save_json(dest/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),sha256=hashes))

def audit():
    snapshot()
    a=json.loads((KAMP/'audit.json').read_text(encoding='utf-8'))
    d=pd.read_csv(KAMP/'joined_batches.csv',dtype={'batch_id':str}).sort_values('process_end_at')
    splits=pd.read_csv(KAMP/'split_manifest.csv',dtype={'batch_id':str})[['batch_id','split']]
    d=d.merge(splits,on='batch_id',validate='one_to_one')
    train=d[d.split=='train']; features=a['features']
    q=float(train.defect_rate_percent.quantile(.75))
    original=ROOT/'참고자료/KAMP_열처리_품질보증/원본/data'
    quality=pd.read_excel(original/'열처리_품질데이터.xlsx').rename(columns={'배정번호':'batch_id'})
    quality.batch_id=quality.batch_id.astype(str)
    joined=d.merge(quality,on='batch_id',validate='one_to_one',suffixes=('','_raw'))
    assert set(d.batch_id)==set(quality.batch_id) and len(joined)==136
    for c in ['양품수량','불량수량','총수량']:
        assert np.array_equal(joined[c],joined[c+'_raw'])
    assert (d['양품수량']+d['불량수량']==d['총수량']).all()
    model=joblib.load(KAMP/'release/model.joblib')
    reproduced={}
    for name in ['validation','test']:
        sub=d[d.split==name]; y=(sub.defect_rate_percent>=q).astype(int)
        p=model['pipeline'].predict_proba(sub[features])[:,1]
        reproduced[name]=metrics(y,p,model['alert_threshold'])
    old=json.loads((KAMP/'release/benchmark.json').read_text(encoding='utf-8'))
    for key in ['recall','precision','f1','f2','roc_auc','average_precision']:
        assert abs(reproduced['test'][key]-old['test'][key])<1e-12,key
    corr=train[features].corr(method='spearman').abs()
    correlated=[dict(a=features[i],b=features[j],abs_spearman=corr.iloc[i,j])
        for i in range(len(features)) for j in range(i) if corr.iloc[i,j]>=.95]
    drift={}
    for name in ['train','validation','test']:
        x=d[d.split==name]; y=x.defect_rate_percent>=q
        denom=train[features].std().replace(0,np.nan)
        shift=((x[features].mean()-train[features].mean())/denom).abs().dropna().sort_values(ascending=False)
        outlier=((x[features]<train[features].quantile(.25)-3*(train[features].quantile(.75)-train[features].quantile(.25)))|
                 (x[features]>train[features].quantile(.75)+3*(train[features].quantile(.75)-train[features].quantile(.25))))
        drift[name]=dict(n=len(x),positive=int(y.sum()),prevalence=y.mean(),
            defect_rate_percent=x.defect_rate_percent.describe().to_dict(),
            mean_shift_in_train_std=shift.head(8).to_dict(),rows_outside_train_3iqr=int(outlier.any(axis=1).sum()),
            missing_cells=int(x[features].isna().sum().sum()),start=x.process_start_at.min(),end=x.process_end_at.max())
    # Label uncertainty is illustrative: Wilson intervals assume independent Bernoulli items,
    # whereas parts within a production batch may be correlated.
    n=d['총수량'].to_numpy(); k=d['불량수량'].to_numpy(); rate=k/n; z=1.96
    centre=(rate+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    intervals=(centre-half<=q/100)&(centre+half>=q/100)
    changes=((100*np.maximum(k-1,0)/n>=q)!=(rate*100>=q))|((100*(k+1)/n>=q)!=(rate*100>=q))
    report=dict(source=a['source_url'],independent_quality_rows=len(d),sensor_rows=a['process_rows'],
        original_files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in [original/'data.csv',original/'열처리_품질데이터.xlsx']},
        joined_sha256=sha(KAMP/'joined_batches.csv'),model_sha256=sha(KAMP/'release/model.joblib'),
        join_verified=True,quantity_identity_verified=True,duplicate_feature_rows=int(d[features].duplicated().sum()),
        raw_missing_counts=a['raw_null_counts'],risk_boundary_percent=q,split=drift,
        high_correlation_pairs=correlated,quantity_summary=d['총수량'].describe().to_dict(),
        one_defect_label_flip_rows=int(changes.sum()),wilson_interval_crossing_boundary=int(intervals.sum()),
        wilson_caveat='Descriptive binomial approximation; within-batch correlation unmeasured.',
        date_alignment_counts=(pd.to_datetime(d['작업일']).dt.normalize()==pd.to_datetime(d.process_start_at).dt.normalize()).value_counts().to_dict(),
        reproduced=reproduced,
        limitations=['136 batch labels are an upper bound on independent samples; shared product/equipment dependence unknown.',
                    'No inspection timestamp: training label availability before next production is not verifiable.',
                    'Q3 is relative prioritization, not factory pass/fail. Full-batch features imply end-of-process use.',
                    'Test36 was previously inspected. Descriptive drift audit is not a fresh validation.'])
    # JSON object keys must be strings (numpy bool keys are not serializable).
    report['date_alignment_counts']={str(k):int(v) for k,v in report['date_alignment_counts'].items()}
    save_json(HERE/'results/data_audit.json',report)
    compact_features(d).assign(batch_id=d.batch_id).to_csv(HERE/'results/compact_features.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({k:report[k] for k in ['independent_quality_rows','one_defect_label_flip_rows','wilson_interval_crossing_boundary']},ensure_ascii=False))

def temporal():
    out=HERE/'results/temporal_features.csv'
    if out.exists(): raise FileExistsError('Temporal artifact exists; use a new experiment directory to rerun extraction.')
    source=ROOT/'참고자료/KAMP_열처리_품질보증/원본/data/data.csv'
    groups={'quench':[f'소입로 온도 {i} Zone' for i in range(1,5)],
        'salt':[f'솔트조 온도 {i} Zone' for i in [1,2]],
        'dryer':[f'건조로 온도 {i} Zone' for i in [1,2]],
        'conveyor':[f'솔트 컨베이어 온도 {i} Zone' for i in [1,2]],'cp':['소입로 CP 값']}
    parts={}; rows=0
    cols=['TAG_MIN','배정번호']+sum(groups.values(),[])
    for chunk in pd.read_csv(source,encoding='cp949',usecols=cols,chunksize=150000):
        small=pd.DataFrame({k:chunk[v].mean(axis=1) for k,v in groups.items()})
        small['time']=pd.to_datetime(chunk.TAG_MIN); small['batch_id']=chunk['배정번호'].astype(str)
        for bid,g in small.groupby('batch_id',sort=False): parts.setdefault(bid,[]).append(g.drop(columns='batch_id'))
        rows+=len(chunk)
    records=[]
    for bid,frames in parts.items():
        g=pd.concat(frames,ignore_index=True).sort_values('time')
        hours=(g.time-g.time.min()).dt.total_seconds()/3600
        phase=hours/max(hours.max(),1e-9)
        record={'batch_id':bid}
        for name in groups:
            v=g[name]; ok=v.notna(); x=hours[ok].to_numpy(); values=v[ok].to_numpy()
            early=v[phase<=.2].mean(); late=v[phase>=.8].mean()
            record.update({name+'_min':v.min(),name+'_max':v.max(),name+'_p95_p05':v.quantile(.95)-v.quantile(.05),
                name+'_early_mean':early,name+'_late_mean':late,name+'_late_minus_early':late-early,
                name+'_trend_per_hour':float(np.polyfit(x-x.mean(),values,1)[0]) if len(x)>1 and np.ptp(x)>0 else 0.})
        records.append(record)
    pd.DataFrame(records).sort_values('batch_id').to_csv(out,index=False,encoding='utf-8-sig')
    save_json(HERE/'results/temporal_manifest.json',dict(rows_read=rows,batches=len(records),
        features=35,groups=groups,source_sha256=sha(source),output_sha256=sha(out),
        definition='Group sensor mean per timestamp; min/max/p95-p05; first/last20% elapsed-time means; late-early; least-squares slope per hour.',
        stage_caveat='Elapsed-time windows only; no annotated metallurgical stages or setpoints are supplied.',
        input_timing='All input measurements precede batch completion; inspection availability is unknown. Not real-time early prediction.'))
    print(f'Extracted {len(records)} batches, 35 temporal features from {rows} sensor rows',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--temporal',action='store_true');args=p.parse_args()
    temporal() if args.temporal else audit()
