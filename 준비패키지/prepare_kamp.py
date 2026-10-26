"""Build batch-level features directly from the downloaded KAMP source files.

Source: 중소벤처기업부, KAMP, 열처리 품질보증 AI 데이터셋,
스마트제조혁신추진단(㈜임픽스), 2022.12.23., https://www.kamp-ai.kr/
No code supplied inside the downloaded archive is executed.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT.parent/'참고자료'/'KAMP_열처리_품질보증'/'원본'/'data'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=DEFAULT)
    args=ap.parse_args()
    out=ROOT/'kamp';out.mkdir(exist_ok=True)
    raw=args.source/'data.csv'
    quality_path=args.source/'열처리_품질데이터.xlsx'
    header=pd.read_csv(raw,encoding='cp949',nrows=0).columns.tolist()
    excluded=['TAG_MIN','배정번호','소입로 CP 모니터 값']
    sensors=[c for c in header if c not in excluded]
    totals={}; seen=set(); duplicate_count=0; rows=0; nulls=pd.Series(0,index=header,dtype='int64')
    invalid_time=0
    for chunk in pd.read_csv(raw,encoding='cp949',chunksize=100000):
        rows+=len(chunk); nulls+=chunk.isna().sum()
        hashes=pd.util.hash_pandas_object(chunk[['배정번호','TAG_MIN']],index=False).to_numpy()
        unique_hashes=set(map(int,hashes))
        duplicate_count+=len(hashes)-len(unique_hashes)+len(seen & unique_hashes)
        seen.update(unique_hashes)
        chunk['TAG_MIN']=pd.to_datetime(chunk['TAG_MIN'],errors='coerce')
        invalid_time+=int(chunk['TAG_MIN'].isna().sum())
        if chunk['배정번호'].isna().any():raise ValueError('Missing batch identifier')
        for bid,g in chunk.groupby('배정번호',sort=False):
            values=g[sensors].astype(float)
            count=values.count().to_numpy(dtype=float)
            means=values.mean().to_numpy()
            m2=((values-means)**2).sum().to_numpy()
            if bid not in totals:
                totals[bid]={'count':count,'mean':means,'m2':m2,'rows':len(g),
                             'start':g.TAG_MIN.min(),'end':g.TAG_MIN.max()}
            else:
                t=totals[bid];n=t['count']+count
                a=np.nan_to_num(t['mean']);b=np.nan_to_num(means);delta=b-a
                t['mean']=np.divide(t['count']*a+count*b,n,out=np.full_like(n,np.nan),where=n>0)
                t['m2']=t['m2']+m2+np.divide(delta**2*t['count']*count,n,out=np.zeros_like(n),where=n>0)
                t['count']=n;t['rows']+=len(g)
                t['start']=min(t['start'],g.TAG_MIN.min());t['end']=max(t['end'],g.TAG_MIN.max())
    if duplicate_count or invalid_time:
        raise ValueError(f'Raw key duplicates={duplicate_count}, invalid timestamps={invalid_time}; resolve before modeling')
    records=[]
    for bid,t in totals.items():
        r={'batch_id':int(bid),'process_start_at':t['start'],'process_end_at':t['end'],
           'sensor_rows':t['rows'],'data_source':'KAMP_heat_quality_60'}
        for i,c in enumerate(sensors):
            r[c+'_Avg']=t['mean'][i]
            r[c+'_Std']=np.sqrt(t['m2'][i]/(t['count'][i]-1)) if t['count'][i]>1 else np.nan
        r['missing_fraction']=float(1-t['count'].sum()/(t['rows']*len(sensors)))
        records.append(r)
    inputs=pd.DataFrame(records).sort_values(['process_end_at','batch_id']).reset_index(drop=True)
    q=pd.read_excel(quality_path).rename(columns={'배정번호':'batch_id'})
    assert q.batch_id.is_unique
    assert (q['양품수량']+q['불량수량']==q['총수량']).all()
    assert (q['총수량']>0).all() and (q['불량수량']>=0).all()
    assert set(inputs.batch_id)==set(q.batch_id)
    q['defect_rate_percent']=100*q['불량수량']/q['총수량']
    q['data_source']='KAMP_heat_quality_60'
    merged=inputs.merge(q,on=['batch_id','data_source'],validate='one_to_one')
    inputs.to_csv(out/'process_batches.csv',index=False,encoding='utf-8-sig')
    q.to_csv(out/'quality_batches.csv',index=False,encoding='utf-8-sig')
    merged.to_csv(out/'joined_batches.csv',index=False,encoding='utf-8-sig')
    features=[c for c in inputs if c.endswith(('_Avg','_Std'))]
    audit={'source_url':'https://www.kamp-ai.kr/aidataDetail?DATASET_SEQ=60','download_date':'2026-10-26',
           'process_rows':rows,'process_columns':len(header),'quality_rows':len(q),'quality_columns':7,
           'batch_count':len(inputs),'sensors':sensors,'features':features,'feature_count':len(features),
           'removed_sensor':'소입로 CP 모니터 값 (guide p.37; redundant measurement channel)',
           'raw_null_counts':{k:int(v) for k,v in nulls.items()},'duplicate_batch_timestamp_keys':duplicate_count,
           'invalid_timestamp_count':invalid_time,'batch_join':'136-to-136 one-to-one verified',
           'quantity_identity_valid':True,'process_start':str(inputs.process_start_at.min()),
           'process_end':str(inputs.process_end_at.max()),'quality_date_start':str(q['작업일'].min()),
           'quality_date_end':str(q['작업일'].max()),'hardness_measurements_available':False,
           'aggregation':'Within each batch, observed-value mean and sample std (ddof=1); no global imputation before split.',
           'risk_label_plan':'Training-only defect-rate third quartile; freeze and apply to validation and test.',
           'provided_train_csv':'Not used: starts after source preprocessing/splitting and lacks batch/time identifiers.',
           'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in [raw,quality_path]}}
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['process_rows','process_columns','batch_count','feature_count','process_start','process_end','duplicate_batch_timestamp_keys']},ensure_ascii=False))

if __name__=='__main__':main()
