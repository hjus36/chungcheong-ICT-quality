"""Create distribution assets without copying KAMP batch rows or changing models.

Synthetic examples use declared hand-set ranges, not resampled observations.
The public benchmark contains aggregate experiment statistics only.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
KAMP=ROOT/'준비패키지/kamp'
SHARE=KAMP/'share'

def main():
    SHARE.mkdir(exist_ok=True)
    features=json.loads((KAMP/'audit.json').read_text(encoding='utf-8'))['features']
    rng=np.random.default_rng(20261102)
    data={};ranges={}
    for name in features:
        if name.endswith('_Std'):
            lo,hi=(.001,.02) if 'CP' in name else (.05,5.)
        elif 'CP' in name: lo,hi=.3,.7
        elif 'OP' in name: lo,hi=20.,90.
        elif '소입로 온도' in name: lo,hi=820.,880.
        elif '솔트조' in name: lo,hi=310.,350.
        elif '컨베이어' in name: lo,hi=260.,310.
        elif '세정기' in name: lo,hi=50.,80.
        else: lo,hi=80.,120.
        ranges[name]=[lo,hi];data[name]=rng.uniform(lo,hi,120).round(6)
    df=pd.DataFrame(data)
    df.insert(0,'batch_id',[f'DEMO-{i+1:04}' for i in range(len(df))])
    df['process_start_at']=pd.date_range('2026-01-01',periods=len(df),freq='4h').astype(str)
    df['process_end_at']=(pd.to_datetime(df.process_start_at)+pd.Timedelta(hours=3)).astype(str)
    df['data_source']='synthetic_kamp60'
    # Independent virtual quality counts; never a training/evaluation target.
    df['total_count']=10000;df['defect_count']=rng.integers(0,15,len(df));df['good_count']=df.total_count-df.defect_count
    df.to_csv(SHARE/'synthetic_demo.csv',index=False,encoding='utf-8-sig')
    df.head(20).to_csv(SHARE/'test_upload.csv',index=False,encoding='utf-8-sig')
    definition=dict(version='standalone-demo-20261101',seed=20261102,rows=120,
        source='synthetic_kamp60',method='Independent uniform random values within hand-set demonstration ranges; no raw data or learned distribution is read.',
        ranges=ranges,quality='Independent uniform integer defect counts0..14 of10000; no physical quality relationship.',
        purpose='Functional demonstration and API regression only; not for model training, accuracy evaluation, or process settings.')
    (SHARE/'synthetic_definition.json').write_text(json.dumps(definition,ensure_ascii=False,indent=2),encoding='utf-8')
    private=KAMP/'release/benchmark.json'
    public=KAMP/'release/benchmark_public.json'
    if private.exists():
        report=json.loads(private.read_text(encoding='utf-8'))
        for feature in report['feature_analysis']:
            points=pd.DataFrame(feature['quality_relation']).sort_values('value')
            # Each point is the mean of >=5 training batches, never an individual row.
            groups=np.array_split(np.arange(len(points)),min(8,len(points)//5))
            feature['quality_relation']=[dict(value=round(float(points.iloc[g].value.mean()),6),
                defect_rate_percent=round(float(points.iloc[g].defect_rate_percent.mean()),6),count=len(g)) for g in groups]
        report['quality_relation_mode']='group_means_min5'
        report['distribution_notice']='Repository edition: aggregate training statistics; no batch-level KAMP observations or identifiers.'
        report['source_benchmark_sha256']=hashlib.sha256(private.read_bytes()).hexdigest()
        public.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    elif not public.exists():
        raise SystemExit('A trusted benchmark_public.json or local original benchmark.json is required.')
    print('Prepared120independent synthetic examples,20test rows and aggregate benchmark; deployed model unchanged.')

if __name__=='__main__': main()
