from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'준비패키지/kamp'

def main():
    audit=json.loads((DATA/'audit.json').read_text(encoding='utf-8'))
    p=pd.read_csv(DATA/'process_batches.csv'); q=pd.read_csv(DATA/'quality_batches.csv'); j=pd.read_csv(DATA/'joined_batches.csv')
    assert p.batch_id.is_unique and q.batch_id.is_unique and j.batch_id.is_unique
    assert set(p.batch_id)==set(q.batch_id)==set(j.batch_id) and len(j)==136
    assert (q['양품수량']+q['불량수량']==q['총수량']).all()
    assert np.allclose(q.defect_rate_percent,100*q['불량수량']/q['총수량'])
    merged=p.merge(q,on=['batch_id','data_source'],validate='one_to_one').sort_values('batch_id')
    j=j.sort_values('batch_id')
    assert np.allclose(merged[audit['features']].to_numpy(),j[audit['features']].to_numpy(),equal_nan=True)
    assert np.allclose(merged.defect_rate_percent,j.defect_rate_percent)
    hashes={}
    for name,expected in audit['files'].items():
        raw=ROOT/'참고자료/KAMP_열처리_품질보증/원본/data'/name
        if raw.exists():
            with raw.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            assert digest==expected['sha256'],name
            hashes[name]={'verified':True,'bytes':raw.stat().st_size,'sha256':digest}
        else: hashes[name]={'verified':False,'reason':'Original source not distributed; processed-data checks still ran.'}
    result={'status':'passed','joined_batches':136,'features':36,'one_to_one':True,'quantity_identity':True,
            'joined_feature_values_match':True,'raw_files':hashes}
    (ROOT/'docs/verification/data_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()
