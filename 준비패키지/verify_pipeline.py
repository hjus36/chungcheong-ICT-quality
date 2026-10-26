from pathlib import Path
from io import BytesIO
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from fastapi.testclient import TestClient
from app import app,ROOT,CFG

client=TestClient(app)
checks=[]

def check(name,condition):
    assert condition,name
    checks.append(name)

def upload(name,mapping='{}'):
    return client.post('/predict',files={'file':(name,(ROOT/'data'/name).read_bytes(),'text/csv')},data={'mapping':mapping})

plain=upload('demo_upload.csv');check('20개 LOT 예측',plain.status_code==200 and plain.json()['predicted_rows']==20)
mapped=upload('demo_upload_korean.csv',(ROOT/'data/mapping_korean.json').read_text(encoding='utf-8'))
check('한글 컬럼 매핑 후 점수 동일',mapped.status_code==200 and mapped.json()['results']==plain.json()['results'])
bad=upload('demo_errors.csv');check('중복 포함 오류 6행 예측 보류',bad.status_code==200 and bad.json()['error_rows']==6 and bad.json()['predicted_rows']==0)
df=pd.read_csv(ROOT/'data/demo_upload.csv')
df['hardness_hrc']=999;df['lot_nonconforming']=1
extra=client.post('/predict',files={'file':('extra.csv',df.to_csv(index=False).encode(),'text/csv')})
check('검사결과 추가가 예측에 영향 없음',extra.status_code==200 and extra.json()['results']==plain.json()['results'])
leak=client.post('/predict',files={'file':('extra.csv',df.to_csv(index=False).encode(),'text/csv')},data={'mapping':json.dumps({'hardness_hrc':'austenitize_mean_c'})})
check('검사결과를 입력으로 매핑 차단',leak.status_code==422)
df.loc[0,'quench_oil_c']=None
optional=client.post('/predict',files={'file':('optional.csv',df.to_csv(index=False).encode(),'text/csv')})
check('선택 항목 결측은 학습 중앙값 대체 표시',optional.status_code==200 and any('quench_oil_c' in r['imputed_columns'] for r in optional.json()['results']))
check('불러오기 결과 동일',client.get('/runs/'+plain.json()['run_id']).json()==plain.json())
check('잘못된 매핑 422',upload('demo_upload.csv','[]').status_code==422)
check('잘못된 확장자 415',client.post('/predict',files={'file':('x.txt',b'x','text/plain')}).status_code==415)
check('없는 결과 404',client.get('/runs/does-not-exist').status_code==404)
splits=pd.read_csv(ROOT/'data/split_manifest.csv')
check('같은 열처리 작업이 분할 간 겹치지 않음',splits.groupby('heat_run_id')['split'].nunique().max()==1)
ins=pd.read_csv(ROOT/'data/inspection.csv');label=pd.read_csv(ROOT/'data/labels.csv').set_index('lot_id')
derived=((ins.hardness_hrc<60)|(ins.hardness_hrc>66)).groupby(ins.lot_id).any().astype(int)
check('5000개 검사값과 1000개 LOT 정답 일치',derived.equals(label.lot_nonconforming.sort_index()))
book=ROOT/'data/열처리_합성데이터.xlsx'
if book.exists():
    r=client.post('/predict',files={'file':(book.name,book.read_bytes(),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
    check('Excel 공정 시트 1000행 예측',r.status_code==200 and r.json()['predicted_rows']==1000)
report={'passed':len(checks),'checks':checks,'synthetic_only':True}
(ROOT/'results/verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'results/demo_prediction.json').write_text(json.dumps(plain.json(),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
