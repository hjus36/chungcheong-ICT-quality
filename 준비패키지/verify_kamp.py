from pathlib import Path
from io import BytesIO
import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app_kamp import app

ROOT=Path(__file__).resolve().parent/'kamp';checks=[];client=TestClient(app)
def call(df,mapping=None):
    return client.post('/predict',files={'file':('input.csv',df.to_csv(index=False).encode('utf-8-sig'),'text/csv')},data={'mapping':json.dumps(mapping or {},ensure_ascii=False)})
def ok(name,condition):
    assert condition,name
    checks.append(name)

df=pd.read_csv(ROOT/'demo_upload.csv');r=call(df);data=r.json()
ok('KAMP 작업 20건 예측·출처 표시',r.status_code==200 and data['predicted_rows']==20 and data['data_scope']=='KAMP_heat_quality_60')
scores={x['batch_id']:x['risk_score'] for x in data['results']}
mapped=call(df.rename(columns={'batch_id':'배정번호'}),{'배정번호':'batch_id'}).json()
ok('한글 작업번호 매핑 결과 동일',scores=={x['batch_id']:x['risk_score'] for x in mapped['results']})
with_labels=df.copy();with_labels['불량수량']=999;with_labels['defect_rate_percent']=100
ok('품질 정답 추가 시 점수 불변',scores=={x['batch_id']:x['risk_score'] for x in call(with_labels).json()['results']})
feat=json.loads((ROOT/'audit.json').read_text(encoding='utf-8'))['features'][0]
ok('품질 정답을 공정 입력으로 매핑 차단',call(with_labels,{'불량수량':feat}).status_code==422)
wrong=df.head(1).copy();wrong['data_source']='synthetic_demo'
ok('강구 합성자료와 KAMP 출처 혼용 차단',call(wrong).json()['error_rows']==1)
dup=pd.concat([df.head(1),df.head(1)],ignore_index=True)
ok('중복 작업 모두 보류',call(dup).json()['error_rows']==2)
saved=client.get('/runs/'+data['run_id']).json()
ok('저장한 예측 결과 재조회',saved==data)
manifest=pd.read_csv(ROOT/'split_manifest.csv');q=pd.read_csv(ROOT/'quality_batches.csv')
report=json.loads((ROOT/'benchmark.json').read_text(encoding='utf-8'))
expected=manifest.loc[manifest.split=='train','defect_rate_percent'].quantile(.75)
ok('위험 기준은 학습 구간에서만 계산',np.isclose(expected,report['defect_rate_threshold_percent']))
ok('정답은 불량수량·총수량에서 재현',np.allclose(q.defect_rate_percent,100*q['불량수량']/q['총수량']))
ok('3구간 작업번호 중복 없음',manifest.batch_id.is_unique and len(manifest)==136)
bad=df.head(1).copy();bad[feat]='not-a-number'
ok('문자형 센서 요약값 보류',call(bad).json()['error_rows']==1)
output={'checks_passed':len(checks),'checks':checks}
(ROOT/'verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=False))
