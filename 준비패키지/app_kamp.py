"""KAMP batch upload/prediction prototype. Run uvicorn app_kamp:app locally."""
from pathlib import Path
from functools import lru_cache
import json, sqlite3, uuid
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from app import frame

ROOT=Path(__file__).resolve().parent/'kamp'
AUDIT=json.loads((ROOT/'audit.json').read_text(encoding='utf-8'))
FEATURES=AUDIT['features'];META=['batch_id','process_start_at','process_end_at','data_source']
app=FastAPI(title='무한루프 KAMP 염욕 열처리 분석 시제품',version='0.2.0')
NOTICE='실측자료 분석 시제품입니다. 최종 평가 재현율 28.6%로 성능 개선이 필요하며 출하판정·설비제어에 사용하지 않습니다.'

@lru_cache
def bundle():return joblib.load(ROOT/'pilot_model.joblib')

@app.get('/health')
def health():return {'status':'ok','data_scope':'KAMP_heat_quality_60','readiness':'research_prototype','notice':NOTICE}

@app.get('/schema')
def schema():return {'required_columns':META+FEATURES,'data_source':'KAMP_heat_quality_60','input':'작업번호별 공정 요약 CSV 또는 Excel, 최대 10MB','reference_range':'학습자료의 5~95백분위 범위이며 정상 운전 규격이 아님'}

@app.get('/model')
def model_info():return json.loads((ROOT/'benchmark.json').read_text(encoding='utf-8'))

@app.post('/preview')
async def preview(file:UploadFile=File(...)):
    df=await frame(file)
    return {'columns':list(df.columns),'rows':len(df),'sample':json.loads(df.head(3).to_json(orient='records'))}

@app.post('/predict')
async def predict(file:UploadFile=File(...),mapping:str=Form('{}')):
    df=await frame(file)
    try:
        links=json.loads(mapping)
        if not isinstance(links,dict) or not all(isinstance(k,str) and isinstance(v,str) and k in df.columns and v in META+FEATURES for k,v in links.items()):raise ValueError()
    except (ValueError,TypeError):raise HTTPException(422,'매핑은 원본 열 이름을 공통 입력 항목에 연결하는 JSON 객체여야 합니다.')
    for src,dst in links.items():
        if dst in FEATURES and any(s in src.lower() for s in ['불량','양품','총수량','정답','판정','defect','label','risk','hardness','경도']):
            raise HTTPException(422,'품질 정답을 공정 입력으로 연결할 수 없습니다.')
    df=df.rename(columns=links)
    if df.columns.duplicated().any():raise HTTPException(422,'매핑 후 열 이름이 중복됩니다.')
    missing=[c for c in META+FEATURES if c not in df]
    if missing:raise HTTPException(422,{'missing_columns':missing})
    duplicates=df.batch_id.astype(str).duplicated(keep=False)
    valid=[];errors=[];imputed={};numeric=df[FEATURES].apply(pd.to_numeric,errors='coerce')
    for i,r in df.iterrows():
        reasons=[]
        if pd.isna(r.batch_id) or not str(r.batch_id).strip():reasons.append('batch_id 누락')
        if duplicates.iloc[i]:reasons.append('batch_id 중복')
        if r.data_source!='KAMP_heat_quality_60':reasons.append('데이터 출처·공정이 현재 모델과 다름')
        start=pd.to_datetime(r.process_start_at,errors='coerce',utc=True);end=pd.to_datetime(r.process_end_at,errors='coerce',utc=True)
        if pd.isna(start) or pd.isna(end) or end<start:reasons.append('공정 시작·종료 시각 오류')
        missing_cells=[]
        for c in FEATURES:
            v=numeric.loc[i,c]
            if pd.isna(v):
                if pd.notna(r[c]):reasons.append(c+': 숫자 형식 오류')
                else:missing_cells.append(c)
            elif not np.isfinite(v) or (c.endswith('_Std') and v<0):reasons.append(c+': 유효하지 않은 수치')
        if len(missing_cells)>3:reasons.append('공정 요약값 결측 과다(4개 이상)')
        if reasons:errors.append({'row':int(i)+2,'reasons':reasons})
        else:valid.append(i);imputed[i]=missing_cells
    model=bundle();results=[]
    if valid:
        scores=model['pipeline'].predict_proba(numeric.loc[valid,FEATURES])[:,1]
        for i,score in zip(valid,scores):
            deviations=[]
            for c in FEATURES:
                value=numeric.loc[i,c];ref=model['reference']
                if pd.notna(value) and (value<ref['q05'][c] or value>ref['q95'][c]):
                    deviations.append({'feature':c,'value':float(value),'train_q05':ref['q05'][c],'train_q95':ref['q95'][c]})
            results.append({'batch_id':str(df.loc[i,'batch_id']),'risk_score':round(float(score),6),
                            'review_priority':'우선 검토' if score>=model['alert_threshold'] else '통상 검토',
                            'imputed_columns':imputed[i],'outside_training_reference':deviations})
    payload={'run_id':str(uuid.uuid4()),'input_rows':len(df),'predicted_rows':len(valid),'error_rows':len(errors),
             'model_version':model['version'],'data_scope':model['data_source'],'notice':NOTICE,
             'alert_threshold':model['alert_threshold'],'defect_rate_threshold_percent':model['defect_rate_threshold_percent'],
             'ignored_columns':[c for c in df if c not in META+FEATURES],
             'results':sorted(results,key=lambda x:x['risk_score'],reverse=True),'errors':errors}
    with sqlite3.connect(ROOT/'runs.sqlite3') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        conn.execute('INSERT INTO runs VALUES (?,?)',(payload['run_id'],json.dumps(payload,ensure_ascii=False)))
    return payload

@app.get('/runs/{run_id}')
def saved_run(run_id:str):
    if not (ROOT/'runs.sqlite3').exists():raise HTTPException(404,'저장된 결과가 없습니다.')
    with sqlite3.connect(ROOT/'runs.sqlite3') as conn:r=conn.execute('SELECT payload FROM runs WHERE id=?',(run_id,)).fetchone()
    if r is None:raise HTTPException(404,'저장된 결과가 없습니다.')
    return json.loads(r[0])
