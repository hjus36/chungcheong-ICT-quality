"""Local demo API. Start: python -m uvicorn app:app --host 127.0.0.1 --port 8765"""
from pathlib import Path
from io import BytesIO
from functools import lru_cache
import json, sqlite3, uuid
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
FEATURES=CFG['features']
META=['lot_id','heat_run_id','process_end_at','material_grade','diameter_mm']
app=FastAPI(title='무한루프 열처리 품질위험 시연 API',version='0.1.0')

@lru_cache
def bundle():
    # Load only the model generated inside this package. Uploaded models are never read.
    return joblib.load(ROOT/'models/selected_model.joblib')

async def frame(file):
    raw=await file.read(10*1024*1024+1)
    if len(raw)>10*1024*1024:raise HTTPException(413,'파일은 10MB 이하만 허용합니다.')
    suffix=Path(file.filename or '').suffix.lower()
    try:
        if suffix=='.csv':df=pd.read_csv(BytesIO(raw),encoding='utf-8-sig',dtype={'lot_id':'string','heat_run_id':'string'})
        elif suffix=='.xlsx':df=pd.read_excel(BytesIO(raw),engine='openpyxl')
        else:raise HTTPException(415,'UTF-8 CSV 또는 xlsx 파일을 사용하세요.')
    except HTTPException:raise
    except Exception:raise HTTPException(422,'파일을 읽지 못했습니다. 형식과 UTF-8 인코딩을 확인하세요.')
    if len(df)==0 or len(df)>10000:raise HTTPException(422,'입력은 1~10,000행이어야 합니다.')
    if len(df.columns)>100:raise HTTPException(422,'입력은 100열 이하여야 합니다.')
    return df

@app.get('/health')
def health():return {'status':'ok','data_scope':'synthetic_demo','frontend_status':'React 화면은 후속 구현'}

@app.get('/schema')
def schema():return {'required_columns':META+FEATURES,'config':CFG}

@app.get('/model')
def model_info():return json.loads((ROOT/'results/benchmark.json').read_text(encoding='utf-8'))

@app.post('/preview')
async def preview(file:UploadFile=File(...)):
    df=await frame(file)
    return {'columns':list(df.columns),'rows':len(df),'sample':json.loads(df.head(3).to_json(orient='records'))}

@app.post('/predict')
async def predict(file:UploadFile=File(...),mapping:str=Form('{}')):
    df=await frame(file)
    try:
        mapping_obj=json.loads(mapping)
        if not isinstance(mapping_obj,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in mapping_obj.items()):raise ValueError()
        if any(k not in df.columns for k in mapping_obj):raise ValueError()
        if any(v not in META+FEATURES+['recipe_id','data_source'] for v in mapping_obj.values()):raise ValueError()
    except (ValueError,TypeError):raise HTTPException(422,'매핑은 존재하는 원본 열 이름을 공통 입력 열 이름에 연결하는 JSON 객체여야 합니다.')
    for src,dst in mapping_obj.items():
        if any(term in src.lower() for term in ['hardness','judgment','nonconforming','경도','판정']) and dst in FEATURES:
            raise HTTPException(422,'검사결과를 공정 입력으로 연결할 수 없습니다.')
    df=df.rename(columns=mapping_obj)
    if df.columns.duplicated().any():raise HTTPException(422,'여러 원본 열을 같은 공통 항목에 연결할 수 없습니다.')
    missing=[x for x in META+FEATURES if x not in df.columns]
    if missing:raise HTTPException(422,{'missing_columns':missing})
    ignored=[str(c) for c in df.columns if c not in META+FEATURES+['recipe_id','data_source']]
    errors=[]; valid=[]; warnings={}
    duplicates=df.lot_id.astype(str).duplicated(keep=False)
    for i,row in df.iterrows():
        reasons=[]; imputed=[]
        for key in META:
            if pd.isna(row[key]) or not str(row[key]).strip():reasons.append(f'{key}: 필수값 누락')
        if duplicates.iloc[i]:reasons.append('lot_id: 파일 안에서 중복')
        if row['material_grade']!=CFG['material_grade']:reasons.append('material_grade: 대상 재질은 100Cr6')
        try:
            if float(row['diameter_mm'])!=CFG['diameter_mm']:reasons.append('diameter_mm: 대상 직경은 10mm')
        except (ValueError,TypeError):reasons.append('diameter_mm: 숫자 형식 필요')
        if pd.isna(pd.to_datetime(row['process_end_at'],errors='coerce')):reasons.append('process_end_at: 시각 형식 오류')
        for key in FEATURES:
            value=row[key]
            if pd.isna(value):
                if key in CFG['optional_features']:imputed.append(key); df.loc[i,key]=np.nan
                else:reasons.append(f'{key}: 필수값 누락')
            else:
                try:
                    value=float(value); lo,hi=CFG['input_bounds'][key]
                    if not np.isfinite(value) or not lo<=value<=hi:reasons.append(f'{key}: 시연 입력 범위 오류')
                    else:df.loc[i,key]=value
                except (ValueError,TypeError):reasons.append(f'{key}: 숫자 형식 오류')
        if reasons:errors.append({'row':int(i)+2,'lot_id':None if pd.isna(row.lot_id) else str(row.lot_id),'reasons':reasons})
        else:valid.append(i);warnings[i]=imputed
    results=[]
    model=bundle()
    if valid:
        scores=model['pipeline'].predict_proba(df.loc[valid,FEATURES].astype(float))[:,1]
        for i,score in zip(valid,scores):
            results.append({'row':int(i)+2,'lot_id':str(df.loc[i,'lot_id']),'risk_score':round(float(score),6),
                'status':'주의' if score>=model['threshold'] else '일반',
                'imputed_columns':warnings[i],
                'deviation_from_demo_nominal':{f:None if pd.isna(df.loc[i,f]) else round(float(df.loc[i,f])-CFG['demo_nominals'][f],3) for f in FEATURES}})
    result={'run_id':str(uuid.uuid4()),'input_rows':len(df),'predicted_rows':len(valid),'error_rows':len(errors),
        'model':model['name'],'model_version':model['version'],'threshold':round(float(model['threshold']),4),
        'data_scope':'synthetic_demo_model','notice':'시연 점수입니다. 실제 불량확률이나 최종 출하판정을 뜻하지 않습니다.',
        'ignored_columns':ignored,'results':sorted(results,key=lambda r:r['risk_score'],reverse=True),'errors':errors}
    with sqlite3.connect(ROOT/'results/runs.sqlite3') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        conn.execute('INSERT INTO runs VALUES (?,?)',(result['run_id'],json.dumps(result,ensure_ascii=False)))
    return result

@app.get('/runs/{run_id}')
def saved_run(run_id:str):
    path=ROOT/'results/runs.sqlite3'
    if not path.exists():raise HTTPException(404,'결과가 없습니다.')
    with sqlite3.connect(path) as conn:
        row=conn.execute('SELECT payload FROM runs WHERE id=?',(run_id,)).fetchone()
    if row is None:raise HTTPException(404,'결과가 없습니다.')
    return json.loads(row[0])
