"""FastAPI entry point; run from the root with --app-dir 준비패키지."""
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import quality_service as service

app = FastAPI(title='무한루프 · 제조공정 품질 분석', version='1.0.0')

@app.get('/health')
def health():
    b = service.bundle()
    return {'status': 'ok', 'model_version': b['version'], 'data_scope': b['data_source'],
            'readiness': 'research_prototype', 'notice': service.NOTICE}

@app.get('/schema')
def schema():
    return service.schema()

@app.get('/model')
def model():
    return service.model_info()

@app.get('/sample')
def sample(kind: str = Query('real', pattern='^(real|synthetic)$')):
    path = service.sample_path(kind)
    if not path.exists():
        raise HTTPException(404, '실측 예제는 저장소에 배포하지 않습니다. 합성 예제를 사용하거나 이용 권한이 있는 실측자료를 업로드하세요.')
    return FileResponse(path, filename='kamp_real_demo.csv' if kind == 'real' else 'synthetic_demo.csv', media_type='text/csv')

@app.post('/preview')
async def preview(file: UploadFile = File(...)):
    df, meta = await service.read_upload(file)
    return {**meta, 'columns': list(df.columns), 'rows': len(df), 'sample': service.json_safe(df.head(5)),
            'suggested_mapping': {c: c for c in df.columns if c in service.META + service.FEATURES + service.OPTIONAL}}

@app.post('/validate')
async def validate(file: UploadFile = File(...), mapping: str = Form('{}'), units: str = Form('{}')):
    df, meta = await service.read_upload(file)
    summary, *_ = service.prepare(df, mapping, units)
    return {**meta, **summary}

@app.post('/predict')
async def predict(file: UploadFile = File(...), mapping: str = Form('{}'), units: str = Form('{}')):
    df, meta = await service.read_upload(file)
    return service.analyze(df, meta, mapping, units)

@app.get('/runs')
def runs(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
         search: str = Query('', max_length=100), source: str = Query('', max_length=50)):
    return service.list_runs(limit, offset, search, source)

@app.get('/runs/{run_id}/export')
def export(run_id: str, kind: str = Query('results', pattern='^(results|errors)$')):
    return Response(service.export_run(run_id, kind), media_type='text/csv',
                    headers={'Content-Disposition': f'attachment; filename="{kind}.csv"'})

@app.get('/runs/{run_id}')
def run(run_id: str):
    return service.get_run(run_id)

DIST = Path(__file__).resolve().parent.parent / 'frontend' / 'dist'
if DIST.exists():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='web')
