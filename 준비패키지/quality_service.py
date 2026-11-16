"""Input contract, inference and non-destructive local history storage."""
from pathlib import Path
from io import BytesIO, StringIO
from functools import lru_cache
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import re
import sqlite3
import uuid
import zipfile
import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent / 'kamp'
RELEASE = ROOT / 'release'
AUDIT = json.loads((ROOT / 'audit.json').read_text(encoding='utf-8'))
FEATURES = AUDIT['features']
META = ['batch_id', 'process_start_at', 'process_end_at', 'data_source']
OPTIONAL = ['item_code', 'total_count', 'good_count', 'defect_count']
SOURCES = {'KAMP_heat_quality_60': 'KAMP 실측', 'synthetic_kamp60': '합성 기능검증'}
MAX_BYTES = 10 * 1024 * 1024
NOTICE = '공정 완료 후 검토용 연구 MVP입니다. 위험 점수는 실제 불량률이 아니며 출하판정·설비제어에 사용하지 않습니다.'

def public_demo():
    return os.environ.get('ICT_PUBLIC_DEMO') == '1'


def max_rows():
    return 500 if public_demo() else 10000


@lru_cache
def bundle():
    return joblib.load(RELEASE / 'model.joblib')

@lru_cache
def model_info():
    path = RELEASE / ('benchmark_public.json' if public_demo() else 'benchmark.json')
    if not path.exists(): path = RELEASE / 'benchmark_public.json'
    return json.loads(path.read_text(encoding='utf-8'))

def sample_path(kind):
    if kind == 'real': return ROOT / 'demo_upload.csv'
    if public_demo(): return ROOT / 'share' / 'synthetic_demo.csv'
    local = RELEASE / 'synthetic_demo.csv'
    return local if local.exists() else ROOT / 'share' / 'synthetic_demo.csv'

def unit_for(feature):
    return '%' if ('OP' in feature or 'CP' in feature) else '°C'

def schema():
    return {'required_columns': META + FEATURES, 'optional_columns': OPTIONAL,
            'sample_availability': {'real': not public_demo() and sample_path('real').exists(), 'synthetic': sample_path('synthetic').exists()},
            'fields': [{'name': k, 'kind': 'meta', 'required': True} for k in META] +
                      [{'name': k, 'kind': 'feature', 'unit': unit_for(k),
                        'units': ['%', 'fraction'] if unit_for(k) == '%' else ['°C', '°F', 'K'],
                        'required': True} for k in FEATURES] +
                      [{'name': k, 'kind': 'quality' if k != 'item_code' else 'meta', 'required': False} for k in OPTIONAL],
            'sources': SOURCES, 'max_bytes': MAX_BYTES, 'max_rows': max_rows(), 'max_columns': 100,
            'public_demo': public_demo(), 'history_limit': 20 if public_demo() else None,
            'max_missing_features_per_row': 3, 'timezone': 'Asia/Seoul',
            'reference_notice': '학습 자료 5~95백분위이며 기업 규격이 아닙니다.', 'notice': NOTICE}

def json_safe(value):
    return json.loads(pd.DataFrame(value).to_json(orient='records', date_format='iso'))

async def read_upload(file):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, '파일은 10MB 이하만 허용합니다. 원시 센서는 작업별로 먼저 집계하세요.')
    suffix = Path(file.filename or '').suffix.lower()
    try:
        if suffix == '.csv':
            try:
                content = raw.decode('utf-8-sig'); encoding = 'UTF-8'
            except UnicodeDecodeError:
                content = raw.decode('cp949'); encoding = 'CP949'
            header = next(csv.reader(StringIO(content)))
            if len(header) != len(set(h.strip() for h in header)):
                raise HTTPException(422, '원본 파일에 중복된 열 이름이 있습니다.')
            df = pd.read_csv(StringIO(content), dtype=str, nrows=max_rows()+1, keep_default_na=False)
        elif suffix == '.xlsx':
            with zipfile.ZipFile(BytesIO(raw)) as archive:
                if sum(z.file_size for z in archive.infolist()) > 100 * 1024 * 1024 or len(archive.infolist()) > 6000:
                    raise HTTPException(413, 'Excel 압축 해제 크기가 제한을 초과합니다.')
            from openpyxl import load_workbook
            wb = load_workbook(BytesIO(raw), read_only=True, data_only=True)
            try:
                sheet = wb.worksheets[0]
                if sheet.max_column > 100:
                    raise HTTPException(422, '입력은 100열 이하여야 합니다.')
                rows = list(sheet.iter_rows(max_row=max_rows()+2, values_only=True))
                while rows and not any(v is not None for v in rows[-1]): rows.pop()
                if not rows: raise HTTPException(422, '비어 있는 파일입니다.')
                header = [str(v).strip() if v is not None else '' for v in rows[0]]
                if '' in header or len(header) != len(set(header)):
                    raise HTTPException(422, 'Excel 첫 행의 열 이름이 비어 있거나 중복되었습니다.')
                df = pd.DataFrame(rows[1:], columns=header).fillna('').astype(str)
            finally:
                wb.close()
            encoding = 'xlsx 첫 번째 시트'
        else:
            raise HTTPException(415, 'CSV 또는 xlsx 파일을 선택하세요. xls는 xlsx로 저장 후 사용하세요.')
    except HTTPException: raise
    except Exception:
        raise HTTPException(422, '파일을 읽지 못했습니다. 손상 여부와 CSV 인코딩/Excel 형식을 확인하세요.')
    if not 1 <= len(df) <= max_rows(): raise HTTPException(422, f'입력은 1~{max_rows():,}행이어야 합니다.')
    if len(df.columns) > 100: raise HTTPException(422, '입력은 100열 이하여야 합니다.')
    df.columns = [str(c).strip() for c in df.columns]
    if '' in df.columns or df.columns.duplicated().any(): raise HTTPException(422, '열 이름은 비어 있거나 중복될 수 없습니다.')
    return df, {'filename': Path((file.filename or 'upload').replace('\\', '/')).name,
                'file_sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'encoding': encoding}

def parse_object(value, name):
    try:
        obj = json.loads(value)
        if not isinstance(obj, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()): raise ValueError()
        return obj
    except (ValueError, TypeError): raise HTTPException(422, f'{name}은 문자열 키와 값을 가진 JSON 객체여야 합니다.')

def prepare(df, mapping_text='{}', units_text='{}'):
    mapping = parse_object(mapping_text, '매핑'); units = parse_object(units_text, '단위')
    allowed = META + FEATURES + OPTIONAL
    if any(k not in df.columns or v not in allowed for k, v in mapping.items()):
        raise HTTPException(422, '매핑에는 존재하는 원본 열과 지원되는 공통항목만 사용할 수 있습니다.')
    for src, dst in mapping.items():
        if dst in FEATURES and any(s in src.lower() for s in ['불량', '양품', '총수량', '정답', '판정', 'defect', 'label', 'risk', 'hardness', '경도', 'good_count', 'total_count']):
            raise HTTPException(422, '검사·품질 결과를 공정 특성으로 연결할 수 없습니다.')
    df = df.rename(columns=mapping).copy()
    if df.columns.duplicated().any(): raise HTTPException(422, '여러 원본 열이 같은 공통항목에 연결되었습니다.')
    missing_columns = [c for c in META + FEATURES if c not in df]
    if missing_columns: raise HTTPException(422, {'message': '필수 항목을 연결하세요.', 'missing_columns': missing_columns})
    for key, unit in units.items():
        valid_units = ['%', 'fraction'] if key in FEATURES and unit_for(key) == '%' else ['°C', '°F', 'K']
        if key not in FEATURES or unit not in valid_units: raise HTTPException(422, f'{key}: 지원하지 않는 단위입니다.')
    df['batch_id'] = df.batch_id.astype(str).str.strip()
    duplicate = df.batch_id.duplicated(keep=False)
    df['data_source'] = df.data_source.astype(str).str.strip()
    source_values = set(df.data_source)
    if len(source_values) > 1: raise HTTPException(422, '한 파일에 서로 다른 자료 출처를 섞을 수 없습니다. 실측과 합성 파일을 분리하세요.')
    numeric = df[FEATURES].apply(pd.to_numeric, errors='coerce'); original = numeric.copy()
    for key, unit in units.items():
        if unit == 'fraction': numeric[key] *= 100
        elif unit == '°F': numeric[key] = numeric[key] * 5 / 9 if key.endswith('_Std') else (numeric[key] - 32) * 5 / 9
        elif unit == 'K' and key.endswith('_Avg'): numeric[key] -= 273.15
    errors, valid, missing, row_meta = [], [], {}, {}
    for i, row in df.iterrows():
        reasons, absent = [], []
        if not row.batch_id or row.batch_id.lower() in ['nan', 'none', 'null']: reasons.append('작업 ID: 필수값 누락')
        if duplicate.iloc[i]: reasons.append('작업 ID: 파일 안에서 중복(해당 ID의 모든 행 제외)')
        if row.data_source not in SOURCES: reasons.append('자료 출처: KAMP_heat_quality_60 또는 synthetic_kamp60만 허용')
        dates = []
        for key in ['process_start_at', 'process_end_at']:
            value = str(row[key]).strip()
            try:
                if not re.match(r'^\d{4}-\d{2}-\d{2}(?:[T ]|$)', value): raise ValueError()
                stamp = pd.Timestamp(value)
                if pd.isna(stamp): raise ValueError()
                stamp = stamp.tz_localize('Asia/Seoul') if stamp.tzinfo is None else stamp
                dates.append(stamp)
            except (ValueError, TypeError, OverflowError):
                dates.append(None); reasons.append(f'{key}: YYYY-MM-DD HH:MM:SS 날짜 형식 필요')
        if all(d is not None for d in dates) and dates[1] < dates[0]: reasons.append('공정 종료 시각이 시작보다 빠릅니다.')
        for feature in FEATURES:
            x = numeric.loc[i, feature]
            if pd.isna(x):
                if str(row[feature]).strip().lower() in ['', 'nan', 'na', 'null', 'none']: absent.append(feature)
                else: reasons.append(f'{feature}: 숫자 형식 오류')
            elif not np.isfinite(x): reasons.append(f'{feature}: 유한한 숫자가 필요합니다.')
            elif abs(x) > np.finfo(np.float32).max: reasons.append(f'{feature}: 모델이 처리할 수 있는 수치 범위를 초과합니다.')
            elif feature.endswith('_Std') and x < 0: reasons.append(f'{feature}: 표준편차는 0 이상이어야 합니다.')
            elif feature.endswith('_Avg') and unit_for(feature) == '%' and not 0 <= x <= 100:
                reasons.append(f'{feature}: % 평균은 0~100 범위여야 합니다. 입력 단위를 확인하세요.')
            elif feature.endswith('_Avg') and unit_for(feature) == '°C' and x < -273.15:
                reasons.append(f'{feature}: 절대영도 미만입니다. 입력 단위를 확인하세요.')
        if len(absent) > 3: reasons.append('특성 결측이 4개 이상입니다(최대 3개만 학습 중앙값 대체).')
        quality = {}
        for key in ['total_count', 'good_count', 'defect_count']:
            if key not in df or not str(row[key]).strip(): continue
            try:
                x = float(row[key])
                if not np.isfinite(x) or x < 0 or x > 2**53-1 or x != int(x) or (key == 'total_count' and x == 0): raise ValueError()
                quality[key] = int(x)
            except (ValueError, OverflowError): reasons.append(f'{key}: 총수량은 양의 정수, 검사수량은 0 이상 정수가 필요합니다.')
        if quality and not {'total_count', 'defect_count'} <= quality.keys(): reasons.append('실제 불량률 비교에는 총수량과 불량수량이 모두 필요합니다.')
        if {'total_count', 'defect_count'} <= quality.keys():
            if quality['defect_count'] > quality['total_count']: reasons.append('불량수량은 총수량을 초과할 수 없습니다.')
            if 'good_count' in quality and quality['good_count'] + quality['defect_count'] != quality['total_count']:
                reasons.append('양품수량+불량수량이 총수량과 다릅니다.')
        if reasons: errors.append({'row': int(i) + 2, 'batch_id': row.batch_id, 'reasons': reasons})
        else:
            valid.append(i); missing[i] = absent
            row_meta[i] = {'batch_id': row.batch_id, 'process_start_at': dates[0].isoformat(), 'process_end_at': dates[1].isoformat(),
                           'item_code': row.get('item_code', ''), **quality,
                           'actual_defect_rate_percent': (100 * quality['defect_count'] / quality['total_count']) if quality else None}
    summary = {'input_rows': len(df), 'valid_rows': len(valid), 'error_rows': len(errors), 'errors': errors,
               'missing_cells': sum(len(v) for v in missing.values()), 'imputed_rows': sum(bool(v) for v in missing.values()),
               'converted_columns': [k for k, u in units.items() if u != unit_for(k)],
               'ignored_columns': [c for c in df if c not in allowed], 'data_scope': next(iter(source_values)),
               'mapping': mapping, 'units': units, 'warning': '원본의 숫자만으로 단위를 추정할 수 없습니다. 선택한 단위를 확인하세요.',
               'preview': json_safe(numeric.loc[valid].head(5).fillna(bundle()['reference']['median']))}
    return summary, numeric, original, valid, missing, row_meta

def analyze(df, file_meta, mapping='{}', units='{}'):
    summary, numeric, original, valid, missing, meta = prepare(df, mapping, units)
    if not valid: raise HTTPException(422, {'message': '분석할 수 있는 유효행이 없습니다.', **summary})
    b = bundle(); values = numeric.loc[valid, FEATURES]
    probabilities = b['pipeline'].predict_proba(values)[:, 1]
    reference = b['reference']; processed = values.fillna(reference['median']); results = []
    for i, risk in zip(valid, probabilities):
        deviations = []
        for key in FEATURES:
            value = float(processed.loc[i, key]); low, high = reference['q05'][key], reference['q95'][key]
            if not low <= value <= high:
                deviations.append({'feature': key, 'value': value, 'train_q05': low, 'train_q95': high,
                                   'direction': '높음' if value > high else '낮음', 'unit': unit_for(key), 'imputed': key in missing[i]})
        results.append({**meta[i], 'row': int(i)+2, 'risk_score': round(float(risk), 8),
                        'review_priority': '우선 검토' if risk >= b['alert_threshold'] else '통상 검토',
                        'imputed_columns': missing[i], 'outside_training_reference': deviations,
                        'features': {k: float(processed.loc[i, k]) for k in FEATURES},
                        'original_numeric_features': {k: None if pd.isna(original.loc[i, k]) else float(original.loc[i, k]) for k in FEATURES}})
    payload = {**summary, **file_meta, 'run_id': str(uuid.uuid4()), 'created_at': datetime.now(timezone.utc).isoformat(),
               'predicted_rows': len(valid), 'model_version': b['version'], 'model_name': b['selected_model'],
               'model_sha256': model_info()['model_sha256'], 'notice': NOTICE, 'reference': reference,
               'alert_threshold': b['alert_threshold'], 'defect_rate_threshold_percent': b['defect_rate_threshold_percent'],
               'results': sorted(results, key=lambda r: r['risk_score'], reverse=True)}
    payload.pop('preview', None)
    payload['summary'] = {'priority_count': sum(r['review_priority'] == '우선 검토' for r in results),
                          'mean_risk_score': float(np.mean(probabilities)),
                          'deviation_batch_count': sum(bool(r['outside_training_reference']) for r in results),
                          'quality_available_rows': sum(r['actual_defect_rate_percent'] is not None for r in results)}
    save_run(payload)
    return payload

def connect():
    path = Path(os.environ.get('ICT_DB_PATH', str(ROOT / 'runs.sqlite3'))); path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=15); db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    columns = {r['name'] for r in db.execute('PRAGMA table_info(runs)')}
    for name, kind in {'created_at': 'TEXT', 'filename': 'TEXT', 'data_scope': 'TEXT', 'model_version': 'TEXT', 'predicted_rows': 'INTEGER'}.items():
        if name not in columns: db.execute(f'ALTER TABLE runs ADD COLUMN {name} {kind}')
    for row in db.execute('SELECT id,payload FROM runs WHERE predicted_rows IS NULL').fetchall():
        old = json.loads(row['payload'])
        db.execute('UPDATE runs SET created_at=?,filename=?,data_scope=?,model_version=?,predicted_rows=? WHERE id=?',
                   (old.get('created_at'), old.get('filename', '이전 버전 기록'), old.get('data_scope'),
                    old.get('model_version'), old.get('predicted_rows', 0), row['id']))
    db.execute('CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at DESC)'); db.commit()
    return db

def save_run(payload):
    db = connect()
    try:
        with db:
            db.execute('INSERT INTO runs (id,payload,created_at,filename,data_scope,model_version,predicted_rows) VALUES (?,?,?,?,?,?,?)',
                       (payload['run_id'], json.dumps(payload, ensure_ascii=False, allow_nan=False), payload['created_at'],
                        payload['filename'], payload['data_scope'], payload['model_version'], payload['predicted_rows']))
            if public_demo():
                db.execute('DELETE FROM runs WHERE id NOT IN (SELECT id FROM runs ORDER BY created_at DESC,rowid DESC LIMIT 20)')
    finally: db.close()

def get_run(run_id):
    db = connect()
    try:
        row = db.execute('SELECT payload FROM runs WHERE id=?', (run_id,)).fetchone()
        if row is None: raise HTTPException(404, '해당 분석 이력이 없습니다.')
        return json.loads(row['payload'])
    finally: db.close()

def list_runs(limit=20, offset=0, search='', source=''):
    db = connect()
    try:
        where = "WHERE (filename LIKE ? OR id LIKE ?) AND (?='' OR data_scope=?)"
        args = (f'%{search}%', f'%{search}%', source, source)
        total = db.execute('SELECT COUNT(*) FROM runs ' + where, args).fetchone()[0]
        rows = db.execute('SELECT id AS run_id,created_at,filename,data_scope,model_version,predicted_rows FROM runs ' + where +
                          ' ORDER BY created_at DESC,rowid DESC LIMIT ? OFFSET ?', (*args, limit, offset)).fetchall()
        return {'total': total, 'limit': limit, 'offset': offset, 'items': [dict(r) for r in rows]}
    finally: db.close()

def export_run(run_id, kind):
    payload = get_run(run_id)
    if kind == 'errors':
        rows = [{'행': e['row'], '작업_ID': e.get('batch_id', ''), '오류': '; '.join(e['reasons'])} for e in payload['errors']]
        fields = ['행', '작업_ID', '오류']
    else:
        fields = ['작업_ID', '품목', '위험점수', '검토구분', '실제불량률_pct', '참고구간이탈수', '자료출처', '모델버전']
        rows = [dict(zip(fields, [r['batch_id'], r.get('item_code', ''), r['risk_score'], r['review_priority'],
                                 r.get('actual_defect_rate_percent'), len(r['outside_training_reference']), payload['data_scope'], payload['model_version']])) for r in payload['results']]
    output = StringIO(); writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
    for row in rows:
        writer.writerow({k: "'"+v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else v for k, v in row.items()})
    return output.getvalue().encode('utf-8-sig')
