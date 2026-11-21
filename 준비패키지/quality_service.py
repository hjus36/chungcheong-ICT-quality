"""입력 계약·추론·이력 저장 (사출성형 부품 단위).

계약은 코드에 하드코딩하지 않고 승격된 release에서 읽는다.
- release/model.joblib          번들 (파이프라인·특징·판정 기준·참고 구간·결측 정책)
- release/data_dictionary.json  메타·특징·단위·허용 범위·필수 여부
- release/benchmark_public.json 화면용 집계 지표
새 모델은 ml_workflow/promote.py로만 교체한다.
"""
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

ROOT = Path(__file__).resolve().parent
RELEASE = Path(os.environ.get('ICT_RELEASE_DIR') or ROOT / 'release')
SHARE = ROOT / 'share'
LOCAL = ROOT / 'local'           # 실측 예제 등 배포하지 않는 자료 (git 제외)
MAX_BYTES = 10 * 1024 * 1024
LABEL = 'actual_label'
META = ['part_id', 'cycle_start_at', 'cycle_end_at', 'data_source']
OPTIONAL = ['material', LABEL]
PRIORITY = ('우선 검토', '통상 검토')
DEFAULT_EXCLUDED = ['LBL_', 'IR_', 'CV_', 'SCA_', 'SIM_']
LEAK_WORDS = ['불량', '양품', '정답', '판정', '검사', 'defect', 'label', 'nok', 'risk', 'sink', 'underfill']
NOTICE = ('사출 사이클 종료 직후 검토 우선순위를 정하는 연구 MVP입니다. 위험 점수는 보정된 추정 확률이지만 '
          '실제 불량 여부가 아니며 출하판정·설비제어에 사용하지 않습니다.')


def public_demo():
    return os.environ.get('ICT_PUBLIC_DEMO') == '1'


def max_rows():
    return 500 if public_demo() else 10000


@lru_cache
def bundle():
    b = joblib.load(RELEASE / 'model.joblib')
    if b.get('schema_version') != 2:
        raise RuntimeError('release/model.joblib이 schema_version 2가 아닙니다. ml_workflow/promote.py로 다시 승격하세요.')
    return b


@lru_cache
def dictionary():
    return json.loads((RELEASE / 'data_dictionary.json').read_text(encoding='utf-8'))


@lru_cache
def model_info():
    path = RELEASE / ('benchmark_public.json' if public_demo() else 'benchmark.json')
    if not path.exists():
        path = RELEASE / 'benchmark_public.json'
    return json.loads(path.read_text(encoding='utf-8'))


def features():
    return list(bundle()['features'])


def feature_spec():
    return {f['name']: f for f in dictionary()['features']}


def required_features():
    return [f for f in features() if f not in bundle()['missing_policy']['optional']]


def sources():
    d = dictionary()
    return {d['data_source']: '실측 (ProBayes 사출성형)', d['synthetic_source']: '합성 기능검증'}


def excluded_prefixes():
    return dictionary().get('excluded_input_prefixes') or DEFAULT_EXCLUDED


def sample_path(kind):
    if kind == 'real':
        return LOCAL / 'real_demo.csv'
    if kind == 'test':
        return SHARE / 'test_upload.csv'
    return SHARE / 'synthetic_demo.csv'


def schema():
    spec, b = feature_spec(), bundle()
    req = required_features()
    return {'required_columns': META + req,
            'optional_columns': [f for f in features() if f not in req] + OPTIONAL,
            'sample_availability': {'real': not public_demo() and sample_path('real').exists(),
                                    'synthetic': sample_path('synthetic').exists(), 'test': sample_path('test').exists()},
            'fields': [{'name': m['name'], 'kind': 'meta', 'required': True, 'description': m['description']}
                       for m in dictionary()['meta']] +
                      [{'name': f, 'kind': 'feature', 'unit': spec[f]['unit'], 'required': f in req,
                        'min': spec[f]['min'], 'max': spec[f]['max'], 'description': spec[f]['description'],
                        'reference_q05': b['reference']['q05'][f], 'reference_q95': b['reference']['q95'][f]}
                       for f in features()] +
                      [{'name': 'material', 'kind': 'meta', 'required': False, 'description': '재료 (예: PP, ABS)'},
                       {'name': LABEL, 'kind': 'quality', 'required': False,
                        'description': '실제 판정 0(정상)/1(불량). 예측에 쓰지 않고 사후 비교에만 사용'}],
            'sources': sources(), 'max_bytes': MAX_BYTES, 'max_rows': max_rows(), 'max_columns': 100,
            'public_demo': public_demo(), 'history_limit': 20 if public_demo() else None,
            'missing_policy': b['missing_policy'], 'excluded_input_prefixes': excluded_prefixes(),
            'alert_threshold': b['alert_threshold'], 'model_version': b['version'], 'timezone': 'Asia/Seoul',
            'reference_notice': f"{b['reference'].get('basis', '학습 정상 부품 5~95백분위')}이며 공정 규격이 아닙니다.",
            'notice': NOTICE}


def json_safe(value):
    return json.loads(pd.DataFrame(value).to_json(orient='records', date_format='iso'))


async def read_upload(file):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, '파일은 10MB 이하만 허용합니다. 센서 시계열은 사이클별로 먼저 집계하세요.')
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



def suggested_mapping(columns):
    allowed = META + features() + OPTIONAL
    d = dictionary()
    src = {m['source_column']: m['name'] for m in d['meta'] + d['optional'] if m.get('source_column')}
    out = {}
    for c in columns:
        if c in allowed: out[c] = c
        elif src.get(c) in allowed and src[c] != LABEL: out[c] = src[c]
    return out


def is_leaky(src):
    s = src.strip()
    return s.upper().startswith(tuple(p.upper() for p in excluded_prefixes())) or any(w in s.lower() for w in LEAK_WORDS)


def _absent(v):
    return str(v).strip().lower() in ['', 'nan', 'na', 'null', 'none']


def prepare(df, mapping_text='{}'):
    mapping = parse_object(mapping_text, '매핑')
    feats, spec = features(), feature_spec()
    req = set(required_features())
    allowed = META + feats + OPTIONAL
    if any(k not in df.columns or v not in allowed for k, v in mapping.items()):
        raise HTTPException(422, '매핑에는 존재하는 원본 열과 지원되는 공통항목만 사용할 수 있습니다.')
    for src, dst in mapping.items():
        if dst in feats and src != dst and is_leaky(src):
            raise HTTPException(422, '검사 결과·사후측정·라벨 열을 공정 특징으로 연결할 수 없습니다.')
    df = df.rename(columns=mapping).copy()
    if df.columns.duplicated().any(): raise HTTPException(422, '여러 원본 열이 같은 공통항목에 연결되었습니다.')
    missing_columns = [c for c in META + [f for f in feats if f in req] if c not in df]
    if missing_columns: raise HTTPException(422, {'message': '필수 항목을 연결하세요.', 'missing_columns': missing_columns})
    for f in feats:                       # 선택 특징 열이 없으면 전부 결측으로 보고 대체
        if f not in df: df[f] = ''
    df['part_id'] = df.part_id.astype(str).str.strip()
    duplicate = df.part_id.duplicated(keep=False)
    df['data_source'] = df.data_source.astype(str).str.strip()
    source_values = set(df.data_source)
    if len(source_values) > 1:
        raise HTTPException(422, '한 파일에 서로 다른 자료 출처를 섞을 수 없습니다. 실측과 합성 파일을 분리하세요.')
    numeric = df[feats].apply(pd.to_numeric, errors='coerce')
    srcs = sources()
    errors, valid, missing, row_meta = [], [], {}, {}
    for i, row in df.iterrows():
        reasons, absent = [], []
        if _absent(row.part_id): reasons.append('부품 ID: 필수값 누락')
        if duplicate.iloc[i]: reasons.append('부품 ID: 파일 안에서 중복(해당 ID의 모든 행 제외)')
        if row.data_source not in srcs: reasons.append(f"자료 출처: {' 또는 '.join(srcs)}만 허용")
        dates = []
        for key in ['cycle_start_at', 'cycle_end_at']:
            value = str(row[key]).strip()
            try:
                if not re.match(r'^\d{4}-\d{2}-\d{2}(?:[T ]|$)', value): raise ValueError()
                stamp = pd.Timestamp(value)
                if pd.isna(stamp): raise ValueError()
                stamp = stamp.tz_localize('Asia/Seoul') if stamp.tzinfo is None else stamp
                dates.append(stamp)
            except (ValueError, TypeError, OverflowError):
                dates.append(None); reasons.append(f'{key}: YYYY-MM-DD HH:MM:SS 날짜 형식 필요')
        if all(d is not None for d in dates) and dates[1] < dates[0]: reasons.append('사이클 종료 시각이 시작보다 빠릅니다.')
        for f in feats:
            x, s = numeric.loc[i, f], spec[f]
            if pd.isna(x):
                if not _absent(row[f]): reasons.append(f'{f}: 숫자 형식 오류')
                elif f in req: reasons.append(f'{f}: 필수 특징 누락 (학습 자료에 결측이 없던 항목)')
                else: absent.append(f)
            elif not np.isfinite(x): reasons.append(f'{f}: 유한한 숫자가 필요합니다.')
            elif (s['min'] is not None and x < s['min']) or (s['max'] is not None and x > s['max']):
                reasons.append(f"{f}: 허용 범위 {s['min']}~{s['max']} {s['unit']} 밖입니다. 입력 단위를 확인하세요.")
        label = None
        if LABEL in df and not _absent(row[LABEL]):
            v = str(row[LABEL]).strip()
            if v in ('0', '1', '0.0', '1.0'): label = int(float(v))
            else: reasons.append(f'{LABEL}: 0(정상) 또는 1(불량)만 허용')
        if reasons:
            errors.append({'row': int(i) + 2, 'part_id': row.part_id, 'reasons': reasons})
        else:
            valid.append(i); missing[i] = absent
            row_meta[i] = {'part_id': row.part_id, 'cycle_start_at': dates[0].isoformat(), 'cycle_end_at': dates[1].isoformat(),
                           'material': str(row.get('material', '') or '').strip(), LABEL: label}
    summary = {'input_rows': len(df), 'valid_rows': len(valid), 'error_rows': len(errors), 'errors': errors,
               'missing_cells': sum(len(v) for v in missing.values()), 'imputed_rows': sum(bool(v) for v in missing.values()),
               'ignored_columns': [c for c in df if c not in allowed], 'data_scope': next(iter(source_values)),
               'mapping': mapping, 'units': {f: spec[f]['unit'] for f in feats},
               'warning': '단위 변환은 하지 않습니다. 표시된 단위로 입력했는지 확인하세요.',
               'preview': json_safe(numeric.loc[valid].head(5))}
    return summary, numeric, valid, missing, row_meta


def compare_labels(results):
    rows = [r for r in results if r[LABEL] is not None]
    if not rows: return None
    tp = sum(r[LABEL] == 1 and r['review_priority'] == PRIORITY[0] for r in rows)
    fn = sum(r[LABEL] == 1 and r['review_priority'] == PRIORITY[1] for r in rows)
    fp = sum(r[LABEL] == 0 and r['review_priority'] == PRIORITY[0] for r in rows)
    tn = len(rows) - tp - fn - fp
    return {'count': len(rows), 'positive_count': tp + fn, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'recall': tp / (tp + fn) if tp + fn else None, 'precision': tp / (tp + fp) if tp + fp else None,
            'note': '업로드한 actual_label과 비교한 결과입니다. 합성 자료의 라벨은 가상값입니다.'}


def analyze(df, file_meta, mapping='{}'):
    summary, numeric, valid, missing, meta = prepare(df, mapping)
    if not valid: raise HTTPException(422, {'message': '분석할 수 있는 유효행이 없습니다.', **summary})
    b, feats, spec = bundle(), features(), feature_spec()
    values = numeric.loc[valid, feats]
    probabilities = b['pipeline'].predict_proba(values)[:, 1]
    reference = b['reference']
    processed = values.fillna({f: reference['median'][f] for f in feats})
    results = []
    for i, risk in zip(valid, probabilities):
        deviations = []
        for f in feats:
            if f in missing[i]: continue
            value, low, high = float(values.loc[i, f]), reference['q05'][f], reference['q95'][f]
            if not low <= value <= high:
                deviations.append({'feature': f, 'value': value, 'train_q05': low, 'train_q95': high,
                                   'direction': '높음' if value > high else '낮음', 'unit': spec[f]['unit']})
        results.append({**meta[i], 'row': int(i) + 2, 'risk_score': round(float(risk), 8),
                        'review_priority': PRIORITY[0] if risk >= b['alert_threshold'] else PRIORITY[1],
                        'imputed_columns': missing[i], 'outside_training_reference': deviations,
                        'features': {f: float(processed.loc[i, f]) for f in feats},
                        'original_numeric_features': {f: None if pd.isna(values.loc[i, f]) else float(values.loc[i, f]) for f in feats}})
    payload = {**summary, **file_meta, 'run_id': str(uuid.uuid4()), 'created_at': datetime.now(timezone.utc).isoformat(),
               'predicted_rows': len(valid), 'model_version': b['version'], 'model_name': b['selected_model'],
               'model_sha256': model_info()['model_sha256'], 'notice': NOTICE, 'reference': reference,
               'alert_threshold': b['alert_threshold'], 'risk_score_kind': b['risk_score_kind'],
               'results': sorted(results, key=lambda r: r['risk_score'], reverse=True)}
    payload.pop('preview', None)
    payload['summary'] = {'priority_count': sum(r['review_priority'] == PRIORITY[0] for r in results),
                          'mean_risk_score': float(np.mean(probabilities)),
                          'deviation_part_count': sum(bool(r['outside_training_reference']) for r in results),
                          'label_available_rows': sum(r[LABEL] is not None for r in results),
                          'label_comparison': compare_labels(results)}
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
        rows = [{'행': e['row'], '부품_ID': e.get('part_id', e.get('batch_id', '')), '오류': '; '.join(e['reasons'])}
                for e in payload['errors']]
        fields = ['행', '부품_ID', '오류']
    else:
        fields = ['부품_ID', '사이클_시작', '재료', '위험점수', '검토구분', '실제판정', '결측대체', '참고구간이탈수', '자료출처', '모델버전']
        rows = [dict(zip(fields, [r.get('part_id', r.get('batch_id', '')), r.get('cycle_start_at', ''), r.get('material', ''),
                                  r['risk_score'], r['review_priority'], r.get(LABEL), ','.join(r.get('imputed_columns', [])),
                                  len(r['outside_training_reference']), payload['data_scope'], payload['model_version']]))
                for r in payload['results']]
    output = StringIO(); writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
    for row in rows:
        writer.writerow({k: "'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else v
                         for k, v in row.items()})
    return output.getvalue().encode('utf-8-sig')
