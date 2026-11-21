# API 및 데이터베이스 명세

버전 2.0.0 · 기본 주소 `http://127.0.0.1:8765` · 자동 API 명세 `/docs`, `/openapi.json`.

React 개발 서버만 `/api` 접두사를 프록시에서 제거한다. 정적 빌드와 FastAPI는 같은 주소를 쓴다. 인증은 없다(로컬 단일 사용자 MVP, 공개 체험은 문서 10).

## 입력 계약

계약은 `준비패키지/release/data_dictionary.json`과 번들에서 읽는다. 아래는 현재 모델(`probayes-im-v2-im_v2_r3`) 기준.

| 열 | 필수 | 형식 |
|---|---|---|
| `part_id` | ✓ | 문자열, 파일 안에서 중복 불가 (중복 ID의 모든 행 제외) |
| `cycle_start_at`, `cycle_end_at` | ✓ | `YYYY-MM-DD HH:MM:SS`, 시간대 없으면 Asia/Seoul, 종료 ≥ 시작 |
| `data_source` | ✓ | `ProBayes_IM_v2` 또는 `synthetic_im`, 한 파일에 하나만 |
| `DXP_HoldingPressure1` | ✓ | bar, 0~5000 |
| `CALC_InjectionStroke` |  | mm, 0~1000. 비거나 열이 없으면 학습 중앙값 대체 |
| `material` |  | 문자열 |
| `actual_label` |  | 0 또는 1. 예측에 쓰지 않고 사후 비교만 |

- 원본 데이터셋 열 이름(`MET_MachineCycleID`, `MET_TimeCycleStart`, `MET_TimeCycleEnd`, `MET_MaterialName`)은 자동으로 제안 매핑된다.
- `LBL_`, `IR_`, `CV_`, `SCA_`, `SIM_`로 시작하거나 불량·판정·label 등이 들어간 원본 열은 특징에 연결할 수 없다(422).
- 단위 변환은 없다. 허용 범위는 물리적으로 불가능한 값을 거르는 범위이며 공정 규격이 아니다.
- 파일: CSV(UTF-8/CP949) 또는 xlsx 첫 시트, 10MB, 10,000행(공개 모드 500행), 100열.

## API

| 경로 | 요청 | 응답·동작 |
|---|---|---|
| GET /health | 없음 | status, model_version, data_scope, readiness, notice |
| GET /schema | 없음 | required_columns, optional_columns, fields(단위·min·max·설명·참고 구간), sources, missing_policy, excluded_input_prefixes, alert_threshold, 제한 |
| GET /model | 없음 | benchmark_public.json 그대로 |
| GET /sample | kind=synthetic(기본)·test·real | CSV. real은 승격 때 생기는 `준비패키지/local/real_demo.csv`(원본 holdout 121행)가 있을 때만(공개 모드 404) |
| POST /preview | multipart file | filename, file_sha256, bytes, encoding, columns, rows, sample(5행), suggested_mapping |
| POST /validate | file, mapping(JSON: 원본 열 → 공통항목) | input_rows, valid_rows, error_rows, errors[{row, part_id, reasons}], missing_cells, imputed_rows, ignored_columns, units, data_scope |
| POST /predict | 위와 동일 | 재검증 → 추론 → SQLite 저장 → 결과. 유효행 0개면 422, 저장 안 함 |
| GET /runs | limit 1~100, offset, search, source | total, items[run_id, created_at, filename, data_scope, model_version, predicted_rows] |
| GET /runs/{id} | 없음 | 저장 당시 payload 그대로 (404) |
| GET /runs/{id}/export | kind=results·errors | UTF-8 BOM CSV. `= + - @`로 시작하는 문자열은 `'`를 붙여 수식 실행 방지 |

### /predict 결과 주요 필드

| 필드 | 내용 |
|---|---|
| results[] | part_id, cycle_start_at, cycle_end_at, material, actual_label, risk_score(0~1 보정 확률), review_priority(우선 검토/통상 검토), imputed_columns, outside_training_reference[{feature, value, train_q05, train_q95, direction, unit}], features(대체 후 값), original_numeric_features(입력값, 결측은 null) |
| summary | priority_count, mean_risk_score, deviation_part_count, label_available_rows, label_comparison{count, positive_count, tp, fn, fp, tn, recall, precision} 또는 null |
| 추적 | run_id, created_at, filename, file_sha256, model_version, model_name, model_sha256, alert_threshold, risk_score_kind |

## 데이터베이스

SQLite 파일 하나(`ICT_DB_PATH`, 기본 `준비패키지/runs.sqlite3`, WAL 모드).

| 열 | 형식 | 내용 |
|---|---|---|
| id | TEXT PK | run_id (UUID) |
| payload | TEXT | /predict 응답 JSON 전체 |
| created_at, filename, data_scope, model_version, predicted_rows | TEXT/INTEGER | 목록·검색용 |

이전 버전(열처리 모델) 기록도 그대로 열리며 내보내기가 가능하다. 공개 모드는 최근 20건만 남긴다. 입력 파일 자체는 저장하지 않는다.
