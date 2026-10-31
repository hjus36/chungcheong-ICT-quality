# API 및 데이터베이스 명세

버전 1.0.0 · 기본 주소 `http://127.0.0.1:8765` · 자동 API 명세 `/docs`, `/openapi.json`.

React 개발 서버만 `/api` 접두사를 프록시에서 제거한다. 운영용 정적 빌드와 FastAPI는 같은 주소를 사용한다. 로컬 단일 사용자 MVP이며 인증·외부 네트워크 공개는 포함하지 않는다.

## API

| 경로 | 요청 | 응답·동작 |
|---|---|---|
| GET /health | 없음 | status, model_version, data_scope, readiness, notice |
| GET /schema | 없음 | required_columns 40개, optional_columns 4개, fields와 단위, 자료 출처, 크기/행/열 제한 |
| GET /model | 없음 | 시간순 CV·3모델 검증 비교·선택 모델·보류 평가·기준선·36변수 분석·해시 |
| GET /sample | kind=real 또는 synthetic | 시연 CSV 다운로드 |
| POST /preview | multipart file | filename, file_sha256, bytes, encoding, columns, rows, sample(5행), suggested_mapping |
| POST /validate | multipart file, mapping, units | valid_rows, error_rows, errors, missing_cells, imputed_rows, converted_columns, ignored_columns, preview(전처리 5행) |
| POST /predict | 위와 동일 | 재검증 후 유효행 추론·SQLite 저장·전체 결과 반환. 유효행 0개이면 422이며 저장하지 않음 |
| GET /runs | limit=1~100(기본20), offset>=0, search(최대100자), source | total, limit, offset, items. 생성시각 내림차순, 과거 시각 없는 기록은 뒤에 표시 |
| GET /runs/{run_id} | 분석 UUID 또는 기존 ID | 당시 저장한 JSON. 재추론하지 않음 |
| GET /runs/{run_id}/export | kind=results 또는 errors | UTF-8 BOM CSV. 수식으로 시작하는 텍스트는 작은따옴표로 보호 |

매핑 방향은 **원본 이름 → 공통항목 이름**이다. 단위는 **공통항목 이름 → 원본 단위**이다.

```json
{
  "mapping": {"배정번호": "batch_id", "3존 평균온도": "소입로 온도 3 Zone_Avg"},
  "units": {"소입로 온도 3 Zone_Avg": "°F", "소입1존 OP_Avg": "fraction"}
}
```

실제 multipart의 mapping/units 필드는 각각 위 객체를 JSON 문자열로 전송한다. 미지정 단위는 사전의 표준 단위로 해석한다. 모든 열 이름은 `/schema` 또는 `데이터사전.json`에서 확인한다. 사용자가 입력한 data_source는 출처 선언이며 서버가 원천 진위를 인증한다는 뜻은 아니다.

## 검증과 결과

CSV는 UTF-8 BOM/UTF-8/CP949, Excel은 xlsx 첫 번째 시트. 최대 10MiB·10,000행·100열, Excel 압축 해제 합계 최대 100MiB. 중복 헤더·같은 대상 매핑·정답을 특성으로 매핑·필수 열 누락은 파일 전체 오류다. 여러 자료 출처가 섞인 파일도 거부한다.

행 오류는 원본 기준 2행부터 표시한다. ID 누락/중복, 형식 불명 날짜, 종료<시작, 문자/무한 숫자, 음의 표준편차, 물리적으로 불가능한 단위 값, 품질수량 불일치, 4개 이상의 특성 결측을 제외한다. 3개까지의 특성 결측은 학습 중앙값으로 대체한다. 참고 구간 이탈값은 삭제·절단하지 않는다. 품질수량을 제공하지 않으면 실제 불량률은 null이다.

HTTP 상태: 200 성공, 404 없는 이력, 413 크기 초과, 415 미지원 파일, 422 파일·매핑·행 검증 오류. 일부 오류행이 있어도 유효행이 있으면 200으로 유효행 분석을 저장하고 errors를 포함한다.

각 결과에는 `batch_id`, `risk_score`(0~1), `review_priority`, `actual_defect_rate_percent`, `features`(추론에 사용한 값), `original_numeric_features`, `imputed_columns`, `outside_training_reference`가 있다. 실제 불량률과 점수를 별도 열로 표시한다. 결과 전체에는 source·filename·input SHA-256·model SHA-256·model_version·created_at·mapping·units·reference를 저장한다. 순위는 점수 내림차순이다.

## SQLite

기본 파일 `준비패키지/kamp/runs.sqlite3`, 테스트는 환경변수 `ICT_DB_PATH`로 별도 경로를 사용한다.

```sql
CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  created_at TEXT,
  filename TEXT,
  data_scope TEXT,
  model_version TEXT,
  predicted_rows INTEGER
);
CREATE INDEX runs_created ON runs(created_at DESC);
```

기존 id/payload 테이블에 나머지 열만 추가한다. 기존 payload 자체는 수정하지 않는다. 과거 기록의 생성시각을 임의로 만들지 않고 null로 둔다. 새 기록의 시각은 UTC ISO, UI는 한국시간으로 표시한다. 매개변수 쿼리·트랜잭션·WAL·15초 연결 대기를 사용한다. 입력 원본은 메모리에서 읽고 보존하지 않으며 모델 업로드/임의 경로 읽기 API는 없다.

기록 보존을 위해 서버 종료 후 DB와 존재하는 `-wal`, `-shm` 파일을 함께 백업하거나 SQLite backup API를 사용한다. 운영 중 DB 파일만 단독 복사하면 최신 WAL 내용이 빠질 수 있다. 삭제 API는 제공하지 않는다. 대규모 다중 사용자 운영·백업 자동화·권한 분리는 후속 범위다.

## 공유 환경의 예제 API

`GET /schema`의 `sample_availability`는 real/synthetic 파일의 존재 여부를 제공합니다. `GET /sample?kind=real`은 로컬 실측 파일이 없으면 404와 안내 문장을 반환합니다. `GET /sample?kind=synthetic`은 로컬 기존 합성자료가 있으면 이를 사용하고, 없으면 GitHub에 포함된 `kamp/share/synthetic_demo.csv`를 제공합니다. 자료 출처를 바꾸어 표시하지 않습니다.
