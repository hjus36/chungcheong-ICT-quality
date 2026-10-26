# 충청권 ICT 현장적용 프로젝트

무한루프팀의 **제조공정 데이터 기반 품질위험 예측 및 공정편차 분석 MVP**입니다.

CSV·Excel 공정자료를 업로드하고 항목을 연결한 뒤, 작업별 품질위험과 공정값의 차이를 확인하는 웹 서비스를 개발합니다. 과업 범위에 따라 MVP에서는 **제조공정 1종과 품질판정 대상 1종**을 선정해 검증합니다.

현재 구현과 데이터 분석의 검증 사례는 **KAMP 자동차 부품 염욕 열처리 데이터**입니다. 열처리는 프로젝트 전체의 필수 공정이 아닙니다. 다른 공정으로 확장할 때에는 입력 항목, 품질 정답, 예측 시점과 모델을 해당 공정에 맞춰 검토합니다.

## 팀원이 먼저 볼 내용

1. 이 README에서 현재 상태와 실행 방법을 확인합니다.
2. [면담 답변 원고 PDF](output/pdf/무한루프_면담_답변원고_20261026.pdf)에서 프로젝트 설명을 읽습니다.
3. 아래 명령으로 API를 실행하고 `준비패키지/kamp/demo_upload.csv`로 시험합니다.

최신 문서는 [요약 실행안 PDF](output/pdf/무한루프_면담준비_실행안_20261026.pdf), [수정용 답변 원고](output/docs/무한루프_면담_답변원고_20261026.docx), [수정용 실행안](output/docs/무한루프_면담준비_실행안_20261026.docx)입니다. 문서의 열처리 설명은 현재 검증 사례를 기준으로 작성되어 있습니다.

## 현재 상태

| 항목 | 준비 상태 |
| --- | --- |
| 실측자료 확보 | KAMP 원본과 가이드북을 로컬에 보관하고 작업별 가공자료를 저장소에 포함 |
| 공정·품질 연결 | 원시 센서 2,939,722행을 품질 결과가 있는 136개 작업으로 연결 |
| 모델 비교 | 로지스틱 회귀·랜덤포레스트·XGBoost 비교 완료 |
| 서버 기능 | 파일 미리보기, 항목 연결, 예측, 공정편차, 결과 저장·조회 구현 |
| 검증 | API 및 데이터 검증 11건 통과 |
| 사용자 화면 | React 업로드·결과 화면은 다음 구현 단계 |
| 예측 성능 | 시제품 단계이며 개선 필요 |

품질 정답은 **136개 작업**입니다. 약 294만 센서 행을 독립적인 품질 표본 294만 개로 해석하지 않습니다.

현재 모델은 작업이 끝난 뒤의 공정 요약값을 사용합니다. 공정 중 조기예측을 구현하려면 예측 시점 이전에 얻을 수 있는 값만 사용하도록 데이터 구성을 바꿔야 합니다.

## 빠른 실행

Python 3.12를 기준으로 준비했습니다. 아래 명령은 저장소 최상위 폴더에서 실행합니다.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\준비패키지\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app_kamp:app --app-dir .\준비패키지 --host 127.0.0.1 --port 8765
```

### macOS 또는 Linux

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r 준비패키지/requirements.txt
./.venv/bin/python -m uvicorn app_kamp:app --app-dir 준비패키지 --host 127.0.0.1 --port 8765
```

브라우저에서 [API 호출 화면](http://127.0.0.1:8765/docs)을 엽니다.

1. `POST /predict`의 **Try it out**을 선택합니다.
2. `file`에 `준비패키지/kamp/demo_upload.csv`를 지정합니다.
3. `mapping`은 기본값 `{}`로 두고 실행합니다.
4. 작업별 위험 점수, 참고 범위에서 벗어난 공정값, 입력 오류를 확인합니다.

화면에 보이는 점수는 확률 보정을 검증한 실제 불량확률이 아닙니다. 학습자료의 5~95백분위 범위도 설비의 안전 운전 규격을 뜻하지 않습니다.

## 주요 파일

```text
준비패키지/
  app_kamp.py               현재 KAMP 분석 API
  app.py                    공통 업로드 읽기 함수와 이전 합성 시연 API
  prepare_kamp.py            KAMP 원시 센서 기록을 작업별로 요약
  train_kamp.py              모델 3종 비교와 시간순 평가
  verify_kamp.py             API 및 데이터 검증
  requirements.txt           Python 의존성
  run_local.ps1              Windows 실행 보조 스크립트
  시작하기.txt               데이터와 검증 방법의 상세 설명
  kamp/
    process_batches.csv     작업별 입력값
    quality_batches.csv     작업별 품질 정답
    joined_batches.csv      공정·품질 연결 자료
    demo_upload.csv         품질 정답을 제외한 예측용 20작업
    audit.json              원본 확인과 데이터 가공 기록
    benchmark.json          모델 비교 및 최종 평가 결과
    pilot_model.joblib      이 프로젝트에서 생성한 시제품 모델
    verification.json       검증 결과
output/
  docs/                     최신 면담 문서의 Word 파일
  pdf/                      최신 면담 문서의 PDF 파일
```

`준비패키지/data`, `models`, `results` 및 `generate_data.py`, `train_models.py`, `verify_pipeline.py`는 앞서 준비한 **강구 경도 합성 시연**입니다. 해당 자료의 `synthetic_demo` 출처와 KAMP 실측자료의 `KAMP_heat_quality_60` 출처를 구분합니다. 기본 실행 대상은 `app_kamp:app`입니다.

현재 `app_kamp.py`가 `app.py`의 파일 읽기 함수를 가져오므로 `app.py`와 `config.json`도 실행에 필요합니다.

## 검증과 모델 재학습

Windows에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe .\준비패키지\verify_kamp.py
.\.venv\Scripts\python.exe .\준비패키지\train_kamp.py
```

macOS·Linux에서는 Python 경로를 `./.venv/bin/python`으로 바꿉니다. 재학습한 모델을 API에 반영하려면 서버를 다시 시작합니다.

원본에서 데이터를 다시 가공하려면 KAMP에서 내려받은 `data.csv`와 `열처리_품질데이터.xlsx`가 있는 폴더를 지정합니다.

```powershell
.\.venv\Scripts\python.exe .\준비패키지\prepare_kamp.py --source "C:\data\kamp\data"
.\.venv\Scripts\python.exe .\준비패키지\train_kamp.py
.\.venv\Scripts\python.exe .\준비패키지\verify_kamp.py
```

동봉 가공자료와 모델로 API를 실행할 때는 원본 다운로드와 재학습을 생략할 수 있습니다. 원시 센서 파일은 약 540MB이므로 웹 업로드가 아니라 `prepare_kamp.py`에서 먼저 요약합니다.

## 현재 모델의 평가 결과

학습 구간의 불량률 3사분위수인 약 **0.04944% 이상**을 상대적 위험 작업으로 정의했습니다. 기업의 제품 합격·불합격 규격은 아닙니다.

| 구간 | 작업 수 | 용도 |
| --- | ---: | --- |
| 2022년 1~4월 | 78 | 학습과 전처리 기준 산출 |
| 2022년 5월 | 22 | 모델과 알림 임계값 선택 |
| 2022년 6~7월 | 36 | 선택한 모델의 최종 평가 |

검증 F2 기준으로 로지스틱 회귀와 알림 점수 임계값 0.55를 선택했습니다. 최종 평가에서는 위험 7건 중 2건을 찾았고, 미탐 5건·오탐 9건이었습니다. 재현율은 28.6%, 정밀도는 18.2%입니다.

이 모델은 현장 활용 성능을 입증하지 못했습니다. 현재 결과는 데이터 연결과 모델 비교를 실행한 기록입니다. 품질 기준, 기간별 공정 조건, 변수 선택과 평가 방식을 검토하면서 개선합니다. 최종 평가를 확인한 뒤 수정한 모델을 같은 자료로 다시 평가하는 경우에는 개발 참고값으로 구분합니다.

## 팀 역할과 개발 계획

| 팀원 | 계획한 담당 |
| --- | --- |
| 이강민 | 개발 범위, 품질 기준, 검증과 일정 총괄 |
| 박상민 | React 업로드·매핑·위험 목록·상세 화면 |
| 양희수 | 데이터 전처리, 모델 비교와 평가 |
| 장정원 | FastAPI, SQLite, 예측 연동과 실행 패키지 |

1인 주 6\~8시간, 팀 전체 주 24\~32시간을 기준으로 합니다. 내부 목표 기간은 2026년 10월 29일부터 12월 2일까지 5주이며 공식 일정에 맞춰 조정합니다.

1. 데이터·위험 기준 검토와 화면 골격
2. 업로드·항목 연결
3. 결과 화면과 변수 검토
4. 통합 검증과 멘토 피드백 반영
5. 오류 수정, 보고서·사용 설명서와 시연 준비

팀 작업은 각자 작업 브랜치에서 진행하고 Pull Request로 변경 내용을 확인한 뒤 `main`에 합치는 방식으로 시작합니다.

## 데이터 출처와 저장소 포함 범위

중소벤처기업부, Korea AI Manufacturing Platform(KAMP), **열처리 품질보증 AI 데이터셋**, 스마트제조혁신추진단(㈜임픽스), 2022.12.23.

[KAMP 데이터셋 상세 페이지](https://www.kamp-ai.kr/aidataDetail?DATASET_SEQ=60)

연구·공식 활용 시 KAMP 출처를 표기하고, KAMP가 안내하는 인용자료 전달 절차를 확인합니다. 원본 가이드북에는 개인 다운로드 정보가 표시되어 있어 저장소에 포함하지 않았습니다.

저장소에는 소스코드, 소규모 가공·합성자료, 모델·검증 결과와 최신 면담 문서를 포함합니다. 원본 ZIP·원시 CSV, 개인화된 가이드북, 신청서·평가메일 등 참고자료, 임시 설치 파일, 실행 중 생성되는 SQLite 파일은 `.gitignore`로 제외합니다.

