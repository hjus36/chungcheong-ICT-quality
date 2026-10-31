# 품질모델 개선 실험 · 2026-11-01

결론: **기존 웹 RandomForest(경고 0.35)를 유지한다.** 새 후보는 시간순 학습 내부에서는 좋아 보였지만 5월 검증에서 악화됐다. 추가 확보한 UCI SECOM도 별도 반도체 사례이며 열처리 성능 입증 자료가 아니다. 전체 수치는 [벤치마크 보고서](../../docs/06_모델_벤치마크_보고서.md)에 있다.

## 로컬 연구 폴더 구조 (GitHub 포함 범위와 다름)

- `protocol.json`: 후보 학습 전에 정한 분할·탐색·선정·교체 기준.
- `prepare.py`, `common.py`, `train.py`: 감사, 원시센서 시간 특성, 학습 전처리와 비교.
- `results/`: 감사·16개 공정 관계·35개 시간 특성 및 정의/해시.
- `runs/kamp_v1`, `runs/secom_v1`: CV, OOF 예측, 검증, 후보 모델, 선정 잠금, 적용 결정, 최종 평가. 기존 결과를 덮어쓰지 않는다.
- `sources/uci_secom`: 공식 원본·다운로드 해시·CC BY 4.0 출처. 큰 원본 파일은 Git 제외.
- `source_review.json`: 동일/유사 공정을 우선 조사한 근거와 채택/제외 이유.
- `baseline_snapshot/`: 이번 개선 전 README·문서·검증 기록·현재/초기 모델·DB 복사와 해시. SQLite는 Git 제외.
- `verification/`: 이번 API·모델·재시작·브라우저 확인. 첫 검증의 Windows 문자 인코딩 실패도 `experiment_checks_attempt1.json`으로 보존.

GitHub에는 원본·KAMP 개별 행 CSV·특성행렬·baseline_snapshot을 포함하지 않습니다. 후보 모델은 사전 선정한 KAMP/SECOM 각 1개만 포함하며 나머지는 로컬에 보존합니다. 코드·설정·집계 비교·출처·해시·검증 JSON은 공유합니다. [Clone 이후 자료 재확보 및 재현 절차](../../docs/11_데이터_재현과_저장소_운영.md)를 먼저 읽으세요.

## 재현

프로젝트 루트, 기존 `.venv`와 `준비패키지/requirements-lock.txt`를 사용한다. 일반 웹 실행은 재학습할 필요가 없다.

```powershell
# 새 PC에 SECOM 원본이 없을 때만: 공개 UCI 원본 약1.9MB, CC BY4.0
.\.venv\Scripts\python.exe experiments/quality_20261101/download_secom.py

# 기존 연구 PC 전용: 원본·작업별 결과·baseline_snapshot을 갖춘 환경에서만 검증
.\.venv\Scripts\python.exe experiments/quality_20261101/verify.py

# 학습 재현: 반드시 아직 없는 새 run 이름 사용
.\.venv\Scripts\python.exe experiments/quality_20261101/train.py --dataset kamp --run kamp_reproduction_01
.\.venv\Scripts\python.exe experiments/quality_20261101/train.py --dataset secom --run secom_reproduction_01
```

`train.py`는 기존 run 이름이면 중단하며 웹의 `kamp/release`나 SQLite 이력은 수정하지 않는다. 재현은 이미 본 평가의 재현일 뿐 새 독립 검증이 아니다. 결과를 본 뒤 후보·임계값을 고쳐 동일 test에서 다시 좋은 숫자를 고르지 않는다.

KAMP 시간 특성은 원본 CSV에서 `prepare.py --temporal`로 생성했다. 저장 결과가 이미 있으면 보호를 위해 중단한다. 재추출은 `prepare.py --temporal --output-dir tmp/temporal_reproduction_01`로 새 폴더에 저장하고 원래 CSV의 해시와 비교한다. `prepare.py`의 기본 감사는 원본과 저장 모델을 읽으며 초기 스냅샷이 있으면 새로 덮어쓰지 않는다. 실제 첫 학습 당시 코드4개는 각 run의 `code/`에 보존했으며 environment.json의 SHA256과 일치한다. 현재 코드에는 재현용 출력 경로, 자동 코드 보존, 다운로드 파일의 해시 검증·재사용 처리를 추가했고 학습·선정 로직은 바꾸지 않았다.

회귀 증거를 기존 문서 기록과 분리하려면:

```powershell
$env:ICT_EVIDENCE_DIR = "$PWD/tmp/test-evidence"
.\.venv\Scripts\python.exe tests/test_system.py
.\.venv\Scripts\python.exe tests/verify_restart.py
Remove-Item Env:ICT_EVIDENCE_DIR
```

이번 보고서 생성: `.\.venv\Scripts\python.exe experiments/quality_20261101/report.py`. 생성 초안은 `tmp/generated_reports/`에 저장하고 현재 README·docs를 덮어쓰지 않는다. 로컬 원본과 baseline_snapshot이 필요하다. 기준 실험은 `kamp_v1`, `secom_v1`로 고정한다. 새 실험을 보고서 기준으로 바꾸려면 모델 선정과 평가 규칙부터 새로 검토한다.

## 모델 입력과 설명 범위

기존 앱은 36개 평균·표준편차 입력을 유지한다. 실험용 16개 공정 관계와 35개 시간 특성, 반도체 590개 익명 신호는 웹 입력에 추가하지 않았다. 저장 후보를 연구용으로 불러올 때는 `common.py`가 Python 경로에 있어야 한다. 모델 파일은 신뢰하는 이 프로젝트 산출물만 `joblib.load`로 연다.

McCann, M. & Johnston, A. (2008). **SECOM [Dataset]**. UCI Machine Learning Repository. [DOI 10.24432/C54305](https://doi.org/10.24432/C54305). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 원본은 그대로 보존했고, 시간 분할·결측 처리·특성 선택·모델링은 본 프로젝트에서 수행했다.
