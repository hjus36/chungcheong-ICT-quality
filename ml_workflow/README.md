# ml_workflow — 품질위험 예측 모델 학습 워크플로우

웹 서비스(`준비패키지/`)가 쓰는 모델을 **만들고, 검증하고, 승격**하는 파이프라인이다.
단계별 설계·누수 방지·선택 규칙·설정 항목·한계는 **[WORKFLOW.md](WORKFLOW.md)**에 있다.

## 원칙
- **사람이 판단할 것** → `config/columns.yaml`: 컬럼 역할, 접두사별 측정 단계(사후측정 제외), 예측 시점, 그룹 키, 서비스 메타·단위·설명
- **분석 기준값** → `config/experiment.yaml`: 분할, FDR, 반복 그룹 CV, k 규칙, 모델, 선택 규칙(Gate 1·2), 위험 구간, 기준선
- **배포 기준** → `config/release.yaml`: 승격 조건, 출처, 화면용 분석 설정, 한계
- **기계적으로 계산할 것** → 코드가 실행마다 데이터에서 다시 계산하고, 보고서의 표·수치도 코드가 만든다.

## 실행
```bash
pip install -r ../준비패키지/requirements-lock.txt -r requirements-train.txt
python run.py --run im_v3            # validate → eda → split → filter → wrapper → model → explain → release → report
python run.py --resume im_v3         # 중단된 단계부터 (설정·코드·데이터가 같을 때만)
python promote.py --run im_v3        # 승격 조건 통과 시 준비패키지/release/로 복사
python make_synthetic.py             # 준비패키지/share/ 합성 예제 다시 생성
python make_upload.py --input data/dataset_V2.parquet --holdout-of im_v3 --output out.csv   # 원본 → 업로드 형식
python -m unittest discover -s tests -v
```
Windows: `scripts\setup.ps1 -Train` 후 `scripts\train.ps1 -Run im_v3 -Promote`.

원본 데이터는 git에 넣지 않는다. 받는 방법과 해시는 [data/README.md](data/README.md).

## 단계
| 폴더 | 내용 |
|---|---|
| `00_validate` | 타깃·그룹 키 확인, 모든 컬럼의 역할 판정. **측정 단계가 정해지지 않은 컬럼이 있으면 중단** |
| `01_eda` | 라벨 분포, 그룹 라벨 순도, 결측, 상수·중복, 단일 변수 검정 |
| `02_split` | 그룹(실험일+실험번호) 단위 holdout을 한 번 정해 저장 |
| `03_filter` | dev만으로 F-test + BH-FDR 특징 순위 |
| `04_wrapper` | 모델별 특징 수(k) 탐색 — 반복 그룹 CV, 1-SE 규칙 |
| `05_model` | 4개 모델 비교 → Gate 1(Recall 0.9 지점 Precision ≥ 0.4) → Gate 2(실험일 제외 Recall ≥ 0.5) → PR-AUC → 1-SE 동점 처리. 위험 구간, 기준선, holdout(개발 참고 평가) |
| `06_explain` | permutation importance, 영향 방향, 정상 부품 참고 구간 |
| `07_release` | 서비스 번들·데이터 사전·공개 벤치마크·승격 조건 판정, 실측 예제(원본 holdout → 업로드 형식)와 판정 일치 검사 |
| `99_report` | `report.md`와 행 단위 자료를 뺀 `public/` 사본 |

## 현재 서비스 모델 (`im_v2_r3`)
ProBayes 사출성형 v2, 564 사이클(불량 27.5%). **Logistic Regression, 특징 2개**(`DXP_HoldingPressure1` 필수, `CALC_InjectionStroke` 선택). 네 모델 중 유일하게 Gate 1·2를 통과했다.
holdout 121부품(개발 참고 평가): PR-AUC 0.841, Recall 1.00, Precision 0.60, F2 0.882 — 전부 우선 검토(F2 0.679)보다 높다.
자세한 수치는 `runs/im_v2_r3/report.md`(로컬)와 `준비패키지/release/benchmark_public.json`. (`im_v2_r2`와 결과 동일, r3에서 실측 예제 생성 추가)

승격하면 holdout 121행 실측 예제가 `준비패키지/local/real_demo.csv`(git 제외)로 복사되어 웹의 **실측 예제 불러오기**가 켜진다. 웹에서 분석한 판정(36/0/24/61)은 리포트 holdout 결과와 같으며, 다르면 `07_release`가 승격을 막는다.
