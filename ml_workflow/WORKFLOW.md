# 제조 공정 품질위험 예측 — 하네스 워크플로우 설계 문서

> 이 문서는 **설계**(무엇을 왜 이렇게 하는지)를 설명한다. 실행 결과는 실행마다 생성되는 `runs/<실행 폴더>/report.md`에 있다.
> 대상 과제: 충청권 ICT 현장적용 프로젝트 **[과제 6] 제조공정 품질위험 예측 MVP** — 공정조건 기반 품질위험 예측 및 주요 영향변수 분석

## 목차
1. [전체 구조](#overview)
2. [단계별 상세](#stages)
3. [데이터 누수 방지 장치](#leakage)
4. [모델 선택 규칙과 위험 구간](#selection)
5. [설정 항목](#config)
6. [데이터셋 변경 절차와 자주 생기는 오류](#dataset-change)
7. [알려진 한계](#limitations)

---

<a id="overview"></a>
## 1. 전체 구조

### 1.1 설계 원칙

| 원칙 | 구현 |
|---|---|
| 사람이 판단할 것과 기계가 계산할 것을 분리 | 판단 → `config/columns.yaml`, 기준값 → `config/experiment.yaml`, 계산 → 코드 |
| 데이터셋이 바뀌어도 그대로 동작 | 컬럼 이름 목록이 아니라 **접두사·역할 규칙**으로 특징을 고름. 특징 수·상수·중복 등은 매번 데이터에서 다시 계산 |
| 모르는 것은 조용히 넘기지 않음 | 측정 단계가 정해지지 않은 컬럼이 있으면 실행 중단 |
| 평가는 실제 사용 상황과 같게 | 같은 실험 조건이 학습·평가에 섞이지 않는 **그룹 분할**, 처음 보는 실험일에 대한 진단 |
| 재현 가능 | 실행마다 `runs/<시각>_<태그>/`에 결과·설정 스냅샷·데이터 해시·패키지 버전 저장 (덮어쓰지 않음) |
| 문서와 코드가 어긋나지 않게 | 보고서의 표·수치는 코드가 생성 (`99_report`) |

### 1.2 흐름

```
config/columns.yaml ──┐
config/experiment.yaml┤
                      ▼
dataset ─► 00 validate ─► 01 eda ─► 02 split ─┬─► dev ─► 03 filter ─► 04 wrapper ─► 05 model ─► 06 explain ─► 99 report
                                              └─► holdout (05 model의 마지막 평가 전까지 사용 금지)
                                                                                     │
                                                         predict.py ◄─ model.joblib ◄┘
```

### 1.3 폴더 구성

```
harness/
├── config/
│   ├── columns.yaml       사람이 판단: 컬럼 역할, 접두사별 측정 단계, 예측 시점, 그룹 키
│   └── experiment.yaml    분석 기준: 품질 검사, 분할, FDR, CV, k 규칙, 모델, 선택 규칙, 위험 구간
├── src/                   공통 함수
│   ├── columns.py         컬럼 역할 판정
│   ├── groups.py          그룹 키 생성
│   ├── quality.py         상수·완전 중복·근사 중복
│   ├── stats.py           BH-FDR, 단일 변수 AUC·Mann–Whitney
│   ├── design.py          결측 대체 + F-test/FDR 순위
│   ├── models.py          모델 생성
│   ├── evaluate.py        파이프라인·확률 보정·지표·판정 기준
│   ├── selection.py       모델 선택 규칙
│   ├── pipeline.py        분할 읽기·그룹 CV fold
│   └── runio.py           실행 폴더 입출력
├── stages/                단계별 스크립트 (validate, eda, split, feature_filter, wrapper, model, explain, report)
├── run.py                 실행기
├── predict.py             학습된 모델로 새 데이터 예측
└── runs/                  실행 결과
```

### 1.4 실행

```bash
pip install -r requirements.txt
python run.py                                   # 전체 단계
python run.py --stages validate                 # 새 데이터셋 점검만
python run.py --tag no_indicator                # 설정을 바꿔 비교 실행 (폴더 이름에 태그)
python run.py --run-dir runs/<폴더> --stages model explain report   # 기존 실행에 이어서
python predict.py runs/<폴더> new_data.csv -o predictions.csv
```

---

<a id="stages"></a>
## 2. 단계별 상세

<a id="s00"></a>
### 2.1 `00_validate` — 스키마 검사

- **목적:** 분석 전에 "이 데이터를 이 규칙으로 돌려도 되는가"를 확인한다.
- **처리**
  1. 타깃이 0/1 이진값인지, 그룹 키 컬럼이 있고 결측이 없는지 확인
  2. 모든 컬럼의 역할 판정 (순서대로 적용)
     - `target` → 타깃 / `roles.id`, `roles.time`, `roles.meta` → 제외 / `roles.label_prefixes`(LBL_) → 사후 라벨
     - 접두사의 측정 단계(`prefix_stage`, 컬럼별 `column_overrides` 우선)가 없으면 → **미분류**
     - `manual_drop` → 사유와 함께 제외
     - 측정 단계가 예측 시점(`prediction_time`)보다 늦으면 → 제외 (`offline`은 `allow_offline`일 때만 허용)
     - 숫자형이 아니면(시계열 배열·문자열) → 제외
     - 나머지 → **입력 후보**
  3. 미분류 컬럼이 있거나 입력 후보가 0개면 **중단**
- **산출물:** `column_roles.csv`(모든 컬럼의 역할·단계·사유), `validation.json`
- **설계 이유:** 측정 시점은 데이터 통계로 알 수 없는 공정 지식이다. 접두사 단위로 한 번만 정해두면 새 데이터셋에서 컬럼 수가 바뀌어도 그대로 적용된다. 처음 보는 접두사는 사람이 분류할 때까지 진행하지 않는다.

<a id="s01"></a>
### 2.2 `01_eda` — 탐색 (전체 행)

- **목적:** 데이터 이해와 이후 설계의 근거 확보. 탐색 전용이므로 전체 행을 쓰고, 여기서는 아무것도 선택하지 않는다.
- **처리와 산출물**

| 분석 | 파일 |
|---|---|
| 자료형, 제외 컬럼 요약, 입력 후보 목록 | `dtype_counts.csv`, `excluded_summary.csv`, `candidate_features.csv` |
| 라벨 분포 | `label_distribution.csv` |
| **그룹별 라벨 순도** (그룹 안에 정상·불량이 섞여 있는가) | `group_label_purity.csv` |
| 그룹 키 첫 부분(실험일)별 구성 | `composition_by_<컬럼>.csv` |
| 결측률, 결측률 구간, 행 단위 결측 패턴, 패턴 × 실험일 | `missing_rates.csv`, `missing_bins.csv`, `missing_patterns*.csv` |
| 결측 위치가 같은 특징 묶음별 결측 여부와 NOK | `missingness_vs_target.csv` |
| 상수, 완전 중복, 근사 중복(\|Spearman ρ\| ≥ 기준) | `constant_features.csv`, `exact_duplicates.csv`, `near_duplicates.csv` |
| 단일 변수 AUC 구분력 + Mann–Whitney U + BH-FDR (관측 행만) | `univariate.csv` |

- **설계 이유:** 모든 그룹의 라벨이 고정이면 행 단위 무작위 분할은 "어느 그룹인지"만 맞혀도 점수가 높아진다. 이 경고가 그룹 분할을 쓰는 근거다. 결측 패턴이 실험일과 1:1이면 결측 indicator가 실험일 대리 변수가 된다.

<a id="s02"></a>
### 2.3 `02_split` — Holdout 분리

- **목적:** 마지막 평가에만 쓸 데이터를 가장 먼저 떼어놓는다.
- **처리:** 그룹 키(기본: 실험일 + 실험번호) 단위로 나누고, 그룹의 다수 라벨로 층화한다(`split.strategy: group`). 비교용으로 `random`(행 단위 층화)도 지원한다. dev·holdout 양쪽에 두 클래스가 모두 없으면 중단한다.
- **산출물:** `split.csv`(행별 dev/holdout), `split_summary.json`
- **설계 이유:** 실험번호(A01 등)는 실험일마다 다시 쓰이므로 날짜와 함께 써야 같은 조건을 구분할 수 있다. 이후 모든 단계가 이 파일을 읽어 같은 분할을 쓴다.

<a id="s03"></a>
### 2.4 `03_filter` — 특징 순위 (dev만)

- **목적:** 후보 특징을 정리하고, Wrapper에서 특징을 넣을 **순서**를 정한다.
- **처리**
  1. 품질 제거 (dev 기준 재판정): 전체 결측 → 상수(관측값 1종) → 완전 중복(앞쪽 컬럼 유지) → 근사 중복(`action: drop`일 때만)
  2. dev 중앙값으로 결측 대체, 결측이 있는 특징에 결측 indicator 추가(`missing_indicators`)
  3. 값 열과 indicator 각각 ANOVA F-test (`f_classif`) → 전체 p-value에 BH-FDR 보정
  4. 값 또는 indicator 중 하나라도 q < `fdr_alpha`이면 원본 특징을 후보로 두고, 통과 항목 중 최대 F 점수로 순위
- **산출물:** `feature_set.csv`(사용/제거와 사유), `filter_scores.csv`(항목별 F·p·q), `filter_ranking.csv`(특징 순위), `near_duplicates_dev.csv`
- **설계 이유:** F-test는 특징별 평균 차이만 보는 1차 필터이고, 행들이 독립이라고 가정하므로 같은 조건의 반복 생산이 있으면 p-value가 실제보다 작게 나온다. 최종 판단은 Wrapper의 그룹 CV에 맡긴다. 결측 indicator로만 순위에 든 특징은 보고서에서 경고한다.

<a id="s04"></a>
### 2.5 `04_wrapper` — 모델별 특징 수 탐색

- **목적:** 모델마다 예측에 도움이 되는 특징 개수(k)를 정한다.
- **처리**
  1. dev를 그룹 단위 층화 K-fold(`StratifiedGroupKFold`)로 나눈다. 각 검증 fold에 두 클래스가 모두 있어야 한다.
  2. fold마다 **학습 부분만으로** 03의 순위를 다시 계산한다. 순위는 모델과 무관하므로 fold·indicator 조건마다 한 번만 계산해 모든 모델이 공유한다.
  3. 모델 × indicator 조건마다 `k_grid`의 각 k에 대해 상위 k개로 학습하고, 검증 fold의 PR-AUC·Recall을 기록한다.
  4. `k_rule`로 모델별 k 선택
     - `one_se`(기본): 최고 PR-AUC − (그 지점 표준편차 / √fold 수) 이상인 가장 작은 k
     - `tolerance`: 최고 PR-AUC의 `pr_auc_relative_tolerance` 비율 이내인 가장 작은 k
  5. dev 전체 순위의 상위 k개를 그 모델의 최종 특징으로 저장
- **산출물:** `cv_folds.csv`, `fold_rankings_<조건>.csv`, `curve_<모델>_<조건>.csv`, `selected_features_<모델>.csv`, `model_feature_sets.json`(모델 단계 입력), `wrapper_comparison.csv`, `feature_overlap.csv`, `feature_count_curves.png`
- **설계 이유**
  - 한 모델(예: SVM)로 k를 정해 모든 모델에 쓰면 그 모델에 유리한 비교가 된다. 모델마다 자기 k를 고르게 해 비교를 공정하게 한다.
  - 특징 순서를 공유하므로 모델 간 특징 집합이 포함 관계가 되어 해석이 쉽다.
  - 그룹 CV 곡선은 흔들림이 커서 최고점 기준 규칙은 우연한 봉우리를 고르기 쉽다. 1-SE 규칙은 오차 범위 안에서 가장 단순한 쪽을 고른다.
  - `param_overrides`로 트리 모델은 탐색 중에만 트리 수를 줄인다(모델 단계는 원래 설정).

<a id="s05"></a>
### 2.6 `05_model` — 모델 비교, 선택, 위험 구간, Holdout

- **목적:** 과제 6의 "3종 이상 비교 후 1종 선정"과 품질위험도 구간을 결정한다.
- **처리**
  1. **그룹 CV + OOF:** 04와 같은 fold. fold마다 순위를 다시 계산해 모델별 k개로 학습하고(`calibrate`면 학습 데이터 안에서 확률 보정), 검증 fold 예측을 모아 dev 전체의 OOF 예측을 만든다.
  2. **모델별 지표:** fold별 PR-AUC·ROC-AUC, OOF 기반 Brier, OOF에서 Recall ≥ `target_recall`을 만족하는 가장 높은 판정 기준과 그때의 Precision
  3. **실험일 제외 진단(모든 모델):** dev 안에서 `leave_one_out_by`(실험일)의 한 값을 통째로 빼고 나머지로 순위 계산·학습 → 빠진 실험일에서 PR-AUC, CV 판정 기준을 적용한 Recall·Precision
  4. **선택 규칙** 적용 → [4장](#selection)
  5. **위험 구간:** 선택 모델의 OOF에서 `defect_recall`, `caution_recall` 지점의 기준을 구해 정상/품질주의/불량 위험 구간을 정한다.
  6. **Holdout:** dev 전체로 각 모델을 학습하고, **CV에서 정한 판정 기준을 고정 적용**해 평가한다. 모든 모델을 기록하지만 선택에는 쓰지 않는다.
- **산출물:** `cv_fold_metrics.csv`, `oof_predictions.csv`, `leave_one_out.csv`, `selection_trace.csv`(선택 기록표), `oof_bands.csv`, `thresholds.json`, `holdout_metrics.csv`, `holdout_predictions.csv`, `holdout_bands.csv`, `model_selection.png`, `model_summary.json`
- **설계 이유:** 모델 기본 판정 기준(확률 0.5)은 데이터에 맞지 않을 수 있다. 기준을 holdout이 아닌 OOF로 정해 고정하면, "기준을 미리 정해두고 새 데이터에 적용했을 때"의 정직한 성능이 된다.

<a id="s06"></a>
### 2.7 `06_explain` — 영향변수, 정상범위, MVP 패키지

- **목적:** 과제 6의 "품질결과에 영향을 미치는 주요 공정변수와 정상범위 확인 기능"
- **처리**
  1. 선택 모델을 dev 전체로 다시 학습 (05와 같은 설정)
  2. **영향변수:** holdout permutation importance (특징을 섞었을 때 PR-AUC 감소량). 원본 특징을 섞으면 결측 indicator도 함께 바뀐다.
  3. **영향 방향:** dev에서 특징값과 OOF 위험도의 Spearman 상관 부호
  4. **정상범위:** dev 정상(0) 샘플의 `normal_range_percentiles` 구간, 불량 중 범위 밖 비율, 실험일별 정상범위
  5. **패키지:** `model.joblib`(전처리 포함 파이프라인), `model_card.json`(특징 순서, 위험 구간 기준, 정상범위, 버전)
- **산출물:** `feature_importance.csv`/`.png`, `normal_ranges.csv`, `normal_ranges_by_<컬럼>.csv`, `model.joblib`, `model_card.json`
- **`predict.py`:** `model_card.json`의 특징이 모두 있는지 확인 → 위험도(보정 확률) → 구간 → 특징별 정상범위 이탈 여부를 CSV로 출력. 웹 MVP가 같은 방식으로 불러 쓰면 된다.

<a id="s99"></a>
### 2.8 `99_report` — 보고서

- 모든 단계 산출물로 `runs/<폴더>/report.md`를 만든다. 표·수치·그래프는 자동, 섹션마다 `> 해석:` 자리만 사람이 작성한다.
- 구성: 실행 정보 → 데이터·특징 구성 → 결측 → 변수 품질 → 타깃 연관성 → Filter → Wrapper → 모델 비교·선정(벤치마크) → 영향변수·정상범위 → 결과 해석 및 한계
- 선행 단계 결과가 없으면 해당 섹션만 건너뛴다.

---

<a id="leakage"></a>
## 3. 데이터 누수 방지 장치

| 위험 | 장치 | 단계 |
|---|---|---|
| 예측 시점 이후 측정값 사용 (탈형 후 IR·CV, 냉각 후 SCA) | 접두사별 측정 단계 + 예측 시점 비교로 자동 제외 | 00 |
| 처음 보는 컬럼이 그대로 입력에 들어감 | 미분류 컬럼 발견 시 중단 | 00 |
| 같은 실험 조건이 학습·평가에 섞임 | 그룹 단위 holdout, 그룹 K-fold | 02, 04, 05 |
| 평가 데이터가 특징 선택에 영향 | holdout 격리, fold마다 학습 부분으로 순위 재계산, 대체값·스케일링도 학습 부분 기준 | 03~05 |
| 판정 기준을 평가 데이터로 맞춤 | 판정·구간 기준은 OOF로 결정 후 holdout에 고정 적용 | 05 |
| 결측 패턴으로 실험일을 맞힘 | indicator 켬/끔 비교, indicator로만 순위에 든 특징 경고 | 03, 04 |
| 처음 보는 실험일에서의 성능 과대평가 | 실험일 제외 진단 (모든 모델) | 05 |
| 행 단위 분할 점수 과대평가 | 그룹 라벨 순도 경고 | 01 |

---

<a id="selection"></a>
## 4. 모델 선택 규칙과 위험 구간

### 4.1 과제 요구와 기준의 연결

| 과제 6 요구 | 선택 기준 |
|---|---|
| 생산 중 품질위험을 사전에 확인하는 **예방형** 품질관리 | 불량을 놓치지 않는가 → Recall 목표 지점에서 평가 (gate) |
| 정상·품질주의·불량 가능성 예측, **품질위험도** 표시 | 위험도 순서의 정확도 → PR-AUC / 위험도 %의 신뢰도 → Brier |
| 주요 영향변수와 정상범위 | 해석 가능성 (동점일 때) |
| 향후 실제 제조데이터로 확장 | 처음 보는 실험일에서의 성능 (동점일 때) |
| 3종 이상 비교 후 1종 선정, 벤치마크 보고서 | 규칙과 선택 기록표로 근거 제시 |

### 4.2 규칙 (`model.selection`)

1. **Gate:** OOF 예측에서 Recall ≥ `target_recall`(기본 0.90)을 만족하는 가장 높은 판정 기준을 찾고, 그때 Precision ≥ `min_precision`(기본 0.40)인 모델만 남긴다. 통과 모델이 없으면 `if_none_pass`에 따라 gate를 생략(경고)하거나 중단한다.
2. **주 지표:** 그룹 CV fold별 `primary`(PR-AUC)의 평균이 가장 높은 모델이 기준점.
3. **동점 처리:** 기준점 모델의 평균 − (표준편차 / √fold 수) 이상인 모델을 동점 집합으로 본다. 동점 집합 안에서 `tie.order` 순서로 비교하며, 각 기준에서 `tie.tolerance` 이내 차이는 같다고 보고 다음 기준으로 넘어간다.
   - `lodo_PR_AUC`: 실험일 제외 진단 PR-AUC 평균 (높을수록)
   - `Brier`: OOF Brier score (낮을수록)
   - `interpretability`: 해석 가능성 순서 (기본: Logistic Regression > XGBoost > Random Forest > SVM)
4. 모든 판정은 `selection_trace.csv`에 모델별로 기록된다 (gate 통과 여부, 동점 집합 포함 여부, 밀려난 기준).

### 4.3 위험 구간 (`model.risk_bands`)

선택 모델의 OOF 예측에서 다음 기준을 구한다.

| 구간 | 조건 | 기본 기준점 |
|---|---|---|
| 불량 위험 | 위험도 ≥ defect 기준 | OOF Recall 0.90 지점 |
| 품질주의 | caution 기준 ≤ 위험도 < defect 기준 | OOF Recall 0.98 지점 |
| 정상 | 위험도 < caution 기준 | |

기준값은 `thresholds.json`과 `model_card.json`에 저장되어 `predict.py`와 웹 MVP가 그대로 쓴다.

---

<a id="config"></a>
## 5. 설정 항목

### 5.1 `config/columns.yaml` (사람이 판단)

| 키 | 설명 | 현재 값 |
|---|---|---|
| `target` | 타깃 컬럼 | `LBL_NOK` |
| `roles.id` / `roles.time` / `roles.meta` | 항상 제외할 식별자·시간·메타 컬럼 | Job/MachineCycleID, TimeCycleStart/End, ExperimentNumber/MaterialName |
| `roles.label_prefixes` | 사후 라벨 접두사 | `LBL_` |
| `group.columns`, `group.transforms` | 그룹 키와 변환 (`date`: 날짜만) | `MET_TimeCycleStart`(date) + `MET_ExperimentNumber` |
| `prediction_time` | 예측 시점. `stage_order`에서 이 단계까지만 입력 후보 | `in_cycle` |
| `stage_order` | 측정 단계 순서 | pre_production → in_cycle → post_ejection → post_cooling |
| `allow_offline` | `offline` 단계(시뮬레이션) 허용 여부 | `false` |
| `prefix_stage` | 접두사 → 측정 단계 | SET/MET/ENV/DRY: 생산 전, QUA/DXP/TCE/TCN/CALC/DOS/E77: 사이클 중, IR/CV: 탈형 후, SCA: 냉각 후, SIM: offline |
| `column_overrides` | 컬럼별 예외 (`stage`, `reason`) | 없음 |
| `manual_drop` | 컬럼별 수동 제외 (`column`, `reason`) | 없음 |

`prediction_time`을 `pre_production`으로 바꾸면 생산 전 설정값(SET/MET/ENV)만으로 예측하는 모델이 된다.

### 5.2 `config/experiment.yaml` (분석 기준)

| 키 | 설명 | 기본값 |
|---|---|---|
| `data.path`, `data.fallback_csv` | 데이터 경로 (parquet 엔진이 없으면 CSV) | `../dataset/dataset_V2.parquet` |
| `seed` | 난수 시드 | 42 |
| `features.drop_all_missing` / `drop_constant` / `drop_exact_duplicates` | 품질 제거 | 모두 true |
| `features.near_duplicate` | 근사 중복 기준·처리 (`report` \| `drop`) | 0.99, report |
| `features.missing_indicators` | 결측 indicator 사용 (다음 단계로 넘기는 조건) | true |
| `split.strategy`, `split.holdout_size` | holdout 분할 방식·비율 | group, 0.2 |
| `eda.fdr_alpha`, `eda.missing_bins` | EDA 검정 기준, 결측률 구간 | 0.05 |
| `filter.fdr_alpha` | Filter FDR 기준 | 0.05 |
| `cv.n_splits` | 그룹 K-fold 수 | 5 |
| `wrapper.estimators` | k를 탐색할 모델 | 4종 |
| `wrapper.k_rule`, `pr_auc_relative_tolerance`, `min_recall` | k 선택 규칙 | one_se |
| `wrapper.k_grid` | 탐색할 k (`"1-20, 25-max:5"`) | 1~20 전부, 이후 5 간격 |
| `wrapper.compare_missing_indicators` | indicator 켬/끔 모두 계산 | true |
| `wrapper.param_overrides` | 탐색 중에만 쓰는 모델 설정 | RF·XGBoost 트리 200개 |
| `model.candidates` | 비교 모델 | LR, RF, XGBoost, SVM |
| `model.calibrate`, `model.calibration` | 확률 보정 (학습 데이터 안에서) | true, sigmoid, cv 3 |
| `model.selection.*` | 선택 규칙 ([4.2](#selection)) | gate 0.90/0.40, PR-AUC, one_se |
| `model.risk_bands.*` | 위험 구간 기준점 | 0.90 / 0.98 |
| `model.leave_one_out_by` | 하나씩 제외할 그룹 키 부분 | `MET_TimeCycleStart` |
| `model.params.<모델>` | 하이퍼파라미터 | `class_weight` 등 |
| `explain.permutation_repeats`, `normal_range_percentiles` | 중요도 반복 수, 정상범위 | 30, [5, 95] |
| `report.top_n_univariate`, `top_n_ranking` | 보고서 표 길이 | 30 |

---

<a id="dataset-change"></a>
## 6. 데이터셋 변경 절차와 자주 생기는 오류

### 6.1 절차

1. `experiment.yaml`의 `data.path`를 새 파일로 바꾼다.
2. `python run.py --stages validate`
3. 미분류 컬럼이 나오면 `00_validate/column_roles.csv`에서 확인하고, `columns.yaml`의 `prefix_stage`(접두사 단위) 또는 `column_overrides`(컬럼 단위)에 측정 단계를 추가한다. 측정 시점은 데이터 설명서나 공정 담당자에게 확인한다.
4. 타깃·그룹 키 이름이 바뀌었으면 `target`, `roles`, `group`을 수정한다.
5. `python run.py --tag <데이터셋 이름>`
6. `report.md`에서 먼저 확인할 것: 그룹 라벨 순도 경고, indicator 경고, 실험일 제외 진단

### 6.2 자주 생기는 오류

| 메시지 | 원인 | 조치 |
|---|---|---|
| 측정 단계가 정해지지 않은 컬럼 N개 | 새 접두사 또는 접두사 없는 컬럼 | `prefix_stage` / `column_overrides`에 추가 |
| 타깃이 0/1 이진값이 아닙니다 | 타깃 인코딩이 다름 | 데이터 전처리 또는 `target` 확인 |
| 그룹 키에 결측이 있습니다 | 그룹 컬럼 누락 행 | 데이터 정리 또는 `group.columns` 변경 |
| N번째 fold 검증 데이터에 한 클래스만 있습니다 | 그룹 수가 적거나 불균형 | `cv.n_splits` 줄이기, `seed` 변경 |
| dev/holdout에 한 클래스만 있습니다 | 그룹 수 부족 | `split.holdout_size` 조정, `seed` 변경 |
| FDR 후보가 k보다 적습니다 | fold 학습 데이터에서 통과 특징이 적음 | `filter.fdr_alpha` 완화, `wrapper.max_features` 지정 |
| gate를 통과한 모델이 없습니다 (stop) | 목표 Recall에서 Precision 부족 | 기준 재협의 또는 `if_none_pass: skip_gate` |
| xgboost 패키지가 없어 제외 | 미설치 | `pip install xgboost` (없어도 나머지 모델로 진행) |
| parquet 엔진이 없어 CSV로 읽습니다 | pyarrow 미설치 | `pip install pyarrow` (CSV로도 진행) |

---

<a id="limitations"></a>
## 7. 알려진 한계

- **실험 조건 수가 적다.** 라벨이 47개 실험 조건 단위로 고정되어 있어, 실제 독립 표본은 564개가 아니라 47개에 가깝다. CV 점수의 편차가 크고 k·모델 선택이 흔들린다.
- **실험일 일반화가 약하다.** 실험일이 3개(재료 2종)뿐이라 처음 보는 실험일에서 성능이 크게 떨어지는 모델이 있다. 현재 선택 규칙은 실험일 제외 성능을 **동점일 때만** 반영한다. gate 조건으로 올릴지 검토가 필요하다.
- **특징 순서는 F-test 기준이다.** 평균 차이가 작지만 비선형·조합 효과가 있는 특징은 순위 뒤로 밀려 트리 모델이 쓸 기회를 놓칠 수 있다.
- **모델별 k를 같은 CV로 고르고 비교한다.** 네 모델 모두 같은 방식이라 비교는 공정하지만 절대 점수는 다소 낙관적이다. 절대 성능은 holdout과 실험일 제외 진단으로 판단한다. (그룹이 적어 이중 교차검증은 쓰지 않았다.)
- **확률 보정은 그룹을 고려하지 않는다.** `CalibratedClassifierCV`의 내부 분할은 행 단위이므로 보정이 다소 낙관적일 수 있다.
- **영향변수는 holdout 121행(10개 그룹) 기준이다.** 중요도의 편차가 크며, 상관이 매우 높은 특징끼리는 중요도를 나눠 가진다(예: `CALC_InjectionStroke`와 `CALC_DosingStroke`).
- **정상범위는 관측 기반 통계 구간이다.** 공정 규격이나 인과 관계가 아니므로, 실제 관리 기준으로 쓰기 전에 공정 담당자 검토가 필요하다.
