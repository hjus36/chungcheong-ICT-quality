# 품질위험 예측 하네스

> 단계별 상세 설계, 누수 방지 장치, 설정 항목, 한계는 **[WORKFLOW.md](WORKFLOW.md)** 참고.

기존 `ML_model.py`, `WrapperFeatureSelection.py`와 결과 폴더는 건드리지 않는 별도 파이프라인이다.

## 원칙
- **사람이 판단할 것** → `config/columns.yaml` (컬럼 역할, 접두사별 측정 단계, 예측 시점, 수동 제외 사유)
- **분석 기준값** → `config/experiment.yaml` (FDR 기준, 분할 방식, 중복 처리, 결측 indicator)
- **기계적으로 계산할 것** → 코드가 실행할 때마다 데이터를 보고 다시 계산 (상수·중복, 결측률, 순위 등)
- 실행마다 `runs/<시각>_<태그>/`에 결과를 따로 저장한다. 설정 스냅샷, 데이터 해시, 패키지 버전이 `manifest.json`에 남는다.

## 실행
```bash
pip install -r requirements.txt
python run.py                      # 전체: validate → eda → split → filter → wrapper → model → explain → report
python predict.py runs/<실행 폴더> new_data.csv -o predictions.csv   # 학습된 모델로 위험도·구간 예측
python run.py --tag no_indicator   # 설정을 바꿔 비교 실행할 때 태그 지정
python run.py --stages validate    # 새 데이터셋 점검만
```

## 단계
| 폴더 | 내용 |
|---|---|
| `00_validate` | 타깃·그룹 키 확인, 모든 컬럼의 역할 판정(`column_roles.csv`). **측정 단계가 정해지지 않은 컬럼이 있으면 중단** |
| `01_eda` | 라벨 분포, 그룹별 라벨 순도, 실험일별 구성, 결측률·결측 패턴, 상수·중복·근사중복, 단일 변수 AUC·Mann–Whitney |
| `02_split` | Holdout 분할을 한 번 정해 저장 (기본: 그룹 단위). 이후 단계는 모두 이 분할을 사용 |
| `03_filter` | dev 데이터만으로 품질 제거 → F-test + BH-FDR → 특징 순위(`filter_ranking.csv`) |
| `04_wrapper` | **모델별 특징 수 탐색(B안).** dev 그룹 CV에서 fold마다 F-test 순위를 다시 계산(모든 모델 공유)하고, 모델마다 k별 PR-AUC 곡선을 그려 `k_rule`(기본 1-SE)로 각자의 k를 선택. 결측 indicator 켬/끔 비교, 모델별 특징 겹침표 |
| `05_model` | 모델별 자기 특징으로 같은 fold에서 비교, CV 예측(OOF) 저장. **선택 규칙:** gate(Recall 목표 지점의 Precision) → PR-AUC → 1-SE 동점 → 실험일 제외 PR-AUC → Brier → 해석 가능성. 선택 모델 OOF로 **위험 구간(정상/품질주의/불량 위험) 기준**을 정해 holdout에 고정 적용 |
| `06_explain` | 선택 모델의 holdout permutation importance, 영향 방향, 정상범위(전체·실험일별). MVP용 `model.joblib` + `model_card.json` |
| `99_report` | 모든 단계 결과로 `report.md` 생성. 숫자·표는 자동, `> 해석:` 자리만 직접 작성 |

## 데이터셋이 바뀌었을 때
1. `experiment.yaml`의 `data.path` 수정
2. `python run.py --stages validate` 실행
3. 미분류 컬럼이 나오면 `columns.yaml`의 `prefix_stage`(접두사 단위) 또는 `column_overrides`(컬럼 단위)에 측정 단계를 추가
4. `python run.py`

## 그룹 키
`실험일 + MET_ExperimentNumber`. 실험 번호(A01 등)는 실험일마다 재사용되므로 날짜와 함께 쓴다.
현재 데이터는 47개 그룹 모두 라벨이 고정(정상 34, 불량 13)이므로 행 단위 무작위 분할은 점수를 과대평가한다.

## 모델 선택 규칙 (과제 6 기준)
`config/experiment.yaml`의 `model.selection`, `model.risk_bands`에서 조정한다.
1. **gate**: CV 예측에서 Recall ≥ `target_recall`을 만족하는 판정 기준을 찾고, 그때 Precision ≥ `min_precision`인 모델만 남김 (예방형 품질관리: 불량을 놓치지 않으면서 오경보가 감당 가능한 수준)
2. **주 지표**: 그룹 CV PR-AUC 평균 (위험도 순위의 정확도)
3. **동점(1-SE 이내)**: 실험일 제외 PR-AUC → Brier(위험도 % 신뢰도) → 해석 가능성(LR > XGBoost > RF > SVM)
