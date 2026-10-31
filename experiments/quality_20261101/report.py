"""Publish the measured improvement results, keeping the pre-experiment reports intact."""
import json
from pathlib import Path
import pandas as pd
from common import ROOT,HERE,KAMP,save_json,sha

REPORT_ROOT=ROOT/'tmp/generated_reports'

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def f(x): return '—' if x is None else f'{x:.4f}'
def write(name,text): (REPORT_ROOT/'docs'/name).write_text(text,encoding='utf-8')
def table(items):
    text='| 방법 | Precision | Recall | F1 | F2 | ROC-AUC | AP | TP/FN/FP/TN | 경고 비율 |\n|---|---:|---:|---:|---:|---:|---:|---|---:|\n'
    for name,m in items:
        text+='| '+name+' | '+' | '.join(f(m[k]) for k in ['precision','recall','f1','f2','roc_auc','average_precision'])
        text+=' | '+' / '.join(str(m[k]) for k in ['tp','fn','fp','tn'])+f" | {m['alert_rate']:.1%} |\n"
    return text

def main():
    if not (HERE/'baseline_snapshot/docs').exists():
        raise SystemExit('Historical report generation requires the local baseline snapshot and KAMP source files. See docs/11_데이터_재현과_저장소_운영.md.')
    (REPORT_ROOT/'docs').mkdir(parents=True,exist_ok=True)
    kdir=HERE/'runs/kamp_v1';sdir=HERE/'runs/secom_v1'
    k=read(kdir/'final_results.json');s=read(sdir/'final_results.json')
    kv=read(kdir/'validation.json');sv=read(sdir/'validation.json')
    kc=read(kdir/'cv_results.json');sc=read(sdir/'cv_results.json')
    a=read(HERE/'results/data_audit.json');old=read(KAMP/'release/benchmark.json')
    checks=read(HERE/'verification/experiment_checks.json');api=read(HERE/'verification/api_results.json')
    restart=read(HERE/'verification/restart_results.json');ui=read(HERE/'verification/ui_checks.json')
    assert all(x['status']=='passed' for x in [checks,api,restart,ui])
    assert k['decision']['action']=='retain_deployed_RF'
    assert sha(KAMP/'release/model.joblib')==a['model_sha256']
    snapshot=HERE/'baseline_snapshot'
    d=pd.read_csv(KAMP/'joined_batches.csv')
    start=pd.to_datetime(d.process_start_at).dt.normalize();end=pd.to_datetime(d.process_end_at).dt.normalize()
    work=pd.to_datetime(d['작업일'])
    timing=dict(quality_workday_minus_sensor_start_days={str(x):int(n) for x,n in (work-start).dt.days.value_counts().items()},
        quality_workday_minus_sensor_end_days={str(x):int(n) for x,n in (work-end).dt.days.value_counts().items()},
        equipment=d['설비명'].value_counts().to_dict(),
        interpretation='Workday is not inspection timestamp.51rows have workday one calendar day before sensor start; shift convention plausible but unverified. Batch identifier joins are exact.')
    save_json(HERE/'results/timing_audit.json',timing)
    merged=[]
    for dataset,cv,v in [('KAMP',kc,kv),('SECOM',sc,sv)]:
        byid={r['id']:r for r in v['candidates']}
        for r in cv:
            merged.append(dict(dataset=dataset,id=r['id'],feature_set=r['features'],family=r['family'],parameter=r['value'],
                cv_mean_ap=r['cv_mean_ap'],**{key:val for key,val in byid[r['id']].items() if key not in ['id','calibration']}))
    pd.DataFrame(merged).to_csv(HERE/'results/all_validation_comparisons.csv',index=False,encoding='utf-8-sig')
    bestfamilies=[]
    for cv,v in [(kc,kv),(sc,sv)]:
        byid={r['id']:r for r in v['candidates']}
        best=[]
        for family in ['LogisticRegression','RandomForest','XGBoost']:
            r=max([r for r in cv if r['family']==family and not r['baseline']],key=lambda r:r['cv_mean_ap'])
            best.append((r,byid[r['id']]))
        bestfamilies.append(best)
    kt=table([('현재 서비스 RF · 0.35',kv['baselines']['deployed'])]+[(r['id'],m) for r,m in bestfamilies[0]]+
        [('전부 경고',kv['baselines']['all_risk']),('전부 통상',kv['baselines']['all_normal'])])
    st=table([(r['id'],m) for r,m in bestfamilies[1]]+[(n,sv['baselines'][n]) for n in ['all_risk','all_normal']])
    ci=k['decision']['validation_ap_difference_ci']
    cvtable='| 입력 특성 | LR 최고 CV AP | RF 최고 CV AP | XGB 최고 CV AP |\n|---|---:|---:|---:|\n'
    for feature in ['baseline36','compact16','compact16_plus_temporal35']:
        cvtable+='| '+feature+' | '+' | '.join(f(max(r['cv_mean_ap'] for r in kc if r['features']==feature and r['family']==family and not r['baseline']))
            for family in ['LogisticRegression','RandomForest','XGBoost'])+' |\n'
    ranktable='| 자료 / 모델 | 상위20% K | Precision@K | 상위30% K | Precision@K | Brier ↓ |\n|---|---:|---:|---:|---:|---:|\n'
    for name,m in [('KAMP 검증 / 현재 RF',kv['baselines']['deployed']),('KAMP 검증 / 새 후보',k['validation']),
                   ('KAMP 과거 보류 / 현재 RF',k['test_baselines']['deployed']),('KAMP 과거 보류 / 새 후보',k['test']),
                   ('SECOM 검증 / 새 XGB',s['validation']),('SECOM 보류 / 새 XGB',s['test'])]:
        ranktable+=f"| {name} | {m['k_20pct']} | {f(m['precision_at_20pct'])} | {m['k_30pct']} | {f(m['precision_at_30pct'])} | {f(m['brier'])} |\n"
    benchmark=f'''# 품질모델 성능 개선 및 추가 데이터 검증 보고서

2026-11-01 · 실험 `quality_20261101` · seed 20261101

## 결론

**성능 개선을 입증하지 못했으므로 현재 웹 모델 RandomForest(경고 0.35)를 유지했다.** 기존 실측 결과를 재현하고 원시 센서에서 새 특성을 추출했으며, 열처리 18후보와 기존 RF 대조군, 추가 확보한 반도체 데이터 6후보를 비교했다. 새 후보는 학습 내부 결과가 좋아도 다음 기간에서 악화됐다. 후보 파일과 실패한 결과도 모두 보존했다.

## 1. 원본 및 기존 수치 재현

센서 2,939,722행을 작업별 품질표 136행과 배정번호로 1:1 연결했으며 양품+불량=총수량, 원본 두 파일의 해시가 일치한다. 집계 36특성의 결측과 완전히 같은 특성 행은 0이다. 학습78/검증22/기존 보류36의 상대적 위험은 각각20/4/7건이다. 같은 설비 `열처리 염욕_1`에서 나온 자료로, 136작업조차 완전히 독립인지는 입증하지 못했다.

기존 보류 성능은 TP5/FN2/FP23/TN6, Precision0.1786, Recall0.7143, F1 0.2857, F2 0.4464, ROC-AUC0.4286, AP0.2055로 모두 재현됐다. 전부 경고 F2 0.5469보다 낮다. 기존 수치가 잘못된 집계 때문에 낮아진 정황은 발견하지 못했다.

## 2. 성능 저하에 관한 근거

- **작은 정답 수와 불안정한 검증:** 학습 위험20건, 검증 위험4건이다. 검증에서 한 건을 더 맞히거나 놓치면 Recall이25%p 바뀐다. 약294만 센서행은 독립 정답 증가가 아니다.
- **상대적 라벨의 한계:** 학습 Q3 {a['risk_boundary_percent']:.9f}%는 기업 규격이 아니다. 총수량은8,414~104,761개로 다르고, 불량 개수가±1 바뀌면 라벨이 뒤집히는 작업이6개다. 단순 이항 가정의95% Wilson 구간이 경계를 가로지르는 작업은48개다. 같은 작업 내 부품의 상관을 고려하지 않은 민감도 참고값이며 라벨 오류48개라는 뜻은 아니다.
- **시기 차이:** 위험 비율25.6%→18.2%→19.4%, 검증의 솔트 컨베이어1 평균과 세정기 평균은 학습 대비 각각1.34표준편차 이동했다. 제품·재질·설정값이 없어 품종 변경과 설비 변화의 영향을 분리하지 못한다. 이를 성능 저하의 확정 원인으로 단정하지 않는다.
- **결측·이상값:** 원시 소입1존OP 결측4,288개 등은 존재하지만 집계 결측은0이다. 학습 분위수의3×IQR 바깥 특성이 하나라도 있는 작업은 학습28/검증2/기존 보류13개다. 이는 통계적 범위 이탈이며 설비 불량으로 판정하거나 자동 삭제하지 않았다.
- **중복·상관:** 제거된 CP 모니터 외에 학습36특성에서 절대 Spearman상관≥0.95인 쌍은0이다. 단순 중복 특성이 주원인이라고 단정할 수 없다. 모델에는 fold 내부의 상수·결측·Pearson상관≥0.98 제거와 특성 선택을 시험했다.
- **연결과 시점:** 배정번호와 수량의 연결은 정확하다. 다만 품질표 작업일이 센서 시작 날짜보다 하루 앞선 경우51건이다. 교대 작업일 관행일 가능성은 있지만 확인되지 않았다. 작업일을 검사시각으로 사용하지 않았으며 검사 정답이 언제 이용 가능해졌는지는 검증할 수 없다.
- **기존 선정 방식:** 작은 검증셋에서 F2를 최대화하면 경고가 많아도 유리해진다. 기존 검증 경고17/22, 보류28/36이 이 문제를 보여준다. 검증셋에서 여러 모델·경계를 고른 선택 편향도 남아 있다.

라벨은 비교 가능성을 위해 변경하지 않았다. 향후 실제 판정규격·불량유형과 제품별 기준을 확보하면 기준을 사전에 고정한다. 규격이 없을 때는 연속 불량률 예측 및 순위 평가도 별도 목표로 검토하되, 이번 결과를 본 뒤 유리한 새 라벨로 바꾸지 않는다.

## 3. 추가 특성과 학습 절차

18후보는 3입력군×3모델×2설정이다. 기존 RF를 같은 시간 fold에 넣은 대조군1개도 계산했다.

| 입력군 | 설계 | 선택 후 입력 |
|---|---|---:|
| baseline36 | 기존18센서 평균·표준편차 | fold 학습에서 최대8개 선택 |
| compact16 | 소입·솔트조·건조·컨베이어 평균/존간차이/변동, CP 평균/편차, 공정군 온도차 | 최대8개 |
| compact16_plus_temporal35 | 관계16개 + 원본에서 계산한 시간특성35개 | 최대8개 |

시간특성은5신호군별 최솟값, 최댓값,95%-5%폭, 처음/마지막20%시간 평균, 두 구간 차이, 시간당 선형 추세다. 센서 원본 전체2,939,722행을 다시 읽어136행으로 생성했다. 시간 구간은 경과시간 창이며 실제 가열·유지·냉각 단계 라벨이 아니다. 제품 정답·불량수량·검사 이후 정보는 입력에 쓰지 않았다. 전체 작업 종료 이후에만 계산 가능한 특성이므로 조기예측이 아니다.

학습78건 내부의 expanding time3fold에서 다음 검증 작업 시작과 겹치는 학습 작업을 제거한다. 각 fold의 Q3, 결측 중앙값, 상수/상관 제거, ANOVA 상위8개 선택, LR 표준화는 fold학습에서만 fit한다. LR C=0.01/0.1, RF depth=2/3(200트리,leaf6), XGB depth=1/2(80트리,lr0.05,lambda5,child3)를 사용했다. class_weight=balanced 또는 학습 음성/양성 비율로 불균형을 처리했다. 모든 상세 설정은 저장 pipeline과 `train.py`에 있다.

후보는 평균 CV AP가 가장 큰 하나로 **검증 결과 계산 전에** 고정했다. 임계값도 train OOF의 F2→Precision→높은 경계 순으로 골랐으며, OOF경고율40% 이내라는 내부 시연 가정을 적용했다. 이40%는 기업 요구가 아니며 미래 경고율을 보장하지 않는다. fold별 Q3가 달라 OOF 목표가 완전히 동일하지 않다는 제한도 있다.

{cvtable}

전처리 변경 없는 기존 RF 대조군 CV AP는 {next(r['cv_mean_ap'] for r in kc if r['baseline']):.4f}다. 새 승자 compact16_RF_depth3의 {max(r['cv_mean_ap'] for r in kc if not r['baseline']):.4f}와 차이가 작다. 시간특성 추가군의 최고 CV AP도 기존 대조군보다 낮아, 변수 추가만으로 유효 신호가 생겼다고 볼 수 없다.

## 4. KAMP 검증과 모델 적용 결정

다음 표는 각 모델 가족에서 **train CV로만** 선택한 후보이며 표 전체를 보고 승자를 다시 고르지 않았다. KAMP19행과 SECOM6행 전체 비교는 [실험 CSV](../experiments/quality_20261101/results/all_validation_comparisons.csv)에 보존했다.

{kt}

사전 적용 조건: 기존 대비 AP+0.05 이상, F2 이상, Recall≥0.75, Precision+0.10 이상, 경고율≤50%, 날짜 단위 paired bootstrap AP차이95%하한>0, 기존 입력 호환. 새 승자는 AP/F2/Recall/Precision/불확실성 조건을 통과하지 못했다. AP차이95%구간은 [{ci['lower']:.4f}, {ci['upper']:.4f}]다. 양쪽 클래스를 포함한 재표집 {ci['valid']}/{ci['repetitions']}회를 사용했으며 작은 과거 검증자료의 기술통계다.

**적용 결정은 기존 RF 유지**로 저장한 다음, 새 승자 한 개에 대해서만 기존 보류36건을 계산했다. 아래 결과로 재선정하지 않았다.

{table([('현재 RF',k['test_baselines']['deployed']),('새 compact RF · 미적용',k['test']),('전부 경고',k['test_baselines']['all_risk']),('전부 통상',k['test_baselines']['all_normal'])])}

기존 보류는 과거에도 여러 번 본 개발 참고자료다. 위 결과를 신규 일반화 검증으로 부르지 않는다. 기존 모델을 유지한다는 결정은 기존 성능이 충분하다는 뜻이 아니다.

## 5. 추가 데이터 확보 및 별도 실험

동일/유사 공정을 먼저 조사했다. KAMP61은 센서·정답 수와 첫 작업이 KAMP60과 같아 중복 가능성이 높고, KAMP43은 설비 이상 탐지 목적이며 독립 제품 검사 정답을 확인하지 못했다. 명장 플랫폼 후보는 현재 DNS접속 실패로 원본·이용조건을 검증하지 못했다. [출처별 검토 기록](../experiments/quality_20261101/source_review.json)에 접근 상태와 판단을 남겼다.

[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom)을 CC BY4.0으로 다운로드했다. 실제 반도체 생산 개체와 라인 검사 합격/불합격이 연결된 별도 사례다. 로컬 파일 확인 결과1,567행, 실패104건, 숫자 입력590열이다. 제공처의591특성 설명과 파일 열 수 차이는 그대로 기록했다. 익명 신호라 물리 의미·단위와 LOT식별자는 없으며 열처리 모델에 합치거나 외부 검증 자료로 사용하지 않았다.

같은 날짜가 분할을 넘지 않도록 유일 날짜의60/20/20% 경계로 **820/367/380행**을 고정했다. 이 분할은 라벨 통계·모델 평가 전에 저장했다. 학습 실패69, 검증12, 마지막 평가23건이다. 결측40%초과/상수/고상관 제거, 중앙값, ANOVA 상위40개 선택을 fold학습에만 적용했다. 학습820행에는 결측25,812셀과 상수122열이 있다. 동일한3모델·각2설정·3시간fold로6후보를 비교했다.

검증 결과(모델/경계는 이미 train OOF에서 고정):

{st}

선정 XGB depth1, 경계0.525의 마지막 **처음 평가한 동일 데이터셋 시간 보류** 결과:

{table([('SECOM XGB · 별도 연구모델',s['test']),('전부 경고',s['test_baselines']['all_risk']),('전부 통상',s['test_baselines']['all_normal'])])}

검증 AP{f(s['validation']['average_precision'])}가 보류 AP{f(s['test']['average_precision'])}로 떨어졌고, 위험23건 중1건만 찾았다. Accuracy{ s['test']['accuracy']:.1%}만으로 성공이라고 하면 대부분이 합격인 불균형을 숨기게 된다. 이 결과를 본 뒤 경계를 낮추거나 후보를 바꾸지 않았다. 시간·제품·설비 분포와 점수 분포의 변화가 가능한 설명이지만 익명 변수만으로 원인을 확정할 수 없다. 새로운 기업 외부 검증도 아니다.

## 6. 검토 부담·순위·보정 상태

{ranktable}

K=ceil(전체행×비율), 동점은 시간순으로 고정했다. 상수 기준선에는 의미 있는 순위가 없으므로 Precision@K 개선 근거로 사용하지 않는다. KAMP기존 RF의 과거 보류 상위8개 중 위험1개(12.5%)는 전체 위험 비율19.4%보다 낮다. 새 후보도 순위 개선이 없다.

Brier는 낮을수록 좋지만 보정과 구분력을 함께 반영하므로 순수 보정 검정이 아니다. KAMP검증의 학습 유병률 상수 Brier는 {f(kv['baselines']['train_prevalence']['brier'])}, 기존 RF는 {f(kv['baselines']['deployed']['brier'])}이다. SECOM보류의 상수 Brier는 {f(s['test_baselines']['train_prevalence']['brier'])}, XGB는 {f(s['test']['brier'])}다. 5개 고정 점수구간의 평균 점수·실제 비율·건수도 JSON에 기록했다. 점수를 보정된 불량 확률로 사용할 근거가 없으며, 작은 양성 수에서 추가 보정모델을 튜닝하지 않았다.

## 7. 보존·재현·다음 검증 조건

앱의 API·36입력·버전·경계·SQLite payload를 유지했다. 기존 `kamp/release`·초기모델·검증기록과 DB는 원위치 및 `baseline_snapshot`에 보존했다. 후보는 `runs/*/models`에만 저장했다. 새 자료·실측·합성 결과를 분리했고 합성자료는 이번 학습에 쓰지 않았다.

[실험 실행 안내](../experiments/quality_20261101/README.md), 사전 `protocol.json`, 원본SHA·분할명세·실행환경·학습코드 사본·모델SHA·선정잠금·OOF/검증/보류 예측을 함께 남겼다. 시드와 라이브러리는 고정했으나 환경에 따라 미세한 부동소수점 차이는 생길 수 있다. 검증 재실행은 새 데이터의 성능 증명이 아니다.

다음 열처리 자료에는 **같은 공정의 새 기간 작업**, 제품/재질/규격/설비/레시피, 실제 검사시각과 불량유형/수량, 작업 식별자, 예측할 시점 이전 센서구간이 필요하다. LOT·제품·설비 상관을 고려해 그룹과 시간 경계를 고정하고, 검사 누락 비용과 검토 가능한 작업 수를 현장 담당자와 먼저 정한다. 단순 센서행 수보다 독립 작업·위험 사례 수를 늘려야 한다. 목표 사례 수는 필요한 Recall/Precision 신뢰구간과 현장 불량 비율을 정한 뒤 산정한다.
'''
    write('06_모델_벤치마크_보고서.md',benchmark)
    base03=(snapshot/'docs/03_데이터_AI_설계서.md').read_text(encoding='utf-8')
    base03=base03.replace('2026-11-01 · 실측·합성 분리, 기존 자료 재사용.','2026-11-01 · 성능 개선 실험 반영 · 서비스 입력 및 모델은 유지.')
    base03=base03.replace('추가 데이터는 이번 기능 개발에 필요하지 않아 다운로드하지 않는다.','이번 개선에서는 UCI SECOM을 추가 확보해 별도 반도체 모델로 검증했다. 열처리와 합치지 않는다.')
    base03=base03.replace('새 개선 실험은 train 내부 시간순 교차검증으로 제한된 하이퍼파라미터 후보를 비교하고, 검증 자료 F2/Precision으로 모델과 임계값을 선정한다.',
        '기존 배포 모델의 선정은 검증 F2/Precision을 사용했다. 이번 새 실험은 train 내부 시간순 CV AP로 후보를 고정하고, train OOF에서 검토 부담 제약 아래 임계값을 고정한 다음 검증으로 교체 가능 여부만 판단했다. 교체 기준 미달로 기존 모델을 유지한다.')
    base03+='''
## 2026-11-01 개선 실험 설계

원본 센서 전체를 다시 읽어 시간특성35개를 만들고, 기존 요약에서 공정 관계16개를 계산했다. 36/16/51 입력군을 비교하되 fold학습 내부에서 상수·결측·고상관 제거와 ANOVA상위8개 선택을 수행한다. 구체적인 정의·단위는 `실험_데이터사전.json`에 있다. 모든 특성은 작업 종료 시점까지의 센서값으로 계산하며 정답을 참조하지 않는다.

SECOM은1,567생산개체의 라인검사 합격/불합격 별도 사례다. 원본 숫자신호590열, 시간날짜 기준820/367/380분리, fold학습 내부40특성 선택이다. 익명 변수이므로 공정 물리 해석과 열처리 전이는 주장하지 않는다. 웹의 data_source에 추가하지 않으며 실험 전용으로 보관한다.

사전 기준·선정 잠금·모델 적용 결정·구간별 예측은 `experiments/quality_20261101/runs/`에 있다. 최고CV후보가 KAMP검증에서 악화되어 교체하지 않았다. SECOM도 마지막 구간에서 성능이 약해 현장 성능 확보로 표현하지 않는다. 보정된 확률 대신 위험 점수로 표시하는 기존 설계를 유지한다. 상세 수치와 원인은 `06_모델_벤치마크_보고서.md`를 참조한다.
'''
    write('03_데이터_AI_설계서.md',base03)
    write('07_검증_결과보고서.md',f'''# 통합 테스트 및 개선 실험 검증 결과

2026-11-01 · 기존 웹 모델 및 UI 유지 · Python/Node 의존성은 기존 lockfile 유지

## 이번 작업에서 다시 확인한 결과

| 검증 | 결과 | 새 증거 |
|---|---|---|
| API 통합·회귀 | {api['tests_run']}개 통과, 실패0, 오류0 | experiments/quality_20261101/verification/api_results.json |
| 실험 무결성·재현 | {checks['tests_run']}개 통과 | verification/experiment_checks.json |
| 실제 서버 재시작 | 저장 결과 JSON 완전 동일, 정적 React 제공 | verification/restart_results.json |
| 실제 브라우저 | 실측20행 불러오기·40필수 매핑·검증·추론·저장·기존RF표시·이력 복원, 오류 로그 없음 | verification/ui_checks.json 및 ui_dashboard.png, ui_model.png |
| 원본 재집계 재현·배포 입력 | 별도 폴더의35특성 CSV 해시 동일, 학습 당시 코드8파일 해시 일치, 원본파일 복원 시 출처기록 보존 | verification/delivery_checks.json |

위 표의 짧은 verification경로는 `experiments/quality_20261101/` 아래다. 이전 11개 전체 브라우저 시나리오·반응형·설치 검증 결과는 `docs/verification/`와 개선 전 `baseline_snapshot/`에 그대로 보존했다. 이번 브라우저 확인은 주요 흐름 재검증이며 전체11시나리오 재실행으로 표현하지 않는다.

API검증은 CSV/xlsx·CP949·ID·매핑·단위·결측·중복·정답 누출·수량 오류·합성 분리·파일 제한·이력·CSV내보내기·직접 모델 점수 일치를 포함한다. 별도 임시 DB를 사용했다. 로컬 시연의20행/18경고/평균0.523은 기존 모델 동작 확인이며 새로운 모델 성능 평가가 아니다.

실험 검증은 원본두파일·기존모델·기존DB해시, 보존 사본 해시, SECOM다운로드해시, 분할ID/날짜 비중복, 모델 선정잠금, 학습만으로 계산한 중앙값, 저장 후보의 보류 예측 재현, 시간특성 수/유한값/구간 차이 식을 확인했다. 첫 실행의 Windows기본cp949읽기 오류를 UTF-8지정으로 수정했으며 실패 기록도 attempt1파일에 보존했다. 모델이나 평가 선택을 변경한 수정이 아니다.

## 성능 검증의 판정

KAMP최고CV후보의 검증 AP{f(k['validation']['average_precision'])}, F2{f(k['validation']['f2'])}는 기존 AP{f(kv['baselines']['deployed']['average_precision'])}, F2{f(kv['baselines']['deployed']['f2'])}보다 낮다. 적용 조건을 통과하지 못해 기존 RandomForest와 경계0.35를 유지했다. SECOM은 별도 시간 보류에서 AP{f(s['test']['average_precision'])}, Recall{s['test']['recall']:.1%}로 실용 성능을 확보하지 못했다.

테스트 통과는 기능과 기록 보존을 확인한 결과다. 모델의 품질판정 신뢰성이나 공장 일반화 성능을 입증하지 않는다. 현재 앱은 기술검증용 MVP다.

## 보존 및 미검증 범위

기존原본·현재/초기모델·기존평가 JSON/CSV·SQLite이력을 덮어쓰지 않았다. 보고서는 개선 전 사본을 저장한 뒤 갱신했다. 기존 데이터베이스 해시도 동일하다. API형식·모델버전·화면 동작은 그대로이므로 기존 이력 재점수화나 모델 복구 작업이 발생하지 않았다.

신규 열처리 기업/기간자료, 실제 검사시각 기준 온라인 예측, 제품·설비별 그룹 독립성, 현장 사용자 비용·검토량, 실제 불량 감소 효과는 검증하지 못했다. SECOM의 보류도 동일 출처의 과거 시간 구간이며 타기업 외부 검증이 아니다.

실행 명령과 환경 제약은 [실험 안내](../experiments/quality_20261101/README.md)에 있다. 새 회귀 증거는 `ICT_EVIDENCE_DIR`로 별도 경로를 지정해 남길 수 있다.
'''.replace('기존原본','기존 원본'))
    base09=(snapshot/'docs/09_최종_결과보고서.md').read_text(encoding='utf-8')
    base09=base09.replace('## 모델 검증 결과','## 기존 서비스 모델 검증 결과')
    base09=base09.replace('이번에는 추가 데이터 다운로드·외부 배포·GitHub push·이메일 전송을 하지 않았다.',
        '추가 공개자료 UCI SECOM을 다운로드해 별도 연구모델로 검증했다. 외부 배포·GitHub push·이메일 전송은 하지 않았다.')
    intro=f'''## 성능 개선 작업의 최종 판정

**기존 모델을 교체할 만한 개선 근거는 확보하지 못했다.** 열처리 원본294만행 재분석으로 공정 관계16개와 시간특성35개를 만들고, LR/RF/XGBoost 18후보 및 기존RF대조군을 비교했다. 최고CV후보는 검증 AP0.2290/F2 0.3846으로 기존0.3770/0.6061보다 낮았다. 서비스의 RandomForest0.35, 기존API·입력·이력·화면을 유지했다.

추가로 UCI SECOM실측1,567생산개체·590숫자신호·104실패를 확보했다. 별도 시간순820/367/380분리에서6후보를 비교했고, 선정XGB는 최종380개체에서 AP0.0889, Recall4.3%였다. 이 결과는 열처리모델의 외부검증도, 현장 성능 개선 성공도 아니다.

새 분석 코드·분할·전처리·후보모델·선정잠금·원본해시·평가결과를 `experiments/quality_20261101/`에 보존했다. API11개·실험검증8개·프로세스재시작·주요브라우저흐름을 다시 통과했다. 상세 수치/원인/한계는 문서06, 기술검증은 문서07, 추가출처는 문서10에 있다.

'''
    base09=base09.replace('## 완성한 결과',intro+'## 완성한 결과',1)
    write('09_최종_결과보고서.md',base09)
    base10=(snapshot/'docs/10_자료_출처와_데이터사전.md').read_text(encoding='utf-8')
    base10=base10.replace('원본 다운로드 동의는 기존 대화에서 승인되었으며 추가 자료를 다운로드하거나 이메일을 보내지 않았다.',
        'KAMP원본 다운로드 동의는 기존 대화에서 승인되었다. 이번 추가자료는 공개 UCI SECOM이며 CC BY4.0을 확인하고 확보했다. 이메일은 보내지 않았다.')
    base10+='''
## 추가 확보한 SECOM과 검토한 열처리 후보

McCann, M. & Johnston, A. (2008). **SECOM [Dataset]**. UCI Machine Learning Repository. [DOI10.24432/C54305](https://doi.org/10.24432/C54305), [공식 소개](https://archive.ics.uci.edu/dataset/179/secom), [CC BY4.0](https://creativecommons.org/licenses/by/4.0/).

`experiments/quality_20261101/sources/uci_secom/`에1,964,989바이트 ZIP과 원문3개파일을 확보했다. 출처·다운로드UTC·파일별SHA256·용도는 provenance.json에 있다. 원본 숫자행렬은1,567×590, 라벨은1,567행(-1합격/+1불합격,검사시각)이다. 제공처 설명의591특성과 숫자파일590열의 차이를 기록하고 가상열을 추가하지 않았다. 실패104개, 학습/검증/평가820/367/380행이다.

입력은 signal_001~signal_590으로만 이름을 부여했다. 물리 단위·변수 뜻·LOT/설비식별자는 제공되지 않아 미상으로 표시한다. 검사시각은 시간분할에만 사용하고 입력특성에 넣지 않는다. 행 식별 entity_id는 원본 행번호이며 실제 LOT번호가 아니다. 원본은 수정하지 않았고 시간분할·결측처리·특성선택·학습은 본 프로젝트에서 수행했다. 열처리와 합치지 않으며 웹 업로드 대상도 아니다.

KAMP공정최적화61은 기존60과 센서행수/정답수/첫작업이 일치해 별도 독립자료로 세지 않았다. 예지보전43은 제품검사 정답의 독립성을 확인하지 못했다. 명장 플랫폼 후보는 DNS접속 실패로 확보하지 못했다. 상세 근거와 URL은 `experiments/quality_20261101/source_review.json`에 있다. 모든 열처리 공개자료가 부적합하다는 뜻은 아니다.

## 실험용 파생 데이터 사전

기존 웹 입력 사전 `데이터사전.json`은 그대로 두고, 추가 정의를 `실험_데이터사전.json`으로 분리했다. 관계16개는 기존36특성에서 계산하며 온도평균·온도차·표준편차는°C, CP는%이다. 시간특성35개는5신호군의7통계이며 시간당 추세는°C/h 또는%/h다. 범위·기울기는 기업 허용한계가 아니다. 가열·냉각 실제 단계 정답이 없어 경과시간 처음/마지막20% 창으로 정의했다.

이번에는 원본 센서2,939,722행을 다시 읽어 시간특성을 생성했다. 원래36특성 재집계 파일은 덮어쓰지 않았다. 출력136행/35특성과 원본해시·생성정의는 temporal_manifest.json, 품질연결·민감도·기간별분포는 data_audit.json, 작업일과관측시점 관계는 timing_audit.json에 있다.
'''
    write('10_자료_출처와_데이터사전.md',base10)
    manifest=read(HERE/'results/temporal_manifest.json')
    dictionary=dict(web_contract='unchanged; see 데이터사전.json',kamp_compact=[],kamp_temporal=[],secom=dict(
        source='UCI_SECOM_179',signals='signal_001..signal_590',units='not provided',
        entity_id='1-based original row number, not LOT',label='-1 pass ->0; +1fail ->1',
        test_time='DD/MM/YYYY HH:MM:SS; time split only, not model input',
        model_inputs='590source signals; fit-only pruning and ANOVA40 selection',split=dict(train=820,validation=367,test=380)))
    for group,meaning in [('quench','소입로4존'),('salt','솔트조2존'),('dryer','건조로2존'),('conveyor','솔트컨베이어2존')]:
        for suffix,definition in [('mean','존별 Avg의 평균'),('zone_spread','존별 Avg의 max-min'),('variability','존별 Std의 평균')]:
            dictionary['kamp_compact'].append(dict(name=group+'_'+suffix,unit='°C',definition=meaning+' '+definition))
    dictionary['kamp_compact']+= [dict(name='cp_mean',unit='%',definition='소입로 CP 값_Avg'),dict(name='cp_std',unit='%',definition='소입로 CP 값_Std'),
        dict(name='quench_salt_gap',unit='°C',definition='quench_mean-salt_mean'),dict(name='salt_conveyor_gap',unit='°C',definition='salt_mean-conveyor_mean')]
    definitions={'min':'시간군신호 관측 최솟값','max':'시간군신호 관측 최댓값','p95_p05':'관측95%분위수-5%분위수',
        'early_mean':'전체경과시간 첫20% 관측 평균','late_mean':'마지막20% 관측 평균',
        'late_minus_early':'late_mean-early_mean','trend_per_hour':'관측시각(시간단위)과 신호의 최소제곱 직선 기울기'}
    for group,sensors in manifest['groups'].items():
        for suffix,definition in definitions.items():
            unit='%' if group=='cp' else '°C'
            dictionary['kamp_temporal'].append(dict(name=group+'_'+suffix,unit=unit+('/h' if suffix=='trend_per_hour' else ''),
                source_sensors=sensors,definition='같은 시각의 유효 센서 평균을 사용. '+definition))
    save_json(REPORT_ROOT/'docs/실험_데이터사전.json',dictionary)
    base_readme=(snapshot/'README.md').read_text(encoding='utf-8')
    addition='''## 2026-11-01 성능 개선 실험

기존 모델보다 나은 검증 성능을 확보하지 못해 **서비스 RandomForest와 경고0.35를 유지**했습니다. 원본 센서에서 시간특성35개와 공정 관계16개를 계산하고 열처리18후보+RF대조군을 비교했지만, 최고CV후보의 검증 AP0.2290/F2 0.3846은 기존0.3770/0.6061보다 낮았습니다.

추가로 **UCI SECOM실측1,567개체**를 다운로드하여 반도체 별도6후보를 검증했습니다. 마지막 시간보류380개체에서 선정모델의 AP0.0889, Recall4.3%로 현장 성능을 확보하지 못했습니다. 열처리와 자료를 합치지 않았고, 이 자료를 열처리모델 외부검증이라고 표현하지 않습니다.

- [모든 실험과 적용 판단](docs/06_모델_벤치마크_보고서.md)
- [재현 명령·저장 구조](experiments/quality_20261101/README.md)
- [추가 데이터 출처·채택/제외 근거](experiments/quality_20261101/source_review.json)
- [실험 데이터 사전](docs/실험_데이터사전.json)

기존 원본·모델·평가·DB는 보존했습니다. 이번 API11개·실험검증8개·서버재시작·주요브라우저흐름 확인 결과는 `experiments/quality_20261101/verification/`에 있습니다. 아래 실행 방법과 앱의36특성 입력 계약은 그대로입니다. 아래의 기존모델 선정 설명과 `docs/verification` 기록은 최초MVP 릴리스 기준이며 새 실험과 구분합니다.

'''
    base_readme=base_readme.replace('## 실행\n',addition+'## 실행\n',1)
    base_readme=base_readme.replace('scripts/                   설치·실행·테스트·보고서 생성','experiments/quality_20261101/ 개선 코드·추가자료 출처·후보모델·결과·보존본\nscripts/                   설치·실행·테스트·보고서 생성')
    base_readme=base_readme.replace('추가 데이터 다운로드·유료서비스·GitHub push·이메일 전송은 이번 작업에서 하지 않았습니다.',
        'UCI SECOM 공개 데이터를 추가 다운로드했습니다. 유료서비스·GitHub push·이메일 전송은 하지 않았습니다.')
    old_reproduction=next(line for line in base_readme.splitlines() if line.startswith('학습 재현은'))
    base_readme=base_readme.replace(old_reproduction,
        '현재 개선 실험의 재현은 [실험 실행 안내](experiments/quality_20261101/README.md)의 새 run 이름을 사용하는 명령을 따릅니다. '
        '과거 `train_release.py`는 현재 release 파일을 다시 쓰므로 보존된 모델을 사용하는 일반 실행에는 필요하지 않습니다. '
        '같은 보류자료를 재실행해도 독립 검증이 추가되는 것은 아닙니다. 현재 앱은 `kamp/release/` 모델을 읽습니다.')
    (REPORT_ROOT/'README.md').write_text(base_readme,encoding='utf-8')
    print('Historical report drafts saved to tmp/generated_reports. Published README/docs were not changed.')

if __name__=='__main__': main()
