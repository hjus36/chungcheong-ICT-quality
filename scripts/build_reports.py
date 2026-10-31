"""Build traceable Markdown reports from the measured model/test artifacts."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'준비패키지'))
import quality_service as service

def write(name,text): (ROOT/'docs'/name).write_text(text,encoding='utf-8')
def f(value): return f'{value:.4f}' if value is not None else '—'

def main():
    # Current reports must include the subsequent improvement experiment.
    current_report=ROOT/'experiments/quality_20261101/report.py'
    if (ROOT/'experiments/quality_20261101/runs/secom_v1/final_results.json').exists():
        import runpy
        sys.path.insert(0,str(current_report.parent))
        runpy.run_path(str(current_report),run_name='__main__')
        return
    model=service.model_info()
    original=json.loads((service.ROOT/'benchmark.json').read_text(encoding='utf-8'))
    api=json.loads((ROOT/'docs/verification/api_results.json').read_text())
    browser=json.loads((ROOT/'docs/verification/browser_results.json').read_text(encoding='utf-8'))
    timing=json.loads((ROOT/'docs/verification/api_measurements.json').read_text())
    restart=json.loads((ROOT/'docs/verification/restart_results.json').read_text())
    data=json.loads((ROOT/'docs/verification/data_results.json').read_text(encoding='utf-8'))
    assert api['status']==browser['status']==restart['status']==data['status']=='passed'
    dictionary=service.schema()
    dictionary['source_url']=model['source_url']
    for field in dictionary['fields']:
        if field['kind']=='feature':
            field['aggregation']='sample standard deviation, ddof=1' if field['name'].endswith('_Std') else 'observed mean'
            field['reference']={k:service.bundle()['reference'][k][field['name']] for k in ['q05','median','q95']}
    write('데이터사전.json',json.dumps(dictionary,ensure_ascii=False,indent=2))
    headers='| 모델 | 설정 | 경계 | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | AP |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    table=''.join('| '+m['model']+' | '+f"{m['parameter']}={m['value']}"+' | '+f(m['alert_threshold'])+' | '+' | '.join(f(m[k]) for k in ['accuracy','precision','recall','f1','f2','roc_auc','average_precision'])+' |\n' for m in model['validation_comparison'])
    baselines='| 방법 | TP | FN | FP | TN | Accuracy | Precision | Recall | F1 | F2 | AUC | AP |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for name,m in [(model['selected_model'],model['test']),('전부 우선 검토',model['test_all_risk']),('전부 통상 검토',model['test_all_normal'])]:
        baselines+='| '+name+' | '+' | '.join(str(m[k]) for k in ['tp','fn','fp','tn'])+' | '+' | '.join(f(m[k]) for k in ['accuracy','precision','recall','f1','f2','roc_auc','average_precision'])+' |\n'
    top=sorted(model['feature_analysis'],key=lambda x:x['permutation_ap_drop'],reverse=True)[:6]
    importance='| 변수 | 검증 AP 감소 평균 | 표준편차 | 학습 Spearman |\n|---|---:|---:|---:|\n'+''.join('| '+x['feature']+' | '+' | '.join(f(x[k]) for k in ['permutation_ap_drop','permutation_ap_std','spearman'])+' |\n' for x in top)
    write('06_모델_벤치마크_보고서.md',f'''# 모델 벤치마크 및 개선 검토

2026-11-01 · 모델 `{model['version']}` · seed {model['seed']}

현재 선택은 **{model['selected_model']}**, 점수 경계는 **{model['alert_threshold']}**이다. 학습 내부 시간순 비교를 추가했지만 현장 사용 가능한 성능을 확보했다고 결론내리지 않는다. 보류자료에서 위험 누락은 2건, 오경고는 23건이다.

## 자료와 평가 계약

자동차 부품 염욕 열처리 136작업·36특성. 학습 78(1~4월), 검증 22(5월), 보류 36(6~7월)이며 학습 라벨 20, 검증 4, 보류 7건이 상대적 위험이다. 라벨은 학습 불량률 Q3 **{model['defect_rate_threshold_percent']:.9f}%** 이상이다. 이 값은 기업 규격이 아니다.

학습 내부 TimeSeriesSplit 3분할에서 경계 겹침을 제거한다. 각 fold는 해당 fold의 학습 Q3와 학습 전처리를 사용한다. LR C=0.01/0.1/1, RF depth=2/3/5, XGBoost depth=1/2/3의 총 9후보를 시간순 CV 평균 AP로 비교했다. 학습이 78건에 불과해 탐색 범위를 제한했다. 선택된 가족별 설정을 전체 train으로 학습한 뒤 validation에서 0.10~0.90(0.05간격) 임계값을 비교한다. 검증 F2, Precision 순으로 모델을 선택하며, 같은 모델의 경계가 동점이면 높은 경계를 선택한다.

최종 모델 선택 뒤 보류자료를 계산했다. 보류자료는 10월 초기 실험에서 이미 본 자료이므로 **새 독립 검증이 아닌 개발 참고 평가**다. 이번 결과를 보고 후보·임계값을 재조정하지 않았다. 합성자료는 어느 학습·평가에도 포함하지 않았다.

## 검증 22작업 비교

{headers}{table}
검증 결과만으로 선택했다. F2는 누락을 더 무겁게 보지만 정밀도와 경고 수를 함께 확인해야 한다. 모든 자료를 섞은 무작위 교차검증으로 성능을 높게 보이게 만들지 않았다.

## 보류자료 36작업과 기준선

{baselines}
선택 모델은 28작업에 경고하고 그중 5작업이 상대적 위험이다. AUC는 0.4286이며, F2=0.4464는 전부 우선 검토 기준선 0.5469보다 낮다. 정상/불량 자동판정이나 생산 경고 운영에 사용할 근거가 부족하다. 점수를 불량 확률로 보정하지 않았다.

## 초기 모델과 달라진 점

초기 `kamp/benchmark.json`의 Logistic Regression은 검증 F2=0.6250, 보류 Recall=0.2857, Precision=0.1818, F2=0.2564였다. 이번 실험은 train 시간순 CV로 설정을 선정하는 다른 사전 절차를 적용했다. 검증 F2는 초기 모델보다 낮으며, 보류 재현율이 높아진 대신 오경고가 9건에서 23건으로 늘었다. 따라서 **성능 개선 달성**이라고 표현하지 않는다. 기존 모델과 보고서는 보존했다. 제품 구현 완성도와 모델 일반화 성능은 별도 결과이다.

## 영향변수와 범위

{importance}
검증 집합에서 20회 permutation하여 AP 감소를 계산했다. 상위 2변수만 소폭 양수이고, 0 또는 음수 영향값도 많다. 작은 검증 집합과 변수 상관 때문에 순위가 불안정하다. 표의 수치는 원인이나 공정조건 변경 지시가 아니다. 분포와 Spearman 관계, 5~95백분위는 train 78작업으로만 계산했다. 현재 모델 RF의 impurity importance도 JSON에 저장했으나 UI의 주요 영향변수는 위 permutation 기준이다.

## 남은 제약과 다음 검증

공정 전체 집계이므로 조기예측이 아니다. 검사시각·제품·재질·배치별 운전조건 변화·불량유형이 없어 운영 시점 정렬과 원인 분석을 확정할 수 없다. 2,939,722센서행은 136개의 독립 품질라벨을 대체하지 않는다. 신규 기간/기업 자료를 확보하고 예측 시점 이전 구간으로 재설계한 뒤 외부 검증과 기업 비용 기준을 적용해야 한다. 추가 다운로드는 승인 후 진행할 대상이며 이번에는 다운로드하지 않았다.

재현: `.venv\\Scripts\\python.exe 준비패키지\\train_release.py`. 원본 재집계는 선택 사항이며 `prepare_kamp.py`가 별도 담당한다. 동봉 모델로 앱을 실행할 때 재학습은 필요 없다.

모델 SHA-256: `{model['model_sha256']}`
가공자료 SHA-256: `{model['data_sha256']}`
''')
    write('07_검증_결과보고서.md',f'''# 통합 테스트 및 기술검증 결과

2026-11-01 · Windows · Python 3.12.14 / Node 24.22.1 / FastAPI 0.162.0 / React 19.2.0 / Vite 7.1.12. 프로젝트 전용 `.venv`와 브라우저 Edge에서 실제 실행했다. 정확한 의존성은 `requirements-lock.txt`와 `pnpm-lock.yaml`에 고정했다.

## 실행 결과

| 검증 | 실제 결과 | 증거 |
|---|---|---|
| 자동 API 통합·회귀 | {api['tests_run']}개 테스트 통과, 오류 {api['errors']}, 실패 {api['failures']} | `tests/test_system.py`, `verification/api_results.json` |
| 실제 브라우저 | {len(browser['checks'])}개 사용자 흐름 점검 통과, JS 오류 없음 | `frontend/tests/e2e.mjs`, `verification/browser_results.json` |
| 프로세스 재시작 | 두 서버 프로세스 사이 저장 JSON 완전 동일 | `tests/verify_restart.py`, `verification/restart_results.json` |
| 원본·가공자료 | 2원본 SHA-256 일치, 136:136 연결·수량·특성 재검증 | `tests/verify_data.py`, `verification/data_results.json` |
| 20행 API 추론·저장 | {timing['predict_20_rows_seconds']:.3f}초(로컬 측정) | `verification/api_measurements.json` |
| Windows 설치·빌드 | PowerShell 5.1에서 새 `.venv` 설치, 고정 패키지 설치, 정적 빌드 성공 | `scripts/setup.ps1 -RebuildFrontend` 실제 실행 |
| 최종 실행 스크립트 | start.ps1로 UI/health/이력 확인, 기존 6개 payload 보존 및 실측 시연 기록 저장 | `verification/start_script_results.json` |
| 화면 크기 | 1440×1080 데스크톱, 390×844 모바일; 페이지 가로 넘침 없음 | `verification/dashboard.png`, `mobile.png` |

API 테스트는 UTF-8·CP949·xlsx, 앞자리 0 ID, 별칭 매핑, 정답 누출 차단, 중복/잘못된 값/날짜/부분 성공, 결측 대체, °F/K/비율 변환, 품질수량 무결성, 합성 분리, 파일/행/열 제한, 손상 파일, 없는 이력, 비파괴 DB 마이그레이션, CSV 수식 보호, 직접 모델 점수 일치와 분할·모델 해시를 포함한다.

브라우저는 파일 선택→원본 ID 매핑→검증→매핑 변경으로 검증 무효화→추론→상세 점수 일치→CSV 다운로드→변수 차트→3모델 화면→오류 1행 제외/19행 저장→이력 검색·복원→페이지 새로고침→120합성 작업→모바일을 실제 조작했다. API의 기존 DB를 테스트 데이터로 덮어쓰지 않고 별도 임시 DB를 사용했다.

## 수정한 문제

Windows 격리 환경의 asyncio 로컬 socketpair 대기로 테스트가 멈추는 환경 제약을 확인했고, 승인된 로컬 프로세스 권한에서 정상 실행했다. PowerShell 5.1에서 한글 경로가 깨지는 문제는 UTF-8 BOM으로, Python `-c` 내부 따옴표 문제는 인수 단순화로 수정했다. 브라우저 검증에서 발견한 파비콘 404를 수정했다. 이력 검색 응답 대기와 Edge 표 헤더 선택자 문제를 자동화 테스트에 반영했다. 최신 실행은 모두 위 표의 통과 결과이다.

## 요구사항 추적

| ID | 상태 | 구현·증거 |
|---|---|---|
| R01~04 | 충족 | schema, preview, validate, React 매핑/단위; API 02~08, 브라우저 업로드 |
| R05~07 | 기능 충족, 현장 성능 미입증 | train_release, benchmark, 분포/관계/편차/영향 화면; 모델 보고서 참조 |
| R08~09 | 충족 | React 실연동·SQLite 저장/조회; 브라우저 및 프로세스 재시작 |
| R10 | 충족 | 동일공정 120합성 작업, 명시 출처; API07·브라우저 합성 |
| R11~12 | 충족 | 제한/오류/시간순 분리·train fit; API04~08·11 |
| R13 | 충족 | docs 01~10·사용자 PDF·README·소스/모델/빌드 |
| R14~16 | 충족 | 설치/실행 스크립트·결과 CSV·해시/버전 |
| R17 | 후속 범위 | 실시간·MES·기업 현장 적용은 구현/검증 완료로 표시하지 않음 |

## 검증하지 않은 범위

실제 기업 설비 연결, 현장 사용자 평가, 신규 외부 품질자료, Windows 외 OS 및 모든 브라우저, 다중 사용자 부하, 운영 보안 인증·백업 복원은 검증하지 않았다. 20행 지연 수치는 단일 로컬 측정이며 10,000행 성능 SLA가 아니다. 화면의 모든 변수를 도메인 전문가가 검증한 것은 아니다. 테스트 통과가 모델의 품질판정 신뢰성을 증명하지 않는다.

재실행: `scripts/test.ps1`, `.venv\\Scripts\\python.exe tests/verify_data.py`, `.venv\\Scripts\\python.exe tests/verify_restart.py`. 브라우저 테스트는 테스트 DB로 실행한 서버와 프런트 개발 의존성이 필요하며 README에 순서를 적었다.
''')
    write('09_최종_결과보고서.md',f'''# 프로젝트 최종 결과보고서

무한루프 · 충청권 특화산업연계 수요기반 ICT 현장적용 프로젝트 · 과제 6
기준일 2026-11-01 · 제조공정 데이터 기반 품질위험 예측 및 공정편차 분석 MVP

## 완성한 결과

**CSV/Excel 업로드 → 원천 항목 매핑 → 단위/행 검증 → 모델 추론 → 공정편차/변수 분석 → React 대시보드 → SQLite 저장·이력 조회**를 실제로 동작하는 로컬 웹 MVP로 완성했다. 정적 목업이 아니라 FastAPI와 저장된 모델을 호출한다. UI는 한국어이며 데스크톱과 모바일 화면을 검증했다.

대상은 자동차 부품 염욕 열처리 한 공정과 작업별 상대적 고불량률 위험 한 종류이다. 원본 2,939,722센서행과 품질 136작업을 연결한 기존 집계·모델 기반을 재사용했다. 강구 경도 합성자료는 별도 과거 시연으로 보존하고 현재 분석에 섞지 않았다. 현재 동일공정 합성 기능자료 120작업을 추가했다.

## 기능별 구현

업로드는 UTF-8/CP949 CSV와 xlsx, 파일 미리보기, 40필수·4선택 항목 연결, 명시적 °F/K/비율 변환을 지원한다. 행별 중복·결측·형식·수량 오류를 설명하고 유효행만 분석한다. 모델은 품질수량을 입력으로 사용하지 않는다. 점수·실제/가상 불량률·학습 통계 구간을 각각 구분한다.

위험 작업 목록은 검색·검토 필터·페이지 이동·상세 편차·원본/대체/변환값 확인·CSV 내보내기를 제공한다. 변수 화면은 학습 분포·품질 관계·검증 permutation 영향값을 보여준다. 이력은 시각·파일명·자료 출처·모델 버전·입력 해시와 결과를 보관하고 당시 결과를 그대로 복원한다. 기존 DB는 열을 추가해 읽으며 과거 payload는 유지한다.

## 모델 검증 결과

LR/RF/XGBoost 3종에 train 시간순 CV를 적용했고 검증 기준으로 {model['selected_model']}을 선택했다. 보류 36작업 결과는 TP5/FN2/FP23/TN6, Recall 71.4%, Precision 17.9%, F1 0.2857, F2 0.4464, AUC 0.4286이다. 전부 경고하는 F2 기준선 0.5469보다 낮다. 초기 모델보다 재현율은 높지만 오경고가 크게 늘어 성능 개선 달성으로 결론내리지 않는다.

보류자료는 이미 본 개발 참고 자료이고 신규 독립 검증이 아니다. 공정 완료 후 요약값을 사용하므로 조기예측·생산 중 경고를 구현한 것이 아니다. 현재 완료는 **기술적 기능 검증용 MVP**이며, 현장 품질예측 성능이나 불량 감소 효과를 확보한 상태가 아니다.

## 실행·테스트

자동 API 테스트 {api['tests_run']}개와 브라우저 점검 {len(browser['checks'])}개 통과, 실제 프로세스 재시작 후 결과 동일, 원본 해시·작업 연결 일치, Windows 신규 가상환경 설치·프런트 빌드 성공을 확인했다. 상세 증거와 미검증 범위는 `07_검증_결과보고서.md`에 있다.

현재 PC 실행: `powershell -ExecutionPolicy Bypass -File .\\scripts\\start.ps1` → `http://127.0.0.1:8765`. 다른 PC 최초 설치는 `scripts/setup.ps1`이다. 원본 KAMP 다운로드·추가 학습 없이 동봉 가공 예제와 모델로 시연할 수 있다.

## 공식 산출물 대조

| 공식 과업 18쪽 산출물 | 실제 파일 |
|---|---|
| 공통 입력·합성자료 정의서 | docs/03, docs/10, 데이터사전.json, synthetic_definition.json |
| 머신러닝 벤치마크 | docs/06, kamp/release/benchmark.json, validation_comparison.csv |
| 가상 제조자료 1식 | kamp/release/synthetic_demo.csv (120가상작업) |
| 품질위험 모델 소스 | prepare_kamp.py, train_release.py, quality_service.py |
| 웹 MVP 실행패키지 | frontend/src 및 dist, app_kamp.py, model.joblib, scripts/setup.ps1·start.ps1 |
| 기술검증·사용자 매뉴얼 | docs/07·08, output/pdf/무한루프_MVP_사용자매뉴얼.pdf |

과제 6의 구체적 필수 MVP 기능은 구현했다. 공식 수행목표의 운영계획 문장과 기대효과의 실시간 예방형 방향은 요구사항 정의서에서 구분했다. 공식 산출물을 구현했다는 사실이 주관기관의 평가·검수 통과를 뜻하지 않는다.

## 다음 현장 적용 조건

기업별 품목·재질·불량유형·검사시각·판정규격과 예측 시점 이전 센서구간을 확보하고, 새 기간/기업의 독립 자료로 누락/오경고 비용을 검증해야 한다. 그 후에 운영 알림 기준·사용자 권한·백업·설비/MES 연결을 설계한다. 이번에는 추가 데이터 다운로드·외부 배포·GitHub push·이메일 전송을 하지 않았다. 원본과 사용자 자료는 삭제하지 않았다.
''')
    print('Reports and data dictionary generated from measured artifacts.')

if __name__=='__main__': main()
