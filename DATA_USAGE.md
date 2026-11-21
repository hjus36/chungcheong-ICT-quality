# 데이터·모델·문서의 공유 범위

이 저장소는 무한루프팀의 개발 저장소이며 2026-11-16부터 Public이다. 코드와 연구 산출물을 공유하며, 데이터 제공처의 권리를 대신 허가하는 저장소 라이선스를 부여하지 않는다.

## 사용 데이터

| 항목 | 내용 |
|---|---|
| 자료명 | Injection-Molding Production Data with Quality Labels, **v2** (2022-10-25) |
| 제공 | German Plastics Center (SKZ), Fraunhofer IPA — 연구 프로젝트 ProBayes |
| 작성자 | Lambers, Jonathan; Schüder, Jakob; Krauß, Jonas |
| 게시처 | EUDAT B2SHARE — https://b2share.eudat.eu/records/k0v7s-jf859 |
| DOI | [10.23728/b2share.4c5692b886db419180f716acf895bf06](https://doi.org/10.23728/b2share.4c5692b886db419180f716acf895bf06) |
| 라이선스 | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) |

인용: Lambers, J., Schüder, J., & Krauß, J. (2022). *Injection-Molding Production Data with Quality Labels* (v2). German Plastics Center (SKZ) / Fraunhofer IPA, EUDAT B2SHARE. https://doi.org/10.23728/b2share.4c5692b886db419180f716acf895bf06 — CC BY 4.0.

**변경 내용(CC BY 4.0 표기 의무):** 원본 파일은 수정하지 않았다. 본 프로젝트는 원본에서 예측 시점(사이클 종료 직후) 이전에 측정되는 특징만 골라 그룹 분할·특징 선택·학습·평가를 수행했고, `LBL_NOK`(싱크마크 또는 미충전)를 불량 라벨로 사용했다. 학습된 모델, 집계 통계, 참고 구간은 이 자료에서 파생된 결과다. 제공처가 이 프로젝트를 보증하지 않는다.

## 저장소에 넣는 것과 넣지 않는 것

| 자료 | 처리 |
|---|---|
| 원본 `dataset_V2.parquet`, `data_description.pdf` | **넣지 않는다.** 라이선스상 재배포는 가능하지만, 저장소 크기와 출처 관리를 위해 게시처에서 직접 받도록 한다. 받는 방법과 해시: [ml_workflow/data/README.md](ml_workflow/data/README.md) |
| 행 단위 파생 자료 (분할표, fold 배정, OOF·holdout 예측, 개별 부품 값) | **넣지 않는다.** `ml_workflow/runs/`는 git 제외. 보고서 공개 사본(`runs/<이름>/public/`)도 이 파일들을 뺀다 |
| 서비스 모델 `준비패키지/release/model.joblib` | 넣는다. 전처리(학습 중앙값)·계수·참고 구간 등 집계 값만 포함한다 |
| `benchmark_public.json`, `data_dictionary.json`, `release_check.json`, `release_info.json` | 넣는다. 지표·분포는 집계값이며 공정변수 관계는 5부품 이상 묶은 구간 평균이다. 부품 ID·개별 측정값은 없다 |
| 합성 예제 `준비패키지/share/` | 넣는다. 승격된 데이터 사전의 참고 구간(집계값)을 넓힌 범위에서 독립 난수로 새로 뽑았다. 원본 행을 복사·재표집하지 않았다. `actual_label`은 모델 위험도로 뽑은 **가상값**이며 성능 평가에 쓰지 않는다 |
| 실측 예제 `준비패키지/local/` | 넣지 않는다 (git 제외). 있으면 로컬 화면에서만 실측 예제 버튼이 켜진다 |
| SQLite 분석 이력, 가상환경·캐시, 공식 신청·평가·면담 문서 | 넣지 않는다 |

## 이전 자료(KAMP 열처리·UCI SECOM)

이전 버전은 KAMP 열처리 품질보증 데이터셋과 UCI SECOM을 사용했다. 2026-11-21에 서비스·모델·실험을 사출성형 자료 기준으로 교체하면서 관련 코드·모델·실험·자산을 현재 트리에서 삭제했다. 과거 커밋에는 남아 있으며(이력 재작성이나 Force Push는 하지 않았다), 초기 커밋에는 면담 문서와 KAMP 작업별 가공 CSV가 포함되어 있었다. Public 저장소이므로 과거 이력의 공개 범위는 별도 검토가 필요하다. 현재 파일 감사 통과는 과거 이력의 공개 적합성을 뜻하지 않는다.

## 공유 전 확인

`python scripts/audit_repository.py --staged`로 업로드 대상의 금지 경로(원본·실행 결과·행 단위 파일·미승격 모델), 크기, 개인 경로, 인증정보 패턴, Markdown 링크를 검사한다. 알려진 패턴과 명시된 규칙만 확인하며 모든 민감정보를 자동으로 보증하지 않는다.
