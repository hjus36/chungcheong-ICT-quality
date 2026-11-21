# 원본 데이터 위치

이 폴더의 데이터 파일은 Git에 올리지 않는다 (`.gitignore`). 아래 공개 데이터를 직접 내려받아 이 폴더에 둔다.

| 항목 | 내용 |
|---|---|
| 자료명 | Injection-Molding Production Data with Quality Labels, **v2** (2022-10-25) |
| 제공 | German Plastics Center (SKZ), Fraunhofer IPA — 연구 프로젝트 ProBayes |
| 작성자 | Lambers, Jonathan; Schüder, Jakob; Krauß, Jonas |
| 게시처 | EUDAT B2SHARE — https://b2share.eudat.eu/records/k0v7s-jf859 |
| DOI | 10.23728/b2share.4c5692b886db419180f716acf895bf06 |
| 라이선스 | Creative Commons Attribution 4.0 International (CC BY 4.0) |

## 필요한 파일

| 파일 | 용도 | MD5 (게시처 표기) | SHA-256 |
|---|---|---|---|
| `dataset_V2.parquet` | 학습 데이터 (필수) | `f04efe419e63db5fb4a392e1569ea417` | `009d64ae4d77d2ed2de817c7d921d2c5ca58065f62b2680ededdf6b5576468bd` |
| `data_description.pdf` | 컬럼 설명·측정 시점 근거 (권장) | `2455efb8cb3fcf930a84f4a82347190a` | `493a06e02ae74e7d8618f87c5d75bf93682164ff89d30410c83019cfa78e5daf` |

내려받은 뒤 해시를 확인한다.

```powershell
Get-FileHash ml_workflow\data\dataset_V2.parquet -Algorithm SHA256
```

```bash
shasum -a 256 ml_workflow/data/dataset_V2.parquet
```

다른 위치에 둘 때는 환경변수 `ICT_DATA_PATH`에 파일 경로를 지정한다. SHA-256은 `config/experiment.yaml`의 `data.expected_sha256`과 비교되며, 다르면 실행 시 경고가 나온다. 새 데이터셋으로 바꿀 때는 이 값을 함께 갱신한다.
