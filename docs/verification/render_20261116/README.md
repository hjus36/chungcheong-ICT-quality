# Render 공개 배포 확인 — 2026-11-16

- 주소: https://chungcheong-ict-quality.onrender.com
- Render 서비스: `srv-dbt73r9srm7s73ads6vg` / My Workspace / Singapore / Free
- 배포 ID: `dep-dbt73s1srm7s73ads9n0` / 상태 `live`
- 실행 코드: `codex/public-demo` 브랜치의 `16f5b622c340779ea5fe2161c8604d547ecd7b99`
- 배포 완료: 2026-11-16 12:02:57 KST
- 자동 배포: 끔. 이후 문서 커밋은 실행 코드 변경이 없어 재배포하지 않음.

## 실제 공개 주소 검증

- 브라우저 16개 점검 통과: 홈 합성 120작업 추론, CSV 매핑·검증·추론·이력 저장, 점수 일치, 필터·페이지 선택, CSV 다운로드, 변수 차트, 모델 비교, 잘못된 숫자 처리, 이력 검색·복원, 새로고침, 합성 출처 표시, 390px 모바일 화면, JavaScript 오류 없음.
- HTTP 4개 점검 통과: `/health` 모델 로딩, 실측 예제 404, 공개용 통계 일치, 독립 생성 합성 예제 일치.
- 기존 로컬 검증: API 16개 테스트 통과. 공개 모드의 CSV/Excel 500행 제한과 최근 20건 이력 보관 포함.
- 확인 시점의 Render 앱 오류 로그 없음.
- 초기 관측 메모리 약 220MiB / 한도 512MiB. 짧은 기능 시험의 관측값이며 동시 접속 부하를 보장하는 결과는 아님.

결과 파일: [브라우저 결과](browser_results.json), [HTTP 결과](http_checks.json).

무료 서버의 이력은 임시 저장이며 방문자끼리 공유됩니다. 원본 데이터·면담 문서·로컬 SQLite는 서버 실행 패키지에 포함하지 않았습니다. GitHub의 과거 커밋 공개 범위는 [데이터 공유 안내](../../../DATA_USAGE.md)의 별도 검토 대상으로 남습니다.
