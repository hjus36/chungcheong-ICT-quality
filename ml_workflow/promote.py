"""완료된 run의 07_release 산출물을 서비스 폴더로 승격한다.

    python promote.py --run im_v2_release            # 승격 조건 확인 후 복사
    python promote.py --run im_v2_release --dry-run  # 확인만
    python promote.py --run X --target 다른/경로       # 대상 폴더 지정 (테스트용)

- release_check.json이 통과가 아니면 중단한다 (--force로만 무시).
- 기존 release/ 내용은 release/previous/<기존 버전>/으로 옮겨 보관한다 (git 제외).
- 승격 기록은 release/release_info.json에 남긴다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parent
REPO = WORKFLOW_DIR.parent
DEFAULT_TARGET = REPO / "준비패키지" / "release"
RUNS_DIR = Path(os.environ.get("ICT_RUNS_DIR") or WORKFLOW_DIR / "runs")
FILES = ("model.joblib", "benchmark_public.json", "data_dictionary.json", "release_check.json")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="승격 조건 미통과여도 복사 (권장하지 않음)")
    a = ap.parse_args(argv)

    run_dir = RUNS_DIR / a.run
    src = run_dir / "07_release"
    man_path = run_dir / "manifest.json"
    if not man_path.exists():
        print(f"✗ run이 없습니다: {run_dir}"); return 2
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    st = manifest.get("stages", {}).get("release", {}).get("status")
    if st != "ok":
        print(f"✗ release 단계가 완료되지 않았습니다 (상태: {st})"); return 2
    missing = [f for f in FILES if not (src / f).exists()]
    if missing:
        print(f"✗ 산출물 누락: {missing}"); return 2
    check = json.loads((src / "release_check.json").read_text(encoding="utf-8"))
    bench = json.loads((src / "benchmark_public.json").read_text(encoding="utf-8"))
    if sha256(src / "model.joblib") != bench["model_sha256"]:
        print("✗ model.joblib 해시가 benchmark_public.json과 다릅니다"); return 2

    print(f"승격 후보: {check['version']} ({bench['selected_model']}, 특징 {bench['feature_count']}개)")
    for c in check["checks"]:
        print(f"  {'✓' if c['passed'] else '✗'} {c['check']}")
    print(f"  holdout 상태: {check.get('test_status')}")
    if not check["passed"] and not a.force:
        print("✗ 승격 조건 미통과 — 중단합니다."); return 1
    if a.dry_run:
        print(f"(dry-run) 대상: {a.target}"); return 0

    target = a.target
    target.mkdir(parents=True, exist_ok=True)
    info_path = target / "release_info.json"
    if info_path.exists():
        old = json.loads(info_path.read_text(encoding="utf-8")).get("version", "unknown")
        keep = target / "previous" / old
        if keep.exists():
            keep = keep.with_name(f"{old}-{dt.datetime.now():%Y%m%d%H%M%S}")
        keep.mkdir(parents=True)
        for f in (*FILES, "release_info.json"):
            if (target / f).exists():
                shutil.move(str(target / f), keep / f)
        print(f"  이전 버전 보관: {keep}")
    for f in FILES:
        shutil.copy2(src / f, target / f)
    info = {"version": check["version"], "run": a.run, "promoted_at": dt.datetime.now().isoformat(timespec="seconds"),
            "forced": bool(a.force and not check["passed"]), "model_sha256": bench["model_sha256"],
            "data_sha256": manifest.get("data_sha256"), "config_sha256": manifest.get("config_sha256"),
            "code_sha256": manifest.get("code_sha256"), "versions": manifest.get("versions"),
            "files": {f: sha256(target / f) for f in FILES}}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 승격 완료 → {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
