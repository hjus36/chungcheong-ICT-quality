"""ML 워크플로우 실행기.

실행 규약
- 새 실행은 이름을 정해서 시작한다. 같은 이름의 실행이 이미 있으면 중단한다(덮어쓰기 금지).
- 중단된 실행은 --resume으로 이어간다. 이미 끝난 단계는 다시 실행하지 않는다.
- 실행을 만들 때 설정 파일과 코드(src, stages, run.py)의 사본·SHA-256을 남기고,
  이어서 실행할 때 설정이나 코드가 바뀌었으면 중단한다.

사용 예:
    python run.py --run im_v2_baseline                 # 전체 단계
    python run.py --run im_v2_check --stages validate  # 새 데이터셋 점검만
    python run.py --resume im_v2_baseline              # 끝나지 않은 단계 이어서
    python run.py --list                               # 실행 목록
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
import warnings
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

for stream in (sys.stdout, sys.stderr):          # Windows 콘솔에서도 한글이 깨지지 않게
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="Features .* are constant")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

from src.config import WORKFLOW_DIR, load_config  # noqa: E402
from src.data import file_sha256  # noqa: E402
from src.runio import RunContext, StageError  # noqa: E402
from stages import eda, explain, feature_filter, model, release, report, split, validate, wrapper  # noqa: E402

STAGES = {"validate": validate, "eda": eda, "split": split, "filter": feature_filter,
          "wrapper": wrapper, "model": model, "explain": explain, "release": release, "report": report}
ORDER = list(STAGES)
PACKAGES = ["pandas", "numpy", "scikit-learn", "scipy", "xgboost", "joblib", "pyyaml", "pyarrow", "matplotlib"]
RUNS_DIR = Path(os.environ.get("ICT_RUNS_DIR") or WORKFLOW_DIR / "runs")   # 테스트에서 임시 폴더 지정
CODE_PATHS = ["run.py", "src", "stages"]
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def _code_files():
    files = []
    for rel in CODE_PATHS:
        p = WORKFLOW_DIR / rel
        files += [p] if p.is_file() else sorted(q for q in p.rglob("*.py") if "__pycache__" not in q.parts)
    return files


def code_hashes() -> dict[str, str]:
    return {str(f.relative_to(WORKFLOW_DIR)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in _code_files()}


def _snapshot(run_dir: Path, cfg):
    snap = run_dir / "config_snapshot"
    snap.mkdir()
    for f in cfg.files():
        shutil.copy2(f, snap / f.name)
    code = run_dir / "code"
    for f in _code_files():
        dst = code / f.relative_to(WORKFLOW_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)


def list_runs():
    if not RUNS_DIR.exists():
        print("실행 기록이 없습니다.")
        return 0
    for d in sorted(p for p in RUNS_DIR.iterdir() if (p / "manifest.json").exists()):
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        done = [s for s, r in m.get("stages", {}).items() if r.get("status") == "ok"]
        print(f"{d.name:32s} {m.get('created', ''):20s} 완료 {len(done)}/{len(ORDER)}: {', '.join(done)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="새 실행 이름 (영문·숫자·_.-)")
    g.add_argument("--resume", help="이어서 실행할 기존 실행 이름")
    g.add_argument("--list", action="store_true", help="실행 목록")
    ap.add_argument("--stages", nargs="+", choices=ORDER, help="실행할 단계 (기본: 전체)")
    args = ap.parse_args(argv)
    if args.list:
        return list_runs()

    cfg = load_config()
    name = args.run or args.resume
    if not NAME_RE.match(name):
        ap.error("실행 이름은 영문·숫자로 시작하고 영문·숫자·_ . - 만 쓸 수 있습니다 (최대 64자).")
    run_dir = RUNS_DIR / name
    manifest_path = run_dir / "manifest.json"

    if args.run:
        if run_dir.exists():
            ap.error(f"이미 있는 실행입니다: {name}. 결과를 덮어쓰지 않습니다. 새 이름을 쓰거나 --resume을 쓰세요.")
        run_dir.mkdir(parents=True)
        _snapshot(run_dir, cfg)
        manifest = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
                    "config_sha256": cfg.sha256(), "code_sha256": code_hashes(),
                    "versions": _versions(), "stages": {}}
    else:
        if not manifest_path.exists():
            ap.error(f"실행을 찾을 수 없습니다: {name}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_sha256"] != cfg.sha256():
            ap.error("설정 파일이 실행을 만들 때와 다릅니다. 같은 실행을 다른 설정으로 이어갈 수 없습니다. 새 실행을 만드세요.")
        changed = sorted(k for k, v in code_hashes().items() if manifest["code_sha256"].get(k) != v)
        changed += sorted(set(manifest["code_sha256"]) - set(code_hashes()))
        if changed:
            ap.error(f"코드가 실행을 만들 때와 다릅니다: {changed[:5]}. 새 실행을 만드세요.")

    requested = [s for s in ORDER if s in (args.stages or ORDER)]
    done = {s for s, r in manifest["stages"].items() if r.get("status") == "ok"}
    if args.resume and args.stages:
        again = [s for s in requested if s in done]
        if again:
            ap.error(f"이미 끝난 단계는 다시 실행하지 않습니다: {again}")
    todo = [s for s in requested if s not in done]

    ctx = RunContext(run_dir, cfg)
    print(f"실행: {name}  ({run_dir})")
    if not todo:
        print("실행할 단계가 없습니다 (모두 완료).")
        return 0
    try:
        data_sha = file_sha256(ctx.data_file)
    except StageError as e:
        print(f"✗ {e}")
        return 1
    if manifest.get("data_sha256") not in (None, data_sha):
        print("✗ 데이터 파일이 실행을 만들 때와 다릅니다. 새 실행을 만드세요.")
        return 1
    manifest.update({"data_file": ctx.data_file.name, "data_sha256": data_sha})
    if cfg.expected_sha256 and data_sha != cfg.expected_sha256:
        print(f"[경고] 데이터 해시가 experiment.yaml의 expected_sha256과 다릅니다: {data_sha[:16]}…")
        manifest["data_sha256_matches_expected"] = False

    status = 0
    for stage in todo:
        print(f"\n[{stage}]", flush=True)
        t0 = time.time()
        rec = {"started": datetime.now().isoformat(timespec="seconds")}
        try:
            STAGES[stage].run(ctx)
            rec["status"] = "ok"
        except StageError as e:
            rec.update(status="failed", error=str(e))
            print(f"  ✗ 중단: {e}")
            status = 1
        rec["seconds"] = round(time.time() - t0, 2)
        manifest["stages"][stage] = rec
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if status:
            break
    print("\n완료" if status == 0 else "\n실패 — 위 메시지와 manifest.json을 확인하세요.")
    return status


if __name__ == "__main__":
    sys.exit(main())
