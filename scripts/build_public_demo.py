"""Build an allowlisted runtime tree; original data and local history never enter it."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'tmp' / 'public-demo'
FILES = (
    '준비패키지/app_kamp.py',
    '준비패키지/quality_service.py',
    '준비패키지/kamp/audit.json',
    '준비패키지/kamp/release/model.joblib',
    '준비패키지/kamp/release/benchmark_public.json',
    '준비패키지/kamp/share/synthetic_demo.csv',
)


def build():
    # Refuse redirected output before removing this generated, task-owned directory.
    expected = ROOT / 'tmp' / 'public-demo'
    if DEST.resolve() != expected or (ROOT / 'tmp').is_symlink() or DEST.is_symlink():
        raise RuntimeError('Unsafe build destination')
    assets = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'frontend' / 'dist').rglob('*') if p.is_file())
    names = [*FILES, *assets]
    if 'frontend/dist/index.html' not in assets:
        raise RuntimeError('Build frontend/dist before packaging')
    for name in names:
        if not (ROOT / name).is_file(): raise FileNotFoundError(name)
    if DEST.exists(): shutil.rmtree(DEST)
    manifest = []
    for name in names:
        content = (ROOT / name).read_bytes()
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest.append({'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(manifest), 'bytes': sum(f['bytes'] for f in manifest)}, ensure_ascii=False))
    return DEST


if __name__ == '__main__': build()
