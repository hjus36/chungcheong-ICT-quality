"""Package only audited Git index contents, never local datasets or runtime files."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'output/무한루프_MVP_GitHub_20261101.zip')
    args=parser.parse_args()
    subprocess.run([sys.executable,str(ROOT/'scripts/audit_repository.py'),'--staged'],cwd=ROOT,check=True)
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(filter(None,names)):
            content=subprocess.check_output(['git','show',':'+name],cwd=ROOT)
            archive.writestr(name,content)
            manifest.append(dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'))
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
    print(json.dumps(dict(path=str(args.output),files=len(manifest),bytes=args.output.stat().st_size,
                         sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()),ensure_ascii=False))

if __name__=='__main__': main()
