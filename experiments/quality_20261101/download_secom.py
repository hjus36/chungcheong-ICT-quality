"""Download only documented public UCI data files; do not execute archive content."""
import urllib.request
import zipfile
import hashlib
import json
from datetime import datetime, timezone
from common import HERE,sha,save_json

out=HERE/'sources/uci_secom'
out.mkdir(parents=True,exist_ok=True)
archive=out/'secom.zip'
url='https://archive.ics.uci.edu/static/public/179/secom.zip'
expected_sha='eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e'
if not archive.exists():
    with urllib.request.urlopen(url,timeout=60) as response:
        payload=response.read(20*1024*1024)
    if hashlib.sha256(payload).hexdigest()!=expected_sha:
        raise ValueError('Public source changed; review before adopting a different data version')
    with archive.open('xb') as f: f.write(payload)
if sha(archive)!=expected_sha: raise ValueError('Existing archive differs from the recorded original')
with zipfile.ZipFile(archive) as z:
    expected={'secom.data','secom_labels.data','secom.names'}
    if set(z.namelist())!=expected: raise ValueError(f'Unexpected archive members: {z.namelist()}')
    if sum(i.file_size for i in z.infolist())>20*1024*1024: raise ValueError('Unexpected uncompressed size')
    for name in sorted(expected):
        content=z.read(name)
        if (out/name).exists():
            if (out/name).read_bytes()!=content: raise ValueError(f'Preserve and investigate changed original: {name}')
        else:
            with (out/name).open('xb') as f: f.write(content)
provenance=dict(source='https://archive.ics.uci.edu/dataset/179/secom',url=url,
    downloaded_utc=datetime.now(timezone.utc).isoformat(),license='CC BY 4.0',
    license_url='https://creativecommons.org/licenses/by/4.0/',
    citation='McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305',
    purpose='Independent semiconductor case, not pooled with heat-treatment data; original -1/+1 quality test labels.',
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.name in expected|{'secom.zip'}})
if not (out/'provenance.json').exists(): save_json(out/'provenance.json',provenance)
else:
    original=json.loads((out/'provenance.json').read_text(encoding='utf-8'))
    if original['files']!=provenance['files']: raise ValueError('Original provenance mismatch')
print(f'Verified {archive.stat().st_size} bytes and3source files; existing originals and provenance preserved')
