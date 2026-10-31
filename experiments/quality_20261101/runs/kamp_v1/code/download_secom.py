"""Download only documented public UCI data files; do not execute archive content."""
import urllib.request
import zipfile
from datetime import datetime, timezone
from common import HERE,sha,save_json

out=HERE/'sources/uci_secom'
out.mkdir(parents=True,exist_ok=True)
archive=out/'secom.zip'
if archive.exists(): raise FileExistsError('Existing original download is preserved')
url='https://archive.ics.uci.edu/static/public/179/secom.zip'
with urllib.request.urlopen(url,timeout=60) as response:
    with archive.open('xb') as f: f.write(response.read(20*1024*1024))
with zipfile.ZipFile(archive) as z:
    expected={'secom.data','secom_labels.data','secom.names'}
    if set(z.namelist())!=expected: raise ValueError(f'Unexpected archive members: {z.namelist()}')
    if sum(i.file_size for i in z.infolist())>20*1024*1024: raise ValueError('Unexpected uncompressed size')
    for name in sorted(expected):
        with (out/name).open('xb') as f: f.write(z.read(name))
save_json(out/'provenance.json',dict(source='https://archive.ics.uci.edu/dataset/179/secom',url=url,
    downloaded_utc=datetime.now(timezone.utc).isoformat(),license='CC BY 4.0',
    license_url='https://creativecommons.org/licenses/by/4.0/',
    citation='McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305',
    purpose='Independent semiconductor case, not pooled with heat-treatment data; original -1/+1 quality test labels.',
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()}))
print(f'Downloaded and verified {archive.stat().st_size} bytes; extracted 3 documented data files')
