"""Verify publication inputs and a fresh extraction without touching old evaluations."""
import json
import shutil
import subprocess
import sys
from common import HERE,ROOT,KAMP,sha,save_json

expected=HERE/'results/temporal_features.csv'
replica=ROOT/'tmp/temporal_reproduction_01/temporal_features.csv'
assert sha(expected)==sha(replica),'Raw temporal extraction changed'
frozen=[]
for name in ['kamp_v1','secom_v1']:
    run=HERE/'runs'/name
    env=json.loads((run/'environment.json').read_text())
    for filename,digest in env['source_code_sha256'].items():
        assert sha(run/'code'/filename)==digest
        frozen.append(f'{name}/{filename}')
    result=json.loads((run/'final_results.json').read_text())
    assert result['selection_lock_sha256']==sha(run/'selection_lock.json')
    assert result['deployment_decision_sha256']==sha(run/'deployment_decision.json')

# Simulate a distributed checkout with existing citation metadata but missing raw files.
dest=ROOT/'tmp/download_reproduction'
dest.mkdir(exist_ok=False)
for name in ['download_secom.py','common.py']: shutil.copy2(HERE/name,dest/name)
source=dest/'sources/uci_secom';source.mkdir(parents=True)
for name in ['secom.zip','secom.names','provenance.json']:
    shutil.copy2(HERE/'sources/uci_secom'/name,source/name)
metadata_before=sha(source/'provenance.json')
subprocess.run([sys.executable,str(dest/'download_secom.py')],check=True)
assert sha(source/'provenance.json')==metadata_before
for name in ['secom.data','secom_labels.data']:
    assert sha(source/name)==sha(HERE/'sources/uci_secom'/name)
manifest=json.loads((HERE/'baseline_snapshot/manifest.json').read_text(encoding='utf-8'))['sha256']
for name,digest in manifest.items():
    if 'kamp' in name and ('release' in name or name.endswith('.sqlite3')): assert sha(ROOT/name)==digest
readme=(ROOT/'README.md').read_text(encoding='utf-8')
assert '추가 데이터 다운로드·유료서비스' not in readme
save_json(HERE/'verification/delivery_checks.json',dict(status='passed',
    temporal_reproduction_sha256=sha(replica),frozen_training_code_verified=frozen,
    downloader_recovers_missing_raw_preserving_provenance=True,deployed_release_and_history_unchanged=True))
print('PASS: fresh raw-feature reproduction, frozen code, downloader recovery, release/history preservation')
