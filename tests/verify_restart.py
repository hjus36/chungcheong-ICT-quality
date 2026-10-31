"""Start two real server processes sequentially against the same isolated DB."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
PORT = 8767

def launch(db):
    env = {**os.environ, 'ICT_DB_PATH': str(db), 'PYTHONIOENCODING': 'utf-8'}
    process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app_kamp:app', '--app-dir', str(ROOT/'준비패키지'),
                                '--host', '127.0.0.1', '--port', str(PORT)], cwd=ROOT, env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    for _ in range(100):
        if process.poll() is not None: raise RuntimeError('Test server failed to start; check port 8767.')
        try:
            response=httpx.get(f'http://127.0.0.1:{PORT}/health',timeout=.5,trust_env=False)
            if response.status_code==200: return process
        except httpx.TransportError: pass
        time.sleep(.1)
    process.terminate();process.wait(timeout=10)
    raise RuntimeError('Test server readiness timeout')

def main():
    with tempfile.TemporaryDirectory(prefix='ict-restart-') as temp:
        db=Path(temp)/'runs.sqlite3'; process=launch(db)
        with httpx.Client(base_url=f'http://127.0.0.1:{PORT}',timeout=15,trust_env=False) as client:
            try:
                assert client.get('/').status_code==200
                response=client.post('/predict',files={'file':('restart.csv',(ROOT/'준비패키지/kamp/share/test_upload.csv').read_bytes())})
                response.raise_for_status(); first=response.json()
            finally: process.terminate();process.wait(timeout=10)
            process=launch(db)
            try:
                second=client.get('/runs/'+first['run_id']).json()
                assert second==first
                assert client.get('/runs').json()['total']==1
                assert client.get('/favicon.svg').status_code==200
            finally: process.terminate();process.wait(timeout=10)
        evidence=Path(os.environ.get('ICT_EVIDENCE_DIR',str(ROOT/'tmp/test-evidence')))
        evidence.mkdir(parents=True,exist_ok=True)
        (evidence/'restart_results.json').write_text(json.dumps({'status':'passed','process_restarted':True,
            'predicted_rows':first['predicted_rows'],'payload_identical':True,'static_frontend_served':True},indent=2),encoding='utf-8')
        print('PASS: server process restart preserves exact stored payload; React and favicon served.')

if __name__=='__main__': main()
