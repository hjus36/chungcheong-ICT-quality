param([switch]$Browser, [switch]$Workflow, [string]$Url = 'http://127.0.0.1:8766')
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$python=Join-Path $repoRoot '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING='utf-8'
Push-Location $repoRoot
try {
    & $python -u tests/verify_data.py
    if ($LASTEXITCODE -ne 0) { throw 'Release integrity check failed.' }
    & $python -u tests/test_system.py
    if ($LASTEXITCODE -ne 0) { throw 'API tests failed.' }
    & $python -m unittest tests.test_public_demo
    if ($LASTEXITCODE -ne 0) { throw 'Public demo tests failed.' }
    & $python -u tests/verify_restart.py
    if ($LASTEXITCODE -ne 0) { throw 'Restart test failed.' }
    if ($Workflow) {
        # 작은 합성 데이터셋으로 ml_workflow 전 단계 (setup.ps1 -Train 필요)
        Push-Location (Join-Path $repoRoot 'ml_workflow')
        try {
            & $python -m unittest discover -s tests -v
            if ($LASTEXITCODE -ne 0) { throw 'Workflow tests failed.' }
        } finally { Pop-Location }
    }
    if ($Browser) {
        $env:ICT_TEST_URL=$Url
        & node frontend/tests/e2e.mjs
        if ($LASTEXITCODE -ne 0) { throw 'Browser tests failed.' }
    }
} finally { Pop-Location }
