param([switch]$Browser, [string]$Url = 'http://127.0.0.1:8766')
$ErrorActionPreference='Stop'
$repoRoot=Split-Path -Parent $PSScriptRoot
$python=Join-Path $repoRoot '.venv\Scripts\python.exe'
$env:PYTHONIOENCODING='utf-8'
Push-Location $repoRoot
try {
    & $python -u tests/test_system.py
    if ($LASTEXITCODE -ne 0) { throw 'API tests failed.' }
    if ($Browser) {
        $env:ICT_TEST_URL=$Url
        & node frontend/tests/e2e.mjs
        if ($LASTEXITCODE -ne 0) { throw 'Browser tests failed.' }
    }
} finally { Pop-Location }
