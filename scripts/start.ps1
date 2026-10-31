param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimePython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) { throw 'Run scripts/setup.ps1 first.' }
if (-not (Test-Path (Join-Path $repoRoot 'frontend\dist\index.html'))) { throw 'Run scripts/setup.ps1 -RebuildFrontend first.' }
$env:PYTHONIOENCODING = 'utf-8'
if (-not $NoBrowser) {
    Write-Host "Open http://127.0.0.1:$Port after the server is ready."
}
Push-Location $repoRoot
try {
    & $runtimePython -m uvicorn app_kamp:app --app-dir (Join-Path $repoRoot '준비패키지') --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE." }
} finally { Pop-Location }
