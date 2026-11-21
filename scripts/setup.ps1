param([string]$PythonExe = '', [switch]$RebuildFrontend, [switch]$Train)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PIP_CACHE_DIR = Join-Path $repoRoot 'tmp\pip-cache'
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
function Assert-Exit([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)." } }
if (-not (Test-Path -LiteralPath $venvPython)) {
    if (-not $PythonExe) {
        $bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $bundled) { $PythonExe = $bundled }
        elseif (Get-Command py -ErrorAction SilentlyContinue) { $PythonExe = (& py -3.12 -c 'import sys;print(sys.executable)').Trim(); Assert-Exit 'Find Python 3.12' }
        elseif (Get-Command python -ErrorAction SilentlyContinue) { $PythonExe = (Get-Command python).Source }
        else { throw 'Install Python 3.12, then rerun scripts/setup.ps1.' }
    }
    & $PythonExe -c 'import sys;sys.exit(0 if sys.version_info[:2]==(3,12) else 1)'
    Assert-Exit 'Python version check'
    & $PythonExe -m venv (Join-Path $repoRoot '.venv')
    Assert-Exit 'Create virtual environment'
}
$requirements = Join-Path $repoRoot '준비패키지\requirements-lock.txt'
if (-not (Test-Path $requirements)) { $requirements = Join-Path $repoRoot '준비패키지\requirements.txt' }
& $venvPython -m pip install --disable-pip-version-check -r $requirements
Assert-Exit 'Install Python dependencies'
& $venvPython -c 'import fastapi,numpy,pandas,sklearn,xgboost,openpyxl;print(fastapi.__version__)'
Assert-Exit 'Dependency check'
if ($Train) {
    # 학습 워크플로우(ml_workflow) 전용 의존성: yaml 설정, parquet 읽기, 그래프
    & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $repoRoot 'ml_workflow\requirements-train.txt')
    Assert-Exit 'Install training dependencies'
}
if ($RebuildFrontend -or -not (Test-Path (Join-Path $repoRoot 'frontend\dist\index.html'))) {
    $bundleRoot = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies'
    $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
    if (-not $nodeCommand) { throw 'Install Node.js 22.12+ (24 LTS recommended) to build the frontend.' }
    Push-Location (Join-Path $repoRoot 'frontend')
    try {
        $bundledPnpm = Join-Path $bundleRoot 'node\node_modules\pnpm\bin\pnpm.cjs'
        if (Test-Path -LiteralPath $bundledPnpm) {
            & $nodeCommand.Source $bundledPnpm install --frozen-lockfile --store-dir (Join-Path $repoRoot 'tmp\pnpm-store')
        } elseif (Get-Command pnpm.cmd -ErrorAction SilentlyContinue) {
            & pnpm.cmd install --frozen-lockfile
        } elseif (Get-Command npx.cmd -ErrorAction SilentlyContinue) {
            & npx.cmd --yes pnpm@11.25.0 install --frozen-lockfile
        } else { throw 'Install pnpm 11.25.0 (or Node.js with npm) to build the frontend.' }
        Assert-Exit 'Install frontend dependencies'
        & $nodeCommand.Source 'node_modules\vite\bin\vite.js' build
        Assert-Exit 'Build frontend'
    } finally { Pop-Location }
}
Write-Host 'Setup complete. Run: powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1'
if ($Train) { Write-Host 'Training: powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1 -Run <name>' }
