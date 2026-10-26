$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = Split-Path -Parent $packageRoot
$repositoryPython = Join-Path $repositoryRoot '.venv\Scripts\python.exe'
$packagePython = Join-Path $packageRoot '.venv\Scripts\python.exe'
$bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$taskDependencies = Join-Path $repositoryRoot 'tmp\ml_deps'
$previousPythonPath = $env:PYTHONPATH
$useBundledRuntime = $false

if (Test-Path -LiteralPath $repositoryPython) { $runtimePython = $repositoryPython }
elseif (Test-Path -LiteralPath $packagePython) { $runtimePython = $packagePython }
elseif ((Test-Path -LiteralPath $bundledPython) -and (Test-Path -LiteralPath $taskDependencies)) {
    $runtimePython = $bundledPython
    $useBundledRuntime = $true
}
else { throw 'Create a Python 3.12 virtual environment and install requirements.txt as described in README.md.' }

Push-Location -LiteralPath $packageRoot
try {
    if ($useBundledRuntime) { $env:PYTHONPATH = $taskDependencies }
    & $runtimePython -m uvicorn app_kamp:app --host 127.0.0.1 --port 8765
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
