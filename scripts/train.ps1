param([Parameter(Mandatory=$true)][string]$Run, [switch]$Resume, [switch]$Promote, [string]$Stages = '')
# 학습 워크플로우 실행 → (선택) 서비스로 승격 → 합성 예제 재생성
#   powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1 -Run im_v3
#   powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1 -Run im_v3 -Resume       (중단된 단계부터)
#   powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1 -Run im_v3 -Promote      (끝나면 승격)
# 원본 데이터: ml_workflow\data\dataset_V2.parquet (또는 환경변수 ICT_DATA_PATH)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/setup.ps1 -Train first.' }
$env:PYTHONIOENCODING = 'utf-8'
Push-Location (Join-Path $repoRoot 'ml_workflow')
try {
    $runArgs = @('run.py', $(if ($Resume) { '--resume' } else { '--run' }), $Run)
    if ($Stages) { $runArgs += @('--stages', $Stages) }
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Workflow failed (exit $LASTEXITCODE). Rerun with -Resume after fixing." }
    if ($Promote) {
        & $python promote.py --run $Run
        if ($LASTEXITCODE -ne 0) { throw 'Promotion refused: release_check did not pass.' }
        & $python make_synthetic.py
        if ($LASTEXITCODE -ne 0) { throw 'Synthetic example generation failed.' }
        Write-Host 'Promoted. Restart the server (scripts/start.ps1) to load the new model.'
    } else {
        Write-Host "Done. Review ml_workflow\runs\$Run\report.md, then promote: .\scripts\train.ps1 -Run $Run -Promote -Resume -Stages report"
    }
} finally { Pop-Location }
