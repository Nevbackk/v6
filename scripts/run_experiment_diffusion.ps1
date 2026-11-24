<#
.SYNOPSIS
Complete Execution Pipeline — Round 5 Minimal Genuine Diffusion (VP-OrbitDiff)

.EXAMPLE
.\scripts\run_experiment_diffusion.ps1 -Stage "test"
.\scripts\run_experiment_diffusion.ps1 -PreparedRoot "D:\G4\prepared" -ProtocolDir "D:\G4\protocol" -Device "cuda"
#>
param(
    [string]$PreparedRoot = $env:PREPARED_ROOT,
    [string]$ProtocolDir = $env:PROTOCOL_DIR,
    [string]$OutputRoot = "",
    [string]$PythonBin = $(if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }),
    [string]$Device = "cuda",
    [int]$Epochs = 10,
    [int]$BatchSize = 32,
    [double]$Lr = 0.001,
    [int[]]$Seeds = @(20261028, 20261029, 20261030),
    [int]$AblSeed = $Seeds[0],
    [int]$SplitSeed = 20261028,
    [ValidateSet("all", "test", "main", "k_curve", "abl_guidance", "abl_budget_delta", "b1", "b2", "paired_val", "paired_b1", "paired_b2")]
    [string]$Stage = "main",
    [int]$MaxRecordsPerEpoch = 0,
    [int]$MaxEvalRecords = 0,
    [switch]$RunPairedVal
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (Test-Path (Join-Path $RepoRoot "v6\v6")) {
    $V6Dir = Join-Path $RepoRoot "v6"
} else {
    $V6Dir = $RepoRoot
}
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot "runs\exp_diffusion" }
function Invoke-ExperimentPython {
    & $PythonBin @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
}

Write-Host "======================================================================"
Write-Host "[Round 5 VP-OrbitDiff] Stage='$Stage' | Device='$Device' | SplitSeed='$SplitSeed'"
Write-Host "======================================================================"

if ($Stage -in @("all", "test")) {
    Write-Host "[Diff][Stage 0] Running full unit test suite..."
    Push-Location $V6Dir
    try { Invoke-ExperimentPython -s -m unittest discover -s tests -v } finally { Pop-Location }
    if ($Stage -eq "test") { exit 0 }
}

if (-not $PreparedRoot -or -not $ProtocolDir) {
    throw "PreparedRoot and ProtocolDir are required for data stages."
}

$CommonArgs = @(
    "--prepared-root", $PreparedRoot,
    "--protocol-dir", $ProtocolDir,
    "--device", $Device,
    "--split-seed", "$SplitSeed",
    "--max-records-per-epoch", "$MaxRecordsPerEpoch",
    "--max-eval-records", "$MaxEvalRecords"
)

if ($Stage -in @("all", "main")) {
    foreach ($seed in $Seeds) {
        $runDir = "$OutputRoot\stage1_main_K4_P8\seed_$seed"
        Write-Host "[Diff][Stage 1] Training VP-OrbitDiff (K=4, P=8, delta=0.10, VJP) | seed=$seed..."
        Invoke-ExperimentPython "$V6Dir\scripts\train_diffusion.py" @CommonArgs `
            --run-dir $runDir --K 4 --P 8 --delta 0.10 --guidance-mode vjp `
            --epochs "$Epochs" --batch-size "$BatchSize" --lr "$Lr" --seed "$seed" --eval-split dev_select
    }
}

if ($Stage -in @("all", "k_curve")) {
    $mainCkpt = "$OutputRoot\stage1_main_K4_P8\seed_$AblSeed\best.pt"
    if (-not (Test-Path -LiteralPath $mainCkpt -PathType Leaf)) { throw "Missing required checkpoint: $mainCkpt" }
    foreach ($kVal in @(1, 2, 4, 8)) {
        Write-Host "[Diff][Stage 2] Evaluating trained VP-OrbitDiff at K=$kVal on dev_select..."
        Invoke-ExperimentPython "$V6Dir\scripts\validate_diffusion.py" @CommonArgs `
            --run-dir "$OutputRoot\stage2_k_curve\K${kVal}_seed_$AblSeed" `
            --K "$kVal" --P 8 --delta 0.10 --guidance-mode vjp `
            --resume $mainCkpt --eval-split dev_select
    }
}

if ($Stage -in @("all", "abl_guidance")) {
    $mainCkpt = "$OutputRoot\stage1_main_K4_P8\seed_$AblSeed\best.pt"
    if (-not (Test-Path -LiteralPath $mainCkpt -PathType Leaf)) { throw "Missing required checkpoint: $mainCkpt" }
    foreach ($gMode in @("identity_jac", "none")) {
        Invoke-ExperimentPython "$V6Dir\scripts\validate_diffusion.py" @CommonArgs `
            --run-dir "$OutputRoot\stage3_abl_guidance\${gMode}_seed_$AblSeed" `
            --K 4 --P 8 --delta 0.10 --guidance-mode $gMode `
            --resume $mainCkpt --eval-split dev_select
    }
    Invoke-ExperimentPython "$V6Dir\scripts\train_diffusion.py" @CommonArgs `
        --run-dir "$OutputRoot\stage3_abl_guidance\fixed_isotropic_v_seed_$AblSeed" `
        --K 4 --P 8 --delta 0.10 --guidance-mode vjp --fixed-isotropic-v `
        --epochs "$Epochs" --batch-size "$BatchSize" --lr "$Lr" --seed "$AblSeed" --eval-split dev_select
    Invoke-ExperimentPython "$V6Dir\scripts\train_diffusion.py" @CommonArgs `
        --run-dir "$OutputRoot\stage3_abl_guidance\no_rollout_priors_seed_$AblSeed" `
        --K 4 --P 8 --delta 0.10 --guidance-mode vjp --no-rollout-priors `
        --epochs "$Epochs" --batch-size "$BatchSize" --lr "$Lr" --seed "$AblSeed" --eval-split dev_select
}

if ($Stage -in @("all", "abl_budget_delta")) {
    $mainCkpt = "$OutputRoot\stage1_main_K4_P8\seed_$AblSeed\best.pt"
    if (-not (Test-Path -LiteralPath $mainCkpt -PathType Leaf)) { throw "Missing required checkpoint: $mainCkpt" }
    foreach ($dVal in @(0.05, 0.20)) {
        Invoke-ExperimentPython "$V6Dir\scripts\validate_diffusion.py" @CommonArgs `
            --run-dir "$OutputRoot\stage4_budget_delta\delta${dVal}_seed_$AblSeed" `
            --delta $dVal --K 4 --P 8 --resume $mainCkpt --eval-split dev_select
    }
}

foreach ($route in @("b1", "b2")) {
    if ($Stage -in @("all", $route)) {
        foreach ($seed in $Seeds) {
            Invoke-ExperimentPython "$V6Dir\scripts\train_diffusion.py" @CommonArgs `
                --route $route --run-dir "$OutputRoot\$route\seed_$seed" `
                --K 4 --P 8 --epochs $Epochs --batch-size $BatchSize --lr $Lr --seed $seed
        }
    }
}

$lockedRoutes = @()
if ($Stage -eq "paired_val") { $lockedRoutes = @("a") }
elseif ($Stage -in @("paired_b1", "paired_b2")) { $lockedRoutes = @($Stage.Substring(7)) }
elseif ($RunPairedVal) {
    if ($Stage -eq "all") { $lockedRoutes = @("a", "b1", "b2") }
    elseif ($Stage -in @("b1", "b2")) { $lockedRoutes = @($Stage) }
    else { $lockedRoutes = @("a") }
}
foreach ($route in $lockedRoutes) {
    foreach ($seed in $Seeds) {
        $dir = Join-Path $OutputRoot "$route\seed_$seed"
        if ($route -eq "a") { $dir = Join-Path $OutputRoot "stage1_main_K4_P8\seed_$seed" }
        $ckpt = Join-Path $dir "best.pt"
        if (-not (Test-Path -LiteralPath $ckpt -PathType Leaf)) { throw "Missing required checkpoint: $ckpt" }
        Invoke-ExperimentPython "$V6Dir\scripts\validate_diffusion.py" @CommonArgs --route $route --run-dir $dir --resume $ckpt --eval-split paired_val
    }
}

Write-Host "[Round 5 VP-OrbitDiff] Completed successfully."
