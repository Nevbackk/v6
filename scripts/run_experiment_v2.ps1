<#
.SYNOPSIS
Train one continuous PPC posterior per track or run the independent ASD-SC controls.

.EXAMPLE
.\scripts\run_experiment_v2.ps1 -PreparedRoot "D:\G4\prepared" -ProtocolDir "D:\G4\protocol" -Device "cuda"
.\scripts\run_experiment_v2.ps1 -Stage "test" -PythonBin "python"
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
    [int]$SplitSeed = $(if ($env:SPLIT_SEED) { [int]$env:SPLIT_SEED } else { 20261028 }),
    [ValidateSet("all", "test", "main", "asd_sc", "abl1_synergy", "abl2_gate", "abl3_replay_unroll", "abl4_horizon_prog", "abl5_memory_reanchor", "paired_val")]
    [string]$Stage = "main",
    [int]$MaxRecordsPerEpoch = 0,
    [int]$MaxEvalRecords = 0,
    [string]$SensorCheckpoint = $env:SENSOR_CHECKPOINT,
    [switch]$RunPairedVal
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (Test-Path (Join-Path $RepoRoot "v6\v6")) {
    $V6Dir = Join-Path $RepoRoot "v6"
} else {
    $V6Dir = $RepoRoot
}
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot "runs\exp_v2" }
function Invoke-ExperimentPython {
    & $PythonBin @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
}

Write-Host "======================================================================"
Write-Host "[V2] Stage='$Stage' | Device='$Device' | Output='$OutputRoot'"
Write-Host "======================================================================"

if ($Stage -in @("all", "test")) {
    Write-Host "[V2][Stage 0] Running V2 unit tests..."
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
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "$Lr",
    "--max-records-per-epoch", "$MaxRecordsPerEpoch",
    "--max-eval-records", "$MaxEvalRecords"
)

if ($Stage -in @("all", "main")) {
    foreach ($seed in $Seeds) {
        $runDir = Join-Path $OutputRoot "ppc\seed_$seed"
        $initArgs = @()
        if ($SensorCheckpoint -and -not (Test-Path (Join-Path $runDir "last.pt"))) { $initArgs = @("--sensor-checkpoint", $SensorCheckpoint) }
        Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs --arch ppc --run-dir $runDir --seed $seed @initArgs
    }
}

if ($Stage -in @("all", "asd_sc")) {
    foreach ($seed in $Seeds) {
        $runDir = "$OutputRoot\stage1_asd_sc_full\seed_$seed"
        Write-Host "[V2][Stage 1] Training Full ASD-SC | seed=$seed..."
        Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
            --run-dir $runDir --ablation-mode full `
            --lambda-forecast 1.0 --lambda-prog 0.25 --lambda-good-harm 2.0 `
            --counterfactual-prob 0.25 --seed "$seed" --eval-split dev_select
    }
}

if ($Stage -in @("all", "abl1_synergy")) {
    foreach ($mode in @("obs_only", "state_only")) {
        Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
            --run-dir "$OutputRoot\abl1_synergy\${mode}_seed_$AblSeed" `
            --ablation-mode $mode --seed "$AblSeed" --eval-split dev_select
    }
    Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
        --run-dir "$OutputRoot\abl1_synergy\no_counterfactual_seed_$AblSeed" `
        --ablation-mode full --counterfactual-prob 0.0 --seed "$AblSeed" --eval-split dev_select
}

if ($Stage -in @("all", "abl2_gate")) {
    foreach ($mode in @("fixed_g", "unconstrained_mlp6d", "no_quantile_aux")) {
        Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
            --run-dir "$OutputRoot\abl2_gate\${mode}_seed_$AblSeed" `
            --ablation-mode $mode --seed "$AblSeed" --eval-split dev_select
    }
}

if ($Stage -in @("all", "abl3_replay_unroll")) {
    foreach ($seed in $Seeds) {
        Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
            --run-dir "$OutputRoot\abl3_replay_unroll\static_1step\seed_$seed" `
            --ablation-mode full --no-dagger-replay --single-step-grad --seed "$seed" --eval-split dev_select
    }
    Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
        --run-dir "$OutputRoot\abl3_replay_unroll\static_2step_seed_$AblSeed" `
        --ablation-mode full --no-dagger-replay --seed "$AblSeed" --eval-split dev_select
    Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
        --run-dir "$OutputRoot\abl3_replay_unroll\dagger_1step_seed_$AblSeed" `
        --ablation-mode full --single-step-grad --seed "$AblSeed" --eval-split dev_select
}

if ($Stage -in @("all", "abl4_horizon_prog")) {
    Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
        --run-dir "$OutputRoot\abl4_horizon_prog\current_only_seed_$AblSeed" `
        --ablation-mode full --lambda-forecast 0.0 --seed "$AblSeed" --eval-split dev_select
    Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
        --run-dir "$OutputRoot\abl4_horizon_prog\next_frame_seed_$AblSeed" `
        --ablation-mode full --lambda-forecast 1.0 --seed "$AblSeed" --eval-split dev_select
    Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
        --run-dir "$OutputRoot\abl4_horizon_prog\no_progressive_aux_seed_$AblSeed" `
        --ablation-mode full --lambda-prog 0.0 --seed "$AblSeed" --eval-split dev_select
    Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
        --run-dir "$OutputRoot\abl4_horizon_prog\no_good_harm_penalty_seed_$AblSeed" `
        --ablation-mode full --lambda-good-harm 0.0 --seed "$AblSeed" --eval-split dev_select
}

if ($Stage -in @("all", "abl5_memory_reanchor")) {
    Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
        --run-dir "$OutputRoot\abl5_memory_reanchor\local_only_seed_$AblSeed" `
        --ablation-mode no_pending_memory --disable-reanchor --seed "$AblSeed" --eval-split dev_select
    Invoke-ExperimentPython "$V6Dir\scripts\train_v2.py" @CommonArgs `
        --run-dir "$OutputRoot\abl5_memory_reanchor\memory_no_reanchor_seed_$AblSeed" `
        --ablation-mode full --disable-reanchor --seed "$AblSeed" --eval-split dev_select
}

if ($RunPairedVal -or ($Stage -eq "paired_val")) {
    foreach ($seed in $Seeds) {
        $runDir = Join-Path $OutputRoot "ppc\seed_$seed"
        $best = Join-Path $runDir "best.pt"
        if (-not (Test-Path -LiteralPath $best -PathType Leaf)) { throw "Missing required checkpoint: $best" }
        Invoke-ExperimentPython "$V6Dir\scripts\validate_v2.py" @CommonArgs --arch ppc --run-dir $runDir --resume $best --eval-split paired_val
    }
}
if ($RunPairedVal -and ($Stage -eq "all")) {
    Write-Host "[V2][Stage 7] Evaluating locked ASD-SC vs Static-1Step on historical VAL..."
    foreach ($seed in $Seeds) {
        $fullBest = "$OutputRoot\stage1_asd_sc_full\seed_$seed\best.pt"
        if (-not (Test-Path -LiteralPath $fullBest -PathType Leaf)) { throw "Missing required checkpoint: $fullBest" }
        Invoke-ExperimentPython "$V6Dir\scripts\validate_v2.py" `
            --prepared-root $PreparedRoot --protocol-dir $ProtocolDir `
            --run-dir "$OutputRoot\stage1_asd_sc_full\seed_$seed" `
            --ablation-mode full --resume $fullBest --eval-split paired_val `
            --split-seed $SplitSeed --device $Device --max-eval-records "$MaxEvalRecords"
        $statBest = "$OutputRoot\abl3_replay_unroll\static_1step\seed_$seed\best.pt"
        if (-not (Test-Path -LiteralPath $statBest -PathType Leaf)) { throw "Missing required checkpoint: $statBest" }
        Invoke-ExperimentPython "$V6Dir\scripts\validate_v2.py" `
            --prepared-root $PreparedRoot --protocol-dir $ProtocolDir `
            --run-dir "$OutputRoot\abl3_replay_unroll\static_1step\seed_$seed" `
            --ablation-mode full --resume $statBest --eval-split paired_val `
            --split-seed $SplitSeed --device $Device --max-eval-records "$MaxEvalRecords"
    }
}

Write-Host "[V2] Completed successfully."
