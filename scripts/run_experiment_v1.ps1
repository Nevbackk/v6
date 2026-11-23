<#
.SYNOPSIS
Complete Execution Pipeline — Model Plan V1 (JOINT_RESEARCH_PLAN + Flagship FD-ASF)

.EXAMPLE
.\scripts\run_experiment_v1.ps1 -PreparedRoot "D:\G4\prepared" -ProtocolDir "D:\G4\protocol" -Device "cuda"
.\scripts\run_experiment_v1.ps1 -Stage "test" -PythonBin "C:\Users\skywalker\.conda\envs\temporal2seq-gpu\python.exe"
#>
param(
    [string]$PreparedRoot = $env:PREPARED_ROOT,
    [string]$ProtocolDir = $env:PROTOCOL_DIR,
    [string]$OutputRoot = "",
    [string]$PythonBin = "C:\Users\skywalker\.conda\envs\temporal2seq-gpu\python.exe",
    [string]$Device = "cuda",
    [int]$Epochs = 10,
    [int]$BatchSize = 32,
    [double]$Lr = 0.001,
    [int[]]$Seeds = @(20261028, 20261029, 20261030),
    [ValidateSet("all", "test", "classical", "abl_3x2", "ltr_lm", "fd_asf", "paired_val")]
    [string]$Stage = "all",
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
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot "runs\exp_v1" }
if (-not (Test-Path $PythonBin)) { $PythonBin = "python" }

Write-Host "======================================================================"
Write-Host "[Model Plan V1] Stage='$Stage' | Device='$Device' | Output='$OutputRoot'"
Write-Host "======================================================================"

if ($Stage -in @("all", "test")) {
    Write-Host "[V1][Stage 0] Running V6 + V1 + V2 unit tests..."
    Push-Location $V6Dir
    & $PythonBin -s -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { Pop-Location; throw "Unit tests failed" }
    Pop-Location
    if ($Stage -eq "test") { exit 0 }
}

if (-not $PreparedRoot -or -not $ProtocolDir) {
    throw "PreparedRoot and ProtocolDir are required for data stages."
}

$CommonArgs = @(
    "--prepared-root", $PreparedRoot,
    "--protocol-dir", $ProtocolDir,
    "--device", $Device,
    "--max-records-per-epoch", "$MaxRecordsPerEpoch",
    "--max-eval-records", "$MaxEvalRecords"
)

if ($Stage -in @("all", "classical")) {
    foreach ($arch in @("cl_gn1_pure", "cl_gn1_matched", "cl_window_lm", "cl_full_prefix_wls")) {
        Write-Host "[V1][Stage 1] Evaluating classical baseline '$arch' on dev_select..."
        & $PythonBin "$V6Dir\scripts\validate_v1.py" @CommonArgs `
            --run-dir "$OutputRoot\stage1_classical\$arch" `
            --arch $arch --writeback-mode joint_6d --eval-split dev_select
        if ($LASTEXITCODE -ne 0) { throw "Classical validation failed for $arch" }
    }
}

if ($Stage -in @("all", "abl_3x2")) {
    foreach ($loss in @("scaled_mse", "huber", "log_charbonnier")) {
        foreach ($jfMode in @("vel_only", "homog_6d")) {
            foreach ($seed in $Seeds) {
                $runDir = "$OutputRoot\stage2_jfbr_3x2\${jfMode}_${loss}\seed_${seed}"
                Write-Host "[V1][Stage 2] Training JF-BR ($jfMode) | loss=$loss | seed=$seed..."
                & $PythonBin "$V6Dir\scripts\train_v1.py" @CommonArgs `
                    --run-dir $runDir --arch jf_br --jfbr-mode $jfMode --loss-type $loss `
                    --epochs "$Epochs" --batch-size "$BatchSize" --lr "$Lr" --seed "$seed" --eval-split dev_select
                if ($LASTEXITCODE -ne 0) { throw "JF-BR training failed" }
            }
        }
    }
}

if ($Stage -in @("all", "ltr_lm")) {
    foreach ($wb in @("joint_6d", "schur_vel", "lock_pos_vel")) {
        foreach ($seed in $Seeds) {
            $runDir = "$OutputRoot\stage3_ltr_lm\$wb\seed_$seed"
            Write-Host "[V1][Stage 3] Training LTR-LM ($wb) | seed=$seed..."
            & $PythonBin "$V6Dir\scripts\train_v1.py" @CommonArgs `
                --run-dir $runDir --arch ltr_lm --writeback-mode $wb --loss-type log_charbonnier `
                --epochs "$Epochs" --batch-size "$BatchSize" --lr "$Lr" --seed "$seed" --eval-split dev_select
            if ($LASTEXITCODE -ne 0) { throw "LTR-LM training failed" }
        }
    }
}

if ($Stage -in @("all", "fd_asf")) {
    foreach ($p in @(4, 12, 16)) {
        Write-Host "[V1][Stage 4] Evaluating Sobol-Lambert-LM | P=$p..."
        & $PythonBin "$V6Dir\scripts\validate_v1.py" @CommonArgs `
            --run-dir "$OutputRoot\stage4_flagship\sobol_lambert_P$p" `
            --arch sobol_lambert_lm --num-particles "$p" --eval-split dev_select
        if ($LASTEXITCODE -ne 0) { throw "Sobol-Lambert evaluation failed" }
    }
    foreach ($arch in @("posterior_cnf", "fd_asf")) {
        foreach ($seed in $Seeds) {
            $runDir = "$OutputRoot\stage4_flagship\${arch}_P12\seed_$seed"
            Write-Host "[V1][Stage 4] Training $arch (P=12) | seed=$seed..."
            & $PythonBin "$V6Dir\scripts\train_v1.py" @CommonArgs `
                --run-dir $runDir --arch $arch --num-particles 12 --dsm-weight 0.05 `
                --epochs "$Epochs" --batch-size "$BatchSize" --lr "$Lr" --seed "$seed" --eval-split dev_select
            if ($LASTEXITCODE -ne 0) { throw "$arch training failed" }
        }
    }
}

if ($RunPairedVal -or ($Stage -eq "paired_val")) {
    Write-Host "[V1][Stage 5] Running locked paired replication on historical VAL..."
    & $PythonBin "$V6Dir\scripts\validate_v1.py" @CommonArgs `
        --run-dir "$OutputRoot\stage1_classical\cl_gn1_matched" --arch cl_gn1_matched --eval-split paired_val
    foreach ($seed in $Seeds) {
        $ltrBest = "$OutputRoot\stage3_ltr_lm\joint_6d\seed_$seed\best.pt"
        if (Test-Path $ltrBest) {
            & $PythonBin "$V6Dir\scripts\validate_v1.py" @CommonArgs `
                --run-dir "$OutputRoot\stage3_ltr_lm\joint_6d\seed_$seed" `
                --arch ltr_lm --writeback-mode joint_6d --resume $ltrBest --eval-split paired_val
        }
        $fdBest = "$OutputRoot\stage4_flagship\fd_asf_P12\seed_$seed\best.pt"
        if (Test-Path $fdBest) {
            & $PythonBin "$V6Dir\scripts\validate_v1.py" @CommonArgs `
                --run-dir "$OutputRoot\stage4_flagship\fd_asf_P12\seed_$seed" `
                --arch fd_asf --num-particles 12 --resume $fdBest --eval-split paired_val
        }
    }
}

Write-Host "[Model Plan V1] Completed successfully."
