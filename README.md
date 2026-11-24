# V6 NoHistoricalState orbit filters

A shared PyTorch backbone implements V6-NoHistoricalState-Gain and
V6-NoHistoricalState-DirectCorrection. Both use two-body RK4 prediction,
current LOS innovation, scalar/vector tokens, relation-conditioned attention
and the original LOS Direct channels. Attention retains the current state
token and the full observation history while masking historical state keys.

With `D = diag(1e6, 1e6, 1e6, 1e3, 1e3, 1e3)`, the Gain head
(`240 → 128 → GELU → 18`) emits a 6-by-3 matrix and applies
`x_post = D (D⁻¹ x_prior + K e)`. The DirectCorrection head
(`240 → 128 → GELU → 6`) applies `x_post = D (D⁻¹ x_prior + delta)`.
LOS innovation remains an input feature in both modes.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

CUDA fusion uses Triton 3.4.0 on Linux/NVIDIA. The production environment used
PyTorch 2.8.0 with CUDA 12.8. The CPU entry points use the PyTorch blockwise
backend. Run the CPU smoke check with `python -m unittest discover -s tests -v`.

## External data

Pass the original frozen prepared root via `--prepared-root`. It must contain
`initialization_cache/{train,val}/shard_*.jsonl` and
`prepared_cache/{train,val}/shards/shard_*.pt`. Supply a separate
`--protocol-dir` containing the original baseline `config.json` and the
`train_source_allowlist.csv` and `val_source_allowlist.csv` files named by it.
The loader verifies the allowlist digests, frozen record membership, the
shared Gauss-LS B0 and the strict position error threshold below 10,000 km.
The formal snapshot contains 43,783 train and 10,857 validation records.
Use trusted cache and checkpoint files because PyTorch loads their Python objects.

## Train and validate

```bash
python scripts/train_gain.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir /path/to/gain
python scripts/train_direct.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir /path/to/direct
python scripts/validate_gain.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir /path/to/gain --resume /path/to/gain/epoch_10.pt
python scripts/validate_direct.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir /path/to/direct --resume /path/to/direct/epoch_10.pt
```

Defaults are 10 epochs, effective batch 128, seed 20261028, Adam at 0.001,
weight decay 0.0001, gradient clipping at 1.0, FP64, and AMP off.
The frame-weighted six-state scaled MSE is accumulated over the full effective
batch before one Adam update. Recursion uses full BPTT and the full observation
history. The Gain default is `blockwise_checkpoint` with microbatch 8.
CUDA Direct defaults to `fused_tiles_save` with microbatch capped at 32 and
length-aware grouping. See `--help` for device, epochs, batch, LR and seed options.

Both heads initialize the final linear layer to zero. Direct pairs its shared
weights and hidden head layer to a seeded Gain initialization by default.
Supply the original external Gain initialization via `--gain-step-zero` to
reproduce the server run's exact paired tensors.

Training saves `last.pt` after each update, runs recursive validation each
epoch, and writes per-epoch checkpoints and aggregate metrics. It resumes
automatically from the release run's `last.pt`, or from `--resume`.
The portable trainer has its own resume metadata. Server checkpoints with a
`model` state dict can be used by the validation entry points. Validation in
this release reports scaled MSE, position and velocity RMSE, and nonfinite
frame counts. The server's paired reports and diagnostic pipeline are separate.

## Model source

- Gain launch: AutoDL `v6_nohistoricalstate_optimized_resume_20261122/launch/run_resume.sh`,
  which places the accepted `variants/exact_geometry` source on `PYTHONPATH`.
- Direct launch: AutoDL `v6_directcorrection_knetmatched_b0lt10000_fused_20261123/code/run_pipeline.sh`,
  which uses the isolated vendor model and the `fused_tiles_save` training backend.

The shared source incorporates the Direct run's dual-mode integration and the
Gain path's exact-geometry blockwise backend. The two-body RK4 batch implementation
is shared by the portable model; the serial adapter uses it for a single track.
No KalmanNet network or experiment infrastructure is needed at runtime.

## Model Plan V1 & Model Plan V2 Extensions

Building directly on the frozen `Prepared` data loader and differentiable `float64` two-body RK4 physics, this repository also implements **Model Plan V1** and **Model Plan V2**:

- **Shared Physics, STM & Stratified Evaluation**:
  - `v6/models/reprojection.py`: analytical RK4 tangent-linear STM $\Phi(t_j, t_k)$, branchless $\mathrm{SO}(3)$ LOS tangent frames, spherical retraction $\mathrm{Exp}_{S^2}$, local Log-MSC chart $B_k$, sign-verified sensitivity $A = -D_y\nu(0)$, and `pois10`-compatible unequal-time triplet noise-floor calibration $\tau_k$.
  - `v6/data/splits.py` & `v6/eval_metrics.py`: `source_id`-isolated `train_fit / train_calib / dev_select` splits, macro-checkpoint & `Block 2 + 30 s` scheduling, 4-cadence + `source_id`-equal tables, fair prior-referenced good-orbit harm rates, and cluster-paired bootstrap 95% CIs.
- **Model Plan V1 (`v6/models/v1_ltr_lm.py`, `v6/models/v1_fd_asf.py`, `v6/training_v1.py`)**:
  - Classical non-learned controls (`cl_gn1_pure`, `cl_gn1_matched`, `cl_window_lm`, `cl_full_prefix_wls`), Jacobian-free bounded refiners (`JF-BR-Vel`, `JF-BR-6D`), and learned trust-region preconditioned 1-step LM (`LTR-LM` with `joint_6d`, `schur_vel`, `lock_pos_vel`).
  - Flagship `FD-ASF`: batch-invariant continuous-time Legendre tracklet encoder $(H_s, B_s)$, $2+4$ Lambert ($M=0$) & Attributable ambiguity charts with analytic Attributable and finite-difference Lambert volume factors, 4-layer Conditional RealNVP proposal trained on SNIS optimal forecast disagreement $q_k^\star \propto \pi_k\|\mathbf{h}_{k+1}-\bar{\mathbf{h}}_{k+1}\|_2$ with defensive mixing, prefix likelihood reweighting, and 3-condition dominance-gated publishing.
  - Entry points: `scripts/train_v1.py`, `scripts/validate_v1.py`.
- **Model Plan V2 (`v6/models/v2_asd_sc.py`, `v6/training_v2.py`)**:
  - `ASD-SC`: progressive subspace bounded blind denoiser ($L=2$ microsteps, 0.5/0.9 directional log-error quantile head, deadzone $\chi_i$, bounded shrinkage $g_i \in [g_{\min}, 1]$), cross-gap thin-QR pending subspace transport $D^{-1}\Phi D V_p = QR$, $K_p$ conditional unlock & $E_a$ restricted writeback ($E_h^\top\Delta\mathbf{y}=0$), A/B/C causal re-anchoring, DAgger rollout buffer, 2-checkpoint cross-gap unrolling, $\{0, 60, 300\}\text{ s}$ multi-horizon risk, and switchable progressive step-increment supervision $T_\ell\mathbf{d}_\ell$.
  - Entry points: `scripts/train_v2.py`, `scripts/validate_v2.py`.

## Diffusion filters for optical orbit estimation

Use `scripts/train_diffusion.py` and `scripts/validate_diffusion.py` with the same frozen prepared root and protocol directory as the V1 and V2 trainers.

| `--route` | Estimator |
| --- | --- |
| `a` | A cosine variance-preserving diffusion filter denoises the six-dimensional initialization error. Each reverse step uses a locally linearized angular likelihood and a denoiser vector–Jacobian product. TRAIN-fit sources calibrate the initial mean error and covariance. |
| `b1` | A cosine flow-matching proposal samples range coordinates on an Attributable or Lambert chart. Its invertible substeps have explicit log-determinants. Importance weights include the angular likelihood, physical chart volume and proposal density. |
| `b2` | A stochastic Gaussian reverse chain uses learned proposal drifts, intermediate angular potentials and ancestry-aware resampling. The terminal importance ratio targets the same feasible chart as B1. |

```bash
python scripts/train_diffusion.py --route b1 --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/b1 --K 4 --P 8
python scripts/validate_diffusion.py --run-dir runs/b1 --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --eval-split dev_select
```

`--K` sets the number of artificial-time intervals and `--P` the particle count. Route A uses `--delta` for its covariance floor. `--B-los` optionally limits likelihood evaluations to a subset of observations. The selected checkpoint includes the route and source split. Validation defaults to `best.pt`, and training resumes from `last.pt`. Use `--eval-split paired_val` for a locked configuration.

The Bash and PowerShell experiment runners provide `main`, `b1`, `b2` and `test` stages. `paired_val`, `paired_b1` and `paired_b2` validate saved Route A, B1 and B2 runs, respectively. The CPU tests include two-body synthetic tracks and temporary prepared data that exercise all three training and validation entry points.

