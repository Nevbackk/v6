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

## Per-frame orbit estimation

V1 learns the angular sensor from a known initial state. V2 starts at the supplied noisy Gauss–LS estimate and updates **one normalized six-dimensional probability density** on each valid LOS arrival. Both use the actual elapsed time, known station state and deterministic two-body dynamics. Batch replay calls the same per-frame recursion as a live stream.

| Entry point | Orbit state carried between observations | What each arrival contributes |
| --- | --- | --- |
| `scripts/train_v1.py` | A known state propagated by RK4 | A predictive angular bias and covariance from a learned reverse Markov chain, followed by a sensor innovation and score. |
| `scripts/train_v2.py` (default `--arch ppc`) | One invertible spline density with a physical reference, a full-rank factor and 150 shape coefficients | One reference orbit and its first two variations propagate the previous shape. A QR update absorbs the local Gaussian information. The remaining spherical likelihood changes the shape and relative mass within that same density. |

For V2, call `state = model.start_track(x0, t0)` once and `result = model.step(state, tk, los, station)` at each arrival. `result["distribution"].log_prob(x_si)` gives the full state log density in SI units. `model.readout(state.posterior)` returns quadrature estimates of the mean, covariance and marginal intervals. Integration nodes serve individual integrals and carry no trajectory identity.

The default sensor uses four cosine VP reverse transitions. Within V2, `--K` also sets the number of fractional likelihood updates whose powers sum to one. `--P` controls angular sampling and the number of CFM training pairs, `--warp-nodes` sets the integration rule per reference chart, and `--unroll-frames` sets the training gradient span. Physical propagation always uses the observation timestamps.

```bash
python scripts/train_v1.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/v1
python scripts/train_v2.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/v2 --sensor-checkpoint runs/v1/best.pt
python scripts/validate_v2.py --prepared-root /path/to/prepared --protocol-dir /path/to/protocol --run-dir runs/v2 --eval-split paired_val
```

V1 selects on held-out angular Gaussian moment NLL. V2 selects on source-equal full-record scaled MSE. Evaluation also reports failed records, the four sampling cadences, initialization strata, six-dimensional density and 90% highest-density coverage, and the next observed LOS predictive score and cone coverage for PPC and its Gaussian controls. Sensor transfer requires the same source split and sensor model. The main Bash and PowerShell V2 runners save under `ppc/seed_*`.

## Matched controls and CPU checks

| Option | What it measures |
| --- | --- |
| `train_v2.py --arch qr --observation-model calibrated` | Classical local Gaussian reference with causal angular calibration. |
| `train_v2.py --arch gaussian --observation-model direct` | Same nonlinear evidence and quadrature as PPC, projected onto one Gaussian. |
| `train_v2.py --arch ppc --observation-model direct` | Direct heteroscedastic sensor moments in place of the angular diffusion chain. |
| `train_v1.py --arch cl_full_prefix_wls --initialization gauss_ls` | Causal full-prefix least squares starting from the same noisy initial state as V2. |
| `train_v2.py --arch diffusion` | Previous state VP/VJP filter. Archived ASD-SC and B1/B2 experiments have their own explicit entries. |

On a controlled range slice with two positive solutions near 4,200 and 8,358 km, three independently trained splines assigned 2.58–3.00% to the interval between the peaks, compared with 34.56% for the matched projected Gaussian. The reference grid assigns about 0.00000085%. This experiment isolates shape estimation in one physical dimension with five Gaussian nuisance coordinates. Full G4 orbit accuracy and good-initial-orbit performance require the corresponding data evaluation.

```bash
python scripts/check_ppc_two_ranges.py --seed 17 --iterations 500 --output range.json --figure range.png
python scripts/profile_frame_filter.py --arch ppc
python -s -m unittest discover -s tests -v
```

The physical prediction uses a local second-order approximation. The shipping estimator takes an external initial state. Detailed equations, sensor alternatives, training losses and experiment definitions appear in the [method description](docs/METHODS_V1_V2_CN.md).
