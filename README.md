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
