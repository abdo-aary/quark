# QuaRK: A Quantum Reservoir Kernel for Time Series Learning

This repository implements the experimental pipeline accompanying the paper [*QuaRK: A Quantum Reservoir Kernel for Time 
Series Learning*](https://example.com). ...

## Project layout

- `src/data/` — dataset abstractions, synthetic generators (e.g., VARMA), labeling functionals, and dataset artifact I/O (save/load).
- `src/qrc/` — circuit configurations, circuit families, runners (Aer / other backends), feature-map retrievers (exact or sampling-based).
- `src/models/` — learning models built on top of QRC features (notably `QRCMaternKRRRegressor`, including regularization sweeps).
- `src/experiment/` — orchestration layer and scripts to reproduce experiments (load artifact → build model → fit → evaluate → save plots/metrics).
- `src/utils/` — small shared utilities (logging, math helpers, persistence helpers, etc.).
- `docs/` — human-readable design notes describing objects and how the packages interact (see below).

## Documentation map

The `docs/` folder is meant to be the “tour guide” to the codebase:

- `docs/data_overview.md` — how dataset artifacts are structured on disk; what metadata is stored; how to load/save.
- `docs/circuits_overview.md` — how QRC circuits are specified, built, and executed; what gets cached and where.
- `docs/models_overview.md` — model classes and how they consume features/kernels; tuning vs sweeping; expected inputs/outputs.
- `docs/experiment_overview.md` — experiment orchestration, Hydra configs, and the artifact outputs (plots, CSV/NPZ results).
- `docs/run_overview.md` — recommended run patterns and Hydra override examples.

## Quick start

### 0) Setup

Create/activate your environment, install in editable mode, and define the repository root used by configs:

```bash
# from repo root
export PROJECT_ROOT=$(pwd)

pip install -e .
```

### 1) Generate a tiny synthetic dataset (optional)

The generator is driven by Hydra configs under `src/experiment/conf/data/…`. Example for a small VARMA dataset:

```bash
python -m src.experiment.scripts.generate_data \
  data.sampling.N=16 data.sampling.w=6 data.sampling.d=2 data.sampling.s=2
```

This writes a dataset artifact under `storage/data/...` (data + metadata + config snapshot).

### 2) Run a Matérn-KRR regularization sweep (main “figure” workflow)

The regularization sweep fits the model once (tuning Matérn hyperparameters) and then sweeps \(\lambda_{\mathrm{reg}}\) to produce train/test curves, with one curve per labeling functional.

```bash
python -m src.experiment.scripts.run_reg_sweep_experiment \
  dataset_path="/abs/path/to/storage/data/.../N=16__w=6__d=2__s=2"
```

Useful overrides:

```bash
# Choose the grid you want (short grid for quick smoke tests)
python -m src.experiment.scripts.run_reg_sweep_experiment \
  dataset_path="/abs/path/to/dataset_artifact" \
  reg_sweep.reg_grid='[1e-12,1e-9,1e-6,1e-3]'

# Control where artifacts are written
python -m src.experiment.scripts.run_reg_sweep_experiment \
  dataset_path="/abs/path/to/dataset_artifact" \
  output.save_dir="${oc.env:PROJECT_ROOT}/storage/results"
```

Outputs (written under Hydra’s run directory by default):

- `reg_sweep.npz` — raw sweep results
- `reg_sweep_metrics.csv` — tidy metrics table
- `reg_sweep_train.pdf`, `reg_sweep_test.pdf` — train/test MSE vs regularization (one curve per functional)

## Reproducibility notes

- Every run is controlled by Hydra configs; configs are saved alongside outputs (Hydra `.hydra/` folder).
- Dataset artifacts include metadata/config snapshots so experiments can be reproduced from an on-disk artifact alone.
- The `Experiment` wrapper caches QRC features during `.fit()` so that sweeps and follow-up evaluations do not re-run expensive circuit executions.

## Paper results
- All empirical results of the paper can be found in notebooks under `notebooks/main-results.ipynb`.
- To run the cells of such a notebook, download first `storage.zip` from the v1.0.0 release assets here 
https://github.com/abdo-aary/effective-efficient-qrc/releases/tag/v1.0.0.
- The zip file should be unzipped and should be `storage/` with the expected structure of subfolders and files).

## Where to look next

If you want to understand the codebase quickly:

1. Read `docs/experiment_overview.md` to see the end-to-end run flow and artifacts.
2. Skim `docs/models_overview.md` to see how `QRCMaternKRRRegressor` is structured (tuning vs sweeping).
3. Use `docs/data_overview.md` to understand what a “dataset artifact” path should point to on disk.

---

If you are adding a new experiment, the intended workflow is: add a small Hydra config under `src/experiment/conf/`, implement a short script under `src/experiment/scripts/`, and wire the logic via `src.experiment.experiment.Experiment` so the run produces consistent artifacts (NPZ/CSV + plots).
