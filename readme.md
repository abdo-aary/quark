# QuaRK — Effective & Efficient Quantum Reservoir Computing

This repository contains the **experimental codebase** used to generate the **empirical results** of the paper:

> **QuaRK: A Quantum Reservoir Kernel for Time Series Learning**  
> Abdallah Aaraba · Soumaya Cherkaoui · Ola Ahmad · Shengrui Wang

QuaRK is an end-to-end quantum–classical pipeline for time-series learning that:
- **projects** each windowed input using a **Johnson–Lindenstrauss (JL)** map (so circuit width is decoupled from ambient dimension),
- **featurizes** the projected sequence with a **hardware-realistic quantum reservoir** (ring architecture, spatial multiplexing),
- **measures** compact features via **classical shadows** on families of *k*-local Pauli observables,
- **learns** a **kernel ridge regression (KRR)** readout (Matérn kernel in the paper) with explicit regularization control.

---

## Table of contents
- [Repository at a glance](#repository-at-a-glance)
- [Quick start](#quick-start)
- [Reproducing the paper’s empirical section](#reproducing-the-papers-empirical-section)
- [Outputs and artifacts](#outputs-and-artifacts)
- [Configuration (Hydra)](#configuration-hydra)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)

---

## Repository at a glance

**Top-level**
- `docs/` — short design notes (“tour guide” to the codebase)
- `notebooks/` — notebooks that *aggregate* and *plot* final paper results
- `src/` — the implementation (data, QRC, models, experiments)
- `tests/` — unit/integration tests for critical components
- `requirements.txt` — Python dependencies

**Code layout (high-level)**

```text
.
├── docs/
├── notebooks/
├── src/
│   ├── data/         # datasets, synthetic generators (e.g., VARMA), artifact I/O
│   ├── qrc/          # circuit configs, circuit families, runners, feature retrievers
│   ├── models/       # learning models consuming QRC features (e.g., Matérn-KRR)
│   ├── experiment/   # orchestration layer + scripts + Hydra configs
│   └── utils/        # misc helpers (logging, persistence, math, ...)
└── tests/
```

---

## Quick start

### 0) Environment setup

Create a virtual environment (or conda env) and install dependencies:

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt
```

So that `python -m src....` works from the repository root, export:

```bash
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}"
```

> Tip: if you use Windows PowerShell, use:
> `setx PROJECT_ROOT $PWD` and `setx PYTHONPATH $PWD`

---

## Reproducing the paper’s empirical section

### Option A (recommended): run the paper notebook with the released `storage/` artifacts

The paper figures are reproduced in:

- `notebooks/main-results.ipynb`

To run it **as-is**, download the **`storage.zip`** artifact from the **v1.0.0 release** and unzip it so you have:

```text
storage/
  data/
  results/
  ...
```

Release link (contains `storage.zip`):  
https://github.com/abdo-aary/effective-efficient-qrc/releases/tag/v1.0.0

Once `storage/` is in the repo root, open and run:

```bash
jupyter lab
# or
jupyter notebook
```

---

### Option B: regenerate data and rerun experiment scripts

This repo uses an **artifact-based workflow**:

1) **Generate** a dataset artifact under `storage/data/...` (data + metadata + config snapshot)  
2) **Run** an experiment script that loads the artifact, builds the model, fits, and writes results/plots

#### 1) Generate a synthetic dataset artifact (VARMA)

A minimal smoke-test (tiny dataset):

```bash
python -m src.experiment.scripts.generate_data   data.sampling.N=16 data.sampling.w=6 data.sampling.d=2 data.sampling.s=2
```

Paper-style settings (as described in Section 4.1 of the paper):
- dimension `d=3`, window length `w=25`
- VARMA with `p=q=3`
- stride `s=100` (gap 75)

The exact Hydra keys for `p,q` depend on the generator config; the notebook + release artifact (Option A)
is the most reliable way to reproduce the paper without chasing config names.

#### 2) Run the Matérn-KRR regularization sweep (Figure 3-style workflow)

```bash
python -m src.experiment.scripts.run_reg_sweep_experiment   dataset_path="/abs/path/to/storage/data/.../your_dataset_artifact"
```

Common overrides:

```bash
# shorten the sweep grid for quick runs
python -m src.experiment.scripts.run_reg_sweep_experiment   dataset_path="/abs/path/to/dataset_artifact"   reg_sweep.reg_grid='[1e-12,1e-9,1e-6,1e-3]'

# write artifacts to a custom folder
python -m src.experiment.scripts.run_reg_sweep_experiment   dataset_path="/abs/path/to/dataset_artifact"   output.save_dir="${PROJECT_ROOT}/storage/results"
```

#### 3) Sweep training-set size (Figure 5-style workflow)

The paper’s second experiment sweeps the number of training windows (e.g., `N=100` to `8000`) while keeping
the same featurizer + kernel + relaxed regularization.

In this repo, the “sweep” utilities typically live under `src/experiment/` and/or are invoked from the main notebook.
If you want to re-run this from scripts rather than notebooks, start from:
- `docs/experiment_overview.md`
- `notebooks/main-results.ipynb` (the most direct reference implementation)

---

## Outputs and artifacts

Most runs write artifacts under Hydra’s run directory (and optionally also under `storage/results/`), including:

- `*.npz` — raw arrays for sweeps
- `*.csv` — tidy metrics tables
- `*.pdf` — publication-ready plots

Example (regularization sweep):
- `reg_sweep.npz`
- `reg_sweep_metrics.csv`
- `reg_sweep_train.pdf`, `reg_sweep_test.pdf`

---

## Configuration (Hydra)

All scripts are driven by **Hydra configs** under `src/experiment/conf/`.

Good starting points:
- `docs/run_overview.md` — recommended run patterns + Hydra override examples
- `docs/experiment_overview.md` — end-to-end run flow and what artifacts are produced
- `docs/data_overview.md` — what a “dataset artifact” is and where it lives on disk

---

## Testing

Run tests from the repo root:

```bash
pytest -q
```

---

## Citation

If you use this codebase in academic work, please cite the QuaRK paper.

```bibtex
@misc{quark,
  title        = {QuaRK: A Quantum Reservoir Kernel for Time Series Learning},
  author       = {Aaraba, Abdallah and Cherkaoui, Soumaya and Ahmad, Ola and Wang, Shengrui},
  note         = {Preprint},
}
```

---

## License

This project is released under the **MIT License** (see `LICENSE`).
