from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.settings import PROJECT_ROOT_PATH
from src.experiment.scripts import generate_data


def _conf_dir() -> Path:
    root = Path(PROJECT_ROOT_PATH)
    candidates = [
        root / "experiment" / "conf" / "data",
        root / "src" / "experiment" / "conf" / "data",
    ]
    for c in candidates:
        if (c / "config.yaml").exists():
            return c
    raise FileNotFoundError(f"Could not find data/config.yaml under {candidates}.")


def test_generate_data_run_unitish(tmp_path: Path):
    """UNIT-ISH: compose cfg and call generate_data.run(cfg) (no subprocess)."""
    GlobalHydra.instance().clear()
    conf_dir = _conf_dir()

    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "sampling=tiny",
                "process=varma",
                "functionals=one_step",
                "noise=none",
                f"output.save_dir={tmp_path.as_posix()}",
                "output.name=pytest_script_unit",
                "output.format=npz",
                "output.overwrite=true",
                "output.save_meta=true",
                "output.save_config=true",
            ],
        )

    ds, art = generate_data.run(cfg)
    assert art.data_path.exists()
    assert ds.X.shape[0] == int(cfg.sampling.N)


def test_generate_data_script_subprocess_integration(tmp_path: Path):
    """INTEGRATION: run the Hydra script exactly as users will (python -m ...)."""
    root = Path(PROJECT_ROOT_PATH)
    env = dict(os.environ)
    # ensure subprocess can import `src.*`
    env["PYTHONPATH"] = str(root)

    hydra_run_dir = tmp_path / "hydra_runs"
    cmd = [
        sys.executable,
        "-m",
        "src.experiment.scripts.generate_data",
        "sampling=tiny",
        "process=varma",
        "functionals=one_step",
        "noise=none",
        f"output.save_dir={tmp_path.as_posix()}",
        "output.name=pytest_script_integration",
        "output.format=npz",
        "output.overwrite=true",
        f"hydra.run.dir={hydra_run_dir.as_posix()}",
    ]

    res = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    expected = tmp_path / "pytest_script_integration.npz"
    assert expected.exists(), f"Did not find expected dataset at {expected}"
