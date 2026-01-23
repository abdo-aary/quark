from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import hashlib
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig

from hydra.utils import instantiate

from src.data.generate.base import WindowsDataset
from src.settings import PROJECT_ROOT_PATH


@dataclass(frozen=True)
class DatasetArtifact:
    root: Path
    data_path: Path
    meta_path: Path | None
    config_path: Path | None


def _as_project_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return Path(PROJECT_ROOT_PATH) / p


def _stable_cfg_hash(cfg: DictConfig) -> str:
    """Short stable hash of the fully-resolved config (for traceability)."""
    obj = OmegaConf.to_container(cfg, resolve=True)
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def build_generator(cfg: DictConfig):
    """Instantiate the process generator from cfg.process.generator."""
    return instantiate(cfg.process.generator)


def build_label_functionals(cfg: DictConfig) -> List[Any]:
    """Instantiate label functionals from cfg.functionals['items']."""
    items = cfg.functionals["items"]  # IMPORTANT: avoid cfg.functionals.items (method)
    if not isinstance(items, (list, ListConfig)):
        raise TypeError(f"cfg.functionals['items'] must be a list, got {type(items)}")

    return [instantiate(item) for item in items]


def make_dataset(cfg: DictConfig) -> WindowsDataset:
    """Generate dataset in-memory."""
    N = int(cfg.sampling.N)
    w = int(cfg.sampling.w)
    d = int(cfg.sampling.d)

    gen = build_generator(cfg)
    labs = build_label_functionals(cfg)

    ds = gen.make_windows_dataset(N=N, w=w, d=d, label_functionals=labs)

    # augment meta with some high-level info
    meta = dict(ds.meta)
    meta.update(
        {
            "N": N,
            "w": w,
            "d": d,
            "L": int(ds.y.shape[0]),
            "process_kind": str(cfg.process.kind),
            "functionals_kind": str(cfg.functionals.kind),
            "seed": int(cfg.seed),
            "cfg_hash": _stable_cfg_hash(cfg),
        }
    )
    ds.meta = meta
    return ds


def _auto_name(cfg: DictConfig) -> str:
    N, w, d = int(cfg.sampling.N), int(cfg.sampling.w), int(cfg.sampling.d)
    s = int(cfg.sampling.s)
    proc = str(cfg.process.kind)
    fun = str(cfg.functionals.kind)
    seed = int(cfg.seed)
    return f"{proc}__{fun}__N={N}__w={w}__d={d}__s={s}__seed={seed}"


def save_dataset(ds: WindowsDataset, cfg: DictConfig) -> DatasetArtifact:
    """Save dataset according to cfg.output.*"""
    out = cfg.output
    save_dir = _as_project_path(out.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    name = str(out.name)
    if name.lower() == "auto":
        name = _auto_name(cfg)

    fmt = str(out.format).lower()
    overwrite = bool(out.get("overwrite", False))

    if fmt == "npz":
        data_path = save_dir / f"{name}.npz"
        if data_path.exists() and not overwrite:
            raise FileExistsError(f"{data_path} exists (set output.overwrite=true to overwrite).")
        np.savez_compressed(data_path, X=ds.X, y=ds.y)

    elif fmt == "npy":
        # store as two separate arrays, but still return a "data_path" pointing to X
        x_path = save_dir / f"{name}.X.npy"
        y_path = save_dir / f"{name}.y.npy"
        if (x_path.exists() or y_path.exists()) and not overwrite:
            raise FileExistsError(f"{name} .npy exists (set output.overwrite=true to overwrite).")
        np.save(x_path, ds.X)
        np.save(y_path, ds.y)
        data_path = x_path
    else:
        raise ValueError(f"Unknown output.format={fmt!r}")

    meta_path = None
    if bool(out.get("save_meta", True)):
        meta_path = save_dir / f"{name}.meta.json"
        meta_path.write_text(json.dumps(ds.meta, indent=2), encoding="utf-8")

    config_path = None
    if bool(out.get("save_config", True)):
        config_path = save_dir / f"{name}.config.yaml"
        OmegaConf.save(cfg, config_path, resolve=True)

    return DatasetArtifact(root=save_dir, data_path=data_path, meta_path=meta_path, config_path=config_path)


def generate_and_save_dataset(cfg: DictConfig) -> Tuple[WindowsDataset, DatasetArtifact]:
    ds = make_dataset(cfg)
    art = save_dataset(ds, cfg)
    return ds, art
