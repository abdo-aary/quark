"""
Dummy objects used to unit-test model configuration wiring.

These are intentionally tiny, fast-to-instantiate stand-ins for:
- QR config objects
- circuit runners
- feature map retrievers
- observables builders
- featurizer (optional patch target)

They allow testing `.from_config(cfg)` without requiring Qiskit Aer or running simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class DummyQRConfig:
    """
    Minimal QR config object.

    Parameters
    ----------
    input_dim : int
        Input dimension `d`.
    num_qubits : int
        Number of qubits.
    seed : int
        RNG seed.
    """
    input_dim: int
    num_qubits: int
    seed: int = 0


class DummyRunner:
    """
    Minimal runner stand-in.

    Parameters
    ----------
    qr_cfg : Any
        The instantiated QR config (e.g., DummyQRConfig).
    """
    def __init__(self, qr_cfg: Any):
        self.qr_cfg = qr_cfg


class DummyRetriever:
    """
    Minimal feature map retriever stand-in.

    Parameters
    ----------
    qr_cfg : Any
        The instantiated QR config (e.g., DummyQRConfig).
    observables : list
        Observables passed by `.from_config`.
    """
    def __init__(self, qr_cfg: Any, observables: List[Any]):
        self.qr_cfg = qr_cfg
        self.observables = list(observables)


def make_dummy_observables(locality: int = 1, num_qubits: int = 1) -> List[str]:
    """
    Build a dummy observables list.

    Parameters
    ----------
    locality : int, default=1
        Locality parameter (unused except for labeling).
    num_qubits : int, default=1
        Number of qubits (unused except for labeling).

    Returns
    -------
    list of str
        Dummy observable identifiers.
    """
    return [f"OBS(locality={locality},Q={num_qubits})"]


class DummyFeaturizer:
    """
    Minimal featurizer stand-in mirroring the signature used by your regressor.

    This class stores inputs so tests can assert correct wiring / forwarding.

    Parameters
    ----------
    cfg : Any
        QR config.
    runner : Any
        Runner.
    fmp_retriever : Any
        Feature maps retriever.
    pubs_family : str
        PUBS family name.
    angle_positioning_name : str
        Name of the angle positioning strategy.
    pubs_kwargs : dict
        Remaining PUBS kwargs.
    runner_kwargs : dict
        Runner runtime kwargs.
    fmp_kwargs : dict
        Retriever runtime kwargs.
    """
    def __init__(
        self,
        *,
        cfg: Any,
        runner: Any,
        fmp_retriever: Any,
        pubs_family: str,
        angle_positioning_name: str,
        pubs_kwargs: dict,
        runner_kwargs: dict,
        fmp_kwargs: dict,
    ):
        self.cfg = cfg
        self.runner = runner
        self.fmp_retriever = fmp_retriever
        self.pubs_family = pubs_family
        self.angle_positioning_name = angle_positioning_name
        self.pubs_kwargs = dict(pubs_kwargs)
        self.runner_kwargs = dict(runner_kwargs)
        self.fmp_kwargs = dict(fmp_kwargs)
