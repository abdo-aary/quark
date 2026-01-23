from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class BaseLabelFunctional(ABC):
    """Base interface for synthetic labels.

    Given a window set X with shape (N, w, d), return a set of scalar labels Y.

    Reason of separation from the time-series generator.
    - We can reuse the *same* beta-mixing process with different tasks.
    - We can ablate label difficulty independently from mixing properties.
    """

    @abstractmethod
    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class LabelNoise:
    """Optional additive noise model for labels."""

    std: float = 0.0

    def add(self, y, rng: np.random.Generator):
        """
        Add i.i.d. Gaussian noise with std self.std to t.
        Supports scalars or numpy arrays.
        """
        if self.std <= 0:
            return y

        # If Y is a scalar, draw a scalar noise
        if np.isscalar(y):
            return float(y) + float(rng.normal(loc=0.0, scale=self.std))

        y = np.asarray(y)
        noise = rng.normal(loc=0.0, scale=self.std, size=y.shape)
        return y + noise
