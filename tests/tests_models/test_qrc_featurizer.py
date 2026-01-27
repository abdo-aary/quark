import numpy as np
import pytest

import src.models.qrc_featurizer as qf


class DummyRunner:
    def __init__(self):
        self.called = False
        self.last_pubs = None
        self.last_kwargs = None

    def run_pubs(self, pubs, **kwargs):
        self.called = True
        self.last_pubs = pubs
        self.last_kwargs = dict(kwargs)
        return {"results": "OK"}


class DummyRetriever:
    def __init__(self, D=7):
        self.D = D
        self.called = False
        self.last_results = None
        self.last_kwargs = None

    def get_feature_maps(self, results, **kwargs):
        self.called = True
        self.last_results = results
        self.last_kwargs = dict(kwargs)
        # produce deterministic features (N, D) using kwargs for sanity
        N = kwargs["_N"]
        Phi = np.zeros((N, self.D), dtype=float)
        Phi[:, 0] = np.arange(N)
        return Phi


class DummyCfg:
    pass


def test_featurizer_transform_happy_path(monkeypatch):
    # Arrange
    N, w, d = 5, 4, 3
    X = np.random.default_rng(0).normal(size=(N, w, d))

    runner = DummyRunner()
    retriever = DummyRetriever(D=11)

    # Patch CircuitFactory call so we don't touch Qiskit/circuits here
    def fake_create_pubs(*, qrc_cfg, angle_positioning, X, **kwargs):
        assert qrc_cfg is cfg_obj
        assert callable(angle_positioning)
        assert X.shape == (N, w, d)
        assert kwargs == {"a": 1}
        return {"pubs": "DATASET"}

    cfg_obj = DummyCfg()
    monkeypatch.setattr(
        qf.CircuitFactory,
        "create_pubs_dataset_reservoirs_IsingRingSWAP",
        staticmethod(fake_create_pubs),
    )

    featurizer = qf.QRCFeaturizer(
        qrc_cfg=cfg_obj,
        runner=runner,
        fmp_retriever=retriever,
        pubs_family="ising_ring_swap",
        angle_positioning_name="linear",
        pubs_kwargs={"a": 1},
        runner_kwargs={"rk": 2},
        fmp_kwargs={"_N": N, "fk": 3},
    )

    # Act
    Phi = featurizer.transform(X)

    # Assert
    assert Phi.shape == (N, 11)
    assert runner.called is True
    assert runner.last_pubs == {"pubs": "DATASET"}
    assert runner.last_kwargs == {"rk": 2}

    assert retriever.called is True
    assert retriever.last_results == {"results": "OK"}
    assert retriever.last_kwargs == {"_N": N, "fk": 3}


def test_featurizer_rejects_bad_X_shape():
    runner = DummyRunner()
    retriever = DummyRetriever()

    featurizer = qf.QRCFeaturizer(
        qrc_cfg=DummyCfg(),
        runner=runner,
        fmp_retriever=retriever,
        pubs_family="ising_ring_swap",
        angle_positioning_name="linear",
        pubs_kwargs={},
        runner_kwargs={},
        fmp_kwargs={"_N": 1},
    )

    with pytest.raises(ValueError):
        featurizer.transform(np.zeros((10, 3)))  # not (N,w,d)


def test_featurizer_rejects_unknown_pubs_family():
    runner = DummyRunner()
    retriever = DummyRetriever()
    featurizer = qf.QRCFeaturizer(
        qrc_cfg=DummyCfg(),
        runner=runner,
        fmp_retriever=retriever,
        pubs_family="unknown",
        angle_positioning_name="linear",
        pubs_kwargs={},
        runner_kwargs={},
        fmp_kwargs={"_N": 1},
    )
    with pytest.raises(ValueError):
        featurizer.transform(np.zeros((2, 3, 4)))


def test_featurizer_rejects_unknown_angle_positioning():
    runner = DummyRunner()
    retriever = DummyRetriever()
    featurizer = qf.QRCFeaturizer(
        qrc_cfg=DummyCfg(),
        runner=runner,
        fmp_retriever=retriever,
        pubs_family="ising_ring_swap",
        angle_positioning_name="not_a_key",
        pubs_kwargs={},
        runner_kwargs={},
        fmp_kwargs={"_N": 2},
    )
    with pytest.raises(ValueError):
        featurizer.transform(np.zeros((2, 3, 4)))


def test_featurizer_rejects_wrong_feature_shape(monkeypatch):
    N, w, d = 4, 2, 2
    X = np.ones((N, w, d))

    runner = DummyRunner()

    class BadRetriever:
        @staticmethod
        def get_feature_maps(results, **kwargs):
            return np.zeros((N, 3, 2))  # wrong ndim

    retriever = BadRetriever()

    monkeypatch.setattr(
        qf.CircuitFactory,
        "create_pubs_dataset_reservoirs_IsingRingSWAP",
        staticmethod(lambda **kwargs: {"pubs": "OK"}),
    )

    featurizer = qf.QRCFeaturizer(
        qrc_cfg=DummyCfg(),
        runner=runner,
        fmp_retriever=retriever,
        pubs_family="ising_ring_swap",
        angle_positioning_name="linear",
        pubs_kwargs={},
        runner_kwargs={},
        fmp_kwargs={},
    )

    with pytest.raises(ValueError):
        featurizer.transform(X)
