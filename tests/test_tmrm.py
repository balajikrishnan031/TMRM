"""
Unit tests for TMRM package.
"""
import numpy as np
import pytest
from tmrm import TMRM, StreamingTMRM, __version__

def test_version():
    assert __version__ == "4.2.0"

def test_tmrm_classification():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)

    clf = TMRM(task_type="classification", n_subspaces=3, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert preds.shape == (100,)
    assert probs.shape == (100, 2)
    assert np.mean(preds == y) > 0.85

def test_tmrm_regression():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.1 * np.random.randn(100)

    reg = TMRM(task_type="regression", n_subspaces=3, random_state=42)
    reg.fit(X, y)

    preds = reg.predict(X)
    assert preds.shape == (100,)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    assert r2 > 0.80

def test_tmrm_novelty():
    np.random.seed(42)
    X_train = np.random.randn(100, 4)
    y_train = (X_train[:, 0] > 0).astype(int)

    clf = TMRM(n_subspaces=2, random_state=42)
    clf.fit(X_train, y_train)

    # In distribution
    novelty_id = clf.get_epistemic_novelty(X_train)
    # Out of distribution
    X_ood = np.random.randn(20, 4) + 15.0
    novelty_ood = clf.get_epistemic_novelty(X_ood)

    assert np.mean(novelty_ood) > np.mean(novelty_id)

def test_streaming_tmrm():
    np.random.seed(42)
    stream = StreamingTMRM(n_subspaces=2, random_state=42)

    X1 = np.random.randn(50, 6)
    y1 = (X1[:, 0] > 0).astype(int)
    stream.partial_fit(X1, y1)

    X2 = np.random.randn(50, 6)
    y2 = (X2[:, 0] > 0).astype(int)
    stream.partial_fit(X2, y2)

    preds = stream.predict(X2)
    assert len(preds) == 50
