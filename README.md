# TMRM: Topological Manifold Resonant Machine (v4.0)

[![PyPI version](https://img.shields.io/badge/version-4.0.0-blue.svg)](https://github.com/balajikrishnan031/TMRM)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: clean](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A unified, non-parametric, physics-inspired machine learning architecture that operates directly on **Riemannian Topological Energy Manifolds**. TMRM bridges continuous differential geometry, discrete Ollivier-Ricci curvature auto-tuning, and multi-octave wavelet resonance into a single cohesive algorithm for both **classification** and **continuous regression**.

Designed from the ground up as a pure foundation architecture without external dataset bloat or legacy neural net training bottlenecks.

---

## Authors

- **Balaji P**
- **Navaneetham V**
- **Dhavan RG**

---

## Key Features

1. **Native Unified Paradigm**: Handles both multi-class/binary classification and continuous regression with identical differential geometric foundations.
2. **Multi-Octave Wavelet Resonance**: Multi-frequency harmonic kernels capture fine local variations while maintaining global topological coherence.
3. **Ollivier-Ricci Curvature Auto-Tuning**: Automatically detects bottlenecks and clusters on Riemannian manifolds to calibrate metric scaling per feature subspace.
4. **Multi-Faceted Minkowski Metric Spectrum**: Fuses $L_2$ Riemannian geodesic fields, $L_\infty$ Chebyshev hyper-box bounds, and $L_1$ Manhattan sparsity.
5. **Closed-Form Dual Ridge Potential Superposition**: Instant closed-form exact mathematical solutions—zero stochastic gradient drift and zero loss of convergence.
6. **Johnson-Lindenstrauss Latent Projection**: Seamlessly scales to 50,000+ sparse high-dimensional features.
7. **Built-in Epistemic Novelty & OOD Detection**: Inherent self-doubt quantifying distributional novelty without external anomaly detectors.
8. **Dual Geodesic Recourse Generator**: Generates actionable, minimal-distance feature modifications to transition boundary states.
9. **Streaming & Big Data Engine**: `StreamingTMRM` with online running metric moments for continuous online stream ingestion.

---

## Installation

Install directly via `pip` from GitHub:

```bash
pip install git+https://github.com/balajikrishnan031/TMRM.git
```

Or clone and install locally in editable mode:

```bash
git clone https://github.com/balajikrishnan031/TMRM.git
cd TMRM
pip install -e .
```

---

## Quickstart

### 1. Classification

```python
import numpy as np
from tmrm import TMRM

# Generate synthetic tabular classification data
X_train = np.random.randn(200, 10)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

X_test = np.random.randn(50, 10)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

# Initialize TMRM Classifier
model = TMRM(task_type="classification", n_subspaces=6, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Predict class labels and probabilities
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

# Evaluate
accuracy = np.mean(preds == y_test)
print(f"TMRM Classification Accuracy: {accuracy * 100:.2f}%")

# Compute epistemic self-doubt (OOD Novelty)
novelty_scores = model.get_epistemic_novelty(X_test)
print(f"Mean In-Distribution Novelty: {np.mean(novelty_scores):.4f}")
```

### 2. Continuous Regression

```python
import numpy as np
from tmrm import TMRM

# Generate synthetic non-linear continuous regression target
X_train = np.random.uniform(-3, 3, size=(300, 8))
y_train = np.sin(X_train[:, 0]) * np.exp(-0.2 * np.abs(X_train[:, 1])) + 0.5 * X_train[:, 2]

X_test = np.random.uniform(-3, 3, size=(100, 8))
y_test = np.sin(X_test[:, 0]) * np.exp(-0.2 * np.abs(X_test[:, 1])) + 0.5 * X_test[:, 2]

# Initialize TMRM Regressor
model = TMRM(task_type="regression", n_subspaces=6, random_state=42)
model.fit(X_train, y_train)

# Predict continuous target values
y_pred = model.predict(X_test)

# Compute R^2 score
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2_score = 1.0 - (ss_res / ss_tot)
print(f"TMRM Continuous Regression R^2: {r2_score:.4f}")
```

### 3. Streaming Big Data (`StreamingTMRM`)

```python
import numpy as np
from tmrm import StreamingTMRM

stream_model = StreamingTMRM(n_subspaces=4, random_state=42)

# Simulate continuous online batches
for batch_idx in range(5):
    X_batch = np.random.randn(100, 10)
    y_batch = (X_batch[:, 0] > 0).astype(int)
    stream_model.partial_fit(X_batch, y_batch)

X_eval = np.random.randn(50, 10)
stream_preds = stream_model.predict(X_eval)
print(f"Streaming TMRM Predictions shape: {stream_preds.shape}")
```

---

## Mathematical Architecture

TMRM constructs an empirical discrete metric measure space $(M, d, \mu)$ from training observations and computes predictive fields via resonant harmonic potential superposition:

$$\Phi(x) = \sum_{k=1}^{K} w_k \cdot \Psi_k(d_g(x, x_k))$$

where:
- $d_g(x, x_k)$ is the hybrid geodesic Minkowski distance tensor.
- $\Psi_k$ is the multi-octave wavelet resonance kernel operator.
- $w_k$ represents the dual ridge potential field closed-form solution.

---

## Repository Structure

```
TMRM/
├── tmrm/
│   ├── __init__.py          # Package exports (TMRM, StreamingTMRM)
│   └── core.py              # Full pure algorithm implementation (Zero dataset dependencies)
├── examples/
│   ├── quickstart_classification.py
│   └── quickstart_regression.py
├── tests/
│   └── test_tmrm.py         # Complete unit test suite
├── pyproject.toml           # Modern PEP 517/621 build configuration
├── setup.py                 # Setuptools packaging script
├── LICENSE                  # MIT License
├── README.md                # Documentation & Usage Guide
└── .gitignore               # Strict exclusion of data/cache files
```

---

## Citation

If you use TMRM in your research or application, please cite:

```bibtex
@software{tmrm2026,
  author = {Balaji, P. and Navaneetham, V. and Dhavan, R. G.},
  title = {TMRM: Topological Manifold Resonant Machine},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/balajikrishnan031/TMRM}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
