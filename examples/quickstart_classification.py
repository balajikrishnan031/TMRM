"""
TMRM Quickstart: Synthetic Tabular Classification
"""
import numpy as np
from tmrm import TMRM

def main():
    print("=" * 60)
    print("TMRM Quickstart: Binary & Multi-Class Classification")
    print("=" * 60)

    # Generate synthetic tabular dataset
    np.random.seed(42)
    n_samples, n_features = 300, 12
    X = np.random.randn(n_samples, n_features)
    # Non-linear boundary
    y = ((X[:, 0]**2 + X[:, 1]**2 > 1.5) & (X[:, 2] > 0)).astype(int)

    split = int(0.75 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train TMRM model
    clf = TMRM(task_type="classification", n_subspaces=5, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate predictions
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    acc = np.mean(preds == y_test)

    print(f"Dataset Size: {n_samples} samples, {n_features} features")
    print(f"TMRM Accuracy: {acc * 100:.2f}%")
    print(f"Sample Predictions (first 5): {preds[:5]}")
    print(f"True Labels        (first 5): {y_test[:5]}")

    # Self-doubt / Novelty detection
    novelties = clf.get_epistemic_novelty(X_test)
    print(f"Mean Test Novelty Score: {np.mean(novelties):.4f}")

    print("Classification quickstart completed successfully!")

if __name__ == "__main__":
    main()
