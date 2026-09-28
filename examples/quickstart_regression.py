"""
TMRM Quickstart: Continuous Non-Linear Regression
"""
import numpy as np
from tmrm import TMRM

def main():
    print("=" * 60)
    print("TMRM Quickstart: Continuous Regression")
    print("=" * 60)

    np.random.seed(42)
    n_samples, n_features = 400, 10
    X = np.random.uniform(-2, 2, size=(n_samples, n_features))
    # Highly non-linear manifold response
    y = np.sin(X[:, 0] * 2.0) + 0.5 * np.cos(X[:, 1]) + (X[:, 2] * X[:, 3]) + np.random.normal(0, 0.05, n_samples)

    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    reg = TMRM(task_type="regression", n_subspaces=6, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    print(f"TMRM Continuous Regression R^2 Score: {r2:.4f}")
    print(f"TMRM Continuous Regression RMSE:      {rmse:.4f}")
    print("Regression quickstart completed successfully!")

if __name__ == "__main__":
    main()
