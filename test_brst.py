"""Smoke-test: fit on synthetic binary task, check AUC > 0.7."""
import sys, time
sys.path.insert(0, "python")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from brstboost import BRSTBoost

def run(n=5000, D=20, seed=42):
    X, y = make_classification(n, D, n_informative=10, random_state=seed)
    X = X.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)

    m = BRSTBoost(n_estimators=100, learning_rate=0.1, max_depth=4,
                  reg_lambda=1.0, subsample=0.8, n_bins=32, random_state=seed)
    t0 = time.time()
    m.fit(Xtr, ytr)
    elapsed = time.time() - t0

    proba = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    print(f"n={n} D={D}  AUC={auc:.4f}  fit={elapsed:.2f}s  avg_k={m.avg_k():.2f}")
    return auc

if __name__ == "__main__":
    auc = run()
    assert auc > 0.7, f"AUC too low: {auc}"
    print("PASS")
