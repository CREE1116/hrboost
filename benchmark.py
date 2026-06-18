"""
BRSTBoost vs LightGBM / XGBoost / CatBoost — AUC benchmark.

Usage:
    uv run python benchmark.py            # all synthetic + adult
    uv run python benchmark.py --quick    # fast: small n, no adult
    uv run python benchmark.py --no-adult # synth only, full n
"""
import sys, time, argparse
sys.path.insert(0, "python")

import numpy as np
import pandas as pd
from sklearn.datasets import (make_classification, make_moons,
                               make_circles, make_blobs)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle as sk_shuffle

from brstboost import BRSTBoost

# ── optional competitors ───────────────────────────────────────────────────────

def _try(name):
    try: return __import__(name)
    except ImportError: return None

lgb = _try("lightgbm")
xgb = _try("xgboost")
cb  = _try("catboost")

# ── synthetic dataset zoo ─────────────────────────────────────────────────────

def _ds_classification(n, D=30, seed=0):
    X, y = make_classification(n, D, n_informative=15, n_redundant=5,
                                random_state=seed)
    return X.astype(np.float32), y.astype(np.int32), "class-D30"

def _ds_highdim(n, D=100, seed=0):
    X, y = make_classification(n, D, n_informative=8, n_redundant=2,
                                n_repeated=0, random_state=seed)
    return X.astype(np.float32), y.astype(np.int32), "class-D100-sparse"

def _ds_moons(n, seed=0):
    X, y = make_moons(n, noise=0.25, random_state=seed)
    return X.astype(np.float32), y.astype(np.int32), "moons"

def _ds_circles(n, seed=0):
    X, y = make_circles(n, noise=0.1, factor=0.5, random_state=seed)
    return X.astype(np.float32), y.astype(np.int32), "circles"

def _ds_imbalanced(n, seed=0):
    X, y = make_classification(n, 30, n_informative=15, n_redundant=5,
                                weights=[0.9, 0.1], random_state=seed)
    return X.astype(np.float32), y.astype(np.int32), "imbalanced-90/10"

def _ds_blobs(n, seed=0):
    centers = np.array([[-3, 0], [3, 0], [0, 3], [0, -3]], dtype=np.float32)
    X, lbl = make_blobs(n, centers=centers, cluster_std=1.2, random_state=seed)
    y = (lbl % 2).astype(np.int32)
    X = X.astype(np.float32)
    return X, y, "blobs-4cluster"

def _ds_mixed_cat(n, seed=0):
    rng = np.random.default_rng(seed)
    X_num, y = make_classification(n, 10, n_informative=7, random_state=seed)
    X_num = X_num.astype(np.float32)
    cat1 = rng.integers(0, 5, n).astype(np.float32).reshape(-1, 1)
    cat2 = rng.integers(0, 10, n).astype(np.float32).reshape(-1, 1)
    X = np.hstack([X_num, cat1, cat2])
    return X, y.astype(np.int32), "mixed-cat", [10, 11]

def synthetic_suite(quick=False):
    n = 5_000 if quick else 20_000
    return [
        _ds_classification(n),
        _ds_highdim(n),
        _ds_moons(n),
        _ds_circles(n),
        _ds_imbalanced(n),
        _ds_blobs(n),
        _ds_mixed_cat(n),
    ]

# ── adult (real) ──────────────────────────────────────────────────────────────

def load_adult():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "adult/adult.data")
    cols = ["age","workclass","fnlwgt","education","edu_num","marital",
            "occupation","relationship","race","sex","cap_gain","cap_loss",
            "hours","country","income"]
    try:
        df = pd.read_csv(url, header=None, names=cols, na_values=" ?",
                         skipinitialspace=True)
    except Exception as e:
        print(f"  [adult load failed: {e}]"); return None
    df = df.dropna()
    y = (df["income"].str.strip() == ">50K").astype(int).values
    df = df.drop("income", axis=1)
    for c in df.select_dtypes("object").columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    X = df.values.astype(np.float32)
    cat_idx = [i for i, c in enumerate(df.columns)
               if df[c].nunique() < 50 and c not in
               ("age","fnlwgt","edu_num","cap_gain","cap_loss","hours")]
    return X, y.astype(np.int32), "adult", cat_idx

# ── model factory ─────────────────────────────────────────────────────────────

def make_models(cat_features=None, quick=False):
    cat = cat_features or []
    est = 200 if quick else 400
    lr  = 0.1  if quick else 0.05
    models = {}

    nbins = 64 if quick else 255
    models["BRSTBoost"] = BRSTBoost(
        n_estimators=est, learning_rate=lr, max_depth=5, max_leaves=64,
        reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        n_bins=nbins, max_k=2, cat_features=cat, random_state=0)

    if lgb:
        lgb_kw = dict(n_estimators=est, learning_rate=lr, max_depth=5,
                      num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, random_state=0, verbose=-1)
        if cat:
            lgb_kw["categorical_feature"] = cat
        models["LightGBM"] = lgb.LGBMClassifier(**lgb_kw)

    if xgb:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=est, learning_rate=lr, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", random_state=0, verbosity=0)

    if cb:
        cb_kw = dict(iterations=est, learning_rate=lr, depth=5,
                     l2_leaf_reg=1.0, subsample=0.8,
                     random_state=0, verbose=False)
        if cat:
            cb_kw["cat_features"] = cat
        _inner_cb = cb.CatBoostClassifier(**cb_kw)
        # CatBoost rejects float arrays with cat_features; use DataFrame + str cat cols
        class _CBWrap:
            def __init__(self, m, cat_idx):
                self._m = m; self._cat = cat_idx
            def _prep(self, X):
                df = pd.DataFrame(X)
                for c in self._cat:
                    df[c] = df[c].astype(int).astype(str)
                return df
            def fit(self, X, y):
                self._m.fit(self._prep(X), y); return self
            def predict_proba(self, X):
                return self._m.predict_proba(self._prep(X))
        models["CatBoost"] = _CBWrap(_inner_cb, cat) if cat else _inner_cb

    return models

# ── evaluation ────────────────────────────────────────────────────────────────

HDR = f"  {'Model':<12} {'AUC':>8} {'±':>6} {'fit/fold':>9}"
SEP = "  " + "─"*12 + " " + "─"*8 + " " + "─"*6 + " " + "─"*9

def evaluate(X, y, name, cat_features=None, n_splits=5, quick=False):
    print(f"\n{'─'*58}")
    print(f"  {name}  n={len(y):,}  D={X.shape[1]}  pos={y.mean():.3f}")
    print(f"{'─'*58}")
    print(HDR); print(SEP)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for mname, model in make_models(cat_features, quick=quick).items():
        aucs, times = [], []
        for tr, te in cv.split(X, y):
            t0 = time.time()
            model.fit(X[tr], y[tr])
            elapsed = time.time() - t0
            times.append(elapsed)
            p = model.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))

        mu, sd = np.mean(aucs), np.std(aucs)
        results[mname] = mu
        print(f"  {mname:<12} {mu:>8.4f} {sd:>6.4f} {np.mean(times):>8.1f}s")

    return results

# ── decision boundary ─────────────────────────────────────────────────────────

BOUNDARY_DATASETS = [
    ("moons",   lambda: make_moons(1500, noise=0.2, random_state=0)),
    ("circles", lambda: make_circles(1500, noise=0.08, factor=0.5, random_state=0)),
    ("blobs",   lambda: (
        lambda X, lbl: (X.astype(np.float32), (lbl % 2).astype(np.int32)))(
        *make_blobs(1500, centers=np.array([[-2,0],[2,0],[0,2],[0,-2]]),
                    cluster_std=0.9, random_state=0))),
    ("class2d", lambda: make_classification(
        1500, 2, n_informative=2, n_redundant=0,
        n_clusters_per_class=2, random_state=1)),
]

def plot_boundary():
    import matplotlib.pyplot as plt

    n_ds = len(BOUNDARY_DATASETS)
    n_models_est = 1 + bool(lgb) + bool(xgb) + bool(cb)

    fig, axes = plt.subplots(n_ds, n_models_est,
                             figsize=(4.5 * n_models_est, 4 * n_ds),
                             squeeze=False)

    h = 0.05

    for row, (ds_name, ds_fn) in enumerate(BOUNDARY_DATASETS):
        X_raw, y_raw = ds_fn()
        X2 = np.ascontiguousarray(X_raw, dtype=np.float32)
        y2 = np.ascontiguousarray(y_raw, dtype=np.int32)

        x0, x1 = X2[:, 0], X2[:, 1]
        xx, yy = np.meshgrid(
            np.arange(x0.min() - .5, x0.max() + .5, h),
            np.arange(x1.min() - .5, x1.max() + .5, h))
        grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

        model_list = [
            ("BRSTBoost", BRSTBoost(n_estimators=200, learning_rate=0.1,
                                    max_depth=5, max_leaves=64,
                                    reg_lambda=1.0, colsample_bytree=0.8,
                                    n_bins=32, random_state=0)),
        ]
        if lgb:
            model_list.append(("LightGBM", lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                num_leaves=31, verbose=-1, random_state=0)))
        if xgb:
            model_list.append(("XGBoost", xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                eval_metric="logloss", verbosity=0, random_state=0)))
        if cb:
            model_list.append(("CatBoost", cb.CatBoostClassifier(
                iterations=200, learning_rate=0.1, depth=5,
                random_state=0, verbose=False)))

        for col, (mname, model) in enumerate(model_list):
            ax = axes[row][col]
            model.fit(X2, y2)
            Z = model.predict_proba(grid)[:, 1].reshape(xx.shape)
            cf = ax.contourf(xx, yy, Z, levels=50, cmap="RdBu_r",
                             alpha=0.75, vmin=0, vmax=1)
            ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.5)
            ax.scatter(x0[y2==0], x1[y2==0], s=5, c="royalblue", alpha=0.4)
            ax.scatter(x0[y2==1], x1[y2==1], s=5, c="tomato",    alpha=0.4)
            auc = roc_auc_score(y2, model.predict_proba(X2)[:, 1])
            ax.set_title(f"{mname}  AUC={auc:.3f}", fontsize=9)
            if row == 0:
                ax.set_title(f"{mname}\n(AUC={auc:.3f})", fontsize=9)
            ax.set_xlabel(ds_name if col == 0 else "")

    plt.tight_layout()
    out = "decision_boundary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick",    action="store_true",
                    help="Small n (5k), fewer estimators, no adult")
    ap.add_argument("--no-adult", action="store_true",
                    help="Skip adult dataset but use full n")
    ap.add_argument("--no-boundary", action="store_true",
                    help="Skip decision boundary plots")
    args = ap.parse_args()

    print("BRSTBoost benchmark")
    print(f"  LightGBM={'yes' if lgb else 'no'}  "
          f"XGBoost={'yes' if xgb else 'no'}  "
          f"CatBoost={'yes' if cb else 'no'}")

    for entry in synthetic_suite(quick=args.quick):
        if len(entry) == 4:
            X, y, name, cat = entry
            evaluate(X, y, name, cat_features=cat, quick=args.quick)
        else:
            X, y, name = entry
            evaluate(X, y, name, quick=args.quick)

    if not args.no_boundary:
        print("\nPlotting decision boundaries…")
        plot_boundary()

    if not args.quick and not args.no_adult:
        result = load_adult()
        if result:
            X_a, y_a, name_a, cats_a = result
            evaluate(X_a, y_a, name_a, cat_features=cats_a)

if __name__ == "__main__":
    main()
