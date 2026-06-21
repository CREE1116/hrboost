"""
Refactored Decision Boundary Visualization Script for BRSTBoost.
Features highly diverse 2D datasets (including advanced 3-class topologies)
with ultra-soft pastel background contours for crisp boundary interpretation.
Resolves all scikit-learn clone compatibility issues for both wrappers.
"""
import sys
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "python")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from selfcdsb import SelfCDSBClassifier

# Try importing competitors
def _try(name):
    try: return __import__(name)
    except ImportError: return None

lgb = _try("lightgbm")
xgb = _try("xgboost")
cb  = _try("catboost")

# ── 2-Class Synthetic Datasets ──────────────────────────────────────────

def make_complex_checkerboard(n_samples=1500, noise=0.08, random_state=42):
    """비선형 트리 정렬을 교란하는 3x3 격자 형태의 복잡한 체커보드 데이터셋"""
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-2.5, 2.5, (n_samples, 2))
    x_grid = np.floor(X[:, 0] + 2.5).astype(int)
    y_grid = np.floor(X[:, 1] + 2.5).astype(int)
    y = ((x_grid + y_grid) % 2 == 0).astype(np.int32)
    flip = rng.random(n_samples) < noise
    y[flip] = 1 - y[flip]
    return X.astype(np.float32), y

# ── 3-Class (Multiclass) Synthetic Datasets ────────────────────────────

def make_anisotropic_blobs(n_samples=1500, random_state=42):
    """축-정렬 분할을 사용하는 기성 트리를 스트레스 테스트하기 위한 사선 왜곡 블롭"""
    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.8, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X = np.dot(X, transformation)
    return X.astype(np.float32), y.astype(np.int32)

def make_dense_spiral(n_samples=1500, noise=0.10, random_state=42):
    """중심축에서 강하게 휘어지는 형태의 3클래스 고밀도 스파이럴"""
    np.random.seed(random_state)
    n = n_samples // 3
    X = []
    y = []
    for i in range(3):
        r = np.linspace(0.05, 2.2, n)
        t = np.linspace(i * 2.5 * np.pi / 3, (i + 2.2) * 2.5 * np.pi / 3, n) + np.random.randn(n) * noise
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.ones(n, dtype=np.int32) * i)
    return np.vstack(X).astype(np.float32), np.hstack(y).astype(np.int32)

def make_concentric_rings_3class(n_samples=1500, noise=0.06, random_state=42):
    """과적합 성향을 판별하기 위한 3중 동심원 고리 구조"""
    rng = np.random.default_rng(random_state)
    n = n_samples // 3
    X = []
    y = []
    radii = [0.35, 0.95, 1.65]
    for i, r in enumerate(radii):
        theta = rng.uniform(0, 2 * np.pi, n)
        dr = rng.normal(0, noise, n)
        X.append(np.c_[(r + dr) * np.sin(theta), (r + dr) * np.cos(theta)])
        y.append(np.ones(n, dtype=np.int32) * i)
    return np.vstack(X).astype(np.float32), np.hstack(y).astype(np.int32)

# ── 데이터셋 마스터 리스트 ──────────────────────────────────────────────────
BOUNDARY_DATASETS = [
    ("Moons (Binary)", lambda: make_moons(1500, noise=0.15, random_state=42), False),
    ("Circles (Binary)", lambda: make_circles(1500, noise=0.08, factor=0.5, random_state=42), False),
    ("Grid Checkerboard (Binary)", lambda: make_complex_checkerboard(1500, noise=0.08, random_state=42), False),
    ("Anisotropic (3-Class)", lambda: make_anisotropic_blobs(1500, random_state=42), True),
    ("Dense Spiral (3-Class)", lambda: make_dense_spiral(1500, noise=0.10, random_state=42), True),
    ("Concentric Rings (3-Class)", lambda: make_concentric_rings_3class(1500, noise=0.06, random_state=42), True),
]

# ── [개선] scikit-learn Clone 호환성을 완전히 준수하는 글로벌 래퍼 클래스들 ──

# _BRSTBoostSklearnWrapper removed since SelfCDSBClassifier is natively scikit-learn compliant


class _LGBMClassifierSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=5,
                 num_leaves=31, subsample=0.8, colsample_bytree=1.0,
                 reg_lambda=1.0, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, X, y):
        import lightgbm as lgb
        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, num_leaves=self.num_leaves,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda, random_state=self.random_state, verbose=-1
        )
        df = pd.DataFrame(X)
        df.columns = [str(c) for c in df.columns]
        self.model_.fit(df, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        df = pd.DataFrame(X)
        df.columns = [str(c) for c in df.columns]
        return self.model_.predict_proba(df)


def main():
    print("Generating ultra-soft aesthetic decision boundary visualization...")
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.titlesize": 16,
        "figure.dpi": 180
    })

    n_ds = len(BOUNDARY_DATASETS)
    
    def create_model_list(is_multiclass):
        models = [
            ("SelfCDSB", SelfCDSBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5, max_leaves=64,
                reg_lambda=1.0, subsample=0.8, colsample_bytree=1.0,
                n_bins=32, random_state=42, cat_features=[]
            ))
        ]
        
        if lgb:
            models.append(("LightGBM", _LGBMClassifierSklearnWrapper(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                num_leaves=31, subsample=0.8, colsample_bytree=1.0,
                reg_lambda=1.0, random_state=42
            )))
            
        if xgb:
            models.append(("XGBoost", xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5,
                subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                eval_metric="logloss", random_state=42, verbosity=0
            )))
            
        if cb:
            models.append(("CatBoost", cb.CatBoostClassifier(
                iterations=200, learning_rate=0.1, depth=5,
                subsample=0.8, l2_leaf_reg=1.0, random_state=42, verbose=False
            )))

        wrapped_models = []
        for name, model in models:
            if is_multiclass:
                wrapped_models.append((name, OneVsRestClassifier(model)))
            else:
                wrapped_models.append((name, model))
        return wrapped_models

    sample_models = create_model_list(False)
    n_models = len(sample_models)
    
    fig, axes = plt.subplots(n_ds, n_models, figsize=(3.6 * n_models, 3.5 * n_ds), squeeze=False)
    h = 0.03

    cmap_2class = LinearSegmentedColormap.from_list(
        "pastel_blue_pink", ["#EBF8FF", "#FFFFFF", "#FFF5F5"]
    )
    cmap_3class = ListedColormap(["#F0F4F8", "#FFF5F5", "#F0FDF4"])
    
    scatter_colors_2class = ["#3182CE", "#E53E3E"]
    scatter_colors_3class = ["#3182CE", "#E53E3E", "#38A169"]

    for row, (ds_name, ds_fn, is_mc) in enumerate(BOUNDARY_DATASETS):
        X_raw, y_raw = ds_fn()
        X = np.ascontiguousarray(X_raw, dtype=np.float32)
        y = np.ascontiguousarray(y_raw, dtype=np.int32)

        x0_min, x0_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
        x1_min, x1_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
        
        xx, yy = np.meshgrid(
            np.arange(x0_min, x0_max, h),
            np.arange(x1_min, x1_max, h)
        )
        grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

        models_to_test = create_model_list(is_mc)

        for col, (mname, model) in enumerate(models_to_test):
            ax = axes[row][col]
            
            t0 = time.time()
            model.fit(X, y)
            fit_time = time.time() - t0
            
            probs = model.predict_proba(grid)
            
            if is_mc:
                Z = np.argmax(probs, axis=1).reshape(xx.shape)
                ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap_3class, alpha=0.4)
                ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors="#718096", linewidths=0.9, linestyles="solid")
            else:
                Z = probs[:, 1].reshape(xx.shape)
                ax.contourf(xx, yy, Z, levels=50, cmap=cmap_2class, alpha=0.45, vmin=0, vmax=1)
                ax.contour(xx, yy, Z, levels=[0.5], colors="#718096", linewidths=1.1, linestyles="solid")

            colors_to_use = scatter_colors_3class if is_mc else scatter_colors_2class
            classes = [0, 1, 2] if is_mc else [0, 1]
            
            for cl in classes:
                ax.scatter(X[y == cl, 0], X[y == cl, 1], s=8, c=colors_to_use[cl], 
                           alpha=0.50, edgecolors="none", label=f"Class {cl}" if (row==0 and col==0) else None)

            ax.set_xlim(x0_min, x0_max)
            ax.set_ylim(x1_min, x1_max)
            ax.set_xticks([])
            ax.set_yticks([])
            
            y_pred_probs = model.predict_proba(X)
            if is_mc:
                auc = roc_auc_score(y, y_pred_probs, multi_class="ovr")
            else:
                auc = roc_auc_score(y, y_pred_probs[:, 1])

            title_text = f"{mname}\nAUC: {auc:.4f} ({fit_time*1000:.1f}ms)"
            ax.set_title(title_text, pad=5, weight="bold" if mname == "SelfCDSB" else "normal",
                         color="#1A202C" if mname == "SelfCDSB" else "#4A5568")
            
            if col == 0:
                ax.set_ylabel(ds_name, fontsize=9.5, labelpad=6, weight="bold", color="#2D3748")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=6.5)

    plt.suptitle("Advanced Decision Boundary & Topology Comparison Grid", y=0.99, weight="bold", color="#1A202C")
    plt.tight_layout()
    
    out_path = "decision_boundary_enhanced.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Refactored decision boundaries successfully exported to {out_path}!")
    plt.close()

if __name__ == "__main__":
    main()