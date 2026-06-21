"""
HRBoost vs LightGBM / XGBoost / CatBoost — Comprehensive Tabular Benchmarking

Usage:
    uv run python benchmark.py --quick
    uv run python benchmark.py --params best_params_optuna.json
"""
import sys
import time
import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

# Configure library paths
sys.path.insert(0, "python")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

from hrboost import HRBoostClassifier

# ── Optional competitors ───────────────────────────────────────────────────────
def _try(name):
    try: return __import__(name)
    except ImportError: return None

lgb = _try("lightgbm")
xgb = _try("xgboost")
cb  = _try("catboost")

# ── Dataset Loaders ────────────────────────────────────────────────────────────

def load_breast_cancer_ds():
    data = load_breast_cancer()
    return data.data.astype(np.float32), data.target.astype(np.int32), "breast_cancer", [], 2

def load_digits_ds():
    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int32)
    return X, y, "digits", list(range(X.shape[1])), 10

def load_wine_ds():
    data = load_wine()
    return data.data.astype(np.float32), data.target.astype(np.int32), "wine", [], 3

def load_openml_by_id(data_id, name, is_binary=True, cat_index=None):
    try:
        data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
        df_X = data.data.copy()
        
        # Determine labels
        if is_binary:
            unique_targets = data.target.unique()
            pos_label = unique_targets[0]
            y = (data.target == pos_label).astype(np.int32).to_numpy()
        else:
            le_y = LabelEncoder()
            y = le_y.fit_transform(data.target.astype(str)).astype(np.int32)
            
        cat_idx = cat_index if cat_index is not None else []
        if cat_index is None:
            for i, col in enumerate(df_X.columns):
                if df_X[col].dtype.name == 'category' or df_X[col].dtype == object:
                    cat_idx.append(i)
                    le = LabelEncoder()
                    df_X[col] = le.fit_transform(df_X[col].astype(str))
                    
        X = df_X.to_numpy().astype(np.float32)
        return X, y, name, cat_idx, len(np.unique(y))
    except Exception as e:
        print(f"  [Failed to fetch OpenML {name} (ID: {data_id}): {e}]")
        return None

# Wrapper for CatBoost
class CatBoostWrapper:
    def __init__(self, is_multiclass, cat_idx, params):
        self.cat_idx = cat_idx
        loss_fn = "MultiClass" if is_multiclass else "Logloss"
        self.model = cb.CatBoostClassifier(
            iterations=params.get("iterations", 100),
            learning_rate=params.get("learning_rate", 0.1),
            depth=params.get("depth", 5),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            subsample=params.get("subsample", 0.8),
            bootstrap_type=params.get("bootstrap_type", "Bernoulli"),
            loss_function=loss_fn,
            random_seed=0,
            verbose=False,
            cat_features=cat_idx if cat_idx else None
        )
        
    def fit(self, X, y):
        df = pd.DataFrame(X)
        if self.cat_idx:
            for c in self.cat_idx:
                df[c] = df[c].astype(int).astype(str)
        self.model.fit(df, y)
        return self
        
    def predict_proba(self, X):
        df = pd.DataFrame(X)
        if self.cat_idx:
            for c in self.cat_idx:
                df[c] = df[c].astype(int).astype(str)
        return self.model.predict_proba(df)

# ── Model Factory ─────────────────────────────────────────────────────────────

def make_models(ds_name, num_classes, hpo_dict=None, cat_features=None, quick=False):
    cat = cat_features or []
    est = 100 if quick else 200
    lr  = 0.1  if quick else 0.05
    is_multi = num_classes > 2
    obj_type = "multiclass" if is_multi else "binary"
    models = {}
    
    # Extract optimized settings if available
    ds_hpo = hpo_dict.get(ds_name, {}) if hpo_dict else {}
    hrboost_params = ds_hpo.get("HRBoost", {})
    env_config = ds_hpo.get("ENV", {})

    # 1. HRBoost
    sc_params = {
        "n_estimators": est, "learning_rate": lr, "max_depth": 5, "max_leaves": 31,
        "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 1.0, "n_bins": 32,
        "cat_features": cat, "random_state": 0, "objective": obj_type,
        "num_classes": num_classes, "verbose": False
    }
    if hrboost_params:
        sc_params.update(hrboost_params)
        
    # Wrapper to inject COHESION_REG dynamically during fit
    class HRBoostWrapper:
        def __init__(self, params, env_vars):
            self.model = HRBoostClassifier(**params)
            self.env_vars = env_vars
        def fit(self, X, y):
            old_envs = {}
            for k, v in self.env_vars.items():
                old_envs[k] = os.environ.get(k, None)
                os.environ[k] = str(v)
            try:
                self.model.fit(X, y)
            finally:
                for k, v in old_envs.items():
                    if v is not None:
                        os.environ[k] = v
                    elif k in os.environ:
                        del os.environ[k]
            return self
        def predict_proba(self, X):
            return self.model.predict_proba(X)
            
    models["HRBoost"] = HRBoostWrapper(sc_params, env_config)

    # 2. LightGBM
    if lgb:
        lgb_kw = dict(n_estimators=est, learning_rate=lr, max_depth=5,
                      num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, random_state=0, verbose=-1)
        lgb_hpo = ds_hpo.get("LightGBM", {})
        if lgb_hpo:
            lgb_kw.update(lgb_hpo)
        models["LightGBM"] = lgb.LGBMClassifier(**lgb_kw)

    # 3. XGBoost
    if xgb:
        xgb_kw = dict(n_estimators=est, learning_rate=lr, max_depth=5,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=0, verbosity=0)
        if is_multi:
            xgb_kw["objective"] = "multi:softprob"
            xgb_kw["num_class"] = num_classes
        else:
            xgb_kw["eval_metric"] = "logloss"
        xgb_hpo = ds_hpo.get("XGBoost", {})
        if xgb_hpo:
            xgb_kw.update(xgb_hpo)
        models["XGBoost"] = xgb.XGBClassifier(**xgb_kw)

    # 4. CatBoost
    if cb:
        cb_kw = dict(iterations=est, learning_rate=lr, depth=5,
                     l2_leaf_reg=3.0, subsample=0.8, bootstrap_type="Bernoulli")
        cb_hpo = ds_hpo.get("CatBoost", {})
        if cb_hpo:
            cb_kw.update(cb_hpo)
        models["CatBoost"] = CatBoostWrapper(is_multi, cat, cb_kw)

    return models

# ── Evaluation Suite ──────────────────────────────────────────────────────────

def evaluate(X, y, name, num_classes, hpo_dict=None, cat_features=None, n_splits=5, quick=False):
    print(f"\n{'─'*65}")
    print(f"  Dataset: {name} (N={len(y):,}, D={X.shape[1]}, Classes={num_classes})")
    if hpo_dict and name in hpo_dict:
        print(f"  [Status] Using Optimized HPO Settings & Envs from JSON")
    else:
        print(f"  [Status] Using Default Parameters")
    print(f"{'─'*65}")
    
    hdr = f"  {'Model':<12} | {'AUC (macro)':>12} | {'Accuracy':>10} | {'Fit Time':>10}"
    sep = "  " + "─"*12 + "─┬─" + "─"*12 + "─┬─" + "─"*10 + "─┬─" + "─"*10
    print(hdr); print(sep)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for mname, model in make_models(name, num_classes, hpo_dict, cat_features, quick=quick).items():
        aucs, accs, times = [], [], []
        for tr, te in cv.split(X, y):
            t0 = time.time()
            model.fit(X[tr], y[tr])
            times.append(time.time() - t0)
            
            probs = model.predict_proba(X[te])
            preds = np.argmax(probs, axis=1) if num_classes > 2 else (probs[:, 1] >= 0.5).astype(int)
            
            try:
                if num_classes > 2:
                    auc = roc_auc_score(y[te], probs, multi_class="ovr", average="macro")
                else:
                    auc = roc_auc_score(y[te], probs[:, 1])
            except Exception:
                auc = 0.5

                
            aucs.append(auc)
            accs.append(accuracy_score(y[te], preds))

        mu_auc, mu_acc = np.mean(aucs), np.mean(accs)
        print(f"  {mname:<12} | {mu_auc:>12.4f} | {mu_acc:>10.4f} | {np.mean(times):>9.2f}s")

# ── Main Suite Runner ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Run with fewer trees to complete quickly")
    ap.add_argument("--params", type=str, default="best_params_optuna.json",
                    help="Path to the JSON file containing optimized params")
    args = ap.parse_args()

    print("=================================================================")
    print("      HRBoost Multiclass & Binary GBDT Evaluation Suite")
    print("=================================================================")
    
    hpo_dict = None
    if os.path.exists(args.params):
        try:
            with open(args.params, "r", encoding="utf-8") as f:
                hpo_dict = json.load(f)
            print(f"  Loaded hyperparameter configurations from: {args.params}")
        except Exception as e:
            print(f"  [Warning] Failed to read HPO JSON ({e}). Running baseline.")
    else:
        print(f"  [Info] '{args.params}' not found. Running baseline.")

    # 14 Tabular Datasets Suite (Binary and Multiclass)
    dataset_configs = [
        # Built-ins
        ("breast_cancer", lambda: load_breast_cancer_ds()),
        ("digits", lambda: load_digits_ds()),
        ("wine", lambda: load_wine_ds()),
        
        # OpenML Binary tasks
        ("diabetes", lambda: load_openml_by_id(37, "diabetes", is_binary=True)),
        ("credit-g", lambda: load_openml_by_id(31, "credit-g", is_binary=True)),
        ("blood-transfusion", lambda: load_openml_by_id(1464, "blood-transfusion", is_binary=True)),
        ("spambase", lambda: load_openml_by_id(44, "spambase", is_binary=True)),
        ("banknote", lambda: load_openml_by_id(1462, "banknote", is_binary=True)),
        ("qsar-biodeg", lambda: load_openml_by_id(1494, "qsar-biodeg", is_binary=True)),
        
        # OpenML Multiclass tasks
        ("car", lambda: load_openml_by_id(40975, "car", is_binary=False)),
        ("nursery", lambda: load_openml_by_id(26, "nursery", is_binary=False)),
        ("splice", lambda: load_openml_by_id(46, "splice", is_binary=False)),
        ("balance-scale", lambda: load_openml_by_id(11, "balance-scale", is_binary=False)),
        ("segment", lambda: load_openml_by_id(40984, "segment", is_binary=False))
    ]

    for name, loader in dataset_configs:
        res = loader()
        if res is not None:
            X, y, name, cat, num_classes = res
            evaluate(X, y, name, num_classes, hpo_dict=hpo_dict, cat_features=cat, quick=args.quick)

if __name__ == "__main__":
    main()