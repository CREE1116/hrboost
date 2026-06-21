"""
BRSTBoost vs LightGBM / XGBoost / CatBoost — AUC benchmark via JSON parameters.

Usage:
    uv run python benchmark.py --params best_params.json
    uv run python benchmark.py --quick
"""
import sys, time, argparse, json, os
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
sys.path.insert(0, "python")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from hrboost import HRBoostClassifier

# ── optional competitors ───────────────────────────────────────────────────────

def _try(name):
    try: return __import__(name)
    except ImportError: return None

lgb = _try("lightgbm")
xgb = _try("xgboost")
cb  = _try("catboost")

# ── real dataset zoo ──────────────────────────────────────────────────────────

def load_breast_cancer_ds():
    data = load_breast_cancer()
    return data.data.astype(np.float32), data.target.astype(np.int32), "breast_cancer"

def load_diabetes():
    try:
        data = fetch_openml(data_id=37, as_frame=True, parser='auto')
        X = data.data.to_numpy().astype(np.float32)
        y = (data.target == 'tested_positive').astype(np.int32).to_numpy()
        return X, y, "diabetes"
    except Exception as e:
        print(f"  [diabetes load failed: {e}]")
        return None

def load_credit_g():
    try:
        data = fetch_openml(data_id=31, as_frame=True, parser='auto')
        df_X = data.data.copy()
        y = (data.target == 'good').astype(np.int32).to_numpy()
        
        cat_idx = []
        for i, col in enumerate(df_X.columns):
            if df_X[col].dtype.name == 'category' or df_X[col].dtype == object:
                cat_idx.append(i)
                le = LabelEncoder()
                df_X[col] = le.fit_transform(df_X[col].astype(str))
                
        X = df_X.to_numpy().astype(np.float32)
        return X, y, "credit-g", cat_idx
    except Exception as e:
        print(f"  [credit-g load failed: {e}]")
        return None

def load_adult():
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "adult/adult.data")
    cols = ["age","workclass","fnlwgt","education","edu_num","marital",
            "occupation","relationship","race","sex","cap_gain","cap_loss",
            "hours","country","income"]
    try:
        df = pd.read_csv(url, header=None, names=cols, na_values="?",
                         skipinitialspace=True)
    except Exception as e:
        print(f"  [adult load failed: {e}]")
        return None
    y = (df["income"].str.strip() == ">50K").astype(int).values
    df = df.drop("income", axis=1)
    for c in df.select_dtypes("object").columns:
        s = pd.Series(np.nan, index=df.index, dtype=np.float32)
        non_null_mask = df[c].notnull()
        if non_null_mask.any():
            le = LabelEncoder()
            s.loc[non_null_mask] = le.fit_transform(df.loc[non_null_mask, c].astype(str)).astype(np.float32)
        df[c] = s
    X = df.values.astype(np.float32)
    cat_idx = [i for i, c in enumerate(df.columns)
               if df[c].nunique(dropna=True) < 50 and c not in
               ("age","fnlwgt","edu_num","cap_gain","cap_loss","hours")]
    return X, y.astype(np.int32), "adult", cat_idx

# ── model factory with JSON injector ──────────────────────────────────────────

def make_models(ds_name, hpo_dict=None, cat_features=None, quick=False):
    cat = cat_features or []
    est = 200 if quick else 400
    lr  = 0.1  if quick else 0.05
    models = {}
    
    # 해당 데이터셋의 HPO 서브 세트 추출 시도
    ds_hpo = hpo_dict.get(ds_name, {}) if hpo_dict else {}

    # 1. SelfCDSB
    sc_params = {
        "n_estimators": est, "learning_rate": lr, "max_depth": 5, "max_leaves": 31,
        "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 1.0, "n_bins": 255,
        "cat_features": cat, "random_state": 0, "objective": "binary", "verbose": False
    }
    if "SelfCDSB" in ds_hpo:
        sc_hpo = ds_hpo["SelfCDSB"].copy()
        sc_hpo.pop("cdsb_alpha", None)
        sc_params.update(sc_hpo)
    elif "HRBoost" in ds_hpo:
        sc_hpo = ds_hpo["HRBoost"].copy()
        sc_params.update(sc_hpo)
    models["HRBoost"] = HRBoostClassifier(**sc_params)

    # 2. LightGBM
    if lgb:
        lgb_kw = dict(n_estimators=est, learning_rate=lr, max_depth=5,
                      num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, random_state=0, verbose=-1)
        if cat:
            lgb_kw["categorical_feature"] = cat
        if "LightGBM" in ds_hpo:
            lgb_kw.update(ds_hpo["LightGBM"])
        models["LightGBM"] = lgb.LGBMClassifier(**lgb_kw)

    # 3. XGBoost
    if xgb:
        xgb_kw = dict(n_estimators=est, learning_rate=lr, max_depth=5,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      eval_metric="logloss", random_state=0, verbosity=0)
        if "XGBoost" in ds_hpo:
            xgb_kw.update(ds_hpo["XGBoost"])
        models["XGBoost"] = xgb.XGBClassifier(**xgb_kw)

    # 4. CatBoost
    if cb:
        cb_kw = dict(iterations=est, learning_rate=lr, depth=5,
                     l2_leaf_reg=1.0, subsample=0.8,
                     random_state=0, verbose=False)
        if "CatBoost" in ds_hpo:
            cb_kw.update(ds_hpo["CatBoost"])
        if cat:
            cb_kw["cat_features"] = cat
            
        _inner_cb = cb.CatBoostClassifier(**cb_kw)
        
        class _CBWrap:
            def __init__(self, m, cat_idx):
                self._m = m; self._cat = cat_idx
            def _prep(self, X):
                df = pd.DataFrame(X)
                for c in self._cat:
                    df[c] = df[c].map(lambda v: 'nan' if np.isnan(v) else str(int(v)))
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

def evaluate(X, y, name, hpo_dict=None, cat_features=None, n_splits=5, quick=False):
    print(f"\n{'─'*58}")
    print(f"  {name}  n={len(y):,}  D={X.shape[1]}  pos={y.mean():.3f}")
    
    # HPO 설정 주입 유무 피드백
    if hpo_dict and name in hpo_dict:
        print(f"  [Status] Using Optimized HPO Hyperparameters from JSON")
    else:
        print(f"  [Status] Using Default Baseline Hyperparameters")
        
    print(f"{'─'*58}")
    print(HDR); print(SEP)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for mname, model in make_models(name, hpo_dict, cat_features, quick=quick).items():
        aucs, times = [], []
        for tr, te in cv.split(X, y):
            t0 = time.time()
            model.fit(X[tr], y[tr])
            elapsed = time.time() - t0
            times.append(elapsed)
            p = model.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))

        mu, sd = np.mean(aucs), np.std(aucs)
        print(f"  {mname:<12} {mu:>8.4f} {sd:>6.4f} {np.mean(times):>8.1f}s")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--params", type=str, default="best_params.json",
                    help="Path to the JSON file containing optimized params")
    args = ap.parse_args()

    print("Self-CDSB Benchmark (Real Datasets Integration)")
    
    # JSON 로드 시도
    hpo_dict = None
    if os.path.exists(args.params):
        try:
            with open(args.params, "r", encoding="utf-8") as f:
                hpo_dict = json.load(f)
            print(f"  Loaded hyperparameter configurations from: {args.params}")
        except Exception as e:
            print(f"  [Warning] Failed to read JSON ({e}). Falling back to baseline.")
    else:
        print(f"  [Info] '{args.params}' file not found. Running with baseline parameters.")

    datasets = []
    datasets.append(load_breast_cancer_ds())

    db_data = load_diabetes()
    if db_data: datasets.append(db_data)

    cr_data = load_credit_g()
    if cr_data: datasets.append(cr_data)

    adult_data = load_adult()
    if adult_data: datasets.append(adult_data)

    for entry in datasets:
        if len(entry) == 4:
            X, y, name, cat = entry
            evaluate(X, y, name, hpo_dict=hpo_dict, cat_features=cat, quick=args.quick)
        else:
            X, y, name = entry
            evaluate(X, y, name, hpo_dict=hpo_dict, quick=args.quick)

if __name__ == "__main__":
    main()