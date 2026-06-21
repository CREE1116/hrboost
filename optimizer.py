import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import json
import argparse
import numpy as np
import optuna
import pandas as pd
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Ensure hrboost path is configured
sys.path.insert(0, "python")
from hrboost import HRBoostClassifier

# Optional competitors
def _try(name):
    try: return __import__(name)
    except ImportError: return None

lgb = _try("lightgbm")
xgb = _try("xgboost")
cb  = _try("catboost")

SUPPORTED_DATASETS = {
    "breast_cancer": (None, True),
    "digits": (None, False),
    "wine": (None, False),
    "diabetes": (37, True),
    "credit-g": (31, True),
    "blood-transfusion": (1464, True),
    "spambase": (44, True),
    "banknote": (1462, True),
    "qsar-biodeg": (1494, True),
    "car": (40975, False),
    "nursery": (26, False),
    "splice": (46, False),
    "balance-scale": (11, False),
    "segment": (40984, False)
}

def load_data(dataset_name):
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        return data.data.astype(np.float32), data.target.astype(np.int32), [], 2
    elif dataset_name == "digits":
        data = load_digits()
        X = data.data.astype(np.float32)
        return X, data.target.astype(np.int32), list(range(X.shape[1])), 10
    elif dataset_name == "wine":
        data = load_wine()
        return data.data.astype(np.float32), data.target.astype(np.int32), [], 3
    elif dataset_name in SUPPORTED_DATASETS:
        data_id, is_binary = SUPPORTED_DATASETS[dataset_name]
        print(f"Fetching '{dataset_name}' (ID: {data_id}) from OpenML...")
        data = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
        df_X = data.data.copy()
        
        if is_binary:
            unique_targets = data.target.unique()
            pos_label = unique_targets[0]
            y = (data.target == pos_label).astype(np.int32).to_numpy()
        else:
            le_y = LabelEncoder()
            y = le_y.fit_transform(data.target.astype(str)).astype(np.int32)
            
        cat_idx = []
        for i, col in enumerate(df_X.columns):
            if df_X[col].dtype.name == 'category' or df_X[col].dtype == object:
                cat_idx.append(i)
                le = LabelEncoder()
                df_X[col] = le.fit_transform(df_X[col].astype(str))
                
        X = df_X.to_numpy().astype(np.float32)
        return X, y, cat_idx, len(np.unique(y))
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Choose from: {list(SUPPORTED_DATASETS.keys())}")

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

def optimize(dataset_name, model_name, n_trials, output_file):
    print(f"\n>>> Starting HPO for '{dataset_name}' with '{model_name}' ({n_trials} trials)...")
    try:
        X, y, cat_idx, num_classes = load_data(dataset_name)
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}. Skipping.")
        return
        
    objective_type = "binary" if num_classes <= 2 else "multiclass"
    is_multi = num_classes > 2
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Minimize output logs of optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        if model_name == "HRBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "max_leaves": trial.suggest_int("max_leaves", 15, 127),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "n_bins": trial.suggest_int("n_bins", 15, 255),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 5.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "random_state": 42,
                "objective": objective_type,
                "num_classes": num_classes,
                "cat_features": cat_idx,
                "verbose": False
            }
            cohesion_val = trial.suggest_float("cohesion_reg", 0.0, 1.5)
            os.environ["COHESION_REG"] = f"{cohesion_val:.4f}"
            
        elif model_name == "LightGBM":
            if not lgb:
                raise ImportError("lightgbm is not installed.")
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 0,
                "verbose": -1
            }
            
        elif model_name == "XGBoost":
            if not xgb:
                raise ImportError("xgboost is not installed.")
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 0,
                "verbosity": 0
            }
            if is_multi:
                params["objective"] = "multi:softprob"
                params["num_class"] = num_classes
            else:
                params["eval_metric"] = "logloss"
                
        elif model_name == "CatBoost":
            if not cb:
                raise ImportError("catboost is not installed.")
            params = {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                "depth": trial.suggest_int("depth", 3, 7),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "bootstrap_type": "Bernoulli"
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        aucs = []
        for tr, te in cv.split(X, y):
            # Model instantation
            if model_name == "HRBoost":
                model = HRBoostClassifier(**params)
            elif model_name == "LightGBM":
                model = lgb.LGBMClassifier(**params)
            elif model_name == "XGBoost":
                model = xgb.XGBClassifier(**params)
            elif model_name == "CatBoost":
                model = CatBoostWrapper(is_multi, cat_idx, params)
                
            model.fit(X[tr], y[tr])
            probs = model.predict_proba(X[te])
            
            try:
                if objective_type == "binary":
                    auc = roc_auc_score(y[te], probs[:, 1])
                else:
                    auc = roc_auc_score(y[te], probs, multi_class="ovr", average="macro")
            except Exception:
                auc = 0.5
            aucs.append(auc)
            
        return np.mean(aucs)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_val = study.best_value
    best_params = study.best_params
    print(f"Finished '{dataset_name}' with '{model_name}' | Best AUC: {best_val:.4f}")
    
    # Load previous HPO results
    hpo_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                hpo_results = json.load(f)
        except Exception:
            pass
            
    # Structure HPO parameters formatting
    ds_entry = hpo_results.get(dataset_name, {})
    
    if model_name == "HRBoost":
        hrboost_hpo = {
            "n_estimators": int(best_params["n_estimators"]),
            "learning_rate": float(best_params["learning_rate"]),
            "max_depth": int(best_params["max_depth"]),
            "max_leaves": int(best_params["max_leaves"]),
            "reg_lambda": float(best_params["reg_lambda"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "n_bins": int(best_params["n_bins"]),
            "min_child_weight": float(best_params["min_child_weight"]),
            "gamma": float(best_params["gamma"]),
        }
        env_config = {
            "COHESION_REG": float(best_params["cohesion_reg"])
        }
        ds_entry["HRBoost"] = hrboost_hpo
        ds_entry["ENV"] = env_config
        
    elif model_name == "LightGBM":
        ds_entry["LightGBM"] = {
            "n_estimators": int(best_params["n_estimators"]),
            "learning_rate": float(best_params["learning_rate"]),
            "max_depth": int(best_params["max_depth"]),
            "num_leaves": int(best_params["num_leaves"]),
            "reg_lambda": float(best_params["reg_lambda"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
        }
    elif model_name == "XGBoost":
        ds_entry["XGBoost"] = {
            "n_estimators": int(best_params["n_estimators"]),
            "learning_rate": float(best_params["learning_rate"]),
            "max_depth": int(best_params["max_depth"]),
            "reg_lambda": float(best_params["reg_lambda"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
        }
    elif model_name == "CatBoost":
        ds_entry["CatBoost"] = {
            "iterations": int(best_params["iterations"]),
            "learning_rate": float(best_params["learning_rate"]),
            "depth": int(best_params["depth"]),
            "l2_leaf_reg": float(best_params["l2_leaf_reg"]),
            "subsample": float(best_params["subsample"]),
            "bootstrap_type": "Bernoulli"
        }
        
    hpo_results[dataset_name] = ds_entry
    with open(output_file, "w") as f:
        json.dump(hpo_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRBoost & Competitors Optuna Hyperparameter Optimizer")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset name (or 'all' to tune all 14 datasets)")
    parser.add_argument("--model", type=str, default="all",
                        help="Model to optimize: HRBoost, LightGBM, XGBoost, CatBoost, or 'all'")
    parser.add_argument("--n-trials", type=str, default="20", help="Number of trials per model/dataset")
    parser.add_argument("--output", type=str, default="best_params_optuna.json", help="Path to save results")
    args = parser.parse_args()
    
    trials = int(args.n_trials)
    
    target_models = ["HRBoost", "LightGBM", "XGBoost", "CatBoost"]
    if args.model != "all":
        if args.model in target_models:
            target_models = [args.model]
        else:
            raise ValueError(f"Unknown model '{args.model}'. Choose from: HRBoost, LightGBM, XGBoost, CatBoost, or 'all'")
            
    target_datasets = list(SUPPORTED_DATASETS.keys())
    if args.dataset != "all":
        if args.dataset in target_datasets:
            target_datasets = [args.dataset]
        else:
            raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from: {target_datasets} or 'all'")

    print(f"Starting batch HPO optimization for datasets: {target_datasets} and models: {target_models}")
    for ds in target_datasets:
        for model in target_models:
            # Skip if packages are not installed
            if model == "LightGBM" and not lgb:
                continue
            if model == "XGBoost" and not xgb:
                continue
            if model == "CatBoost" and not cb:
                continue
            optimize(ds, model, trials, args.output)
            
    print(f"\nAll optimization processes finished! Hyperparameters saved to {args.output}")
