import os
import sys
import json
import argparse
import numpy as np
import optuna
from sklearn.datasets import load_digits, load_breast_cancer, fetch_openml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Ensure hrboost path is configured
sys.path.insert(0, "python")
from hrboost import HRBoostClassifier

def load_data(dataset_name):
    if dataset_name == "digits":
        data = load_digits()
        return data.data.astype(np.float32), data.target.astype(np.int32), [], len(np.unique(data.target))
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        return data.data.astype(np.float32), data.target.astype(np.int32), [], 2
    elif dataset_name == "credit-g":
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
        return X, y, cat_idx, 2
    elif dataset_name == "splice":
        data = fetch_openml(name='splice', version=1, as_frame=True, parser='auto')
        df_X = data.data.copy()
        le_y = LabelEncoder()
        y = le_y.fit_transform(data.target.astype(str)).astype(np.int32)
        cat_idx = []
        for i, col in enumerate(df_X.columns):
            cat_idx.append(i)
            le = LabelEncoder()
            df_X[col] = le.fit_transform(df_X[col].astype(str))
        X = df_X.to_numpy().astype(np.float32)
        return X, y, cat_idx, len(np.unique(y))
    else:
        # Standard OpenML fetch by name
        print(f"Fetching '{dataset_name}' from OpenML...")
        data = fetch_openml(name=dataset_name, version=1, as_frame=True, parser='auto')
        df_X = data.data.copy()
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

def optimize(dataset_name, n_trials, output_file):
    X, y, cat_idx, num_classes = load_data(dataset_name)
    objective_type = "binary" if num_classes <= 2 else "multiclass"
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    def objective(trial):
        # Hyperparameters
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
        
        # Optuna environments tuning for BHC dynamic cohesion
        cohesion_val = trial.suggest_float("cohesion_reg", 0.0, 1.5)
        os.environ["COHESION_REG"] = f"{cohesion_val:.4f}"
        
        aucs = []
        for tr, te in cv.split(X, y):
            model = HRBoostClassifier(**params)
            model.fit(X[tr], y[tr])
            probs = model.predict_proba(X[te])
            
            if objective_type == "binary":
                auc = roc_auc_score(y[te], probs[:, 1])
            else:
                auc = roc_auc_score(y[te], probs, multi_class="ovr", average="macro")
            aucs.append(auc)
            
        return np.mean(aucs)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\nOptimization Finished for dataset: {dataset_name}")
    print(f"Best Trial Score: {study.best_value:.4f}")
    best_params = study.best_params
    print("Best params found:", best_params)
    
    # Save parameters
    hpo_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                hpo_results = json.load(f)
        except Exception:
            pass
            
    # Format to match benchmark.py HPO style
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
    
    # Environment variable configurations
    env_config = {
        "COHESION_REG": float(best_params["cohesion_reg"])
    }
    
    hpo_results[dataset_name] = {
        "HRBoost": hrboost_hpo,
        "ENV": env_config
    }
    
    with open(output_file, "w") as f:
        json.dump(hpo_results, f, indent=4)
        
    print(f"Saved optimized hyperparameters into: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HRBoost Optuna Hyperparameter Optimizer")
    parser.add_argument("--dataset", type=str, default="credit-g", help="Dataset name (digits, breast_cancer, credit-g, splice or OpenML dataset name)")
    parser.add_argument("--n-trials", type=str, default="30", help="Number of HPO trials")
    parser.add_argument("--output", type=str, default="best_params_optuna.json", help="Path to save best parameters")
    args = parser.parse_args()
    
    optimize(args.dataset, int(args.n_trials), args.output)
