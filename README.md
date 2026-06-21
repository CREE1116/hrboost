# HRBoost (Hierarchical Refined Boost)

HRBoost is a fast, lightweight Gradient Boosting Decision Tree (GBDT) library built in C++ and Python. It introduces a **Non-monotonic Bayesian Hierarchical Clustering (LNM-BHC, $k=3$)** algorithm inside its core engine to find optimal splits for high-cardinality categorical variables with zero manual parameter tuning.

It is designed to be 100% compliant with the `scikit-learn` API, offering both `HRBoostClassifier` and `HRBoostRegressor`.

---

## Key Features

- **Optimal Categorical Splitting (LNM-BHC)**: Implements non-monotonic Bayesian Hierarchical Clustering to capture categorical structure under noise without sorting artifacts.
- **Zero-Parameter Diet**: Slimmed-down hyperparameter interface where BHC regularization uses a robust fixed sliding window size $k=3$ and falls back to `reg_lambda`.
- **Scikit-Learn Compliant**: Direct replacement for `LGBMClassifier/Regressor` or `XGBClassifier/Regressor` in python pipelines.
- **COHESION_REG Tuning**: Keep control of dynamic regularization sensitivity via the `COHESION_REG` environment variable (default: `0.3`).

---

## Installation

### From PyPI
```bash
pip install hrboost
```

### From Source
Ensure you have a C++ compiler supporting C++17.
```bash
git clone https://github.com/yourusername/hrboost.git
cd hrboost
sh build.sh
pip install -e .
```

---

## Quick Start

### 1. Classification (`HRBoostClassifier`)
`HRBoostClassifier` supports binary and multiclass tasks natively.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from hrboost import HRBoostClassifier

# Load digits dataset (10 classes)
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Initialize & fit
clf = HRBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    objective="multiclass"
)
clf.fit(X_train, y_train)

# Predict probabilities and classes
probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)

accuracy = np.mean(preds == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

### 2. Regression (`HRBoostRegressor`)
`HRBoostRegressor` models continuous target values with Mean Squared Error (MSE) objective.

```python
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from hrboost import HRBoostRegressor

# Load diabetes dataset
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Initialize & fit
reg = HRBoostRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    random_state=42
)
reg.fit(X_train, y_train)

# Predict
preds = reg.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"MSE: {mse:.4f}")
```

### 3. Dynamic Regularization Sensitivity (`COHESION_REG`)
You can tune BHC's dynamic regularization cohesion penalty via the environment variable:

```bash
export COHESION_REG=0.5
python your_script.py
```

---

## License

This project is licensed under the MIT License.
