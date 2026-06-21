# HRBoost (Hierarchical Refined Boost)

HRBoost is a fast, lightweight Gradient Boosting Decision Tree (GBDT) library built in C++ and Python. It introduces a **Non-monotonic Bayesian Hierarchical Clustering (LNM-BHC, $k=3$)** algorithm inside its core engine to find optimal splits for high-cardinality categorical variables with zero manual parameter tuning.

HRBoost is 100% compliant with the `scikit-learn` API, offering both `HRBoostClassifier` and `HRBoostRegressor`.

---

## Installation

```bash
pip install hrboost
```

---

## Hyperparameter Reference

`HRBoostClassifier` and `HRBoostRegressor` accept the following parameters in their constructors:

### Core GBDT Parameters
- **`n_estimators`** (*int*, default=`200`): The number of boosting rounds (trees to build).
- **`learning_rate`** (*float*, default=`0.1`): Shrinkage rate applied to each tree's update to prevent overfitting.
- **`max_depth`** (*int*, default=`4`): Maximum depth of each decision tree.
- **`max_leaves`** (*int*, default=`64`): Maximum number of leaves allowed per tree.
- **`reg_lambda`** (*float*, default=`1.0`): L2 regularization term on weights. It also scales the baseline regularization for Bayesian Hierarchical Clustering.
- **`subsample`** (*float*, default=`0.8`): Fraction of training samples randomly chosen to train each tree.
- **`colsample_bytree`** (*float*, default=`1.0`): Fraction of features randomly selected for building each tree.
- **`n_bins`** (*int*, default=`32`): Maximum number of discrete bins to bucket continuous features.

### Split Constraints
- **`min_child_weight`** (*float*, default=`0.1`): Minimum sum of instance Hessian needed in a child node.
- **`gamma`** (*float*, default=`0.0`): Minimum loss reduction required to make a split.
- **`max_delta_step`** (*float*, default=`0.0`): Maximum delta step allowed for each tree's leaf output (useful for highly unbalanced classes).

### System & Features
- **`cat_features`** (*list of int*, default=`None`): List of feature indices to be treated as categorical features.
- **`random_state`** (*int*, default=`0`): Seed for random number generators (subsampling, colsample).
- **`verbose`** (*bool*, default=`True`): Controls C++ engine logging during training.

---

## Environment Variables for Advanced Tuning

HRBoost exposes internal engine dynamics through system environment variables to avoid hyperparameter inflation:

- **`COHESION_REG`** (*float*, default=`0.3`):
  - Controls the intensity of **Dynamic Cohesion Regularization** during tree splitting.
  - Cohesion measures how **similar** the two prospective child nodes are in terms of their leaf weight estimates ($dL = G_L/H_L$, $dR = G_R/H_R$). When children are similar (high cohesion — uninformative split), L2 regularization is dynamically increased to penalize the split. When children diverge (low cohesion — informative split), regularization stays at the base `reg_lambda`.
  - Set `export COHESION_REG=0.0` to disable and revert to standard XGBoost-style gain. High-noise or high-cardinality categorical settings benefit from higher values (e.g., `0.5` or `1.0`).
- **`MIN_CAT_COUNT`** (*float*, default=automatically scaled):
  - The minimum count required for a categorical bin to participate in BHC clustering. It helps filter out extremely rare categorical values.

---

## Quick Start

### 1. Classification
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from hrboost import HRBoostClassifier

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

clf = HRBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
clf.fit(X_train, y_train)

print(f"Test Accuracy: {clf.score(X_test, y_test):.4f}")
```

### 2. Regression
```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from hrboost import HRBoostRegressor

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

reg = HRBoostRegressor(n_estimators=150, learning_rate=0.08, max_depth=4)
reg.fit(X_train, y_train)

print(f"Test R2 Score: {reg.score(X_test, y_test):.4f}")
```
