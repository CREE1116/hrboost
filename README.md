# HRBoost (Hierarchical Refined Boost)

HRBoost is a high-performance Gradient Boosting Decision Tree (GBDT) library built in C++ and Python. It is designed to natively handle high-cardinality categorical variables and prevent overfitting on sparse splits. 

HRBoost is 100% compliant with the `scikit-learn` API, offering both `HRBoostClassifier` and `HRBoostRegressor` as drop-in replacements for traditional boosting frameworks.

---

## Technical Moat: How It Works

HRBoost derives its name from **Hierarchical Refined Boost**. It achieves state-of-the-art accuracy on tabular data through two key innovations inside its C++ Core:

### 1. LNM-BHC (Local Non-monotonic Bayesian Hierarchical Clustering, $k=3$)
Traditional GBDTs (such as LightGBM) handle categorical variables by sorting categories by their target statistics in 1D space, then performing a monotonic split. While fast, this sorting is highly vulnerable to noise, target leakage, and fails when categories do not exhibit monotonic relationships with the target.

HRBoost addresses this by performing **Bayesian Hierarchical Clustering (BHC)** to merge categories into optimal partitions. To prevent ordering artifacts:
- We introduce a **Local Non-monotonic BHC scan** with a sliding window of size $k=3$.
- During hierarchical merging, categories are allowed to merge out-of-order within the window range.
- This captures non-monotonic, non-adjacent category combinations, forming an optimal categorical split mask that traditional GBDTs miss.

```
[Traditional GBDT: 1D Monotonic Split]
Category Sorted:  [A] -> [B] -> [C] -> [D]
Split Boundary:          |  (Only monotonic cuts allowed)

[HRBoost: Local Non-monotonic BHC (k=3)]
Window Scans:     [A, B, C] -> Out-of-order merges allowed
Output Mask:      Left child: {A, D} | Right child: {B, C}  (Complex combinations captured)
```

### 2. Cohesion Dynamic Regularization
In high-cardinality or highly imbalanced datasets, GBDTs often overfit by creating deep splits for rare categories. HRBoost implements **Cohesion Dynamic Regularization**:
- During tree splitting, we measure the similarity ("Cohesion") of the proposed child node leaf values.
- If the child predictions diverge excessively, the L2 regularization lambda ($\lambda$) is dynamically scaled upwards:

$$\lambda_{dyn} = \lambda \times (1.0 + \gamma_{cohesion} \times Cohesion)$$

Where cohesion is computed as:

$$Cohesion = 1.0 - \frac{|dL - dR|}{|dL| + |dR| + 10^{-5}}$$

(where $dL$ and $dR$ represent the change in leaf weights).
- This dynamically penalizes overly aggressive splits on sparse categories, encouraging the tree to search for broader, more generalizable split boundaries. Users can control this sensitivity via the `COHESION_REG` environment variable (default: `0.3`).

---

## Performance Benchmarks

### 1. Binary Classification AUC (5-Fold CV)
Measured using baseline settings vs. competitors' optimized hyperparameter settings.

| Dataset | HRBoost (LNM-BHC) | LightGBM | XGBoost | CatBoost |
| :--- | :---: | :---: | :---: | :---: |
| **breast_cancer** | **0.9956** 🏆 | 0.9948 | 0.9944 | 0.9952 |
| **diabetes** | 0.8035 | 0.7911 | 0.8003 | **0.8186** 🏆 |
| **credit-g (Categorical)** | 0.7831 | 0.7772 | **0.7844** 🏆 | 0.7820 |
| **adult (Large Scale)** | 0.9301 | **0.9305** 🏆 | 0.9296 | 0.9301 |

### 2. OpenML Multiclass Benchmark (Macro AUC / Accuracy)
Evaluation on multiclass datasets using 5-Fold Cross Validation.

| Dataset | Metric | HRBoost | LightGBM | CatBoost |
| :--- | :--- | :---: | :---: | :---: |
| **splice** (D=60, Classes=3) | Macro AUC <br> Accuracy | **0.9956** 🏆 <br> **0.9680** 🏆 | 0.9945 <br> 0.9671 | 0.9926 <br> 0.9502 |
| **digits** (D=64, Classes=10) | Macro AUC <br> Accuracy | **0.9977** 🏆 <br> **0.9432** 🏆 | 0.9975 <br> 0.9416 | 0.9939 <br> 0.8943 |
| **balance-scale** (Classes=3) | Macro AUC <br> Accuracy | **0.9221** 🏆 <br> **0.8832** 🏆 | 0.9188 <br> 0.8752 | 0.9205 <br> 0.8801 |

---

## Installation & Quick Start

```bash
pip install hrboost
```

### Classification Example
```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from hrboost import HRBoostClassifier

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = HRBoostClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

### Regression Example
```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from hrboost import HRBoostRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = HRBoostRegressor(n_estimators=150, learning_rate=0.08, max_depth=4)
reg.fit(X_train, y_train)

print(f"R2 Score: {reg.score(X_test, y_test):.4f}")
```

---

## License
MIT License.
