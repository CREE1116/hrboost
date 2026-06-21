# HRBoost (Hierarchical Refined Boost)

HRBoost is a high-performance Gradient Boosting Decision Tree (GBDT) library built in C++ and Python. It is designed to natively handle high-cardinality categorical variables and prevent overfitting on sparse splits. 

HRBoost is 100% compliant with the `scikit-learn` API, offering both `HRBoostClassifier` and `HRBoostRegressor` as drop-in replacements for traditional boosting frameworks.

---

## Technical Moat: How It Works

HRBoost derives its name from **Hierarchical Refined Boost** — referring to its two-phase categorical split engine: BHC clustering followed by iterative refinement.

### 1. LNM-BHC + Iterative Partition Refinement
Traditional GBDTs (such as LightGBM) handle categorical variables by sorting categories by their target statistics in 1D space, then performing a monotonic split. While fast, this sorting is highly vulnerable to noise, target leakage, and fails when categories do not exhibit monotonic relationships with the target.

HRBoost addresses this with a two-phase approach:

**Phase 1 — LNM-BHC (Local Non-monotonic Bayesian Hierarchical Clustering, $k=3$):**
- Categories are initialized sorted by their gradient/hessian ratio ($G/H$).
- **Bayesian Hierarchical Clustering** greedily merges the most similar adjacent category pair, using gradient statistics as the similarity criterion.
- A **local window of $k=3$** allows merging non-adjacent categories within 3 steps — capturing non-monotonic combinations that monotonic splits miss.
- Merging continues until 2 clusters remain, forming the initial Left/Right partition.

**Phase 2 — Iterative Partition Refinement ("Refined" in HRBoost):**
- Starting from the BHC partition, each category is individually tested for flipping (Left→Right or Right→Left).
- The single move that most improves split gain is applied per iteration.
- Up to 10 iterations, stopping when no improvement exceeds $10^{-7}$.
- This corrects suboptimal greedy decisions made during BHC.

```
[Traditional GBDT: 1D Monotonic Split]
Category Sorted:  [A] -> [B] -> [C] -> [D]
Split Boundary:          |  (Only monotonic cuts allowed)

[HRBoost: LNM-BHC (k=3) + Refinement]
BHC clusters:     {A, D} | {B, C}  (non-adjacent merge via k=3 window)
After Refinement: individual categories re-evaluated and moved if gain improves
```

### 2. Cohesion Dynamic Regularization
In high-cardinality or highly imbalanced datasets, GBDTs often overfit by creating splits that barely separate the data. HRBoost implements **Cohesion Dynamic Regularization**:
- During tree splitting, we measure the similarity ("Cohesion") of the proposed child node leaf value estimates.
- When child predictions are **similar** (high cohesion — the split is uninformative), $\lambda$ is dynamically increased to penalize the split:

$$\lambda_{dyn} = \lambda \times (1.0 + \gamma_{cohesion} \times Cohesion)$$

Where cohesion is computed as:

$$Cohesion = 1.0 - \frac{|dL - dR|}{|dL| + |dR| + 10^{-5}}$$

(where $dL = G_L / H_L$ and $dR = G_R / H_R$ are the left/right leaf weight estimates).
- When children **diverge** (low cohesion — informative split), $\lambda_{dyn} \approx \lambda$ and the split proceeds normally.
- This discourages uninformative splits on rare or noisy categories. Users can control sensitivity via the `COHESION_REG` environment variable (default: `0.3`, set to `0.0` to disable).

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

## Hyperparameter Optimization & Benchmarking

HRBoost provides built-in tools for hyperparameter tuning and model evaluation against other frameworks (`LightGBM`, `XGBoost`, `CatBoost`).

### 1. Hyperparameter Tuning (`optimizer.py`)
You can use Optuna to automatically search for the best hyperparameters (including `COHESION_REG`) for any classification dataset:

```bash
# Run HPO tuning with 50 trials on 'credit-g' dataset
python optimizer.py --dataset credit-g --n-trials 50 --output best_params_optuna.json
```

This will find the optimal parameters and record them into `best_params_optuna.json` under the dataset's name.

### 2. Comprehensive Benchmarking (`benchmark.py`)
HRBoost includes a benchmarking suite containing **14 diverse tabular datasets** (breast_cancer, digits, wine, car, nursery, splice, balance-scale, segment, and more). It dynamically applies HPO parameters from your JSON configuration.

```bash
# Evaluate HRBoost against competitors using optimized parameters
python benchmark.py --params best_params_optuna.json

# Or run a quick evaluation with fewer trees
python benchmark.py --quick
```

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
