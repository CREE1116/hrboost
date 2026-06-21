import os
import numpy as np
from sklearn.datasets import load_digits, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Add python path to ensure hrboost import succeeds
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "python"))

from hrboost import HRBoostClassifier, HRBoostRegressor

def test_classifier():
    print("=== Testing HRBoostClassifier ===")
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # test binary first by filtering classes 0 and 1
    binary_mask_train = (y_train == 0) | (y_train == 1)
    binary_mask_test = (y_test == 0) | (y_test == 1)
    X_bin_train, y_bin_train = X_train[binary_mask_train], y_train[binary_mask_train]
    X_bin_test, y_bin_test = X_test[binary_mask_test], y_test[binary_mask_test]

    clf_bin = HRBoostClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42, objective="binary")
    clf_bin.fit(X_bin_train, y_bin_train)
    preds_bin = clf_bin.predict(X_bin_test)
    acc_bin = accuracy_score(y_bin_test, preds_bin)
    print(f"Binary Classifier Accuracy: {acc_bin:.4f}")
    assert acc_bin > 0.90, f"Binary accuracy {acc_bin} is too low!"

    # test multiclass
    clf_multi = HRBoostClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42, objective="multiclass")
    clf_multi.fit(X_train, y_train)
    preds_multi = clf_multi.predict(X_test)
    acc_multi = accuracy_score(y_test, preds_multi)
    print(f"Multiclass Classifier Accuracy: {acc_multi:.4f}")
    assert acc_multi > 0.85, f"Multiclass accuracy {acc_multi} is too low!"

def test_regressor():
    print("=== Testing HRBoostRegressor ===")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = HRBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Regressor MSE: {mse:.4f}")
    
    # baseline target mean mse for comparison
    baseline_mse = mean_squared_error(y_test, np.full_like(y_test, np.mean(y_train)))
    print(f"Baseline (Mean Predictor) MSE: {baseline_mse:.4f}")
    assert mse < baseline_mse, f"Regressor MSE {mse} should be lower than baseline MSE {baseline_mse}!"

def test_cohesion_reg():
    print("=== Testing COHESION_REG regularization ===")
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test with standard cohesion reg (default is 0.3)
    clf_default = HRBoostClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42, objective="multiclass")
    clf_default.fit(X_train, y_train)
    acc_def = accuracy_score(y_test, clf_default.predict(X_test))

    # Test with COHESION_REG environment variable set to high value (e.g. 2.0)
    os.environ["COHESION_REG"] = "2.0"
    clf_high = HRBoostClassifier(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42, objective="multiclass")
    clf_high.fit(X_train, y_train)
    acc_high = accuracy_score(y_test, clf_high.predict(X_test))

    print(f"Accuracy (default): {acc_def:.4f}")
    print(f"Accuracy (COHESION_REG=2.0): {acc_high:.4f}")
    
    # Clean up
    if "COHESION_REG" in os.environ:
        del os.environ["COHESION_REG"]

if __name__ == "__main__":
    test_classifier()
    test_regressor()
    test_cohesion_reg()
    print("All tests passed successfully!")
