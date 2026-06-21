import os
import ctypes
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from ._lib import _lib

class HRBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        max_leaves=64,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=1.0,
        n_bins=32,
        min_child_weight=0.1,
        gamma=0.0,
        max_delta_step=0.0,
        cat_features=None,
        random_state=0,
        objective="binary",
        num_classes=None,
        verbose=True,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_bins = n_bins
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.max_delta_step = max_delta_step
        self.cat_features = cat_features
        self.random_state = random_state
        self.objective = objective
        self.num_classes = num_classes
        self.verbose = verbose

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y_orig = np.ascontiguousarray(y)
        self.classes_ = np.unique(y_orig)
        n_classes = len(self.classes_)

        if self.objective == "binary" or n_classes <= 2:
            self.objective_ = "binary"
            self.num_classes_ = 1
        else:
            self.objective_ = "multiclass"
            self.num_classes_ = self.num_classes if self.num_classes is not None else n_classes

        # Target must be float32 for C++ HRBoost fit
        y = y_orig.astype(np.float32, copy=False)
        n, D = X.shape

        self.n_features_in_ = D
        self._handle = _lib.hrboost_create()

        cat_list = self.cat_features if self.cat_features is not None else []
        cats = np.asarray(cat_list, dtype=np.int32)
        cat_ptr = cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        obj_bytes = self.objective_.encode("utf-8")

        # Set environment variables for C++ logging control
        old_verbose = os.environ.get("HRBOOST_VERBOSE", None)
        os.environ["HRBOOST_VERBOSE"] = "1" if self.verbose else "0"

        try:
            _lib.hrboost_fit(
                self._handle,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                cat_ptr,
                ctypes.c_char_p(obj_bytes),
                ctypes.c_double(self.learning_rate),
                ctypes.c_double(self.reg_lambda),
                ctypes.c_double(self.subsample),
                ctypes.c_double(self.colsample_bytree),
                ctypes.c_double(self.min_child_weight),
                ctypes.c_double(self.gamma),
                ctypes.c_double(self.max_delta_step),
                ctypes.c_int(n),
                ctypes.c_int(D),
                ctypes.c_int(self.n_estimators),
                ctypes.c_int(self.max_depth),
                ctypes.c_int(self.max_leaves),
                ctypes.c_int(self.n_bins),
                ctypes.c_int(len(cats)),
                ctypes.c_int(self.random_state),
                ctypes.c_int(self.num_classes_)
            )
        finally:
            if old_verbose is not None:
                os.environ["HRBOOST_VERBOSE"] = old_verbose
            elif "HRBOOST_VERBOSE" in os.environ:
                del os.environ["HRBOOST_VERBOSE"]

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, D = X.shape

        if self.objective_ == "binary":
            out = np.empty(n, dtype=np.float64)
            _lib.hrboost_predict_proba(
                self._handle,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(n),
                ctypes.c_int(D),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            return np.column_stack([1.0 - out, out])
        else:
            out = np.empty(n * self.num_classes_, dtype=np.float64)
            _lib.hrboost_predict_proba(
                self._handle,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(n),
                ctypes.c_int(D),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            return out.reshape(n, self.num_classes_)

    def predict(self, X):
        check_is_fitted(self)
        proba = self.predict_proba(X)
        if self.objective_ == "binary":
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(proba, axis=1)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.hrboost_free(self._handle)
            self._handle = None


class HRBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        max_leaves=64,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=1.0,
        n_bins=32,
        min_child_weight=0.1,
        gamma=0.0,
        max_delta_step=0.0,
        cat_features=None,
        random_state=0,
        verbose=True,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_bins = n_bins
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.max_delta_step = max_delta_step
        self.cat_features = cat_features
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        n, D = X.shape

        self.objective_ = "regression"
        self.num_classes_ = 1
        self.n_features_in_ = D
        self._handle = _lib.hrboost_create()

        cat_list = self.cat_features if self.cat_features is not None else []
        cats = np.asarray(cat_list, dtype=np.int32)
        cat_ptr = cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        obj_bytes = self.objective_.encode("utf-8")

        # Set environment variables for C++ logging control
        old_verbose = os.environ.get("HRBOOST_VERBOSE", None)
        os.environ["HRBOOST_VERBOSE"] = "1" if self.verbose else "0"

        try:
            _lib.hrboost_fit(
                self._handle,
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                cat_ptr,
                ctypes.c_char_p(obj_bytes),
                ctypes.c_double(self.learning_rate),
                ctypes.c_double(self.reg_lambda),
                ctypes.c_double(self.subsample),
                ctypes.c_double(self.colsample_bytree),
                ctypes.c_double(self.min_child_weight),
                ctypes.c_double(self.gamma),
                ctypes.c_double(self.max_delta_step),
                ctypes.c_int(n),
                ctypes.c_int(D),
                ctypes.c_int(self.n_estimators),
                ctypes.c_int(self.max_depth),
                ctypes.c_int(self.max_leaves),
                ctypes.c_int(self.n_bins),
                ctypes.c_int(len(cats)),
                ctypes.c_int(self.random_state),
                ctypes.c_int(self.num_classes_)
            )
        finally:
            if old_verbose is not None:
                os.environ["HRBOOST_VERBOSE"] = old_verbose
            elif "HRBOOST_VERBOSE" in os.environ:
                del os.environ["HRBOOST_VERBOSE"]

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, D = X.shape

        out = np.empty(n, dtype=np.float64)
        _lib.hrboost_predict(
            self._handle,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n),
            ctypes.c_int(D),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return out

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.hrboost_free(self._handle)
            self._handle = None
