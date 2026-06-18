"""sklearn-compatible BRSTBoost classifier."""
import ctypes
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from ._lib import _lib


class BRSTBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=4,
                 max_leaves=64, reg_lambda=1.0, bhc_lam=0.0,
                 subsample=0.8, colsample_bytree=1.0, n_bins=32,
                 min_child_weight=1.0, gamma=0.0, max_k=2,
                 cat_features=None, random_state=0):
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.max_depth        = max_depth
        self.max_leaves       = max_leaves
        self.reg_lambda       = reg_lambda
        self.bhc_lam          = bhc_lam
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.n_bins           = n_bins
        self.min_child_weight = min_child_weight
        self.gamma            = gamma
        self.max_k            = max_k
        self.cat_features     = cat_features or []
        self.random_state     = random_state

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)
        n, D = X.shape
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = D
        self._handle = _lib.brst_create()
        cats = np.asarray(self.cat_features, dtype=np.int32)
        cat_ptr = cats.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        _lib.brst_fit(
            self._handle,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n), ctypes.c_int(D),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(self.n_estimators),
            ctypes.c_double(self.learning_rate),
            ctypes.c_int(self.max_depth),
            ctypes.c_int(self.max_leaves),
            ctypes.c_double(self.reg_lambda),
            ctypes.c_double(self.bhc_lam),
            ctypes.c_double(self.subsample),
            ctypes.c_double(self.colsample_bytree),
            ctypes.c_int(self.n_bins),
            ctypes.c_double(self.min_child_weight),
            ctypes.c_double(self.gamma),
            ctypes.c_int(self.max_k),
            cat_ptr,
            ctypes.c_int(len(cats)),
            ctypes.c_int(self.random_state),
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, D = X.shape
        out = np.empty(n, dtype=np.float64)
        _lib.brst_predict_proba(
            self._handle,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(n), ctypes.c_int(D),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return np.column_stack([1 - out, out])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def avg_k(self):
        check_is_fitted(self)
        return _lib.brst_avg_k(self._handle)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.brst_free(self._handle)
            self._handle = None
