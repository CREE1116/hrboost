"""ctypes bindings to libbrstboost.dylib / libbrstboost.so"""
import ctypes, pathlib

def _load():
    here = pathlib.Path(__file__).resolve().parent
    root = here.parent.parent
    for name in ("libbrstboost.dylib", "libbrstboost.so"):
        p = root / name
        if p.exists():
            return ctypes.CDLL(str(p))
    raise FileNotFoundError(
        "libbrstboost not found — run `make` in the brstboost project root")

_lib = _load()

_lib.brst_create.restype  = ctypes.c_void_p
_lib.brst_create.argtypes = []

_lib.brst_free.restype  = None
_lib.brst_free.argtypes = [ctypes.c_void_p]

_lib.brst_fit.restype  = None
_lib.brst_fit.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),  # X
    ctypes.c_int,                    # n
    ctypes.c_int,                    # D
    ctypes.POINTER(ctypes.c_int),    # y
    ctypes.c_int,                    # n_estimators
    ctypes.c_double,                 # learning_rate
    ctypes.c_int,                    # max_depth
    ctypes.c_int,                    # max_leaves
    ctypes.c_double,                 # reg_lambda
    ctypes.c_double,                 # bhc_lam
    ctypes.c_double,                 # subsample
    ctypes.c_double,                 # colsample_bytree
    ctypes.c_int,                    # n_bins
    ctypes.c_double,                 # min_child_weight
    ctypes.c_double,                 # gamma
    ctypes.c_int,                    # max_k
    ctypes.c_double,                 # lambda_depth_decay
    ctypes.POINTER(ctypes.c_int),    # cat_features
    ctypes.c_int,                    # n_cat_features
    ctypes.c_int,                    # random_state
]

_lib.brst_predict_proba.restype  = None
_lib.brst_predict_proba.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]

_lib.brst_avg_k.restype  = ctypes.c_double
_lib.brst_avg_k.argtypes = [ctypes.c_void_p]
