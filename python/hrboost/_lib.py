import ctypes
import os
import pathlib

def _load():
    here = pathlib.Path(__file__).resolve().parent
    root = here.parent.parent
    for name in ("libhrboost.dylib", "libhrboost.so"):
        # Check inside the package directory (for installed package)
        p_pkg = here / name
        if p_pkg.exists():
            return ctypes.CDLL(str(p_pkg))
        # Check in the project root (for local development)
        p_root = root / name
        if p_root.exists():
            return ctypes.CDLL(str(p_root))
    raise FileNotFoundError(
        "libhrboost not found — run `make` in the project root"
    )

_lib = _load()

_lib.hrboost_create.restype  = ctypes.c_void_p
_lib.hrboost_create.argtypes = []

_lib.hrboost_free.restype  = None
_lib.hrboost_free.argtypes = [ctypes.c_void_p]

_lib.hrboost_fit.restype  = None
_lib.hrboost_fit.argtypes = [
    ctypes.c_void_p,                      # 1. model handle
    ctypes.POINTER(ctypes.c_float),       # 2. X
    ctypes.POINTER(ctypes.c_float),       # 3. y (float target)
    ctypes.POINTER(ctypes.c_int),         # 4. cat_features ptr
    ctypes.c_char_p,                      # 5. objective
    ctypes.c_double,                      # 6. learning_rate
    ctypes.c_double,                      # 7. reg_lambda
    ctypes.c_double,                      # 8. subsample
    ctypes.c_double,                      # 9. colsample_bytree
    ctypes.c_double,                      # 10. min_child_weight
    ctypes.c_double,                      # 11. gamma
    ctypes.c_double,                      # 12. max_delta_step
    ctypes.c_int,                         # 13. n
    ctypes.c_int,                         # 14. D
    ctypes.c_int,                         # 15. n_estimators
    ctypes.c_int,                         # 16. max_depth
    ctypes.c_int,                         # 17. max_leaves
    ctypes.c_int,                         # 18. n_bins
    ctypes.c_int,                         # 19. cat_features_len
    ctypes.c_int,                         # 20. random_state
    ctypes.c_int                          # 21. num_classes
]

_lib.hrboost_predict_proba.restype  = None
_lib.hrboost_predict_proba.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),  # X
    ctypes.c_int,                    # n
    ctypes.c_int,                    # D
    ctypes.POINTER(ctypes.c_double), # out_p
]

_lib.hrboost_predict.restype  = None
_lib.hrboost_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),  # X
    ctypes.c_int,                    # n
    ctypes.c_int,                    # D
    ctypes.POINTER(ctypes.c_double), # out_y
]
