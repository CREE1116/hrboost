#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef void* BRSTHandle;

BRSTHandle brst_create();
void       brst_free(BRSTHandle h);

void brst_fit(BRSTHandle h,
              const float* X, int n, int D,
              const int*   y,
              int    n_estimators,
              double learning_rate,
              int    max_depth,
              int    max_leaves,
              double reg_lambda,
              double bhc_lam,
              double subsample,
              double colsample_bytree,
              int    n_bins,
              double min_child_weight,
              double gamma,
              int    max_k,
              const int* cat_features,
              int    n_cat_features,
              int    random_state);

void brst_predict_proba(BRSTHandle h,
                         const float* X, int n, int D,
                         double* out_p1);

double brst_avg_k(BRSTHandle h);

#ifdef __cplusplus
}
#endif
