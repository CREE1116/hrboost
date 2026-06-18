#include "capi.h"
#include "brstboost.h"
#include <new>

using namespace brst;

extern "C" {

BRSTHandle brst_create() { return new(std::nothrow) BRSTBoost(); }
void brst_free(BRSTHandle h) { delete static_cast<BRSTBoost*>(h); }

void brst_fit(BRSTHandle h,
              const float* X, int n, int D, const int* y,
              int n_estimators, double learning_rate,
              int max_depth, int max_leaves,
              double reg_lambda, double bhc_lam,
              double subsample, double colsample_bytree,
              int n_bins, double min_child_weight, double gamma,
              int max_k,
              const int* cat_features, int n_cat_features,
              int random_state) {
    Params p;
    p.n_estimators=n_estimators; p.learning_rate=learning_rate;
    p.max_depth=max_depth; p.max_leaves=max_leaves;
    p.reg_lambda=reg_lambda; p.bhc_lam=bhc_lam;
    p.subsample=subsample; p.colsample_bytree=colsample_bytree;
    p.n_bins=n_bins; p.min_child_weight=min_child_weight;
    p.gamma=gamma; p.max_k=max_k;
    p.cat_features.assign(cat_features, cat_features+n_cat_features);
    p.random_state=random_state;
    static_cast<BRSTBoost*>(h)->fit(X, n, D, y, p);
}

void brst_predict_proba(BRSTHandle h, const float* X, int n, int D, double* out_p1) {
    static_cast<BRSTBoost*>(h)->predict_proba(X, n, D, out_p1);
}

double brst_avg_k(BRSTHandle h) { return static_cast<BRSTBoost*>(h)->avg_k(); }

} // extern "C"
