#include "capi_hrboost.h"

#include <cstring>
#include <string>
#include <vector>

#include "hrboost.h"

extern "C" {

void* hrboost_create() { return new hrboost::HRBoost(); }

void hrboost_free(void* model) {
  if (model) {
    delete static_cast<hrboost::HRBoost*>(model);
  }
}

void hrboost_fit(void* model, const float* X, const float* y,
                  const int* cat_features,
                  const char* objective, double learning_rate,
                  double reg_lambda, double subsample,
                  double colsample_bytree, double min_child_weight,
                  double gamma, double max_delta_step,
                  int n, int D,
                  int n_estimators, int max_depth, int max_leaves,
                  int n_bins, int cat_features_len, int random_state,
                  int num_classes) {
  hrboost::Params params;
  params.learning_rate = learning_rate;
  params.reg_lambda = reg_lambda;
  params.subsample = subsample;
  params.colsample_bytree = colsample_bytree;
  params.min_child_weight = min_child_weight;
  params.gamma = gamma;
  params.max_delta_step = max_delta_step;

  params.n = n;
  params.D = D;
  params.n_estimators = n_estimators;
  params.max_depth = max_depth;
  params.max_leaves = max_leaves;
  params.n_bins = n_bins;
  params.random_state = random_state;
  params.num_classes = num_classes;

  std::string obj_str = objective ? objective : "binary";
  std::vector<int> cats;
  if (cat_features && cat_features_len > 0) {
    cats.assign(cat_features, cat_features + cat_features_len);
  }

  auto* hrboost_model = static_cast<hrboost::HRBoost*>(model);
  hrboost_model->fit(X, y, params, cats, obj_str);
}

void hrboost_predict_proba(void* model, const float* X, int n, int D,
                            double* out_p) {
  auto* hrboost_model = static_cast<const hrboost::HRBoost*>(model);
  hrboost_model->predict_proba(X, n, D, out_p);
}

void hrboost_predict(void* model, const float* X, int n, int D,
                      double* out_y) {
  auto* hrboost_model = static_cast<const hrboost::HRBoost*>(model);
  hrboost_model->predict(X, n, D, out_y);
}
}
