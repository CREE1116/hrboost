#ifndef CAPI_HRBOOST_H
#define CAPI_HRBOOST_H

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

EXPORT void* hrboost_create();
EXPORT void hrboost_free(void* model);

EXPORT void hrboost_fit(void* model, const float* X, const float* y,
                        const int* cat_features,
                        const char* objective, double learning_rate,
                        double reg_lambda, double subsample,
                        double colsample_bytree, double min_child_weight,
                        double gamma, double max_delta_step,
                        int n, int D,
                        int n_estimators, int max_depth, int max_leaves,
                        int n_bins, int cat_features_len, int random_state,
                        int num_classes);

EXPORT void hrboost_predict_proba(void* model, const float* X, int n, int D,
                                  double* out_p);

EXPORT void hrboost_predict(void* model, const float* X, int n, int D,
                            double* out_y);
}

#endif  // CAPI_HRBOOST_H
