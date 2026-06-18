#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>

namespace brst {

static constexpr int HSTRIDE = 3;

struct SplitResult {
    bool valid = false;
    bool is_cat = false;
    double gain = 0;
    int feature = -1;
    int nan_child = 1;
    int best_bin = -1;
    float threshold = 0;
    uint64_t cat_left_mask = 0;
    double G_left = 0, H_left = 0;
    double G_right = 0, H_right = 0;
};

struct Node {
    bool is_leaf = true;
    double value = 0;
    SplitResult split;
    int left = -1, right = -1;
};

struct Tree {
    std::vector<Node> nodes;
    std::vector<std::vector<float>> cat_vals;
    std::vector<bool> is_cat;
    int B = 0;
    double predict_row(const float* x, int D) const;
};

struct BinCtx {
    int B = 32;
    int D = 0;
    int D_cat = 0;
    std::vector<bool> is_cat;
    std::vector<std::vector<float>> edges;
    std::vector<float> ax_min, ax_range;
    std::vector<uint8_t> code;
    std::vector<std::vector<float>> cat_vals;
    int n = 0;
};

struct Params {
    int n_estimators = 200;
    double learning_rate = 0.1;
    int max_depth = 6;
    int max_leaves = 64;
    double reg_lambda = 1.0;
    double bhc_lam = 0.0;
    double subsample = 0.8;
    double colsample_bytree = 1.0;
    int n_bins = 32;
    double min_child_weight = 1.0;
    double gamma = 0.0;
    int max_k = 2;
    std::vector<int> cat_features;
    int random_state = 0;
};

class BRSTBoost {
public:
    void fit(const float* X, int n, int D, const int* y, Params params);
    void predict_proba(const float* X, int n, int D, double* out_p1) const;
    double avg_k() const;
private:
    std::vector<Tree> trees_;
    BinCtx ctx_;
    double base_score_ = 0;
    double lr_ = 0.1;
    double lam_ = 1.0;
    std::vector<int> split_ks_;
};

} // namespace brst
