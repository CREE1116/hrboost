#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace hrboost {

static constexpr int HSTRIDE = 3;

struct SplitResult {
  bool valid = false;
  bool is_cat = false;
  int feature = -1;
  int best_bin = -1;
  float threshold = 0.0f;
  std::vector<uint8_t> cat_left_mask;
  int nan_child = 0;
  double gain = 0.0;
  double GL = 0.0, HL = 0.0;
  double GR = 0.0, HR = 0.0;
};

struct Node {
  bool is_leaf = true;
  double value = 0.0;
  SplitResult split;
  int left = -1;
  int right = -1;
};

struct BinCtx;

struct Tree {
  std::vector<Node> nodes;
  std::vector<std::vector<float>> cat_vals;
  std::vector<bool> is_cat;
  int B = 0;

  double predict_row(const float* x, int D) const;
  int get_leaf_index(const float* x, int D) const;
  int get_leaf_index_binned(int i, const BinCtx& ctx) const;
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

// POD 구조체 (ctypes 연동)
struct Params {
  double learning_rate = 0.1;
  double reg_lambda = 1.0;
  double subsample = 0.8;
  double colsample_bytree = 1.0;
  double min_child_weight = 0.1;
  double gamma = 0.0;
  double max_delta_step = 0.0;

  int n = 0;
  int D = 0;
  int n_estimators = 200;
  int max_depth = 6;
  int max_leaves = 64;
  int n_bins = 32;
  int random_state = 0;
  int num_classes = 1;
};

class HRBoost {
 public:
  void fit(const float* X, const float* y, const Params& params,
           const std::vector<int>& cat_features, const std::string& objective);
  void predict_proba(const float* X, int n, int D, double* out_p) const;
  void predict(const float* X, int n, int D, double* out_y) const; // 회귀 예측 추가

 private:
  std::vector<std::vector<Tree>> trees_;
  BinCtx ctx_;
  std::vector<double> base_scores_;
  int num_classes_ = 1;
  double lr_ = 0.1;
  std::string objective_ = "binary";
};

}  // namespace hrboost
