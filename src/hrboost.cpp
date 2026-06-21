#include "hrboost.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <tuple>
#include <unordered_map>

// OpenMP 병렬화를 위한 헤더 조건부 포함
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace hrboost {

static inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

static inline int route_sample(int i, const SplitResult& sp,
                               const BinCtx& ctx) {
  int f = sp.feature;
  uint8_t b = ctx.code[(size_t)i * ctx.D + f];
  if (b == ctx.B - 1) return sp.nan_child;
  if (sp.is_cat) {
    return (b < (int)sp.cat_left_mask.size() && sp.cat_left_mask[b] == 1) ? 0
                                                                          : 1;
  }
  return (b <= sp.best_bin) ? 0 : 1;
}

static inline double calc_dynamic_gain(double GL, double HL, double GR, double HR,
                                       double G_T, double H_T, double lam, double base) {
  static const double gamma_cohesion = []() {
    const char* env_reg = std::getenv("COHESION_REG");
    if (env_reg) {
      try {
        return std::stod(env_reg);
      } catch (...) {}
    }
    return 0.3; // 기본 최적 민감도 0.3 활성화
  }();

  if (gamma_cohesion <= 0.0) {
    return GL * GL / (HL + lam) + GR * GR / (HR + lam) - base;
  }

  double dL = GL / (HL + 1e-10);
  double dR = GR / (HR + 1e-10);
  double denom = std::abs(dL) + std::abs(dR);
  double cohesion = 1.0;
  if (denom > 1e-5) {
    cohesion = 1.0 - (std::abs(dL - dR) / denom);
  }

  double lam_dyn = lam * (1.0 + gamma_cohesion * cohesion);
  return GL * GL / (HL + lam_dyn) + GR * GR / (HR + lam_dyn) - base;
}

static inline double delta(double Ga, double Ha, double Gb, double Hb,
                           double lam) {
  double ha = std::max(Ha, 1e-10), hb = std::max(Hb, 1e-10);
  double Hab = ha + hb, Gab = Ga + Gb;
  double log_t = 0.0;
  if (lam < 1.0) {
    double val = (ha + lam) * (hb + lam) / (Hab + lam);
    if (val < 1.0) {
      log_t = 0.5 * std::log(val);
    }
  }
  return (Gab * Gab / (Hab + lam) - Ga * Ga / (ha + lam) -
          Gb * Gb / (hb + lam)) *
             0.5 +
         log_t;
}

struct BHCResult {
  double gain = -1.0;
  int best_bin = -1;
  std::vector<uint8_t> cat_left_mask;
  double Gl = 0.0, Hl = 0.0;
  double Gr = 0.0, Hr = 0.0;
  int nan_child = 1;
};

static BHCResult bhc_split_full(const double* G_in, const double* H_in,
                                const int* last_bin_in, int S, double G_T,
                                double H_T, double G_nan, double H_nan,
                                double lam, double min_h, int total_bins) {
  struct Cluster {
    double G;
    double H;
    int prev;
    int next;
    bool active;
  };
  std::vector<Cluster> clusters(S);
  std::vector<int> cluster_assignment(S);
  for (int i = 0; i < S; ++i) {
    clusters[i] = {G_in[i], std::max(H_in[i], 1e-10), i - 1, (i + 1 < S) ? i + 1 : -1, true};
    cluster_assignment[i] = i;
  }

  int cur_S = S;
  while (cur_S > 2) {
    double max_d = -1e30;
    int best_i = -1;
    int best_j = -1;

    for (int i = 0; i != -1; i = clusters[i].next) {
      int j = clusters[i].next;
      int steps = 0;
      while (j != -1 && steps < 3) {
        double d = delta(clusters[i].G, clusters[i].H, clusters[j].G, clusters[j].H, lam);
        if (d > max_d) {
          max_d = d;
          best_i = i;
          best_j = j;
        }
        j = clusters[j].next;
        steps++;
      }
    }
    if (best_i == -1 || best_j == -1) break;

    clusters[best_i].G += clusters[best_j].G;
    clusters[best_i].H += clusters[best_j].H;
    clusters[best_j].active = false;

    for (int k = 0; k < S; ++k) {
      if (cluster_assignment[k] == best_j) {
        cluster_assignment[k] = best_i;
      }
    }

    int prv = clusters[best_j].prev;
    int nxt = clusters[best_j].next;
    if (prv != -1) {
      clusters[prv].next = nxt;
    }
    if (nxt != -1) {
      clusters[nxt].prev = prv;
    }

    --cur_S;
  }

  int c1 = -1, c2 = -1;
  for (int i = 0; i < S; ++i) {
    if (clusters[i].active) {
      if (c1 == -1) c1 = i;
      else { c2 = i; break; }
    }
  }

  BHCResult best;
  best.gain = -1.0;

  if (c1 != -1 && c2 != -1) {
    double base = G_T * G_T / (H_T + lam);
    double GL_c1 = clusters[c1].G;
    double HL_c1 = clusters[c1].H;
    double GR_c2 = clusters[c2].G;
    double HR_c2 = clusters[c2].H;

    std::vector<uint8_t> best_mask(total_bins, 0);
    for (int k = 0; k < S; ++k) {
      if (cluster_assignment[k] == c1) {
        int bin_idx = last_bin_in[k];
        if (bin_idx >= 0 && bin_idx < total_bins) {
          best_mask[bin_idx] = 1;
        }
      }
    }

    {
      double GLv = GL_c1 + G_nan;
      double HLv = HL_c1 + H_nan;
      double GRv = GR_c2;
      double HRv = HR_c2;
      if (HLv >= min_h && HRv >= min_h) {
        double g = calc_dynamic_gain(GLv, HLv, GRv, HRv, G_T, H_T, lam, base);
        if (g > best.gain) {
          best = {g, 0, best_mask, GLv, HLv, GRv, HRv, 0};
        }
      }
    }

    {
      double GLv = GL_c1;
      double HLv = HL_c1;
      double GRv = GR_c2 + G_nan;
      double HRv = HR_c2 + H_nan;
      if (HLv >= min_h && HRv >= min_h) {
        double g = calc_dynamic_gain(GLv, HLv, GRv, HRv, G_T, H_T, lam, base);
        if (g > best.gain) {
          best = {g, 0, best_mask, GLv, HLv, GRv, HRv, 1};
        }
      }
    }

    // Iterative Partition Refinement
    if (best.gain >= 0.0) {
      bool improved = true;
      int max_iters = 10;
      while (improved && max_iters-- > 0) {
        improved = false;
        int best_move_idx = -1;
        double best_new_gain = best.gain;
        double best_GL = best.Gl;
        double best_HL = best.Hl;
        double best_GR = best.Gr;
        double best_HR = best.Hr;

        for (int i = 0; i < S; ++i) {
          int b = last_bin_in[i];
          if (b < 0 || b >= total_bins) continue;

          double Gi = G_in[i];
          double Hi = H_in[i];

          double tGL = best.Gl;
          double tHL = best.Hl;
          double tGR = best.Gr;
          double tHR = best.Hr;

          if (best.cat_left_mask[b] == 1) {
            tGL -= Gi;
            tHL -= Hi;
            tGR += Gi;
            tHR += Hi;
          } else {
            tGL += Gi;
            tHL += Hi;
            tGR -= Gi;
            tHR -= Hi;
          }

          if (tHL >= min_h && tHR >= min_h) {
            double g = calc_dynamic_gain(tGL, tHL, tGR, tHR, G_T, H_T, lam, base);
            if (g > best_new_gain + 1e-7) {
              best_new_gain = g;
              best_move_idx = b;
              best_GL = tGL;
              best_HL = tHL;
              best_GR = tGR;
              best_HR = tHR;
            }
          }
        }

        if (best_move_idx != -1) {
          best.gain = best_new_gain;
          best.cat_left_mask[best_move_idx] = 1 - best.cat_left_mask[best_move_idx];
          best.Gl = best_GL;
          best.Hl = best_HL;
          best.Gr = best_GR;
          best.Hr = best_HR;
          improved = true;
        }
      }
    }
  }
  return best;
}

double Tree::predict_row(const float* x, int D) const {
  int cur = 0;
  while (!nodes[cur].is_leaf) {
    const auto& sp = nodes[cur].split;
    float val = x[sp.feature];
    int child = 0;
    if (std::isnan(val)) {
      child = sp.nan_child;
    } else if (!sp.is_cat) {
      child = (val < sp.threshold) ? 0 : 1;
    } else {
      const auto& cv = cat_vals[sp.feature];
      auto it = std::lower_bound(cv.begin(), cv.end(), val);
      int b = -1;
      if (it != cv.end() && *it == val) {
        b = (int)(it - cv.begin());
      } else {
        b = (int)cv.size();
      }

      if (b < (int)sp.cat_left_mask.size()) {
        child = (sp.cat_left_mask[b] == 1) ? 0 : 1;
      } else {
        child = sp.nan_child;
      }
    }
    cur = (child == 0) ? nodes[cur].left : nodes[cur].right;
  }
  return nodes[cur].value;
}

int Tree::get_leaf_index(const float* x, int D) const {
  int cur = 0;
  while (!nodes[cur].is_leaf) {
    const auto& sp = nodes[cur].split;
    float val = x[sp.feature];
    int child = 0;
    if (std::isnan(val)) {
      child = sp.nan_child;
    } else if (!sp.is_cat) {
      child = (val < sp.threshold) ? 0 : 1;
    } else {
      const auto& cv = cat_vals[sp.feature];
      auto it = std::lower_bound(cv.begin(), cv.end(), val);
      int b = (it != cv.end() && *it == val) ? (int)(it - cv.begin())
                                             : (int)cv.size();

      if (b < (int)sp.cat_left_mask.size()) {
        child = (sp.cat_left_mask[b] == 1) ? 0 : 1;
      } else {
        child = sp.nan_child;
      }
    }
    cur = (child == 0) ? nodes[cur].left : nodes[cur].right;
  }
  return cur;
}

int Tree::get_leaf_index_binned(int i, const BinCtx& ctx) const {
  int cur = 0;
  while (!nodes[cur].is_leaf) {
    const auto& sp = nodes[cur].split;
    int f = sp.feature;
    uint8_t b = ctx.code[(size_t)i * ctx.D + f];
    int child = 0;
    if (b == ctx.B - 1) {
      child = sp.nan_child;
    } else if (sp.is_cat) {
      child = (b < (int)sp.cat_left_mask.size() && sp.cat_left_mask[b] == 1)
                  ? 0
                  : 1;
    } else {
      child = (b <= sp.best_bin) ? 0 : 1;
    }
    cur = (child == 0) ? nodes[cur].left : nodes[cur].right;
  }
  return cur;
}

// ── Binning ───────────────────────
static void build_bin_ctx(const float* X, const Params& params, BinCtx& ctx,
                          const std::vector<int>& cat_features) {
  ctx.D = params.D;
  ctx.n = params.n;
  ctx.is_cat.assign(params.D, false);
  for (int f : cat_features) ctx.is_cat[f] = true;
  ctx.edges.resize(params.D);
  ctx.ax_min.resize(params.D, 0);
  ctx.ax_range.resize(params.D, 0);
  ctx.cat_vals.resize(params.D);

  for (int f = 0; f < params.D; ++f) {
    if (ctx.is_cat[f]) {
      std::vector<float> cats;
      for (int i = 0; i < params.n; ++i) {
        float v = X[(size_t)i * params.D + f];
        if (!std::isnan(v)) {
          cats.push_back(v);
        }
      }
      std::sort(cats.begin(), cats.end());
      cats.erase(std::unique(cats.begin(), cats.end()), cats.end());
      ctx.cat_vals[f] = cats;
    }
  }

  int max_dyn_B = params.n_bins + 1;
  for (int f = 0; f < params.D; ++f) {
    if (ctx.is_cat[f]) {
      int C = (int)ctx.cat_vals[f].size();
      if (C + 2 > max_dyn_B) max_dyn_B = C + 2;
    }
  }
  ctx.B = max_dyn_B;
  ctx.code.resize((size_t)params.n * params.D, (uint8_t)(ctx.B - 1));

  for (int f = 0; f < params.D; ++f) {
    if (ctx.is_cat[f]) {
      std::unordered_map<float, int> cat_map;
      int C = (int)ctx.cat_vals[f].size();
      for (int k = 0; k < C; ++k) cat_map[ctx.cat_vals[f][k]] = k;

      for (int i = 0; i < params.n; ++i) {
        float v = X[(size_t)i * params.D + f];
        if (std::isnan(v)) {
          ctx.code[(size_t)i * params.D + f] = (uint8_t)(ctx.B - 1);
        } else if (cat_map.count(v)) {
          ctx.code[(size_t)i * params.D + f] = (uint8_t)cat_map[v];
        } else {
          ctx.code[(size_t)i * params.D + f] = (uint8_t)C;
        }
      }
    } else {
      std::vector<float> valid;
      valid.reserve(params.n);
      for (int i = 0; i < params.n; ++i) {
        float v = X[(size_t)i * params.D + f];
        if (!std::isnan(v)) valid.push_back(v);
      }
      if (valid.empty()) continue;
      std::sort(valid.begin(), valid.end());
      int nv = (int)valid.size();

      std::vector<float> uniques = valid;
      uniques.erase(std::unique(uniques.begin(), uniques.end()), uniques.end());
      int n_unique = (int)uniques.size();

      auto& edges = ctx.edges[f];
      edges.push_back(std::numeric_limits<float>::lowest());

      if (n_unique <= params.n_bins) {
        for (float val : uniques) {
          edges.push_back(val);
        }
      } else {
        std::vector<float> frequent_vals;
        int run_start = 0;
        float threshold_cnt = 0.05f * nv;
        for (int i = 1; i <= nv; ++i) {
          if (i == nv || valid[i] != valid[run_start]) {
            int run_len = i - run_start;
            if (run_len >= threshold_cnt) {
              frequent_vals.push_back(valid[run_start]);
            }
            run_start = i;
          }
        }

        std::vector<float> remainder;
        remainder.reserve(nv);
        for (float v : valid) {
          if (!std::binary_search(frequent_vals.begin(), frequent_vals.end(),
                                  v)) {
            remainder.push_back(v);
          }
        }

        for (float fv : frequent_vals) {
          edges.push_back(fv);
        }

        int remaining_bins = params.n_bins - (int)frequent_vals.size();
        if (!remainder.empty() && remaining_bins > 0) {
          int n_rem = (int)remainder.size();
          for (int k = 1; k < remaining_bins; ++k) {
            int pos = std::clamp((int)((double)k / remaining_bins * n_rem), 0,
                                 n_rem - 1);
            edges.push_back(remainder[pos]);
          }
        }
      }
      edges.push_back(std::numeric_limits<float>::max());
      std::sort(edges.begin() + 1, edges.end() - 1);
      edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

      ctx.ax_min[f] = valid.front();
      ctx.ax_range[f] =
          valid.back() > valid.front() ? valid.back() - valid.front() : 1.0f;
      int B_real = (int)edges.size() - 1;

      for (int i = 0; i < params.n; ++i) {
        float v = X[(size_t)i * params.D + f];
        if (std::isnan(v)) {
          ctx.code[(size_t)i * params.D + f] = (uint8_t)(ctx.B - 1);
          continue;
        }
        auto it = std::upper_bound(edges.begin() + 1, edges.end() - 1, v);
        int b = std::clamp((int)(it - (edges.begin() + 1)), 0, B_real - 1);
        ctx.code[(size_t)i * params.D + f] = (uint8_t)b;
      }
    }
  }
}

// ── Histogram ops
static void accumulate_hist(const std::vector<int>& rows, const BinCtx& ctx,
                            const double* g, const double* h,
                            std::vector<double>& out) {
  int D = ctx.D, B = ctx.B;
  size_t required_size = (size_t)D * B * HSTRIDE;
  if (out.size() != required_size) {
    out.resize(required_size);
  }
  std::fill(out.begin(), out.end(), 0.0);

  for (int i : rows) {
    const uint8_t* ci = ctx.code.data() + (size_t)i * D;
    for (int f = 0; f < D; ++f) {
      double* sl = out.data() + ((size_t)f * B + ci[f]) * HSTRIDE;
      sl[0] += g[i];
      sl[1] += h[i];
      sl[2] += 1.0;
    }
  }
}

static void subtract_hist(const std::vector<double>& parent,
                          const std::vector<double>& child,
                          std::vector<double>& sibling) {
  if (sibling.size() != parent.size()) sibling.resize(parent.size());
  for (size_t i = 0; i < parent.size(); ++i) sibling[i] = parent[i] - child[i];
}

static SplitResult eval_split(const std::vector<double>& hist,
                              const BinCtx& ctx, double G_T, double H_T,
                              double lam, double min_h, double gamma,
                              const std::vector<int>& feat_subset,
                              size_t node_size) {
  int D = ctx.D, B = ctx.B;
  SplitResult best;

  auto process_feat = [&](int f) {
    const double* fb = hist.data() + (size_t)f * B * HSTRIDE;
    bool is_cat_f = ctx.is_cat[f];
    double G_nan = fb[(B - 1) * HSTRIDE], H_nan = fb[(B - 1) * HSTRIDE + 1];

    if (is_cat_f) {
      struct CatInfo {
        int bin_idx;
        double G;
        double H;
        double score;
      };
      std::vector<CatInfo> cats;
      int active_bins_limit = (int)ctx.cat_vals[f].size() + 1;
      double min_cat_cnt = std::max(2.0, std::min(node_size * 0.01, ctx.n * 0.0005));
      const char* env_mcc = std::getenv("MIN_CAT_COUNT");
      if (env_mcc) {
        try {
          min_cat_cnt = std::stod(env_mcc);
        } catch (...) {}
      }
      for (int b = 0; b < active_bins_limit; ++b) {
        double hb = fb[b * HSTRIDE + 1];
        double cnt = fb[b * HSTRIDE + 2];
        if (hb < 1e-12 || cnt < min_cat_cnt) continue;
        double gb = fb[b * HSTRIDE];
        cats.push_back({b, gb, hb, gb / (hb + lam)});
      }
      int S = (int)cats.size();
      if (S < 2) return;

      std::sort(cats.begin(), cats.end(), [](const CatInfo& a, const CatInfo& b) {
        return a.score < b.score;
      });

      std::vector<double> G_cat(S);
      std::vector<double> H_cat(S);
      std::vector<int> last_bin_cat(S);
      for (int i = 0; i < S; ++i) {
        G_cat[i] = cats[i].G;
        H_cat[i] = cats[i].H;
        last_bin_cat[i] = cats[i].bin_idx;
      }
      BHCResult best_r = bhc_split_full(
          G_cat.data(), H_cat.data(), last_bin_cat.data(), S, G_T, H_T,
          G_nan, H_nan, lam, min_h, B);

      if (best_r.gain <= gamma || best_r.gain <= best.gain) return;

      std::vector<uint8_t> left_mask = std::move(best_r.cat_left_mask);

      best = {true,
              true,
              f,
              best_r.best_bin,
              0.0f,
              std::move(left_mask),
              best_r.nan_child,
              best_r.gain,
              best_r.Gl,
              best_r.Hl,
              best_r.Gr,
              best_r.Hr};
    } else {
      double Gl_curr = 0.0, Hl_curr = 0.0;
      double G_clean = G_T - G_nan, H_clean = H_T - H_nan;
      double base = G_T * G_T / (H_T + lam);
      int B_real = (int)ctx.edges[f].size() - 1;

      for (int b = 0; b < B_real; ++b) {
        Gl_curr += fb[b * HSTRIDE];
        Hl_curr += fb[b * HSTRIDE + 1];

        {
          double GLv = Gl_curr + G_nan;
          double HLv = Hl_curr + H_nan;
          double GRv = G_clean - Gl_curr;
          double HRv = H_clean - Hl_curr;
          if (HLv >= min_h && HRv >= min_h) {
            double g = calc_dynamic_gain(GLv, HLv, GRv, HRv, G_T, H_T, lam, base);
            if (g > gamma && g > best.gain) {
              const auto& edg = ctx.edges[f];
              float thr = (b + 1 < (int)edg.size())
                              ? edg[b + 1]
                              : ctx.ax_min[f] + ctx.ax_range[f];
              best = {true, false, f, b, thr, {}, 0, g, GLv, HLv, GRv, HRv};
            }
          }
        }
        if (H_nan >= 1e-12) {
          double GLv = Gl_curr;
          double HLv = Hl_curr;
          double GRv = G_clean - Gl_curr + G_nan;
          double HRv = H_clean - Hl_curr + H_nan;
          if (HLv >= min_h && HRv >= min_h) {
            double g = calc_dynamic_gain(GLv, HLv, GRv, HRv, G_T, H_T, lam, base);
            if (g > gamma && g > best.gain) {
              const auto& edg = ctx.edges[f];
              float thr = (b + 1 < (int)edg.size())
                              ? edg[b + 1]
                              : ctx.ax_min[f] + ctx.ax_range[f];
              best = {true, false, f, b, thr, {}, 1, g, GLv, HLv, GRv, HRv};
            }
          }
        }
      }
    }
  };

  if (feat_subset.empty()) {
    for (int f = 0; f < ctx.D; ++f) process_feat(f);
  } else {
    for (int f : feat_subset) process_feat(f);
  }
  return best;
}

// ── Tree builder
static Tree build_tree(const BinCtx& ctx, const double* g, const double* h,
                       const std::vector<int>& sub,
                       const Params& params, std::mt19937& rng) {
  Tree tree;
  int max_nodes = 2 * params.max_leaves + 4;
  tree.nodes.resize(max_nodes);
  tree.cat_vals = ctx.cat_vals;
  tree.is_cat = ctx.is_cat;
  tree.B = ctx.B;
  double lam = params.reg_lambda;
  double min_h = params.min_child_weight;
  double gamma = params.gamma;

  std::vector<int> feat_subset;
  if (params.colsample_bytree < 1.0) {
    std::vector<int> all(params.D);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    int ns = std::max(1, (int)(params.D * params.colsample_bytree));
    feat_subset.assign(all.begin(), all.begin() + ns);
    std::sort(feat_subset.begin(), feat_subset.end());
  }

  struct NodeState {
    std::vector<int> rows;
    std::vector<double> hist;
    double G_T = 0.0, H_T = 0.0;
    int depth = 0;
  };
  std::vector<NodeState> nstate(max_nodes);
  nstate[0].rows = sub;
  nstate[0].depth = 0;
  accumulate_hist(sub, ctx, g, h, nstate[0].hist);
  double G_root = 0.0, H_root = 0.0;
  for (int i : sub) {
    G_root += g[i];
    H_root += h[i];
  }
  nstate[0].G_T = G_root;
  nstate[0].H_T = H_root;

  double root_val = -G_root / (H_root + lam);
  if (params.max_delta_step > 0.0)
    root_val =
        std::clamp(root_val, -params.max_delta_step, params.max_delta_step);
  tree.nodes[0].value = root_val;

  using PQ = std::priority_queue<std::pair<double, int>>;
  PQ frontier;
  auto enqueue = [&](int t) {
    if ((int)nstate[t].rows.size() < 2) return;
    auto sp = eval_split(nstate[t].hist, ctx, nstate[t].G_T, nstate[t].H_T, lam,
                         min_h, gamma, feat_subset, nstate[t].rows.size());
    if (!sp.valid || sp.gain <= gamma) return;
    tree.nodes[t].split = sp;
    frontier.push({sp.gain, t});
  };
  enqueue(0);

  int next_id = 1;
  int splits_left = params.max_leaves - 1;
  while (splits_left > 0 && !frontier.empty()) {
    auto [gain, t] = frontier.top();
    frontier.pop();
    if (!tree.nodes[t].is_leaf) continue;
    auto& sp = tree.nodes[t].split;
    if (!sp.valid || sp.gain <= 0) continue;
    if (nstate[t].depth >= params.max_depth) continue;

    int tl = next_id++, tr = next_id++;
    if (next_id > max_nodes) {
      next_id -= 2;
      break;
    }

    for (int i : nstate[t].rows) {
      (route_sample(i, sp, ctx) == 0 ? nstate[tl] : nstate[tr])
          .rows.push_back(i);
    }
    if (nstate[tl].rows.empty() || nstate[tr].rows.empty()) {
      next_id -= 2;
      nstate[tl].rows.clear();
      nstate[tr].rows.clear();
      continue;
    }

    nstate[tl].depth = nstate[t].depth + 1;
    nstate[tr].depth = nstate[t].depth + 1;
    bool ls = nstate[tl].rows.size() <= nstate[tr].rows.size();
    int ts = ls ? tl : tr;
    int tl2 = ls ? tr : tl;
    accumulate_hist(nstate[ts].rows, ctx, g, h, nstate[ts].hist);
    subtract_hist(nstate[t].hist, nstate[ts].hist, nstate[tl2].hist);

    for (int i : nstate[tl].rows) {
      nstate[tl].G_T += g[i];
      nstate[tl].H_T += h[i];
    }
    for (int i : nstate[tr].rows) {
      nstate[tr].G_T += g[i];
      nstate[tr].H_T += h[i];
    }

    if (sp.valid && !sp.is_cat) {
      const auto& edg = ctx.edges[sp.feature];
      int ei = sp.best_bin + 1;
      sp.threshold = (ei < (int)edg.size())
                         ? edg[ei]
                         : ctx.ax_min[sp.feature] + ctx.ax_range[sp.feature];
    }

    double l_val = -nstate[tl].G_T / (nstate[tl].H_T + lam);
    double r_val = -nstate[tr].G_T / (nstate[tr].H_T + lam);
    if (params.max_delta_step > 0.0) {
      l_val = std::clamp(l_val, -params.max_delta_step, params.max_delta_step);
      r_val = std::clamp(r_val, -params.max_delta_step, params.max_delta_step);
    }
    tree.nodes[tl].value = l_val;
    tree.nodes[tr].value = r_val;
    tree.nodes[t].is_leaf = false;
    tree.nodes[t].left = tl;
    tree.nodes[t].right = tr;

    --splits_left;
    nstate[t].hist.clear();
    nstate[t].hist.shrink_to_fit();
    nstate[t].rows.clear();
    enqueue(tl);
    enqueue(tr);
  }
  tree.nodes.resize(next_id > 0 ? next_id : 1);
  return tree;
}

// ── fit / predict
void HRBoost::fit(const float* X, const float* y, const Params& params,
                  const std::vector<int>& cat_features,
                  const std::string& objective) {
  lr_ = params.learning_rate;
  objective_ = objective;
  build_bin_ctx(X, params, ctx_, cat_features);
  std::mt19937 rng(params.random_state);

  auto is_verbose = []() {
    const char* env_v1 = std::getenv("HRBOOST_VERBOSE");
    const char* env_v2 = std::getenv("SELFCDSB_VERBOSE");
    const char* env_v = env_v1 ? env_v1 : env_v2;
    if (env_v &&
        (std::strcmp(env_v, "0") == 0 || std::strcmp(env_v, "false") == 0 ||
         std::strcmp(env_v, "OFF") == 0)) {
      return false;
    }
    return true;
  };

  if (objective_ == "regression") {
    num_classes_ = 1;
    base_scores_.resize(1);
    double sum_y = 0.0;
    for (int i = 0; i < params.n; ++i) sum_y += y[i];
    base_scores_[0] = sum_y / params.n;

    std::vector<double> F(params.n, base_scores_[0]);
    std::vector<double> gvec(params.n), hvec(params.n);
    trees_.clear();
    trees_.resize(params.n_estimators, std::vector<Tree>(1));

    for (int t = 0; t < params.n_estimators; ++t) {
      for (int i = 0; i < params.n; ++i) {
        gvec[i] = F[i] - y[i];
        hvec[i] = 1.0;
      }
      std::vector<int> sub;
      if (params.subsample < 1.0) {
        sub.resize(params.n);
        std::iota(sub.begin(), sub.end(), 0);
        std::shuffle(sub.begin(), sub.end(), rng);
        sub.resize((int)(params.n * params.subsample));
      } else {
        sub.resize(params.n);
        std::iota(sub.begin(), sub.end(), 0);
      }

      auto start_tree = std::chrono::high_resolution_clock::now();
      Tree tree = build_tree(ctx_, gvec.data(), hvec.data(), sub,
                             params, rng);
      auto end_tree = std::chrono::high_resolution_clock::now();
      double t_tree =
          std::chrono::duration<double, std::milli>(end_tree - start_tree)
              .count();

      if (is_verbose() &&
          (t == 0 || t == params.n_estimators - 1 || (t > 0 && t % 50 == 0))) {
        std::cout << "[Tree " << t << "] t_tree=" << t_tree << "ms" << std::endl;
      }

      for (int i = 0; i < params.n; ++i) {
        F[i] += lr_ * tree.predict_row(X + (size_t)i * params.D, params.D);
      }
      trees_[t][0] = std::move(tree);
    }
  } else if (objective_ == "binary") {
    num_classes_ = 1;
    base_scores_.resize(1);
    int n_pos = 0;
    for (int i = 0; i < params.n; ++i) {
      n_pos += (y[i] > 0.5f ? 1 : 0);
    }
    double p0 = std::clamp((double)n_pos / params.n, 1e-6, 1.0 - 1e-6);
    base_scores_[0] = std::log(p0 / (1.0 - p0));

    std::vector<double> F(params.n, base_scores_[0]);
    std::vector<double> gvec(params.n), hvec(params.n);
    trees_.clear();
    trees_.resize(params.n_estimators, std::vector<Tree>(1));

    for (int t = 0; t < params.n_estimators; ++t) {
      for (int i = 0; i < params.n; ++i) {
        double p = sigmoid(F[i]);
        gvec[i] = p - y[i];
        hvec[i] = p * (1.0 - p);
      }
      std::vector<int> sub;
      if (params.subsample < 1.0) {
        sub.resize(params.n);
        std::iota(sub.begin(), sub.end(), 0);
        std::shuffle(sub.begin(), sub.end(), rng);
        sub.resize((int)(params.n * params.subsample));
      } else {
        sub.resize(params.n);
        std::iota(sub.begin(), sub.end(), 0);
      }

      auto start_tree = std::chrono::high_resolution_clock::now();
      Tree tree = build_tree(ctx_, gvec.data(), hvec.data(), sub,
                             params, rng);
      auto end_tree = std::chrono::high_resolution_clock::now();
      double t_tree =
          std::chrono::duration<double, std::milli>(end_tree - start_tree)
              .count();

      if (is_verbose() &&
          (t == 0 || t == params.n_estimators - 1 || (t > 0 && t % 50 == 0))) {
        std::cout << "[Tree " << t << "] t_tree=" << t_tree << "ms" << std::endl;
      }

      for (int i = 0; i < params.n; ++i) {
        F[i] += lr_ * tree.predict_row(X + (size_t)i * params.D, params.D);
      }
      trees_[t][0] = std::move(tree);
    }
  } else {
    num_classes_ = params.num_classes;
    base_scores_.assign(num_classes_, 0.0);
    std::vector<int> class_counts(num_classes_, 0);
    for (int i = 0; i < params.n; ++i) {
      int yi = (int)std::round(y[i]);
      if (yi >= 0 && yi < num_classes_) class_counts[yi]++;
    }
    for (int c = 0; c < num_classes_; ++c) {
      double pc = (double)class_counts[c] / params.n;
      base_scores_[c] = std::log(std::max(pc, 1e-6));
    }

    std::vector<std::vector<double>> F(params.n,
                                       std::vector<double>(num_classes_));
    for (int i = 0; i < params.n; ++i) {
      F[i] = base_scores_;
    }
    std::vector<double> gvec(params.n), hvec(params.n);
    trees_.clear();
    trees_.resize(params.n_estimators, std::vector<Tree>(num_classes_));

    for (int t = 0; t < params.n_estimators; ++t) {
      auto start_t = std::chrono::high_resolution_clock::now();
      for (int c = 0; c < num_classes_; ++c) {
        for (int i = 0; i < params.n; ++i) {
          double max_f = F[i][0];
          for (int cc = 1; cc < num_classes_; ++cc) max_f = std::max(max_f, F[i][cc]);
          double sum_exp = 0.0;
          std::vector<double> exp_F(num_classes_);
          for (int cc = 0; cc < num_classes_; ++cc) {
            exp_F[cc] = std::exp(F[i][cc] - max_f);
            sum_exp += exp_F[cc];
          }
          double p = exp_F[c] / sum_exp;
          int yi = (int)std::round(y[i]);
          double y_onehot = (yi == c) ? 1.0 : 0.0;
          gvec[i] = p - y_onehot;
          hvec[i] = 2.0 * p * (1.0 - p);
        }

        std::vector<int> sub;
        if (params.subsample < 1.0) {
          sub.resize(params.n);
          std::iota(sub.begin(), sub.end(), 0);
          std::shuffle(sub.begin(), sub.end(), rng);
          sub.resize((int)(params.n * params.subsample));
        } else {
          sub.resize(params.n);
          std::iota(sub.begin(), sub.end(), 0);
        }

        Tree tree = build_tree(ctx_, gvec.data(), hvec.data(), sub, params, rng);

        for (int i = 0; i < params.n; ++i) {
          F[i][c] += lr_ * tree.predict_row(X + (size_t)i * params.D, params.D);
        }
        trees_[t][c] = std::move(tree);
      }
      auto end_t = std::chrono::high_resolution_clock::now();
      double t_round = std::chrono::duration<double, std::milli>(end_t - start_t).count();
      if (is_verbose() &&
          (t == 0 || t == params.n_estimators - 1 || (t > 0 && t % 50 == 0))) {
        std::cout << "[Multiclass Round " << t << "] t_round=" << t_round << "ms" << std::endl;
      }
    }
  }
}

void HRBoost::predict_proba(const float* X, int n, int D,
                             double* out_p) const {
  if (objective_ == "binary") {
    std::vector<double> F(n, base_scores_[0]);
    for (const auto& round_trees : trees_) {
      for (int i = 0; i < n; ++i) {
        F[i] += lr_ * round_trees[0].predict_row(X + (size_t)i * D, D);
      }
    }
    for (int i = 0; i < n; ++i) {
      out_p[i] = sigmoid(F[i]);
    }
  } else if (objective_ == "regression") {
    // Regression has no predict_proba, but we fallback to identity predict
    predict(X, n, D, out_p);
  } else {
    std::vector<std::vector<double>> F(n, base_scores_);
    for (const auto& round_trees : trees_) {
      for (int c = 0; c < num_classes_; ++c) {
        for (int i = 0; i < n; ++i) {
          F[i][c] += lr_ * round_trees[c].predict_row(X + (size_t)i * D, D);
        }
      }
    }
    for (int i = 0; i < n; ++i) {
      double max_f = F[i][0];
      for (int c = 1; c < num_classes_; ++c) max_f = std::max(max_f, F[i][c]);
      double sum_exp = 0.0;
      std::vector<double> exp_F(num_classes_);
      for (int c = 0; c < num_classes_; ++c) {
        exp_F[c] = std::exp(F[i][c] - max_f);
        sum_exp += exp_F[c];
      }
      for (int c = 0; c < num_classes_; ++c) {
        out_p[i * num_classes_ + c] = exp_F[c] / sum_exp;
      }
    }
  }
}

void HRBoost::predict(const float* X, int n, int D, double* out_y) const {
  if (num_classes_ == 1) {
    double base = base_scores_.empty() ? 0.0 : base_scores_[0];
    for (int i = 0; i < n; ++i) {
      out_y[i] = base;
    }
    for (const auto& round_trees : trees_) {
      for (int i = 0; i < n; ++i) {
        out_y[i] += lr_ * round_trees[0].predict_row(X + (size_t)i * D, D);
      }
    }
  } else {
    for (int i = 0; i < n * num_classes_; ++i) {
      out_y[i] = base_scores_[i % num_classes_];
    }
    for (const auto& round_trees : trees_) {
      for (int c = 0; c < num_classes_; ++c) {
        for (int i = 0; i < n; ++i) {
          out_y[i * num_classes_ + c] += lr_ * round_trees[c].predict_row(X + (size_t)i * D, D);
        }
      }
    }
  }
}

}  // namespace hrboost
