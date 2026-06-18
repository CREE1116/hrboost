#include "brstboost.h"
#include <unordered_map>
#include <queue>
#include <cstring>

namespace brst {

static inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
template<class T> static inline T clamp(T v, T lo, T hi) { return v<lo?lo:(v>hi?hi:v); }

// ── BHC: Bayesian merge criterion ─────────────────────────────────────────────

static inline double delta(double Ga, double Ha, double Gb, double Hb, double lam) {
    double Hab = Ha+Hb, Gab = Ga+Gb;
    return (Gab*Gab/(Hab+lam) - Ga*Ga/(Ha+lam) - Gb*Gb/(Hb+lam)) * 0.5
           + 0.5 * std::log((Ha+lam)*(Hb+lam)/(Hab+lam));
}

struct BHCResult {
    double gain = -1;
    int    best_bin = -1;
    double Gl = 0, Hl = 0;
    double Gr = 0, Hr = 0;
    int    nan_child = 1;
};

static BHCResult bhc_split(
    double* G, double* H, int* cnt, int* last_bin, int S,
    double G_T, double H_T, double G_nan, double H_nan,
    double lam, double min_h, int max_k)
{
    while (S > max_k) {
        double best_d = 0.0; int best_k = -1;
        for (int k = 0; k+1 < S; ++k) {
            double d = delta(G[k], H[k], G[k+1], H[k+1], lam);
            if (d > best_d) { best_d = d; best_k = k; }
        }
        if (best_k < 0) break;
        G[best_k] += G[best_k+1]; H[best_k] += H[best_k+1];
        cnt[best_k] += cnt[best_k+1]; last_bin[best_k] = last_bin[best_k+1];
        for (int s = best_k+1; s < S-1; ++s) {
            G[s]=G[s+1]; H[s]=H[s+1]; cnt[s]=cnt[s+1]; last_bin[s]=last_bin[s+1];
        }
        --S;
    }

    double G_clean = G_T - G_nan, H_clean = H_T - H_nan;
    double base = G_clean * G_clean / (H_clean + lam);
    double Gl = 0, Hl = 0;
    BHCResult best;

    for (int s = 0; s < S-1; ++s) {
        Gl += G[s]; Hl += H[s];
        {   // NaN → left
            double GLv=Gl+G_nan, HLv=Hl+H_nan, GRv=G_clean-Gl, HRv=H_clean-Hl;
            if (HLv>=min_h && HRv>=min_h) {
                double gain = GLv*GLv/(HLv+lam)+GRv*GRv/(HRv+lam)-base;
                if (gain>best.gain) { best.gain=gain; best.best_bin=last_bin[s];
                    best.Gl=GLv; best.Hl=HLv; best.Gr=GRv; best.Hr=HRv; best.nan_child=0; }
            }
        }
        {   // NaN → right
            double GLv=Gl, HLv=Hl, GRv=G_clean-Gl+G_nan, HRv=H_clean-Hl+H_nan;
            if (HLv>=min_h && HRv>=min_h) {
                double gain = GLv*GLv/(HLv+lam)+GRv*GRv/(HRv+lam)-base;
                if (gain>best.gain) { best.gain=gain; best.best_bin=last_bin[s];
                    best.Gl=GLv; best.Hl=HLv; best.Gr=GRv; best.Hr=HRv; best.nan_child=1; }
            }
        }
    }
    return best;
}

// ── Best split ────────────────────────────────────────────────────────────────

static SplitResult eval_split(
    const std::vector<float>& hist,
    const BinCtx& ctx,
    double G_T, double H_T,
    double lam, double bhc_lam, double min_h, int max_k,
    const std::vector<int>& feat_subset)
{
    int D = ctx.D, B = ctx.B;
    std::vector<double> G(B), H_v(B);
    std::vector<int> cnt(B), last_bin(B);
    SplitResult best;
    double eff_lam = (bhc_lam > 0) ? bhc_lam : lam;
    std::vector<int> cat_order(B);

    auto process_feat = [&](int f) {
        const float* fbuf = hist.data() + (size_t)f * B * HSTRIDE;
        bool is_cat_feat = ctx.is_cat[f];
        double G_nan = fbuf[(B-1)*HSTRIDE+0], H_nan = fbuf[(B-1)*HSTRIDE+1];
        int S = 0;
        for (int b = 0; b < B-1; ++b) {
            double hb = fbuf[b*HSTRIDE+1];
            if (hb < 1e-12) continue;
            G[S]=fbuf[b*HSTRIDE+0]; H_v[S]=hb;
            cnt[S]=(int)fbuf[b*HSTRIDE+2]; last_bin[S]=b; ++S;
        }
        if (S < 2) return;

        if (is_cat_feat) {
            cat_order.resize(S);
            std::iota(cat_order.begin(), cat_order.begin()+S, 0);
            std::sort(cat_order.begin(), cat_order.begin()+S,
                      [&](int a, int b_){ return G[a]/(cnt[a]+1e-9) < G[b_]/(cnt[b_]+1e-9); });
            std::vector<double> Gs(S), Hs(S);
            std::vector<int> cnts(S), orig_bins(S);
            for (int i = 0; i < S; ++i) {
                int si = cat_order[i];
                Gs[i]=G[si]; Hs[i]=H_v[si]; cnts[i]=cnt[si]; orig_bins[i]=last_bin[si];
                last_bin[i]=i;
            }
            auto r = bhc_split(Gs.data(), Hs.data(), cnts.data(), last_bin.data(), S,
                               G_T, H_T, G_nan, H_nan, eff_lam, min_h, max_k);
            if (r.gain <= 0 || r.best_bin < 0 || r.gain <= best.gain) return;
            uint64_t mask = 0;
            for (int i = 0; i <= r.best_bin; ++i) mask |= (uint64_t(1) << orig_bins[i]);
            best = {true, true, r.gain, f, r.nan_child, r.best_bin, 0, mask,
                    r.Gl, r.Hl, r.Gr, r.Hr};
        } else {
            auto r = bhc_split(G.data(), H_v.data(), cnt.data(), last_bin.data(), S,
                               G_T, H_T, G_nan, H_nan, eff_lam, min_h, max_k);
            if (r.gain <= 0 || r.best_bin < 0 || r.gain <= best.gain) return;
            const auto& edg = ctx.edges[f];
            int ei = r.best_bin + 1;
            float thr = (ei < (int)edg.size()) ? edg[ei] : ctx.ax_min[f]+ctx.ax_range[f];
            best = {true, false, r.gain, f, r.nan_child, r.best_bin, thr, 0,
                    r.Gl, r.Hl, r.Gr, r.Hr};
        }
    };

    if (feat_subset.empty()) { for (int f = 0; f < D; ++f) process_feat(f); }
    else                     { for (int f : feat_subset) process_feat(f); }
    return best;
}

// ── Routing ───────────────────────────────────────────────────────────────────

static int route_sample(int sample_i, const SplitResult& sp, const BinCtx& ctx) {
    int b = ctx.code[(size_t)sample_i * ctx.D + sp.feature];
    if (b == ctx.B - 1) return sp.nan_child;
    if (sp.is_cat) return ((sp.cat_left_mask >> b) & 1) ? 0 : 1;
    return (b <= sp.best_bin) ? 0 : 1;
}

// ── Inference ─────────────────────────────────────────────────────────────────

double Tree::predict_row(const float* x, int D) const {
    int cur = 0;
    while (!nodes[cur].is_leaf) {
        const SplitResult& sp = nodes[cur].split;
        float val = x[sp.feature];
        int child;
        if (std::isnan(val)) {
            child = sp.nan_child;
        } else if (!sp.is_cat) {
            child = (val < sp.threshold) ? 0 : 1;
        } else {
            const auto& cv = cat_vals[sp.feature];
            auto it = std::lower_bound(cv.begin(), cv.end(), val);
            int b = (it != cv.end() && *it == val) ? (int)(it - cv.begin()) : (B-1);
            child = (b == B-1) ? sp.nan_child : (((sp.cat_left_mask >> b) & 1) ? 0 : 1);
        }
        cur = (child == 0) ? nodes[cur].left : nodes[cur].right;
    }
    return nodes[cur].value;
}

// ── Binning ───────────────────────────────────────────────────────────────────

static void build_bin_ctx(const float* X, int n, int D,
                           const Params& params, BinCtx& ctx) {
    int B_data = params.n_bins;
    int B = B_data + 1;
    ctx.B=B; ctx.D=D; ctx.n=n;
    ctx.is_cat.assign(D, false);
    for (int f : params.cat_features) ctx.is_cat[f] = true;
    ctx.D_cat = (int)params.cat_features.size();
    ctx.edges.resize(D); ctx.ax_min.resize(D,0); ctx.ax_range.resize(D,0);
    ctx.cat_vals.resize(D);
    ctx.code.resize((size_t)n*D, (uint8_t)(B-1));

    for (int f = 0; f < D; ++f) {
        if (ctx.is_cat[f]) {
            std::unordered_map<float,int> cat_map;
            for (int i = 0; i < n; ++i) {
                float v = X[(size_t)i*D+f];
                if (!std::isnan(v) && !cat_map.count(v)) cat_map[v] = (int)cat_map.size();
            }
            std::vector<float> cats;
            cats.reserve(cat_map.size());
            for (auto& kv : cat_map) cats.push_back(kv.first);
            std::sort(cats.begin(), cats.end());
            int C = std::min((int)cats.size(), B_data);
            ctx.cat_vals[f].resize(C);
            for (int k = 0; k < C; ++k) {
                ctx.cat_vals[f][k] = cats[k];
                cat_map[cats[k]] = k < C-1 ? k : C-1;
            }
            for (int i = 0; i < n; ++i) {
                float v = X[(size_t)i*D+f];
                ctx.code[(size_t)i*D+f] = std::isnan(v) ? (uint8_t)(B-1)
                    : (uint8_t)(cat_map.count(v) ? cat_map[v] : B-1);
            }
        } else {
            std::vector<float> valid;
            valid.reserve(n);
            for (int i = 0; i < n; ++i) {
                float v = X[(size_t)i*D+f]; if (!std::isnan(v)) valid.push_back(v);
            }
            if (valid.empty()) continue;
            std::sort(valid.begin(), valid.end());
            int nv = (int)valid.size();
            auto& edges = ctx.edges[f];
            edges.push_back(std::numeric_limits<float>::lowest());
            for (int k = 1; k < B_data; ++k) {
                int pos = clamp((int)((double)k/B_data*nv), 0, nv-1);
                edges.push_back(valid[pos]);
            }
            edges.push_back(std::numeric_limits<float>::max());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            ctx.ax_min[f] = valid.front();
            ctx.ax_range[f] = valid.back() > valid.front() ? valid.back()-valid.front() : 1.0f;
            int B_real = (int)edges.size() - 1;
            for (int i = 0; i < n; ++i) {
                float v = X[(size_t)i*D+f];
                if (std::isnan(v)) { ctx.code[(size_t)i*D+f] = (uint8_t)(B-1); continue; }
                auto it = std::upper_bound(edges.begin()+1, edges.end()-1, v);
                int b = clamp((int)(it-(edges.begin()+1)), 0, B_real-1);
                ctx.code[(size_t)i*D+f] = (uint8_t)b;
            }
        }
    }
}

// ── Histogram ops ─────────────────────────────────────────────────────────────

static void accumulate_hist(const std::vector<int>& rows, const BinCtx& ctx,
                             const double* g, const double* h,
                             std::vector<float>& out) {
    int D=ctx.D, B=ctx.B;
    out.assign((size_t)D*B*HSTRIDE, 0.0f);
    for (int i : rows) {
        const uint8_t* ci = ctx.code.data() + (size_t)i*D;
        for (int f = 0; f < D; ++f) {
            float* sl = out.data() + ((size_t)f*B + ci[f])*HSTRIDE;
            sl[0]+=(float)g[i]; sl[1]+=(float)h[i]; sl[2]+=1.0f;
        }
    }
}

static void subtract_hist(const std::vector<float>& parent,
                           const std::vector<float>& child,
                           std::vector<float>& sibling) {
    sibling.resize(parent.size());
    for (size_t i = 0; i < parent.size(); ++i) sibling[i] = parent[i] - child[i];
}

// ── Tree builder ──────────────────────────────────────────────────────────────

static Tree build_tree(const BinCtx& ctx,
                        const double* g, const double* h,
                        const std::vector<int>& sub,
                        const Params& params,
                        std::vector<int>& split_ks,
                        std::mt19937& rng) {
    Tree tree;
    int max_nodes = 2 * params.max_leaves + 4;
    tree.nodes.resize(max_nodes);
    tree.cat_vals = ctx.cat_vals;
    tree.is_cat   = ctx.is_cat;
    tree.B = ctx.B;

    int D = ctx.D;
    double lam = params.reg_lambda, bhc_lam = params.bhc_lam;
    double min_h = params.min_child_weight, gamma = params.gamma;

    std::vector<int> feat_subset;
    if (params.colsample_bytree < 1.0) {
        std::vector<int> all(D); std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        int ns = std::max(1, (int)(D*params.colsample_bytree));
        feat_subset.assign(all.begin(), all.begin()+ns);
        std::sort(feat_subset.begin(), feat_subset.end());
    }

    struct NodeState { std::vector<int> rows; std::vector<float> hist; double G_T=0,H_T=0; };
    std::vector<NodeState> nstate(max_nodes);

    nstate[0].rows = sub;
    accumulate_hist(sub, ctx, g, h, nstate[0].hist);
    double G_root=0, H_root=0;
    for (int i : sub) { G_root+=g[i]; H_root+=h[i]; }
    nstate[0].G_T=G_root; nstate[0].H_T=H_root;
    tree.nodes[0].value = -G_root/(H_root+lam);

    using PQ = std::priority_queue<std::pair<double,int>>;
    PQ frontier;

    auto enqueue = [&](int t) {
        if ((int)nstate[t].rows.size() < 4) return;
        auto sp = eval_split(nstate[t].hist, ctx,
                             nstate[t].G_T, nstate[t].H_T,
                             lam, bhc_lam, min_h, params.max_k, feat_subset);
        if (!sp.valid || sp.gain <= gamma) return;
        tree.nodes[t].split = sp;
        frontier.push({sp.gain, t});
    };
    enqueue(0);

    int next_id = 1, splits_left = params.max_leaves - 1;
    while (splits_left > 0 && !frontier.empty()) {
        auto [gain, t] = frontier.top(); frontier.pop();
        if (!tree.nodes[t].is_leaf) continue;
        SplitResult& sp = tree.nodes[t].split;
        if (!sp.valid || sp.gain <= 0) continue;

        int depth = 0; { int cur=t; while (cur>0){cur=(cur-1)/2; ++depth;} }
        if (depth >= params.max_depth) continue;

        int tl = next_id++, tr = next_id++;
        if (next_id > max_nodes) break;

        for (int i : nstate[t].rows) {
            (route_sample(i,sp,ctx)==0 ? nstate[tl] : nstate[tr]).rows.push_back(i);
        }
        if (nstate[tl].rows.empty() || nstate[tr].rows.empty()) {
            --next_id; --next_id;
            nstate[tl].rows.clear(); nstate[tr].rows.clear(); continue;
        }

        bool ls = nstate[tl].rows.size() <= nstate[tr].rows.size();
        int ts = ls?tl:tr, tl2 = ls?tr:tl;
        accumulate_hist(nstate[ts].rows, ctx, g, h, nstate[ts].hist);
        subtract_hist(nstate[t].hist, nstate[ts].hist, nstate[tl2].hist);

        for (int i : nstate[tl].rows) { nstate[tl].G_T+=g[i]; nstate[tl].H_T+=h[i]; }
        for (int i : nstate[tr].rows) { nstate[tr].G_T+=g[i]; nstate[tr].H_T+=h[i]; }
        tree.nodes[tl].value = -nstate[tl].G_T/(nstate[tl].H_T+lam);
        tree.nodes[tr].value = -nstate[tr].G_T/(nstate[tr].H_T+lam);

        tree.nodes[t].is_leaf=false; tree.nodes[t].left=tl; tree.nodes[t].right=tr;
        split_ks.push_back(2); --splits_left;

        nstate[t].hist.clear(); nstate[t].hist.shrink_to_fit(); nstate[t].rows.clear();
        enqueue(tl); enqueue(tr);
    }

    tree.nodes.resize(next_id > 0 ? next_id : 1);
    return tree;
}

// ── fit / predict ─────────────────────────────────────────────────────────────

void BRSTBoost::fit(const float* X, int n, int D, const int* y, Params params) {
    lam_=params.reg_lambda; lr_=params.learning_rate;
    build_bin_ctx(X, n, D, params, ctx_);
    int n_pos=0; for (int i=0;i<n;++i) n_pos+=y[i];
    double p0 = clamp((double)n_pos/n, 1e-6, 1.0-1e-6);
    base_score_ = std::log(p0/(1.0-p0));
    std::vector<double> F(n, base_score_), gvec(n), hvec(n);
    trees_.clear(); trees_.reserve(params.n_estimators);
    split_ks_.clear();
    std::mt19937 rng(params.random_state);
    for (int t = 0; t < params.n_estimators; ++t) {
        for (int i=0;i<n;++i) { double p=sigmoid(F[i]); gvec[i]=p-y[i]; hvec[i]=p*(1-p); }
        std::vector<int> sub;
        if (params.subsample < 1.0) {
            sub.resize(n); std::iota(sub.begin(),sub.end(),0);
            std::shuffle(sub.begin(),sub.end(),rng);
            sub.resize((int)(n*params.subsample));
        } else { sub.resize(n); std::iota(sub.begin(),sub.end(),0); }
        Tree tree = build_tree(ctx_, gvec.data(), hvec.data(), sub, params, split_ks_, rng);
        for (int i=0;i<n;++i) F[i] += lr_*tree.predict_row(X+(size_t)i*D, D);
        trees_.push_back(std::move(tree));
    }
}

void BRSTBoost::predict_proba(const float* X, int n, int D, double* out_p1) const {
    std::fill(out_p1, out_p1+n, base_score_);
    for (const Tree& tree : trees_)
        for (int i=0;i<n;++i) out_p1[i] += lr_*tree.predict_row(X+(size_t)i*D, D);
    for (int i=0;i<n;++i) out_p1[i] = sigmoid(out_p1[i]);
}

double BRSTBoost::avg_k() const {
    if (split_ks_.empty()) return 0.0;
    double s=0; for (int k:split_ks_) s+=k; return s/split_ks_.size();
}

} // namespace brst
