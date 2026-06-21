# HRBoost Architecture

## Overview

HRBoost는 C++ 코어(`libhrboost.dylib`) + Python ctypes 래퍼 구조. sklearn 호환 API.

```
HRBoostClassifier / HRBoostRegressor   (python/hrboost/model.py)
        │ ctypes
        ▼
hrboost_fit / hrboost_predict_proba    (src/capi_hrboost.cpp)
        │
        ▼
HRBoost::fit()                         (src/hrboost.cpp)
        │
        ├── build_bin_ctx()            # float32 → uint8 bin code
        │
        └── per-round boosting
                ├── gradient/hessian 계산
                └── build_tree()
                        └── eval_split()
                                ├── numeric: cumulative scan
                                └── categorical: bhc_split_full()  ← 핵심
```

---

## 1. Binning (`build_bin_ctx`)

학습 전 1회 실행. 모든 샘플을 uint8 bin code로 변환 — 이후 히스토그램 연산은 uint8만 참조.

**수치형:**
- 빈도 ≥5% 값: 독립 bin 배정 (frequent value 보존)
- 나머지: 균등 분위수 샘플링으로 `n_bins` 채움
- NaN → bin index = B-1 (sentinel)

**카테고리형:**
- 각 고유값 → 정수 bin index (sorted unique values)
- NaN → B-1

B = max(n_bins+1, max_cat_cardinality+2) — 카테고리가 많으면 B 자동 확장.

---

## 2. Boosting Loop

**Regression** (MSE 손실):
```
g_i = F_i - y_i
h_i = 1.0
```

**Binary Classification** (log-loss):
```
p_i = sigmoid(F_i)
g_i = p_i - y_i
h_i = p_i * (1 - p_i)
```

**Multiclass** (softmax, one-vs-rest, K trees per round):
```
p_ic = softmax(F_i)[c]
g_ic = p_ic - 1(y_i == c)
h_ic = 2 * p_ic * (1 - p_ic)   ← 2× hessian (XGBoost 관례)
```

---

## 3. Tree Building (`build_tree`)

**Best-first expansion** (priority queue, gain 기준):
- max_leaves로 노드 수 직접 제어 (depth-first 아님)
- colsample_bytree: 피처 무작위 서브샘플링

**히스토그램 연산:**
- `accumulate_hist`: 작은 자식 노드만 직접 누적
- `subtract_hist`: 부모 - 작은 자식 = 큰 자식 (histogram subtraction trick)
- 히스토그램 shape: `[D × B × 3]` — 3 slots = (G_sum, H_sum, count)

---

## 4. Split Evaluation (`eval_split`)

### 4-1. 수치형 피처
표준 cumulative scan. 각 split point `b`에 대해:
```
GLv = Σ G[0..b] + G_nan    (NaN left 방향)
GRv = G_clean - Σ G[0..b]
gain = calc_dynamic_gain(GLv, HLv, GRv, HRv, ...)
```
NaN을 left/right 양쪽 다 시도, gain 높은 방향 선택.

### 4-2. 카테고리형 피처 → `bhc_split_full`
아래 4단계.

---

## 5. `bhc_split_full` — 핵심 알고리즘

### 입력
- S개 카테고리, 각각 (G_i, H_i) — eval_split에서 **G/H score 오름차순 정렬** 후 전달
- 정렬이 BHC 초기화 품질을 높임 (비슷한 gradient 통계끼리 인접)

### Phase 1: BHC 계층적 클러스터링

```
초기 상태: S개 클러스터 (각 카테고리 1개)
linked-list 구조 (prev/next 포인터)

while cur_S > 2:
    for 모든 활성 클러스터 i:
        j를 i.next부터 최대 3칸(steps < 3) 탐색:  ← LNM window
            d = delta(G_i, H_i, G_j, H_j, λ)
    max d 쌍 (best_i, best_j) 병합:
        G_best_i += G_best_j
        H_best_i += H_best_j
        best_j 비활성화, linked-list 재연결

결과: 2개 클러스터 (c1, c2) = 초기 L/R 파티션
```

**`delta` 함수 (병합 기준):**
```
delta(Ga, Ha, Gb, Hb, λ) =
    0.5 × [ (Ga+Gb)² / (Ha+Hb+λ) - Ga²/(Ha+λ) - Gb²/(Hb+λ) ]
    + log_t   (λ < 1일 때 Bayesian prior 항)
```
delta가 **높을수록** 두 클러스터가 비슷함 → 먼저 병합.  
`steps < 3` 윈도우가 비인접 병합(Local Non-monotonic)을 허용.

### Phase 2: NaN 방향 결정

```
c1 → Left 시도:  GLv = G_c1 + G_nan,  GRv = G_c2
c1 → Left (NaN Right): GLv = G_c1,    GRv = G_c2 + G_nan

→ calc_dynamic_gain 더 높은 방향 선택
→ cat_left_mask: c1에 속한 bin index → mask[b] = 1
```

### Phase 3: Iterative Partition Refinement ← "리파인"

BHC 솔루션을 시작점으로, **그리디 개별 카테고리 이동**:

```
max_iters = 10
while improved:
    for 각 카테고리 i:
        현재 mask에서 i를 반대 파티션으로 이동했을 때 gain 계산:
            if mask[i] == 1:  L→R 이동
                tGL -= G_i,  tHL -= H_i
                tGR += G_i,  tHR += H_i
            else:             R→L 이동
                tGL += G_i,  tHL += H_i
                tGR -= G_i,  tHR -= H_i

        if calc_dynamic_gain(tGL, tHL, tGR, tHR) > best.gain + 1e-7:
            best_move_idx = i  (가장 gain 높은 이동 1개만 선택)

    if best_move_idx != -1:
        mask[best_move_idx] ^= 1
        gain/GL/GR/HL/HR 업데이트
        improved = True
```

**핵심:** BHC는 greedy merge라 suboptimal 파티션을 만들 수 있음. Refine이 각 카테고리를 개별적으로 이동해 최대 10회 보정 → BHC 단독 대비 실질적 gain 향상.

---

## 6. Cohesion Dynamic Regularization (`calc_dynamic_gain`)

모든 gain 계산에 적용 (수치형 + 카테고리형 + Refine 내부):

```
dL = GL / (HL + ε)    ← 왼쪽 리프 예측값 추정
dR = GR / (HR + ε)

cohesion = 1 - |dL - dR| / (|dL| + |dR| + ε)

λ_dyn = λ × (1 + γ_cohesion × cohesion)

gain = GL²/(HL + λ_dyn) + GR²/(HR + λ_dyn) - G_T²/(H_T + λ)
```

**동작 원리:**
| 상황 | cohesion | λ_dyn | 효과 |
|------|----------|-------|------|
| dL ≈ dR (양 자식 예측 유사) | → 1.0 | 커짐 | split 억제 (정보 없는 분할 패널티) |
| dL ≫ dR (양 자식 예측 상이) | → 0 | = λ | 정상 regularization |

> **주의:** README 설명("diverge → λ 증가")은 **코드와 반대**. 실제로는 두 자식이 **유사할 때** λ_dyn 증가. "쓸모없는 split" 억제가 실제 의도.

`γ_cohesion` = `COHESION_REG` env (default 0.3). 0으로 설정 시 표준 XGBoost gain으로 대체.

---

## 7. 예측

```
F_i = base_score + Σ_t [ lr × tree_t.predict_row(x_i) ]

predict_row: 트리 루트부터 리프까지 라우팅
  - 수치형: val < threshold → left
  - 카테고리형: cat_left_mask[bin_of(val)] → 0=left, 1=right
  - NaN: nan_child 방향
```

---

## 8. 파라미터 요약

| 파라미터 | 설명 |
|---------|------|
| `n_estimators` | 부스팅 라운드 수 |
| `learning_rate` | shrinkage |
| `max_depth` | 트리 최대 깊이 (best-first와 함께 제어) |
| `max_leaves` | 리프 최대 수 (직접 복잡도 제어) |
| `reg_lambda` | L2 base regularization |
| `subsample` | 행 서브샘플 비율 |
| `colsample_bytree` | 피처 서브샘플 비율 |
| `n_bins` | 수치형 히스토그램 bin 수 |
| `min_child_weight` | 리프 최소 hessian 합 |
| `gamma` | split gain 최소 임계값 |
| `max_delta_step` | 리프값 클리핑 (imbalanced 데이터용) |
| `COHESION_REG` (env) | cohesion regularization 강도 (default 0.3) |
| `MIN_CAT_COUNT` (env) | 카테고리 최소 샘플 수 필터 |

---

## 9. 파일 구조

```
src/
  hrboost.h          - Params, Node, Tree, BinCtx, HRBoost 클래스 선언
  hrboost.cpp        - 전체 C++ 구현
  capi_hrboost.h     - C API (ctypes용) 선언
  capi_hrboost.cpp   - C API 구현 (hrboost_create/fit/predict_proba/predict/free)

python/hrboost/
  _lib.py            - ctypes 함수 시그니처 등록 + dylib 로드
  model.py           - HRBoostClassifier, HRBoostRegressor (sklearn 상속)
  __init__.py

libhrboost.dylib          - 루트 (개발용)
python/hrboost/libhrboost.dylib  - 패키지 배포용

optimizer.py         - Optuna HPO (COHESION_REG 포함)
benchmark.py         - 14개 데이터셋 vs LightGBM/XGBoost/CatBoost
hpo.py               - 추가 HPO 유틸
```
