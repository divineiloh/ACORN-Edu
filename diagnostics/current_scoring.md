# Current ACORN Scoring Logic Analysis

## Scoring Formula

The ACORN scheduler uses a weighted sum of four normalized terms:

```
score = α × slack_term + β × reuse_term + γ × size_term + δ × conn_term
```

Where all terms are normalized to [0,1] with higher values indicating higher priority.

## Term Analysis

### 1. Slack Term (α = 0.30)
**Purpose**: Measures deadline urgency
**Calculation**:
1. `slack_sec = (deadline_ts - now_ts) - dl_time_sec`
2. `slack_nonneg = max(0, slack_sec)`  # Remove negative slack
3. `S_capped = min(slack_nonneg, next_window_sec)`  # Cap to window length
4. `S_eff = min(S_capped, 0.25 × next_window_sec)`  # Dead-zone at 25% of window
5. `slack_term = 1.0 - _norm01(S_eff)`  # Inverted: less slack = higher priority

**Key Features**:
- Dead-zone prevents over-rewarding very early items
- Capped to window length to prevent crowding out urgent items
- Inverted normalization (less slack = higher priority)

### 2. Reuse Term (β = 0.30)
**Purpose**: Measures asset reusability
**Calculation**:
1. `reuse_term = _norm01(reuse_raw)`  # Direct normalization
2. Higher reuse scores = higher priority

### 3. Size Term (γ = 0.30)
**Purpose**: Favors smaller assets (faster to download)
**Calculation**:
1. `size_term = 1.0 - _norm01(size_kb)`  # Inverted: smaller = higher priority
2. Smaller assets get higher priority

### 4. Connectivity Term (δ = 0.10)
**Purpose**: Measures finishability within the next window
**Calculation**:
1. `window_capacity_kb = th × next_window_sec`
2. `frac_of_window = min(size_kb / window_capacity_kb, 1.0)`
3. `conn_term = 1.0 - _norm01(frac_of_window)`  # Inverted: easier to finish = higher priority

## Ablation Logic

Ablations work by zeroing specific weights:

- **alpha0**: `α = 0.0` (removes slack/urgency term)
- **beta0**: `β = 0.0` (removes reuse term)  
- **gamma0**: `γ = 0.0` (removes size term)
- **delta0**: `δ = 0.0` (removes connectivity term)

## Feasibility Gate

Assets are only considered if:
1. `slack_sec >= 0.0` (can be delivered before deadline)
2. `dl_time_sec <= next_window_sec` (can be downloaded in one window)

If not feasible: `score = 0.0`

## Tie-Breaking

When scores are equal, lexicographic sort by:
1. `size_kb` (ascending - smaller first)
2. `-reuse_term` (descending - higher reuse first)  
3. `scores` (descending - higher score first)

## Current Weights

```
α (slack) = 0.30
β (reuse) = 0.30  
γ (size)  = 0.30
δ (conn)  = 0.10
```

## Potential Issues

1. **Normalization Dependencies**: Each term uses `_norm01()` which depends on the min/max of ALL candidates in the current batch
2. **Weight Imbalance**: Slack, reuse, and size all have equal weight (0.30), while connectivity is much lower (0.10)
3. **Dead-zone Effect**: The 25% dead-zone in slack calculation may create non-monotonic behavior
4. **Tie-breaking Order**: Size comes before reuse in tie-breaking, which may not be optimal
