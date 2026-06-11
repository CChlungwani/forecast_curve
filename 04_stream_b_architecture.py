"""
Stream B Model Architecture: Pooled vs Per-Country vs Hierarchical
This is the most important open decision in the framework.

The equity paper had three INDEPENDENT binary classification problems
(MOM vs VAL, MOM vs QUAL, VAL vs QUAL).
Stream B has up to 10 INTERDEPENDENT classification problems (one per country).
The architecture choice fundamentally changes what the model can learn.

Decision framework and full implementation of all three options below.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE COMPARISON MATRIX
# ─────────────────────────────────────────────────────────────────────────────

ARCHITECTURE_COMPARISON = {
    "pooled": {
        "description": "Single model trained on all countries simultaneously. "
                       "Country encoded as a categorical feature.",
        "pros": [
            "Maximises sample size: 10 countries × 929 weeks ≈ 9,290 obs",
            "Can learn cross-country spread co-movement patterns",
            "Single hyperparameter set → computationally efficient",
            "ECB policy variables can be shared features",
        ],
        "cons": [
            "Italian BTP and Finnish OAT have structurally different dynamics",
            "Country-specific shocks contaminate pooled coefficients",
            "Country dummy is a crude way to encode structural heterogeneity",
            "Fiscal variables have very different ranges across countries",
        ],
        "recommended_when": "Phase 1 baseline. Fast to implement; provides upper bound on data efficiency.",
        "risk": "HIGH: peripheral/core heterogeneity may dominate and confuse the model.",
    },
    "per_country": {
        "description": "Separate model trained and evaluated per country. "
                       "Identical to equity paper's approach (3 separate models for 3 pairs).",
        "pros": [
            "Each country's structural dynamics modelled independently",
            "Hyperparameters tuned to each country's signal regime",
            "Italy can have different H, different features than Finland",
            "Cleanest interpretation of feature importance per country",
        ],
        "cons": [
            "10× the computational budget",
            "Small samples for low-volatility countries (AT, FI, NL)",
            "Cannot explicitly learn cross-country contagion",
            "Separate CPCV runs → more chance of overfitting per country",
        ],
        "recommended_when": "Phase 2. Start with 3–4 key countries (IT, ES, PT, FR).",
        "risk": "MEDIUM: overfitting risk for stable countries with weak signals.",
    },
    "hierarchical": {
        "description": "Two-level model: country-specific layers share an ECB policy backbone. "
                       "Conceptually: each country = equity paper's factor pair, "
                       "shared macro features = common level in the hierarchy.",
        "pros": [
            "Borrows statistical strength across countries via shared parameters",
            "Can learn that ECB variables affect all countries similarly",
            "Country-specific parameters capture idiosyncratic components",
            "Elegant solution to the periphery vs core divide",
        ],
        "cons": [
            "Substantially more complex to implement than the equity paper",
            "Requires Bayesian or multi-task learning frameworks",
            "CPCV more complex (need purging at both country and pooled levels)",
            "Interpretation is harder than per-country models",
        ],
        "recommended_when": "Phase 3 extension. Requires Phase 1 & 2 as baselines.",
        "risk": "LOW (statistical): but MEDIUM (complexity): high implementation risk.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# OPTION 1: POOLED MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_pooled_dataset(
    country_spreads: pd.DataFrame,        # cols: IT, ES, FR, PT, ...
    country_labels: pd.DataFrame,         # cols: label_IT, label_ES, ...
    shared_features: pd.DataFrame,        # ECB, macro, global variables
    country_specific_features: Dict[str, pd.DataFrame],  # fiscal, CDS per country
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs a pooled panel dataset: N_obs = T × C (time × countries).

    Country encoding strategy:
      - Add binary country dummies (one-hot, drop one reference = Germany)
      - Alternatively: use embeddings if using neural architecture later
      - Include fiscal/credit features specific to each country
        (debt/GDP, CDS) stacked with their country rows

    CRITICAL: Temporal index must be preserved. Panel stacking must not
    shuffle time ordering, as CPCV requires sorted-by-time data.
    The panel is long-format: first all weeks for IT, then for ES, etc.
    This breaks the single time-axis assumption of CPCV!

    SOLUTION: Use country-by-country CPCV (see below), not naive panel CPCV.
    """
    countries = [c for c in country_spreads.columns]
    all_X = []
    all_y = []

    for cc in countries:
        # Shared features apply to all countries identically
        X_shared = shared_features.copy()

        # Country-specific features
        X_country = country_specific_features.get(cc, pd.DataFrame(index=shared_features.index))

        # Country dummies
        dummy_cols = {f"country_{c}": int(c == cc) for c in countries}
        X_dummy = pd.DataFrame(dummy_cols, index=shared_features.index)

        # Combine
        X_cc = pd.concat([X_shared, X_country, X_dummy], axis=1)
        y_cc = country_labels[f"label_{cc}"]

        # Align indices
        common_idx = X_cc.index.intersection(y_cc.dropna().index)
        all_X.append(X_cc.loc[common_idx])
        all_y.append(y_cc.loc[common_idx])

    X_pooled = pd.concat(all_X, axis=0).sort_index()
    y_pooled = pd.concat(all_y, axis=0).sort_index()

    return X_pooled, y_pooled


def pooled_cpcv_splits(
    index: pd.DatetimeIndex,
    countries: List[str],
    N: int = 5,
    k: int = 1,
    purge_weeks: int = 30,
) -> List[Dict]:
    """
    CPCV for pooled panel: split on TIME, not on observations.
    Each split defines a time window; all countries within that window
    go into train/test together.

    This ensures no country's future data leaks into any country's training,
    and the purge gap applies uniformly across the time dimension.

    WARNING: The equity paper's CPCV treated each time observation as
    independent. In the panel, temporal structure is what matters —
    country independence is assumed (which is violated, but manageable
    with purging).
    """
    unique_dates = sorted(index.unique())
    n_dates = len(unique_dates)
    block_size = n_dates // N
    blocks = [unique_dates[i * block_size: (i + 1) * block_size] for i in range(N)]

    from itertools import combinations
    splits = []
    for test_blocks in combinations(range(N), k):
        test_dates = []
        for tb in test_blocks:
            test_dates.extend(blocks[tb])
        test_dates = set(test_dates)

        # Purged train dates
        purge_dates = set()
        for d in test_dates:
            idx_d = unique_dates.index(d)
            for offset in range(-purge_weeks, purge_weeks + 1):
                pi = idx_d + offset
                if 0 <= pi < n_dates:
                    purge_dates.add(unique_dates[pi])

        train_dates = set(unique_dates) - test_dates - purge_dates

        splits.append({
            "train_dates": sorted(train_dates),
            "test_dates": sorted(test_dates),
            "purged_dates": sorted(purge_dates),
        })

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# OPTION 2: PER-COUNTRY MODEL (directly analogous to equity paper)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CountryModelConfig:
    """Per-country modelling configuration."""
    country_code: str
    H_max: int           # Country-specific label horizon
    n_features: int      # After MI clustering
    min_obs: int = 260   # Minimum training observations

    def __post_init__(self):
        # Country-specific horizon recommendations
        COUNTRY_HORIZONS = {
            "IT": 26,   # High volatility → longer horizon to find stable trend
            "ES": 26,   # Similar to Italy
            "PT": 20,   # Moderate volatility
            "GR": 13,   # Very high volatility; short horizon avoids stale signal
            "FR": 20,   # Semi-core; moderate
            "BE": 16,
            "AT": 13,   # Core-like; shorter H viable
            "NL": 13,
            "FI": 13,
            "IE": 20,
        }
        if self.H_max == 0:  # Sentinel: use recommended
            self.H_max = COUNTRY_HORIZONS.get(self.country_code, 20)


def run_per_country_pipeline(
    country_code: str,
    spread_labels: pd.Series,
    X_features: pd.DataFrame,
    config: CountryModelConfig,
) -> dict:
    """
    Per-country pipeline: identical structure to equity paper.
    Returns performance metrics and feature importance by country.

    This is the recommended Phase 2 implementation.
    Follow steps:
      1. Label construction (01_label_construction.py) per country
      2. MI clustering (from paper's Section 3.6) per country
      3. Logistic / GAM / XGBoost training per country
      4. CPCV hyperparameter optimisation per country
      5. Walk-forward evaluation per country
      6. SHAP per country

    Output: dict of {model_name: {accuracy, roc_auc, log_loss, shap_values}}
    """
    # Validate minimum sample
    valid_labels = spread_labels.dropna()
    if len(valid_labels) < config.min_obs:
        return {
            "country": country_code,
            "error": f"Insufficient observations: {len(valid_labels)} < {config.min_obs}",
        }

    # Align features and labels
    common_idx = X_features.index.intersection(valid_labels.index)
    X = X_features.loc[common_idx]
    y = valid_labels.loc[common_idx]

    return {
        "country": country_code,
        "n_obs": len(y),
        "label_balance": y.mean(),
        "config": config,
        "X_shape": X.shape,
        # Actual model training: call model_training.py equivalents
    }


# ─────────────────────────────────────────────────────────────────────────────
# OPTION 3: HIERARCHICAL / MULTI-TASK SKELETON
# ─────────────────────────────────────────────────────────────────────────────

def build_hierarchical_feature_matrix(
    shared_features: pd.DataFrame,        # ECB, macro — same for all countries
    country_features: Dict[str, pd.DataFrame],  # CDS, fiscal — per country
    interaction_features: Dict[str, pd.DataFrame],  # spread-specific relative metrics
) -> Dict[str, pd.DataFrame]:
    """
    Builds the hierarchical feature matrix for multi-task learning.

    Structure:
      Level 1 (shared ECB backbone): φ_shared ∈ ℝ^{d_shared}
        → Same for all countries; ECB DFR, OIS, HICP, PMI
      Level 2 (country-specific): φ_{country,c} ∈ ℝ^{d_country}
        → CDS spread, fiscal ratios, political risk
      Level 3 (interaction): φ_{interaction,c} ∈ ℝ^{d_int}
        → Cross-country spread z-scores, spread momentum, relvol

    Combined per-country feature vector:
      X_c = [φ_shared | φ_{country,c} | φ_{interaction,c}]

    This is what per-country models use anyway; the hierarchical aspect
    comes from PARAMETER SHARING in multi-task learning:
      - Shared parameters: β_shared learned jointly across all countries
      - Country-specific: Δβ_c adds country idiosyncrasy

    Multi-task XGBoost: not natively supported, but can approximate via:
      - Stacked per-country outputs used as meta-features
      - Group-lasso-style regularisation on shared features
      - Or: simply pool + per-country and blend predictions
    """
    result = {}
    for cc, X_cc in country_features.items():
        X_interaction = interaction_features.get(cc, pd.DataFrame(index=shared_features.index))
        X_combined = pd.concat([shared_features, X_cc, X_interaction], axis=1)
        # Drop duplicate columns (shared features may appear in country features)
        X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
        result[cc] = X_combined
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL INTEGRATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def integrate_stream_signals(
    stream_a_probs: pd.Series,       # P(steepening) from Stream A model
    stream_b_probs: pd.DataFrame,    # P(widening) per country, columns=country codes
    stream_a_threshold: float = 0.55,
    stream_b_threshold: float = 0.55,
    integration_mode: str = "independent",
) -> pd.DataFrame:
    """
    Combines Stream A (curve) and Stream B (spread) signals.

    Modes:
      "independent":
        Each signal acts independently. Duration positioning from A;
        country allocation from B. No cross-signal conditioning.
        Simplest; most transparent; recommended for Phase 1.

      "conditional":
        Stream A signal conditions Stream B signal interpretation.
        E.g., if A predicts flattening (risk-off regime), upweight
        B's widening signals for peripheral countries.
        Economic rationale: curve flattening often coincides with
        spread widening in stress (correlated regime).
        Risk: introduces cross-stream look-ahead bias if not careful.
        Ensure conditioning uses LAGGED Stream A signal.

      "blended":
        Weighted combination of A and B into a single composite score.
        Appropriate if building a single portfolio optimization model.
        Requires careful definition of the common P&L unit.

    Returns DataFrame with columns:
      curve_signal, {cc}_spread_signal, combined_score (if blended)
    """
    result = pd.DataFrame(index=stream_a_probs.index)
    result["curve_signal_prob"] = stream_a_probs
    result["curve_signal_binary"] = (stream_a_probs >= stream_a_threshold).astype(int)

    for cc in stream_b_probs.columns:
        result[f"{cc}_spread_signal_prob"] = stream_b_probs[cc]
        result[f"{cc}_spread_signal_binary"] = (
            stream_b_probs[cc] >= stream_b_threshold
        ).astype(int)

    if integration_mode == "conditional":
        # Conditional: when A signals flattening (low P), boost widening probability
        flat_signal = (stream_a_probs < (1 - stream_a_threshold)).astype(float)
        for cc in stream_b_probs.columns:
            # Adjust probability upward for widening when curve flattens
            # Using lagged stream A signal to avoid look-ahead (lag by 1 week)
            flat_lag = flat_signal.shift(1).fillna(0)
            adj = stream_b_probs[cc] + 0.1 * flat_lag  # 10% boost
            result[f"{cc}_conditional_prob"] = adj.clip(0, 1)

    elif integration_mode == "blended":
        # Equal weight blend across all signals
        all_signal_cols = [c for c in result.columns if "signal_prob" in c]
        result["composite_score"] = result[all_signal_cols].mean(axis=1)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("STREAM B ARCHITECTURE DECISION GUIDE")
    print("=" * 70)
    for arch, info in ARCHITECTURE_COMPARISON.items():
        print(f"\n{arch.upper()}")
        print(f"  Recommended when: {info['recommended_when']}")
        print(f"  Primary risk: {info['risk']}")
        print(f"  Top con: {info['cons'][0]}")

    print("\n\nRECOMMENDED PHASED APPROACH:")
    print("""
  Phase 1 (3 months):
    - Per-country models for IT, ES, FR (highest volatility and economic importance)
    - Use pooled as a robustness check baseline
    - Regime dummies as features (Approach A from 03_regime_breaks.py)
    - Integration mode: independent
    - H_max: 26 weeks for all countries initially

  Phase 2 (3 more months):
    - Extend per-country to remaining 7 countries
    - Within-regime z-score for rate-level features (Approach C)
    - Country-specific H_max tuning based on Phase 1 label diagnostics
    - Integration mode: try conditional; test if it adds value vs independent

  Phase 3 (optional):
    - Hierarchical/multi-task if Phase 2 per-country shows cross-country leakage
    - Regime-segmented training windows (Approach B)
    - Portfolio implementation with transaction cost modelling
    """)
