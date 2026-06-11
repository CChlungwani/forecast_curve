"""
ECB Policy Regime Break Handling
This module has NO equivalent in the equity paper — it is a new methodological
component specific to fixed income and the Euro area.

Background:
  The ECB has operated under four structurally distinct policy regimes since 2007:
  1. Pre-GFC / conventional (2007–2008)
  2. Post-GFC conventional / SMP (2009–2013)
  3. Zero/negative rates + QE APP (2014–2021)
  4. Normalization / hiking + QT (2022–present)

  Each regime alters the relationship between predictor features and the target:
    - In NIRP regime: yield curve inversions occur at levels previously impossible
    - In QE regime: ECB purchases mechanically suppress sovereign spreads, distorting
      carry and valuation signals
    - In hiking regime: front-end OIS becomes the dominant curve driver (bear-flattening)

  Failure to address this means the model trains on structurally incompatible data.

  The equity paper's South African data (2007-2025) also spans multiple regimes but
  does NOT explicitly handle this — a recognised limitation. For Euro fixed income,
  this is NOT optional given the ZLB episode (2014-2021).

Three approaches are implemented below:
  A. Regime dummy features (simplest — inline with the equity paper's approach)
  B. Regime-segmented training (changes what data the model sees)
  C. Structural-break-aware feature transformation (most principled)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# A: REGIME DEFINITION AND ENCODING
# ─────────────────────────────────────────────────────────────────────────────

# Historical ECB policy regime dates (to be updated as policy evolves)
ECB_REGIMES = {
    "pre_gfc_conventional":    ("2007-01-01", "2008-10-07"),
    "post_gfc_conventional":   ("2008-10-08", "2014-06-04"),
    "zirp_nirp_qe":            ("2014-06-05", "2022-07-20"),  # DFR first went to -0.5%
    "normalization_hiking":    ("2022-07-21", "2025-03-28"),  # First 50bp hike
}

# Sub-regimes for more granular conditioning
ECB_SUB_REGIMES = {
    "pre_gfc":                 ("2007-01-01", "2008-09-14"),
    "gfc_acute":               ("2008-09-15", "2009-06-30"),   # Lehman → ECB easing
    "sovereign_debt_crisis":   ("2010-04-23", "2012-09-05"),   # Greece bailout → Draghi "whatever it takes"
    "post_omt_conventional":   ("2012-09-06", "2014-06-04"),
    "qe_app_initial":          ("2015-03-09", "2018-12-31"),   # APP launch → taper
    "pre_pandemic":            ("2019-01-01", "2020-03-17"),
    "pandemic_pepp":           ("2020-03-18", "2022-03-31"),   # PEPP active
    "post_pepp_pre_hike":      ("2022-04-01", "2022-07-20"),
    "hiking_cycle":            ("2022-07-21", "2023-10-26"),   # First hike → peak DFR 4%
    "qt_plateau":              ("2023-10-27", "2025-03-28"),   # QT + rate hold
}


def build_regime_dummies(
    index: pd.DatetimeIndex,
    granularity: str = "main",   # "main" or "sub"
) -> pd.DataFrame:
    """
    Build binary regime dummy columns. Use as additional features.

    APPROACH A: Regime as features — compatible with the equity paper framework.
    The model learns which regime it is in and conditions predictions accordingly.

    Warning: With N=5 CPCV splits over 18 years, some regimes may appear in
    only 1–2 splits. Regime dummies should be monitored for low-variance
    splits that cause estimation instability.
    """
    regimes = ECB_REGIMES if granularity == "main" else ECB_SUB_REGIMES
    df = pd.DataFrame(index=index)

    for name, (start, end) in regimes.items():
        df[f"regime_{name}"] = (
            (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        ).astype(int)

    # Add interaction: hiking cycle × yield level (bull vs bear hiking)
    if granularity == "main":
        hiking = df["regime_normalization_hiking"]
        df["regime_indicator"] = hiking  # Primary conditioning variable

    return df


def build_regime_transition_indicator(
    index: pd.DatetimeIndex,
    transition_window_weeks: int = 12,
) -> pd.Series:
    """
    Binary indicator = 1 in the weeks surrounding a regime transition.
    These observations should either be:
      - Excluded from training (conservative approach)
      - Flagged with a separate dummy
    Reason: the model's learned relationships are most unreliable at
    regime transitions, and the trend-scan labels from these periods
    may be corrupted (the label looks backward AND forward through the break).
    """
    all_starts = [pd.Timestamp(start) for start, _ in ECB_REGIMES.values()]
    all_starts += [pd.Timestamp(end) for _, end in ECB_REGIMES.values()]

    indicator = pd.Series(0, index=index)
    for break_date in all_starts:
        window_start = break_date - pd.Timedelta(weeks=transition_window_weeks // 2)
        window_end = break_date + pd.Timedelta(weeks=transition_window_weeks // 2)
        mask = (index >= window_start) & (index <= window_end)
        indicator[mask] = 1

    return indicator


# ─────────────────────────────────────────────────────────────────────────────
# B: REGIME-SEGMENTED TRAINING WINDOW SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_regime_aware_training_window(
    prediction_date: pd.Timestamp,
    full_history: pd.DatetimeIndex,
    base_window_weeks: int = 773,
    min_window_weeks: int = 260,
    regime_lookback_only: bool = True,
) -> pd.DatetimeIndex:
    """
    APPROACH B: Modify the rolling training window to include/exclude
    observations from incompatible regimes.

    In the equity paper: fixed rolling window of 773 weeks used for all periods.
    Problem for FI: if predicting in 2024 (hiking regime), including 2015–2021
    (NIRP/QE) may introduce noise rather than signal — the feature-to-target
    mapping is structurally different.

    Two options implemented:
      1. regime_lookback_only=True:
         Use only data from the same regime as the prediction date.
         Risk: Sample may be too small if the regime is young.
         Fallback: extend back into prior regime if n < min_window_weeks.

      2. regime_lookback_only=False:
         Use full rolling window but add regime dummies.
         Simpler and consistent with the equity paper approach.

    Recommendation: Start with option 2 (regime dummies) in Phase 1.
    Move to option 1 if feature importance shows regime confusion.
    """
    current_regime = None
    regime_start = None

    for name, (start, end) in ECB_REGIMES.items():
        if pd.Timestamp(start) <= prediction_date <= pd.Timestamp(end):
            current_regime = name
            regime_start = pd.Timestamp(start)
            break

    if regime_lookback_only and current_regime and regime_start:
        regime_obs = full_history[full_history >= regime_start]
        regime_obs = regime_obs[regime_obs < prediction_date]

        if len(regime_obs) >= min_window_weeks:
            # Enough in-regime data
            return regime_obs[-base_window_weeks:]
        else:
            # Fall back to extending into prior regime
            target_start = prediction_date - pd.Timedelta(weeks=base_window_weeks)
            return full_history[
                (full_history >= target_start) & (full_history < prediction_date)
            ]
    else:
        # Standard rolling window
        target_start = prediction_date - pd.Timedelta(weeks=base_window_weeks)
        return full_history[
            (full_history >= target_start) & (full_history < prediction_date)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# C: STRUCTURAL-BREAK-AWARE FEATURE TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────

def apply_regime_relative_transformation(
    series: pd.Series,
    transform: str = "within_regime_zscore",
) -> pd.Series:
    """
    APPROACH C: Transform features relative to the current regime's history,
    rather than full-sample statistics.

    Problem: Standard z-scoring over the full sample is dominated by regime
    differences. E.g., OIS rates at 0% in 2020 look "extreme" on a full-sample
    scale but were "normal" within the NIRP regime.

    Transforms:
      "within_regime_zscore":
        z_t = (x_t - μ_regime_to_date) / σ_regime_to_date
        Captures whether a variable is extreme *relative to the current regime*.

      "zscore_expanding":
        Uses all data up to t; no look-ahead bias but regime dominance issue.

      "quantile_scaled_regime":
        Percentile rank within current regime (0 to 1); robust to outliers.

    This is most important for:
      - ECB_DFR (level drastically different across regimes)
      - OIS rates (clustered near 0 in NIRP era)
      - Yield levels (the fundamental predictor structure differs)
    """
    result = series.copy()

    # Determine which regime each observation belongs to
    for name, (start, end) in ECB_REGIMES.items():
        regime_mask = (series.index >= pd.Timestamp(start)) & \
                      (series.index <= pd.Timestamp(end))
        regime_data = series[regime_mask]

        if transform == "within_regime_zscore":
            # Expanding mean/std within regime (no look-ahead)
            expanding_mean = regime_data.expanding().mean()
            expanding_std = regime_data.expanding().std()
            result[regime_mask] = (
                (regime_data - expanding_mean) / expanding_std.replace(0, np.nan)
            )

        elif transform == "quantile_scaled_regime":
            # Expanding rank within regime (0–1 scale)
            result[regime_mask] = regime_data.expanding().rank(pct=True)

        elif transform == "zscore_expanding":
            result[regime_mask] = (
                (regime_data - regime_data.expanding().mean()) /
                regime_data.expanding().std().replace(0, np.nan)
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

REGIME_HANDLING_RECOMMENDATIONS = {
    "Phase_1_minimum": {
        "approach": "A (regime dummies)",
        "action": "Add 4 binary regime dummies to feature set. "
                  "Mark transition windows; consider up-weighting recent observations.",
        "implementation": "build_regime_dummies() → add columns to X matrix.",
        "risk": "Model may still learn spurious cross-regime mappings.",
    },
    "Phase_2_improved": {
        "approach": "A + C (dummies + within-regime z-score)",
        "action": "Apply within-regime z-score to rate-level features (ECB_DFR, OIS_*, "
                  "EURSWAP_*). Keep raw levels for spread features (less regime-distorted).",
        "implementation": "apply_regime_relative_transformation() on level features only.",
        "risk": "Expanding-window z-score still shrinks σ at start of new regime → "
                "feature noise at regime transitions.",
    },
    "Phase_3_rigorous": {
        "approach": "B + C (segmented window + within-regime transform)",
        "action": "Use only same-regime data for training once ≥260 observations available. "
                  "Exclude ±12-week windows around regime breaks from training.",
        "implementation": "get_regime_aware_training_window() + build_regime_transition_indicator().",
        "risk": "QE regime was 7+ years (360+ weeks); this is sufficient. "
                "Hiking regime (2022-) still accumulating data in 2025.",
    },
}


if __name__ == "__main__":
    dates = pd.date_range("2007-01-05", "2025-03-28", freq="W-FRI")

    # Build regime dummies
    dummies = build_regime_dummies(dates, granularity="main")
    print("Regime dummies shape:", dummies.shape)
    print("Regime obs counts:")
    print(dummies.sum())

    # Transition indicator
    transition = build_regime_transition_indicator(dates, transition_window_weeks=12)
    print(f"\nTransition window observations: {transition.sum()} "
          f"({transition.mean():.1%} of sample)")

    # Phase recommendations
    print("\n--- Regime Handling Recommendations ---")
    for phase, rec in REGIME_HANDLING_RECOMMENDATIONS.items():
        print(f"\n{phase}:")
        print(f"  Approach: {rec['approach']}")
        print(f"  Action:   {rec['action'][:80]}...")
