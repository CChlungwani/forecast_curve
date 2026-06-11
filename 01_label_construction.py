"""
Stream A & B: Trend-Scanning Label Construction
Adapted from equity factor timing (Hlungwani 2026) → Euro area fixed income

Key departures from the equity version:
  - Target is a spread level, not a factor return differential
  - Economic interpretation of label sign differs by stream
  - Purge gap must account for bond settlement (T+2) and data publication lags
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# STREAM A: Yield Curve Steepening / Flattening Label
# ─────────────────────────────────────────────────────────────────────────────

def compute_curve_spread(
    y10: pd.Series,
    y2: pd.Series,
) -> pd.Series:
    """
    Compute the 10Y-2Y Bund slope spread (in basis points).

    Args:
        y10: Weekly 10Y Bund yield (%)
        y2:  Weekly 2Y Bund yield (%)

    Returns:
        spread: 10Y - 2Y in bps (×100 for unit clarity in regressions)

    RISK NOTE: Using *generic* Bund yields conflates on-the-run richness
    effects. If using Bloomberg BGN vs DBR actives, align maturities carefully.
    """
    spread_pct = y10 - y2          # in percentage points
    spread_bps = spread_pct * 100  # convert to bps for readability
    return spread_bps


def trend_scan_labels(
    series: pd.Series,
    H_min: int = 4,
    H_max: int = 26,
    t_threshold: float = 1.0,
    mode: Literal["slope_sign", "t_statistic"] = "slope_sign",
) -> pd.DataFrame:
    """
    López de Prado trend-scanning adapted for fixed income spreads.

    Scans forward windows h ∈ [H_min, H_max] weeks. At each observation t,
    fits OLS on the cumulative spread path over [t, t+h] and selects h*
    that maximises |t-stat(β_h)|.

    FIXED INCOME DEPARTURE FROM EQUITY VERSION:
      - We scan the LEVEL of the spread (not compounded return), because
        bond spread levels are stationary (mean-reverting) over medium horizons,
        whereas equity factor returns are not.
      - If |t*(β_{h*})| < t_threshold, label is set to NaN (uncertain regime);
        these observations should be dropped before model training.
      - H calibration (see HORIZON_GUIDE below).

    Args:
        series:      Spread series (bps), weekly frequency
        H_min:       Minimum forward scan window (weeks)
        H_max:       Maximum forward scan window (weeks), i.e. H in the paper
        t_threshold: Minimum |t-stat| to assign a label (filter noise)
        mode:        "slope_sign" → binary {0,1}; "t_statistic" → raw t-stat

    Returns:
        DataFrame with columns: label, t_stat, optimal_h, label_start, label_end

    HORIZON_GUIDE
    ─────────────
    H_max=4   (~1 month)  → monetary policy meeting window; high noise
    H_max=13  (~3 months) → macro data revision cycle; recommended minimum
    H_max=26  (~6 months) → medium-term positioning; recommended for Stream A
    H_max=52  (~1 year)   → structural regime; risk of excessive label overlap
    Start with H_max=26 and examine label balance (steepen/flatten ratio).
    Target: 40–60% balance. If severely imbalanced, reduce H_max.
    """
    n = len(series)
    index = series.index
    labels = pd.Series(index=index, dtype=float)
    t_stats = pd.Series(index=index, dtype=float)
    opt_h = pd.Series(index=index, dtype=float)

    t_arr = np.arange(n)

    for i in range(n - H_min):
        best_abs_t = -np.inf
        best_t = np.nan
        best_h = np.nan

        for h in range(H_min, min(H_max + 1, n - i)):
            # Fit OLS: spread[i:i+h] ~ α + β·t
            y_window = series.iloc[i: i + h].values
            if np.any(np.isnan(y_window)):
                continue
            x_window = np.arange(h, dtype=float)
            # OLS with scipy for t-stat
            slope, intercept, r, p, se = stats.linregress(x_window, y_window)
            t_val = slope / se if se > 0 else 0.0

            if abs(t_val) > best_abs_t:
                best_abs_t = abs(t_val)
                best_t = t_val
                best_h = h

        t_stats.iloc[i] = best_t
        opt_h.iloc[i] = best_h

        if mode == "slope_sign":
            if best_abs_t >= t_threshold:
                labels.iloc[i] = 1 if best_t > 0 else 0
            # else: NaN (uncertain)
        else:
            labels.iloc[i] = best_t  # raw t-stat for soft labels

    result = pd.DataFrame({
        "label": labels,          # 1=steepening, 0=flattening
        "t_stat": t_stats,
        "optimal_h": opt_h,
    }, index=index)

    # Drop terminal observations where we can't scan a full H_min window
    result.iloc[-(H_min):] = np.nan
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STREAM B: Sovereign Spread Widening / Tightening Label
# ─────────────────────────────────────────────────────────────────────────────

STREAM_B_COUNTRIES = ["IT", "ES", "FR", "PT", "GR", "BE", "AT", "NL", "FI", "IE"]

def compute_sovereign_spread(
    country_yield: pd.Series,
    bund_yield: pd.Series,
    country_code: str,
) -> pd.Series:
    """
    Spread of country 10Y yield vs 10Y Bund (bps).
    Label convention: 1 = widening (country underperforms), 0 = tightening.

    RISK NOTE: For STREAM B the sign convention matters for portfolio
    interpretation:
      - Short Germany + Long Italy → profits when BTP-Bund spread WIDENS
      - Label=1 (widening) means the short-Bund trade wins
    Be consistent with your trading desk's P&L convention.
    """
    spread_bps = (country_yield - bund_yield) * 100
    spread_bps.name = f"spread_{country_code}_bund"
    return spread_bps


def build_stream_b_labels(
    country_spreads: pd.DataFrame,  # columns: IT, ES, FR, ...
    H_min: int = 4,
    H_max: int = 26,
    t_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Builds labels for all Stream B countries.

    Returns a multi-column DataFrame: label_{cc}, t_stat_{cc}, opt_h_{cc}

    ARCHITECTURAL CHOICE NOTE:
    Pooled labels share a single H across countries. Per-country labels
    allow H to be tuned separately (e.g. GR needs shorter H due to volatility).
    For Phase 1, use pooled H; revisit in Phase 2 once you have OOS results.
    """
    all_labels = {}
    for cc in country_spreads.columns:
        result = trend_scan_labels(
            country_spreads[cc],
            H_min=H_min,
            H_max=H_max,
            t_threshold=t_threshold,
        )
        all_labels[f"label_{cc}"] = result["label"]
        all_labels[f"t_stat_{cc}"] = result["t_stat"]
        all_labels[f"opt_h_{cc}"] = result["optimal_h"]

    return pd.DataFrame(all_labels, index=country_spreads.index)


# ─────────────────────────────────────────────────────────────────────────────
# PURGE GAP CALCULATION  ← CRITICAL for CPCV correctness
# ─────────────────────────────────────────────────────────────────────────────

def compute_purge_weeks(
    H_max: int,
    publication_lag_weeks: int = 4,
    settlement_days: int = 2,
) -> int:
    """
    Minimum purge gap for CPCV splits (in weeks).

    Components:
      1. H_max: forward-looking window used in label construction
      2. publication_lag: macro data (GDP, HICP) published 4–6 weeks after period end
      3. settlement: bond settlement T+2 creates a 2-business-day information gap

    EQUITY PAPER comparison: used purge=12 weeks for H_max=12.
    For fixed income with macro features, publication lag is the binding constraint.

    Formula: purge = H_max + publication_lag_weeks + ceil(settlement_days/5)

    Args:
        H_max:                 Maximum trend-scan horizon (weeks)
        publication_lag_weeks: Conservative macro data lag (default: 4 weeks = ~1 month)
        settlement_days:       Bond settlement lag (business days)

    Returns:
        purge_weeks: integer, use this as the embargo in CPCV
    """
    settlement_weeks = int(np.ceil(settlement_days / 5))
    total = H_max + publication_lag_weeks + settlement_weeks
    print(f"Purge gap: {H_max} (H_max) + {publication_lag_weeks} (pub lag) + "
          f"{settlement_weeks} (settlement) = {total} weeks")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# LABEL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def label_diagnostics(labels: pd.Series, stream: str = "A") -> dict:
    """
    Checks label quality before modelling. Flag issues that affect model training.

    Key checks:
      1. Class balance: severe imbalance (>70/30) degrades ROC-AUC; use class_weight
      2. Autocorrelation: high serial dependence means the purge gap may be insufficient
      3. Uncertain labels (NaN) fraction: >30% means H parameters need revisiting
      4. Regime cluster length: very long runs suggest the signal is too persistent
         to be actionable (curve regimes often last 2–3 years)
    """
    valid = labels.dropna()
    balance = valid.mean()
    autocorr_1 = valid.autocorr(lag=1)
    autocorr_4 = valid.autocorr(lag=4)
    nan_pct = labels.isna().mean()

    # Compute run lengths
    runs = (valid != valid.shift()).cumsum()
    run_lengths = valid.groupby(runs).count()
    median_run = run_lengths.median()

    result = {
        "stream": stream,
        "n_valid": len(valid),
        "n_nan": labels.isna().sum(),
        "nan_pct": nan_pct,
        "pct_label_1": balance,
        "pct_label_0": 1 - balance,
        "autocorr_lag1": autocorr_1,
        "autocorr_lag4": autocorr_4,
        "median_run_length_weeks": median_run,
    }

    # Warnings
    if balance < 0.35 or balance > 0.65:
        result["WARNING_balance"] = (
            f"Class imbalance: {balance:.1%} label=1. "
            "Use class_weight='balanced' in Logistic/GAM. "
            "XGBoost: set scale_pos_weight = n_neg / n_pos."
        )
    if autocorr_1 > 0.85:
        result["WARNING_autocorr"] = (
            f"High label autocorrelation ({autocorr_1:.2f}). "
            "Increase purge gap or reduce H_max."
        )
    if nan_pct > 0.30:
        result["WARNING_nan"] = (
            f"High NaN rate ({nan_pct:.1%}). "
            "Lower t_threshold or reduce H_min."
        )
    if median_run > 52:
        result["WARNING_run_length"] = (
            f"Median run length = {median_run:.0f} weeks (>1 year). "
            "Signal may capture structural regimes rather than tactical transitions. "
            "Consider whether H_max is too large for portfolio rebalancing frequency."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# QUICK USAGE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic example: replace with actual Bloomberg/ECB data
    np.random.seed(42)
    dates = pd.date_range("2007-01-05", "2025-03-28", freq="W-FRI")
    n = len(dates)

    # Simulate 10Y and 2Y Bund yields (rough regime structure)
    y2 = 2.5 + np.cumsum(np.random.normal(0, 0.03, n) * 0.1)
    y10 = y2 + 1.0 + np.cumsum(np.random.normal(0, 0.02, n) * 0.1)
    y2 = pd.Series(y2, index=dates, name="Bund_2Y")
    y10 = pd.Series(y10, index=dates, name="Bund_10Y")

    # Stream A: Curve spread
    spread_A = compute_curve_spread(y10, y2)
    labels_A = trend_scan_labels(spread_A, H_min=4, H_max=26, t_threshold=1.0)
    diag_A = label_diagnostics(labels_A["label"], stream="A")
    print("Stream A diagnostics:", diag_A)

    # Purge gap
    purge = compute_purge_weeks(H_max=26, publication_lag_weeks=4)
    print(f"\nUse purge_gap = {purge} weeks in CPCV splits")

    # Stream B: Italian spread
    y_it = y10 + 1.5 + np.cumsum(np.random.normal(0, 0.05, n) * 0.1)
    y_it = pd.Series(y_it, index=dates)
    spread_B_IT = compute_sovereign_spread(y_it, y10, "IT")
    labels_B = trend_scan_labels(spread_B_IT, H_min=4, H_max=26)
    diag_B = label_diagnostics(labels_B["label"], stream="B_IT")
    print("\nStream B (IT) diagnostics:", diag_B)
