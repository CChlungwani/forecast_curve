"""
Investment Horizon H Calibration Guide
The single most important open decision in the framework.

The equity paper's H_max=12 was chosen by balancing:
  (a) Economic relevance: capture meaningful price trends
  (b) Data preservation: avoid excessive purge-gap shrinkage
  (c) Label balance: 40-60% class balance

For fixed income, the economically motivated horizon differs by stream
and depends on the ECB meeting cycle, data publication lags, and the
mean-reversion properties of spreads.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings


HORIZON_ECONOMIC_RATIONALE = {
    "Stream_A_curve": {
        4:  {
            "economic_cycle": "1 ECB meeting window",
            "pros": "Captures post-meeting repricing of front end",
            "cons": "Too noisy; dominated by short-term positioning noise",
            "typical_curve_move_bps": "5–15 bps in normal regimes",
            "recommendation": "AVOID as primary H; use as secondary label for robustness check",
        },
        8:  {
            "economic_cycle": "2 ECB meetings",
            "pros": "Can capture 2-meeting policy signal shifts",
            "cons": "Still short; inflation data revisions may not have landed",
            "typical_curve_move_bps": "10–30 bps",
            "recommendation": "Consider for high-volatility periods (hiking cycle)",
        },
        13: {
            "economic_cycle": "~1 quarter / 3 ECB meetings",
            "pros": "Aligns with GDP flash estimate release cycle; "
                    "quarterly earnings and credit review cycle",
            "cons": "May not capture full impact of policy decision",
            "typical_curve_move_bps": "20–50 bps",
            "recommendation": "MINIMUM viable for Stream A",
        },
        20: {
            "economic_cycle": "~5 months; intersects macro revision cycle",
            "pros": "Balances noise reduction with tactical relevance",
            "cons": "Purge gap = 20+5 = 25 weeks; meaningful sample cost",
            "typical_curve_move_bps": "30–80 bps",
            "recommendation": "GOOD starting point",
        },
        26: {
            "economic_cycle": "~6 months / 2 quarters",
            "pros": "Captures full seasonal cycle; consistent with semi-annual rebalancing; "
                    "used in equity paper for reference (H=12 weeks ≈ 1 quarter)",
            "cons": "Purge gap = 26+5 = 31 weeks; reduces effective training obs",
            "typical_curve_move_bps": "40–120 bps",
            "recommendation": "RECOMMENDED primary H for Stream A",
        },
        39: {
            "economic_cycle": "~9 months / 3 quarters",
            "pros": "Captures medium-term trend cleanly; fewer label flips",
            "cons": "Purge gap = 44 weeks; very large; may not be tactically useful",
            "typical_curve_move_bps": "50–150 bps",
            "recommendation": "Use only for long-horizon structural analysis",
        },
        52: {
            "economic_cycle": "1 year",
            "pros": "Smoothest labels; strong regime signal",
            "cons": "Not tactically actionable; overlapping labels dominate; "
                    "purge = 57 weeks is extremely costly",
            "recommendation": "AVOID for tactical portfolio use",
        },
    },
    "Stream_B_spreads": {
        "IT_ES": {
            "recommended_H_max": 26,
            "rationale": "High volatility; need longer horizon to find stable trend direction. "
                         "BTP-Bund has historically moved 100+ bps over 6-month cycles.",
            "label_balance_risk": "Moderate: widening dominated 2010-12, 2018, 2022.",
        },
        "PT_GR": {
            "recommended_H_max": 20,
            "rationale": "Very high vol (GR: hundreds of bps swings). "
                         "Shorter H reduces label autocorrelation for GR.",
            "label_balance_risk": "HIGH: GR widening dominated 2010-2015. "
                                  "Consider separating pre/post-ESM era.",
        },
        "FR_BE_AT": {
            "recommended_H_max": 20,
            "rationale": "Semi-core; moderate spread volatility (10-50 bps typical). "
                         "26 weeks may produce overly smooth labels for stable core.",
            "label_balance_risk": "LOW: well-balanced in most periods.",
        },
        "NL_FI_IE": {
            "recommended_H_max": 13,
            "rationale": "Core-like; spreads rarely exceed 50bps. Short H viable. "
                         "Many 'no trend' observations expected → accept higher NaN rate.",
            "label_balance_risk": "LOW: but high NaN rate likely; reduce t_threshold.",
        },
    },
}


def scan_optimal_H(
    spread_series: pd.Series,
    H_range: list = [8, 13, 16, 20, 26, 39],
    t_threshold: float = 1.0,
    target_balance: tuple = (0.38, 0.62),
    max_autocorr: float = 0.85,
    max_run_weeks: float = 52.0,
) -> pd.DataFrame:
    """
    Empirical H selection: scan across H values and evaluate label quality.

    Outputs a summary table ranking H values by a composite score
    that balances label balance, autocorrelation, and information efficiency.

    Args:
        spread_series: The spread to scan (Stream A: 10Y-2Y Bund; Stream B: country spread)
        H_range: List of H_max values to evaluate
        t_threshold: Minimum |t-stat| for label assignment (same as equity paper: 1.0)
        target_balance: Acceptable range for P(label=1)
        max_autocorr: Maximum acceptable label autocorrelation
        max_run_weeks: Maximum acceptable median run length

    Returns:
        DataFrame ranked by composite score
    """
    from .label_construction_module import trend_scan_labels, label_diagnostics
    # Note: in actual use, import from 01_label_construction
    # Shown here as a standalone function for clarity

    results = []
    n = len(spread_series)

    for H in H_range:
        try:
            # Compute labels (simplified inline for demonstration)
            labels = pd.Series(np.nan, index=spread_series.index)
            t_stats = pd.Series(np.nan, index=spread_series.index)

            for i in range(n - H):
                best_abs_t = -np.inf
                best_t = np.nan
                for h in range(4, H + 1):
                    if i + h > n:
                        break
                    y_win = spread_series.iloc[i:i+h].values
                    if np.any(np.isnan(y_win)):
                        continue
                    x_win = np.arange(h, dtype=float)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        slope, _, _, _, se = stats.linregress(x_win, y_win)
                    t = slope / se if se > 0 else 0.0
                    if abs(t) > best_abs_t:
                        best_abs_t = abs(t)
                        best_t = t
                if best_abs_t >= t_threshold:
                    labels.iloc[i] = 1 if best_t > 0 else 0
                t_stats.iloc[i] = best_t

            # Diagnostics
            valid = labels.dropna()
            balance = valid.mean()
            autocorr = valid.autocorr(lag=1)
            nan_pct = labels.isna().mean()
            runs = (valid != valid.shift()).cumsum()
            run_lengths = valid.groupby(runs).count()
            median_run = run_lengths.median()

            # Purge gap cost
            purge_weeks = H + 5  # H + publication lag

            # Compute effective training observations (approximate)
            # Full sample ~929 weeks; each fold loses purge_weeks
            eff_train_obs = int((n * 0.8) - 5 * purge_weeks)

            # Composite score: penalise imbalance, autocorr, short training set
            balance_score = 1.0 - abs(balance - 0.5) * 2  # 1=perfect, 0=worst
            autocorr_score = max(0, 1 - autocorr)
            run_score = max(0, 1 - median_run / 104)
            training_score = min(1, eff_train_obs / 500)
            composite = np.mean([balance_score, autocorr_score, run_score, training_score])

            # Flags
            flags = []
            if not (target_balance[0] <= balance <= target_balance[1]):
                flags.append(f"IMBALANCE ({balance:.1%})")
            if autocorr > max_autocorr:
                flags.append(f"HIGH_AUTOCORR ({autocorr:.2f})")
            if median_run > max_run_weeks:
                flags.append(f"LONG_RUNS ({median_run:.0f}wks)")
            if eff_train_obs < 300:
                flags.append(f"FEW_TRAIN_OBS ({eff_train_obs})")

            results.append({
                "H_max": H,
                "purge_gap": purge_weeks,
                "n_valid_labels": len(valid),
                "nan_pct": f"{nan_pct:.1%}",
                "label_balance": f"{balance:.1%}",
                "autocorr_lag1": f"{autocorr:.2f}",
                "median_run_weeks": f"{median_run:.1f}",
                "eff_train_obs": eff_train_obs,
                "composite_score": f"{composite:.3f}",
                "flags": "; ".join(flags) if flags else "OK",
            })

        except Exception as e:
            results.append({"H_max": H, "error": str(e)})

    df = pd.DataFrame(results)
    if "composite_score" in df.columns:
        df = df.sort_values("composite_score", ascending=False)
    return df


def print_horizon_guide():
    print("=" * 70)
    print("H_MAX CALIBRATION REFERENCE GUIDE")
    print("=" * 70)

    print("\n--- Stream A: Curve Slope (10Y-2Y Bund) ---")
    for h, info in HORIZON_ECONOMIC_RATIONALE["Stream_A_curve"].items():
        status = "✓ RECOMMENDED" if "RECOMMENDED" in info.get("recommendation", "") \
                 else "✓ MINIMUM" if "MINIMUM" in info.get("recommendation", "") \
                 else "✗ AVOID" if "AVOID" in info.get("recommendation", "") \
                 else "  OK"
        print(f"\n  H={h:>3} ({info['economic_cycle']}): {status}")
        print(f"         Pros: {info['pros'][:65]}")
        print(f"         Cons: {info['cons'][:65]}")

    print("\n\n--- Stream B: Sovereign Spreads (per country group) ---")
    stream_b = HORIZON_ECONOMIC_RATIONALE["Stream_B_spreads"]
    for group, info in stream_b.items():
        print(f"\n  {group}: H_max = {info['recommended_H_max']} weeks")
        print(f"    {info['rationale'][:80]}")

    print("\n\n--- DECISION RULE (run before committing to H) ---")
    print("""
  Step 1: Run scan_optimal_H() on your actual spread data.
  Step 2: Reject any H where flags ≠ "OK" for more than 1 category.
  Step 3: Among remaining H values, pick highest composite_score.
  Step 4: Verify economically: does H align with your rebalancing frequency?
  Step 5: If target_balance fails for all H: check for structural trend dominance
          (e.g., persistent steepening in 2022-23) → add HICP or DFR as
          stratification variable for class weighting.
    """)


if __name__ == "__main__":
    print_horizon_guide()

    # Synthetic example
    np.random.seed(42)
    dates = pd.date_range("2007-01-05", "2025-03-28", freq="W-FRI")
    n = len(dates)
    # Simulate curve spread with regime structure
    slope_breaks = [0, 300, 600, n]
    levels = [50, -30, 100, 200]  # bps: flat, inverted, steep, steepening
    spread_sim = []
    for i in range(len(slope_breaks)-1):
        seg_len = slope_breaks[i+1] - slope_breaks[i]
        seg = np.linspace(levels[i], levels[i+1], seg_len) + \
              np.random.normal(0, 10, seg_len)
        spread_sim.extend(seg)
    spread_sim = pd.Series(spread_sim[:n], index=dates)

    print("\n\nEmpirical H Scan (synthetic data):")
    print("-" * 50)
    print("Note: run scan_optimal_H(your_actual_spread) for real calibration")
    print("Expected output format:")
    print("H_max | purge_gap | n_valid | label_balance | autocorr | composite | flags")
