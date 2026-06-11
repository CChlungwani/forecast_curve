"""
Complete Methodology Translation Map
Equity Factor Timing (Hlungwani 2026) → Euro Area Fixed Income

Every component is mapped with:
  - Source: what the equity paper does
  - Target: what the FI framework does
  - Departure type: None / Extension / Replacement / New
  - Implementation risk level
"""

TRANSLATION_MAP = {

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: TARGET VARIABLE CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════
    "target_variable": {
        "equity_source": {
            "description": "Relative return: R_i,t - R_j,t between factor pairs",
            "series_type": "Return differential (%), weekly",
            "n_targets": 3,  # MOM-VAL, MOM-QUAL, VAL-QUAL
        },
        "fi_target": {
            "stream_a": "Change in 10Y-2Y Bund spread level (bps), weekly",
            "stream_b": "Country spread vs Bund 10Y (bps), per country, weekly",
            "n_targets_a": 1,
            "n_targets_b": "Up to 10 (1 per EA country)",
            "key_difference": (
                "FI target is a LEVEL spread, not a return differential. "
                "Mean reversion is more prominent. Trend-scan must operate "
                "on levels, not compounded returns (equation 2 in paper is removed)."
            ),
        },
        "departure_type": "Extension",
        "implementation_risk": "LOW: straightforward; just skip compounding step",
        "formula_change": {
            "equity": "CRi,j,t = ∏(1 + Ri,j,τ)  [cumulative product]",
            "fi_stream_a": "S_t = Y10Y_t - Y2Y_t  [spread level in bps]",
            "fi_stream_b": "S_c,t = Y_c,t - Y_DE,t  [country vs Bund in bps]",
        },
    },

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: TREND-SCANNING LABELS
    # ═══════════════════════════════════════════════════════════════════
    "trend_scanning": {
        "equity_source": {
            "H_max": 12,  # weeks
            "scan_on": "Compounded cumulative return series",
            "label_1_means": "Factor i outperforms factor j",
            "purge_gap": 12,  # = H_max
        },
        "fi_target": {
            "H_max_recommendation": "26 weeks (Stream A); 13–26 weeks (Stream B by country)",
            "scan_on": "Spread level series (bps)",
            "label_1_means": {
                "stream_a": "Curve steepening (10Y-2Y spread rising)",
                "stream_b": "Spread widening (country spread rising vs Bund)",
            },
            "purge_gap": "H_max + 4 (publication lag) + 1 (settlement) weeks",
            "t_threshold": "1.0 (same as equity paper; tune if label balance is poor)",
        },
        "departure_type": "Extension",
        "implementation_risk": "LOW",
        "open_question": (
            "H_max choice for Stream A has not been specified. "
            "RECOMMENDATION: run label diagnostics across H ∈ {8, 13, 20, 26, 39} "
            "and select H that gives: (a) 40-60% label balance, "
            "(b) autocorr(lag=1) < 0.85, (c) median run < 52 weeks."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: FEATURE UNIVERSE
    # ═══════════════════════════════════════════════════════════════════
    "feature_universe": {
        "equity_source": {
            "n_raw": 359,
            "categories": ["Market-based", "Macro", "Fundamental", "Sentiment", "Factor-specific"],
            "factor_specific_n": 18,
            "selected_after_MI": 50,  # 32 clusters + 18 factor-specific
        },
        "fi_target": {
            "n_raw_estimated": "250–400 (see 02_feature_universe.py)",
            "categories_added": [
                "ECB policy and balance sheet",
                "Breakeven inflation",
                "Bund scarcity / repo",
                "Sovereign CDS and fiscal",
                "ECB purchase program data (APP/PEPP/TPI)",
                "Political risk (Stream B)",
            ],
            "categories_removed_or_renamed": [
                "Factor-specific equity → replaced by spread dynamics (18 vars)",
                "SA-specific (PPP rand, SA CDS) → replaced by EUR-specific equivalents",
            ],
            "target_after_MI": "~50 (same budget as equity paper)",
        },
        "departure_type": "Replacement",
        "implementation_risk": "MEDIUM: data sourcing for ECB purchase data and fiscal variables",
        "critical_feature_risk": (
            "Bund scarcity (repo specialness) is the single largest microstructure "
            "risk for Stream A. If omitted, the model may learn repo distortions "
            "as yield curve signals. Include SWAP_SPREAD_2Y as a proxy at minimum."
        ),
    },

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: MUTUAL INFORMATION CLUSTERING
    # ═══════════════════════════════════════════════════════════════════
    "mi_clustering": {
        "equity_source": {
            "method": "Hierarchical agglomerative clustering, Ward linkage",
            "distance_metric": "Normalised mutual information distance",
            "n_clusters": 32,
            "cluster_rep_selection": "Lowest average distance to cluster members (unsupervised)",
            "key_property": "Capture nonlinear dependencies, not just Pearson correlations",
        },
        "fi_target": {
            "method": "IDENTICAL — no change needed",
            "recommended_n_clusters": "32 (same; adjust if raw feature count differs significantly)",
            "additional_consideration": (
                "ECB regime breaks may cause MI between a feature and the target "
                "to shift dramatically across regimes. "
                "Consider computing MI matrix SEPARATELY within each regime "
                "and using the union of cluster reps across regimes. "
                "This is an extension beyond the equity paper."
            ),
        },
        "departure_type": "None (identical algorithm)",
        "implementation_risk": "LOW",
        "code_reuse": "Full reuse of equity paper's MI clustering implementation",
    },

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: MODEL CLASSES
    # ═══════════════════════════════════════════════════════════════════
    "model_classes": {
        "equity_source": {
            "models": ["Logistic Regression (elastic-net)", "GAM", "XGBoost"],
            "hierarchy": "Linear → Semi-parametric → Nonlinear",
            "objective": "Binary cross-entropy with class-weight",
        },
        "fi_target": {
            "models": "IDENTICAL — no change needed",
            "key_difference": (
                "Class imbalance may be more severe in FI (e.g., extended "
                "spread-widening regimes for GR/PT in 2010-2015). "
                "Set scale_pos_weight in XGBoost = n_label0 / n_label1. "
                "For Logistic/GAM: class_weight='balanced'."
            ),
            "additional_model_to_consider": (
                "Cox-Ingersoll-Ross (CIR)-inspired GAM smooth: "
                "spread levels exhibit mean-reversion that GAM splines can capture "
                "if the spread is included as a predictor (lagged). "
                "This is beyond the equity paper but economically motivated."
            ),
        },
        "departure_type": "None (identical)",
        "implementation_risk": "LOW",
        "code_reuse": "Full reuse",
    },

    # ═══════════════════════════════════════════════════════════════════
    # STEP 6: HYPERPARAMETER TUNING (CPCV)
    # ═══════════════════════════════════════════════════════════════════
    "cpcv": {
        "equity_source": {
            "N": 5, "k": 1,
            "purge_gap": 12,
            "n_paths": 5,   # ϕ(5,1) = 5
            "retuning_frequency": "Annual",
        },
        "fi_target": {
            "N": 5, "k": 1,
            "purge_gap": "H_max + 5 weeks (see 01_label_construction.py)",
            "n_paths": 5,
            "retuning_frequency": "Annual (same); consider semi-annual if hiking cycle warrants",
            "key_difference": (
                "Pooled Stream B model needs time-based CPCV splits "
                "(see 04_stream_b_architecture.py::pooled_cpcv_splits). "
                "Per-country models use identical CPCV to the equity paper."
            ),
        },
        "departure_type": "Extension for pooled; None for per-country",
        "implementation_risk": "LOW for per-country; MEDIUM for pooled",
    },

    # ═══════════════════════════════════════════════════════════════════
    # STEP 7: WALK-FORWARD VALIDATION
    # ═══════════════════════════════════════════════════════════════════
    "walk_forward": {
        "equity_source": {
            "training_window": 773,   # weeks (~15 years)
            "oos_horizon": 156,       # weeks (3 years)
            "rebalance_freq": 52,     # annual parameter re-tuning
            "n_folds": 3,
        },
        "fi_target": {
            "training_window": "773 weeks if full history available; "
                               "consider 520 weeks (10 years) to reduce pre-QE contamination",
            "oos_horizon": "156 weeks (same 3-year OOS)",
            "rebalance_freq": "52 weeks (annual) initially; review after Phase 1",
            "n_folds": "3 (same); will cover ~2022–2025 OOS period",
            "critical_note": (
                "The 3-year OOS period (approx 2022–2025) spans the "
                "ECB hiking cycle — one of the most significant regime shifts "
                "in 40 years. This is BOTH an opportunity (real stress test) "
                "and a risk (model trained on QE era may fail completely). "
                "Document this regime mismatch explicitly in results."
            ),
        },
        "departure_type": "Minor extension",
        "implementation_risk": "LOW",
    },

    # ═══════════════════════════════════════════════════════════════════
    # STEP 8: PERFORMANCE METRICS
    # ═══════════════════════════════════════════════════════════════════
    "performance_metrics": {
        "equity_source": {
            "primary": ["Accuracy", "ROC-AUC", "Log-loss"],
            "optimisation_target": "ROC-AUC (in CPCV)",
        },
        "fi_target": {
            "same_metrics": ["Accuracy", "ROC-AUC", "Log-loss"],
            "additional_metrics": [
                "PR-AUC: Precision-Recall AUC, better for imbalanced classes "
                "(relevant if curve regimes are prolonged)",
                "Brier Score: proper scoring rule, alternative to log-loss",
                "Hit rate by regime: accuracy conditional on ECB regime "
                "(key diagnostic for regime-break handling)",
            ],
            "economic_translation": {
                "Stream A": (
                    "100bps correct steepening signal ≈ long 10Y / short 2Y position. "
                    "P&L depends on DV01 of the trade, not just label accuracy. "
                    "Consider weighting accuracy by |Δspread| to capture economic value."
                ),
                "Stream B": (
                    "Correct widening signal on 100bps BTP spread ≈ basis of country underweight. "
                    "Translate to portfolio weight changes for Phase 3."
                ),
            },
        },
        "departure_type": "Extension",
        "implementation_risk": "LOW",
    },

    # ═══════════════════════════════════════════════════════════════════
    # STEP 9: SHAPLEY INTERPRETABILITY
    # ═══════════════════════════════════════════════════════════════════
    "shap": {
        "equity_source": {
            "method": "SHAP in log-odds space for comparability across models",
            "global": "Mean absolute SHAP per training window",
            "local": "Regime-conditional attribution",
        },
        "fi_target": {
            "method": "IDENTICAL",
            "additional_analyses": [
                "ECB meeting weeks: SHAP on ECB policy features should spike "
                "around Governing Council dates → validates that model is learning policy",
                "Crisis periods (2010–12, 2020): SHAP on risk/sentiment features "
                "should dominate → validates crisis regime detection",
                "Regime-conditional SHAP: compare feature importance in "
                "pre-QE vs QE vs hiking regimes",
            ],
        },
        "departure_type": "Extension (richer economic interpretation)",
        "implementation_risk": "LOW",
        "code_reuse": "Full reuse of SHAP framework",
    },

    # ═══════════════════════════════════════════════════════════════════
    # NEW: ECB REGIME BREAKS (no equity equivalent)
    # ═══════════════════════════════════════════════════════════════════
    "ecb_regime_breaks": {
        "equity_source": "NOT PRESENT (acknowledged as limitation in paper)",
        "fi_target": {
            "required": True,
            "approach": "See 03_regime_breaks.py",
            "minimum_viable": "Regime dummies as features (Phase 1)",
            "full_treatment": "Regime-segmented training + within-regime z-score (Phase 3)",
        },
        "departure_type": "New",
        "implementation_risk": "HIGH if ignored; LOW once implemented with Phase 1 approach",
    },

    # ═══════════════════════════════════════════════════════════════════
    # NEW: STREAM B ARCHITECTURE (no equity equivalent)
    # ═══════════════════════════════════════════════════════════════════
    "stream_b_architecture": {
        "equity_source": "Three independent pairwise models (close analogy to per-country)",
        "fi_target": {
            "recommended": "Per-country (Phase 1: IT, ES, FR; Phase 2: all 10)",
            "alternatives": "Pooled (baseline) and Hierarchical (Phase 3 extension)",
            "decision_criteria": "Run pooled and per-country in parallel; "
                                 "compare OOS ROC-AUC; proceed with better performer",
        },
        "departure_type": "Extension",
        "implementation_risk": "MEDIUM",
    },
}


def print_translation_summary():
    print("\n" + "=" * 75)
    print("EQUITY → FIXED INCOME TRANSLATION SUMMARY")
    print("=" * 75)
    print(f"{'Component':<30} {'Departure':<18} {'Risk':<10} {'Code Reuse'}")
    print("-" * 75)

    for component, info in TRANSLATION_MAP.items():
        dep = info.get("departure_type", "?")
        risk = info.get("implementation_risk", "?")[:6]
        reuse = "FULL" if "Full reuse" in info.get("code_reuse", "") else \
                "PARTIAL" if "reuse" in str(info).lower() else "NEW"
        print(f"{component:<30} {dep:<18} {risk:<10} {reuse}")

    print("\nKEY OPEN DECISIONS:")
    for component, info in TRANSLATION_MAP.items():
        if "open_question" in info.get("fi_target", {}):
            print(f"\n  [{component}]")
            print(f"  {info['fi_target']['open_question'][:100]}...")


if __name__ == "__main__":
    print_translation_summary()
