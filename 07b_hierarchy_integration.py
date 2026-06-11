"""
Updated Model Hierarchy and Comparison Framework
Integrates LSTM/GRU as the 4th level of the hierarchy.

Equity paper hierarchy (3 levels):
  Logistic → GAM → XGBoost

Extended FI hierarchy (4 levels):
  Logistic → GAM → XGBoost → LSTM/GRU

This module documents:
  1. What LSTM/GRU is expected to ADD over XGBoost
  2. Where XGBoost will likely STILL WIN
  3. Evaluation design to distinguish signal from complexity
  4. How to report the extended comparison
"""

# ─────────────────────────────────────────────────────────────────────────────
# WHAT LSTM/GRU CAN CAPTURE THAT XGBOOST CANNOT
# ─────────────────────────────────────────────────────────────────────────────

THEORETICAL_ADVANTAGES = {
    "path_dependency": {
        "description": (
            "XGBoost sees a SINGLE feature vector at time t (lagged predictors). "
            "LSTM/GRU sees a SEQUENCE of vectors: [X_{t-seq_len}, ..., X_{t-1}]. "
            "This allows the model to condition on how a variable ARRIVED at its "
            "current level, not just its current value."
        ),
        "fi_example": (
            "ECB hiking from 0% to 4% in 12 months vs ECB cutting from 4% to 0%: "
            "DFR level = 2% looks identical to XGBoost at a point-in-time snapshot. "
            "GRU sees the path and can distinguish 'still hiking' vs 'cutting through'. "
            "For yield curve steepening: bear-steepening (hikes) vs bull-steepening "
            "(cuts) have opposite implications for duration positioning."
        ),
        "expected_impact": "HIGH for Stream A (curve regime transitions)",
    },
    "regime_transition_detection": {
        "description": (
            "Recurrent hidden state accumulates evidence across multiple weeks. "
            "A regime transition that unfolds over 4-8 weeks (e.g., ECB pivoting "
            "from hawkish to dovish language) can be captured as the hidden state "
            "evolves — whereas XGBoost can only see the instantaneous signal."
        ),
        "fi_example": (
            "Italian spread widening episodes: typically begin with slow drift "
            "over 3-4 weeks, then accelerate. GRU hidden state can track the "
            "accumulation of early-warning signals (rising CDS, falling PMI, "
            "political news) before the full widening materialises."
        ),
        "expected_impact": "HIGH for Stream B (crisis onset detection)",
    },
    "temporal_attention": {
        "description": (
            "The attention mechanism allows the model to downweight irrelevant "
            "historical weeks and upweight key ECB meeting dates, data releases, "
            "or volatility spikes that contain regime-shift information."
        ),
        "fi_example": (
            "ECB Governing Council meetings (every 6 weeks): the week of and "
            "week after a meeting carry disproportionate information. "
            "Attention weights should spike on these dates."
        ),
        "expected_impact": "MEDIUM — adds interpretability value even if AUC gain is small",
    },
    "gradient_flow_through_time": {
        "description": (
            "GRU's reset and update gates allow gradients to flow back selectively "
            "through time. This means the model can learn that a spread widening "
            "observed 10 weeks ago is still relevant for today's prediction — "
            "something XGBoost cannot represent without explicit feature engineering."
        ),
        "fi_example": (
            "ECB purchase reinvestment schedule: a large maturity 8 weeks ago "
            "(which was reinvested) creates buying pressure that decays over the "
            "following weeks. GRU can learn this decay; XGBoost would need a "
            "hand-crafted decay feature."
        ),
        "expected_impact": "MEDIUM for Stream B (ECB technical flows)",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# WHERE XGBOOST WILL LIKELY STILL WIN
# ─────────────────────────────────────────────────────────────────────────────

XGBOOST_ADVANTAGES_RETAINED = {
    "valuation_mean_reversion": (
        "Value vs Quality dynamics in the equity paper were dominated by "
        "valuation spreads — a point-in-time signal. XGBoost handles "
        "this perfectly. The FI equivalent (carry, ASW spread z-score) "
        "is similar. GRU adds little for valuation-mean-reversion signals."
    ),
    "high_dimensional_interactions": (
        "XGBoost's tree structure explicitly searches all feature interactions. "
        "GRU learns temporal interactions but not necessarily cross-feature "
        "interactions at a single time step. For Stream B, the interaction "
        "between fiscal variables and ECB purchase data may favour XGBoost."
    ),
    "calibration": (
        "The equity paper showed that XGBoost and complex models have poor "
        "probability calibration (high log-loss). GRU will have the same issue. "
        "Logistic regression will likely remain best-calibrated, and Platt "
        "scaling will be needed for GRU outputs in production."
    ),
    "regime_stable_periods": (
        "In the NIRP/QE regime (2014-2021), signals were dominated by "
        "ECB policy anchoring. Point-in-time predictors likely captured "
        "this well. GRU's sequential advantage is smaller when dynamics "
        "are slow and regime-stable."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# UPDATED COMPARISON TABLE (extended from paper's Table 2)
# ─────────────────────────────────────────────────────────────────────────────

EXTENDED_MODEL_COMPARISON = {
    "Logistic Regression": {
        "complexity": "Low",
        "interpretability": "High",
        "temporal_memory": "None (lagged features only)",
        "regime_handling": "Via dummy features",
        "calibration": "Best (consistent with equity paper)",
        "key_hyperparameters": ["λ (regularisation)", "α (L1/L2 mix)", "learning_rate"],
        "cpcv_combinations": 243,
        "expected_role": "Baseline + calibration reference",
    },
    "GAM": {
        "complexity": "Moderate",
        "interpretability": "Moderate–High",
        "temporal_memory": "None (nonlinear marginal effects only)",
        "regime_handling": "Via smooth functions of regime dummies",
        "calibration": "Moderate",
        "key_hyperparameters": ["n_splines", "smoothness_penalty λ_i"],
        "cpcv_combinations": 12,
        "expected_role": "Nonlinearity baseline; ECB policy smooth effect analysis",
    },
    "XGBoost": {
        "complexity": "High",
        "interpretability": "Low (TreeSHAP recovers it)",
        "temporal_memory": "None (hand-crafted lag features only)",
        "regime_handling": "Via regime dummy features + interaction splits",
        "calibration": "Poor (overconfident)",
        "key_hyperparameters": ["max_depth", "learning_rate η", "subsample", "λ/γ"],
        "cpcv_combinations": 243,
        "expected_role": "Primary discriminator; best AUC expected (consistent with equity paper)",
    },
    "GRU (primary recurrent)": {
        "complexity": "High",
        "interpretability": "Low (gradient SHAP + attention weights)",
        "temporal_memory": "EXPLICIT: seq_len weeks of sequential hidden state",
        "regime_handling": "Via hidden state + optional regime embedding channel",
        "calibration": "Poor (requires Platt scaling)",
        "key_hyperparameters": ["seq_len", "hidden_size", "dropout", "lr"],
        "cpcv_combinations": 4,   # Phase 1 grid
        "expected_role": "Regime-transition detector; path-dependent predictions",
    },
    "LSTM (secondary recurrent)": {
        "complexity": "High",
        "interpretability": "Low (gradient SHAP)",
        "temporal_memory": "EXPLICIT: cell state allows longer memory than GRU",
        "regime_handling": "Via cell + hidden state + optional regime embedding",
        "calibration": "Poor (requires Platt scaling)",
        "key_hyperparameters": ["seq_len", "hidden_size", "dropout", "lr"],
        "cpcv_combinations": 4,   # Phase 1 grid
        "expected_role": "Ablation vs GRU; may outperform on multi-year regime cycles",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION DESIGN TO DISTINGUISH SIGNAL FROM COMPLEXITY
# ─────────────────────────────────────────────────────────────────────────────

EVALUATION_DESIGN = {
    "primary_metrics": {
        "ROC_AUC": "Threshold-independent discrimination (primary, consistent with equity paper)",
        "accuracy": "Directional accuracy (secondary)",
        "log_loss": "Calibration quality (Platt-scaled and raw)",
        "brier_score": "Proper scoring rule; combines discrimination + calibration",
    },
    "regime_conditional_metrics": {
        "description": (
            "Split OOS period by ECB regime and report metrics within each regime. "
            "This is the KEY diagnostic for whether GRU adds value over XGBoost."
        ),
        "hypothesis": {
            "H1": "GRU > XGBoost in ROC-AUC DURING regime transitions (±12 weeks of breaks)",
            "H2": "GRU ≈ XGBoost in ROC-AUC WITHIN stable regime periods",
            "H3": "LSTM ≈ GRU overall, with LSTM marginally better on slow multi-year regimes",
        },
        "implementation": (
            "Tag each OOS week with: (a) regime, (b) weeks_since_last_break. "
            "Report ROC-AUC for: all OOS / transition windows / within-regime stable. "
            "If H1 holds: GRU earns its place in the hierarchy."
        ),
    },
    "attention_analysis": {
        "description": (
            "Plot mean attention weights (averaged over OOS predictions) "
            "against the calendar. Hypothesis: attention spikes should "
            "align with ECB meeting dates, macro data releases, and regime breaks."
        ),
        "implementation": (
            "Modify predict_proba() to also return attention_weights tensor. "
            "Average attention[week_offset] across all OOS sequences. "
            "Overlay on ECB meeting calendar."
        ),
    },
    "ablation_sequence_length": {
        "description": "Test seq_len ∈ {12, 20, 26} to isolate optimal memory horizon.",
        "hypothesis": (
            "Stream A: seq_len=20-26 (ECB policy cycles 6-weekly; need 3-4 meetings). "
            "Stream B: seq_len=13-20 (spread regimes are faster-moving)."
        ),
    },
    "feasibility_test_first": {
        "description": (
            "Before full walk-forward: run GRU on a single 4-year in-sample segment "
            "with and without regime conditioning. If in-sample AUC < 0.55: "
            "the model is not learning anything; do not proceed to walk-forward."
        ),
        "threshold": "In-sample AUC > 0.60 required to proceed (generous; financial data is noisy)",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# RISK REGISTER FOR LSTM/GRU ADDITION
# ─────────────────────────────────────────────────────────────────────────────

RISK_REGISTER = {
    "R1_overfitting": {
        "description": "With 690 sequences and ~3k-8k parameters, overfitting is real.",
        "severity": "HIGH",
        "mitigation": [
            "Dropout (0.3 default, tune via CPCV)",
            "Weight decay (L2 = 1e-3)",
            "Early stopping with 20-epoch patience on purged validation fold",
            "Keep hidden_size ≤ 32 in Phase 1",
            "Phase 1 CPCV grid: only 4 combinations (not 96)",
        ],
    },
    "R2_gradient_instability": {
        "description": "LSTMs are prone to exploding/vanishing gradients.",
        "severity": "MEDIUM",
        "mitigation": [
            "Gradient clipping (max_norm=1.0) — implemented in train_recurrent_model()",
            "LayerNorm after projection (stabilises activation variance)",
            "GELU activation (smoother gradient flow than ReLU)",
            "Use GRU as default (fewer gates = stabler gradients than LSTM)",
        ],
    },
    "R3_nondeterminism": {
        "description": (
            "Neural nets have random initialisation. Same hyperparameters can "
            "give different OOS results across runs — problematic for CPCV comparison."
        ),
        "severity": "MEDIUM",
        "mitigation": [
            "Fix seed per walk-forward fold (seed = fold_number * 42)",
            "Average predictions across 3 random seeds for final OOS results",
            "Report mean ± std of ROC-AUC across seeds",
        ],
    },
    "R4_shap_approximation": {
        "description": (
            "GradientExplainer is approximate (unlike TreeSHAP for XGBoost). "
            "SHAP values for LSTM/GRU are less reliable than for tree models."
        ),
        "severity": "LOW",
        "mitigation": [
            "Use GradientExplainer (more stable than DeepExplainer for LSTM)",
            "Interpret SHAP as directional guidance only",
            "Cross-validate against attention weights for ECB-related features",
            "Flag in methodology section: 'approximate Shapley for recurrent models'",
        ],
    },
    "R5_calibration": {
        "description": (
            "GRU/LSTM will likely have high log-loss (same as XGBoost in equity paper). "
            "Raw probabilities are not suitable for probability-weighted allocation."
        ),
        "severity": "MEDIUM",
        "mitigation": [
            "Apply Platt scaling (implemented in platt_scale())",
            "Fit calibration on CPCV validation fold, not training fold",
            "Report both raw and calibrated log-loss",
        ],
    },
    "R6_computational_cost": {
        "description": "GRU/LSTM training is 5-10× slower than XGBoost on CPU.",
        "severity": "LOW (manageable)",
        "mitigation": [
            "Use GPU if available (device='cuda')",
            "Phase 1 grid = 4 combinations only",
            "CPU training of hidden=32 GRU on 690 sequences ≈ 30-60 min per walk-forward fold",
            "Run in parallel across countries for Stream B",
        ],
    },
}


if __name__ == "__main__":
    print("=" * 70)
    print("EXTENDED MODEL HIERARCHY: KEY DESIGN DECISIONS")
    print("=" * 70)

    print("\n--- Where GRU/LSTM adds value over XGBoost ---")
    for adv, info in THEORETICAL_ADVANTAGES.items():
        print(f"\n  {adv} [{info['expected_impact']}]")
        print(f"  FI example: {info['fi_example'][:90]}...")

    print("\n\n--- Extended Model Comparison ---")
    header = f"{'Model':<30} {'Temporal Memory':<30} {'CPCV combos':<14} {'Expected Role'}"
    print(header)
    print("-" * 100)
    for model, info in EXTENDED_MODEL_COMPARISON.items():
        row = (f"{model:<30} {info['temporal_memory']:<30} "
               f"{info['cpcv_combinations']:<14} {info['expected_role']}")
        print(row)

    print("\n\n--- Risk Register Summary ---")
    for risk_id, risk in RISK_REGISTER.items():
        print(f"  {risk_id} [{risk['severity']}]: {risk['description'][:70]}...")
        print(f"    Primary mitigation: {risk['mitigation'][0]}")
