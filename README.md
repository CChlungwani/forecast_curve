# Euro Area Fixed Income ML Framework
## Translation of Hlungwani (2026) Equity Factor Timing → Sovereign Bond Markets

---

## Framework Overview

| Module | File | Status |
|--------|------|--------|
| Label Construction | `01_label_construction.py` | ✓ Complete |
| Feature Universe | `02_feature_universe.py` | ✓ Complete |
| Regime Breaks | `03_regime_breaks.py` | ✓ Complete |
| Stream B Architecture | `04_stream_b_architecture.py` | ✓ Complete |
| Translation Map | `05_methodology_translation_map.py` | ✓ Complete |
| Horizon Calibration | `06_horizon_calibration.py` | ✓ Complete |

---

## Key Decisions Summary

### 1. Investment Horizon H (UNRESOLVED)
**Recommendation:** Run `scan_optimal_H()` on actual data first.
**Starting point:** H_max = 26 weeks for Stream A; H_max = 26 (IT/ES), 20 (FR/PT), 13 (core) for Stream B.
**Binding constraint:** Purge gap = H_max + 4 (pub lag) + 1 (settlement) weeks.

### 2. Stream B Architecture (UNRESOLVED)
**Phase 1:** Per-country models for IT, ES, FR. Pooled as baseline.
**Decision rule:** Compare OOS ROC-AUC pooled vs per-country after Phase 1.
**Hierarchical:** Phase 3 only if per-country shows evidence of cross-country contagion signal.

### 3. ECB Regime Breaks (PARTIALLY RESOLVED)
**Phase 1 minimum:** Regime dummies as features (`build_regime_dummies()`).
**Phase 2:** Within-regime z-score for rate-level variables.
**Phase 3:** Regime-segmented training windows.

### 4. Signal Integration (UNRESOLVED)
**Phase 1:** Independent signals (simplest, no cross-contamination).
**Phase 2:** Test conditional integration (Stream A conditions Stream B).
**Metric:** Does conditional improve portfolio Sharpe vs independent?

---

## Critical Methodological Risks vs Equity Paper

| Risk | Severity | Mitigation |
|------|----------|------------|
| Bund scarcity distorts curve signal | HIGH | Include SWAP_SPREAD_2Y as feature |
| ECB regime breaks invalidate pre-QE training | HIGH | Regime dummies (Phase 1); segmented window (Phase 3) |
| Fiscal data annual frequency → stale | MEDIUM | Use as regime filter, not timing feature |
| Pooled Stream B conflates IT and FI dynamics | MEDIUM | Use per-country as primary |
| Hiking cycle OOS period < 4 years | MEDIUM | Report OOS stats conditional on regime |
| Annual fiscal data has look-ahead in annual rebalancing | LOW | Always lag by full year |

---

## Formula Changes vs Equity Paper

### Removed (equity-specific):
- `CRi,j,t = ∏(1 + Ri,j,τ)` — compounding of factor returns (use spread level directly)

### Added (FI-specific):
- `S_A,t = Y10Y_t - Y2Y_t` — Stream A target (bps)
- `S_B,c,t = Y_c,t - Y_DE,t` — Stream B target per country (bps)
- `purge_gap = H_max + 4 + 1` — publication lag adjustment
- `regime_z(x_t) = (x_t − μ̄_regime) / σ̄_regime` — within-regime normalisation

### Unchanged:
- Trend-scanning OLS: `C_{t+h} = α + β_h·t + ε`; label = sign(β_{h*})
- MI clustering (Ward linkage, distance threshold = 2)
- All three model classes: Logistic, GAM, XGBoost
- CPCV: ϕ(N=5, k=1) = 5 paths; annual retuning
- Walk-forward: 773-week train, 156-week OOS, 52-week rebalance
- SHAP in log-odds space

---

## Data Sources Quick Reference

| Feature Block | Primary Source | Frequency | Lag |
|---------------|----------------|-----------|-----|
| Bund yields, OIS, swaps | Bloomberg BGN | Daily → weekly | T+0 |
| ECB DFR, balance sheet | ECB SDW | Weekly/Monthly | 1–2 weeks |
| HICP, GDP, PMI | Eurostat / Bloomberg | Monthly | 4–6 weeks |
| Sovereign CDS | Bloomberg FXMM | Daily | T+0 |
| ECB APP/PEPP purchases | ECB SDW | Weekly | 1 week |
| Fiscal ratios | Eurostat | Annual | ~3 months |
| VSTOXX, VIX | Bloomberg | Daily | T+0 |
| FX (EURUSD, EURJPY) | Bloomberg | Daily | T+0 |


---

## LSTM / GRU Extension (Module 07)

| File | Content |
|------|---------|
| `07_lstm_gru_module.py` | Architecture, training loop, SHAP, walk-forward wrapper |
| `07b_hierarchy_integration.py` | Where GRU adds value, evaluation design, risk register |

### Updated Model Hierarchy

```
Logistic → GAM → XGBoost → GRU (primary) → LSTM (ablation)
```

### Parameter Counts (n_features=50, with regime embedding)

| Model | Hidden | Params | P/690 seq | Verdict |
|-------|--------|--------|-----------|---------|
| GRU   | 16     | ~2,978 | 4.3       | OK      |
| GRU   | 32     | ~6,594 | 9.6       | OK      |
| LSTM  | 16     | ~3,570 | 5.2       | OK      |
| LSTM  | 32     | ~8,290 | 12.0      | OK      |

### Why GRU/LSTM earns its place here (vs equity paper's decision to exclude it)

The equity paper correctly excluded LSTM for South Africa: short history, high noise, emerging market microstructure. Euro area FI is different:

- **Path dependency is economically meaningful**: DFR at 2% while hiking vs cutting have opposite implications for curve steepening. XGBoost sees a number; GRU sees a trajectory.
- **Regime transitions are slower**: ECB policy shifts unfold over 4–8 weeks of meeting communications. A 26-week sequence window captures this accumulation.
- **Attention maps are interpretable**: Expected to spike on ECB meeting dates — directly testable and economically validating.

### Phase 1 CPCV Grid (4 combinations only)

```python
PHASE1_GRID = {
    "arch":         ["gru"],
    "seq_len":      [13, 26],
    "hidden_size":  [16, 32],
    "dropout":      [0.3],
    "lr":           [5e-4],
    "use_attention":[True],
}
```

### Critical Risks

| Risk | Mitigation |
|------|------------|
| Overfitting (HIGH) | dropout=0.3, weight_decay=1e-3, early stopping, hidden≤32 |
| Exploding gradients | gradient clipping max_norm=1.0, LayerNorm, GRU default |
| Nondeterminism | fixed seed per fold, average over 3 seeds |
| Poor calibration | Platt scaling after CPCV validation fold |
| Approximate SHAP | GradientExplainer + cross-validate vs attention weights |
