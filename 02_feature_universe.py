"""
Fixed Income Feature Universe
Adapted from Hlungwani (2026) equity feature set → Euro area sovereign bonds

The equity paper used ~359 raw features → 32 clusters + 18 factor-specific = 50 total.
The FI equivalent distinguishes:
  - Shared features (used by both Stream A and B)
  - Stream A–specific (curve-level predictors)
  - Stream B–specific (cross-country, fiscal, ECB purchase, political risk)

Each feature includes:
  - Transformation applied (to ensure approximate stationarity)
  - Economic rationale aligned with fixed income microstructure
  - Data source / Bloomberg ticker template
  - Methodological risk flag where applicable
"""

FEATURE_UNIVERSE = {

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 1: YIELD CURVE STRUCTURE  (both streams)
    # ─────────────────────────────────────────────────────────────────

    "BUND_SLOPE_2s10s": {
        "transform": "level_bps",           # 10Y - 2Y Bund, in bps
        "bloomberg": "GDBR10 Index - GDBR2 Index",
        "stream": ["A", "B"],
        "rationale": "Primary target proxy for Stream A; conditioning signal for B. "
                     "Mean-reverting over medium horizons → use level, not return.",
        "risk": "MULTICOLLINEARITY with slope components; use QS (quantile-scaled) version in model.",
    },
    "BUND_SLOPE_2s5s": {
        "transform": "level_bps",
        "bloomberg": "GDBR5 Index - GDBR2 Index",
        "stream": ["A", "B"],
        "rationale": "Belly of curve; sensitive to front-end ECB expectations.",
    },
    "BUND_SLOPE_5s10s": {
        "transform": "level_bps",
        "bloomberg": "GDBR10 Index - GDBR5 Index",
        "stream": ["A"],
        "rationale": "Long-end slope; driven by term premium and global duration supply.",
    },
    "BUND_SLOPE_10s30s": {
        "transform": "level_bps",
        "bloomberg": "GDBR30 Index - GDBR10 Index",
        "stream": ["A"],
        "rationale": "Ultra-long slope; ECB PSPP/PEPP purchases concentrated here.",
    },
    "NS_LEVEL": {
        "transform": "level_pct",           # Nelson-Siegel β₀ (overall yield level)
        "bloomberg": "Fit from GDBR2/5/10/30 Index",
        "stream": ["A", "B"],
        "rationale": "Parallel equivalent of level factor; captures rate regime.",
    },
    "NS_SLOPE": {
        "transform": "level",               # Nelson-Siegel β₁
        "bloomberg": "Fit from GDBR2/5/10/30 Index",
        "stream": ["A"],
        "rationale": "Slope of the fitted yield curve; complements raw 2s10s.",
    },
    "NS_CURVATURE": {
        "transform": "level",               # Nelson-Siegel β₂ (hump/dip)
        "bloomberg": "Fit from GDBR2/5/10/30 Index",
        "stream": ["A"],
        "rationale": "Mid-curve bulge/inversion; captures ZIRP/QE distortions.",
    },
    "EURSWAP_10Y": {
        "transform": "level_pct",
        "bloomberg": "EUSW10 Curncy",
        "stream": ["A", "B"],
        "rationale": "Swap rate provides rate expectation signal without Bund scarcity effects.",
    },
    "SWAP_SPREAD_2Y": {
        "transform": "level_bps",           # EUR 2Y swap minus 2Y Bund
        "bloomberg": "EUSW2 Curncy - GDBR2 Index",
        "stream": ["A", "B"],
        "rationale": "Bund scarcity proxy; widening swap spread = Bund richness distortion.",
        "risk": "CRITICAL for Stream B: sovereign ASW spreads conflate credit and swap-Bund basis.",
    },
    "EURSWAP_2s10s": {
        "transform": "level_bps",
        "bloomberg": "EUSW10 Curncy - EUSW2 Curncy",
        "stream": ["A"],
        "rationale": "Swap curve slope; free of Bund scarcity → cleaner curve signal.",
    },
    "EURSWAP_5s30s": {
        "transform": "level_bps",
        "bloomberg": "EUSW30 Curncy - EUSW5 Curncy",
        "stream": ["A"],
        "rationale": "Long-end swap slope; captures QE/QT impact on term premium.",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 2: ECB POLICY AND EXPECTATIONS  (both streams)
    # ─────────────────────────────────────────────────────────────────

    "ECB_DFR": {
        "transform": "level_pct",           # ECB Deposit Facility Rate
        "bloomberg": "EURR002W Index",       # ECB DFR
        "stream": ["A", "B"],
        "rationale": "Front-end anchor; regime breaks at ZLB, liftoff, normalization.",
        "risk": "REGIME BREAK: requires explicit structural-break indicator (see ECBRegimeBreak).",
    },
    "OIS_1M": {
        "transform": "level_pct",
        "bloomberg": "EUSWFC Curncy",        # EUR OIS 1M (ESTER)
        "stream": ["A", "B"],
        "rationale": "Short-dated rate expectations; more reliable than EURIBOR after reform.",
    },
    "OIS_3M": {
        "transform": "level_pct",
        "bloomberg": "EUSW3M Curncy",
        "stream": ["A", "B"],
        "rationale": "Near-term ECB meeting pricing.",
    },
    "OIS_1Y": {
        "transform": "level_pct",
        "bloomberg": "EUSW1 Curncy",
        "stream": ["A", "B"],
        "rationale": "1-year OIS: forward policy path signal.",
    },
    "OIS_2Y": {
        "transform": "level_pct",
        "bloomberg": "EUSW2 Curncy",
        "stream": ["A"],
        "rationale": "Front-end policy expectations; key steepening/flattening driver.",
    },
    "EURIBOR_3M_FUTURES_STRIP": {
        "transform": "slope_of_strip_bps",  # Derived: near vs far contract spread
        "bloomberg": "ERZ4 Comdty - ERH4 Comdty",
        "stream": ["A"],
        "rationale": "Market-implied policy path curvature; forward guidance proxy.",
        "risk": "Futures roll dates create artificial discontinuities; use continuous series.",
    },
    "REAL_RATE_5Y5Y": {
        "transform": "level_pct",
        "bloomberg": "EUIRSBE55 Index",      # 5Y5Y EUR real rate breakeven
        "stream": ["A", "B"],
        "rationale": "Medium-term real rate; drives term premium component of curve slope.",
    },
    "ECB_BALANCE_SHEET_GROWTH": {
        "transform": "yoy_pct",              # ECB balance sheet YoY growth
        "bloomberg": "Derived from ECB SDW: BSI.M.U2.N.A.T00.A.1.Z5.0000.Z01.E",
        "stream": ["A", "B"],
        "rationale": "QE/QT proxy; APP/PEPP expansion flattens curve; tapering steepens.",
        "risk": "Monthly frequency; forward-fill to weekly. Publication lag ~6 weeks.",
    },
    "ECB_REGIME_BREAK": {
        "transform": "indicator_0_1",        # Structural break dummy
        "bloomberg": "Manual encoding of ECB policy regime dates",
        "stream": ["A", "B"],
        "rationale": "Captures ZLB entry (2014), NIRP (2019), liftoff (2022), QT (2023).",
        "risk": "CRITICAL: equity paper did not address regime breaks. "
                "See 03_regime_breaks.py for full treatment.",
    },
    "HIKING_CYCLE_PROGRESS": {
        "transform": "cumulative_hikes_bps_from_cycle_low",
        "bloomberg": "Derived from ECB DFR history",
        "stream": ["A", "B"],
        "rationale": "Captures how far into a tightening/easing cycle the ECB is. "
                     "Late-cycle hiking → curve inversion; early easing → steepening.",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 3: INFLATION AND BREAKEVENS  (both streams)
    # ─────────────────────────────────────────────────────────────────

    "HICP_EA_YOY": {
        "transform": "level_pct",
        "bloomberg": "ECCPEMUY Index",
        "stream": ["A", "B"],
        "rationale": "Headline EA inflation; drives long-end yields and term premium.",
        "risk": "Monthly publication; ~4-week lag. Forward-fill only within month.",
    },
    "HICP_CORE_EA_YOY": {
        "transform": "level_pct",
        "bloomberg": "EHXZEMU1 Index",
        "stream": ["A", "B"],
        "rationale": "Core HICP strips volatile components; more persistent ECB signal.",
    },
    "BREAKEVEN_5Y_EUR": {
        "transform": "level_bps",           # EUR 5Y breakeven inflation
        "bloomberg": "EUIRSBE55 Index",
        "stream": ["A", "B"],
        "rationale": "Market-implied inflation expectations; key long-end driver.",
    },
    "BREAKEVEN_10Y_EUR": {
        "transform": "level_bps",
        "bloomberg": "EUIRSBE10 Index",
        "stream": ["A", "B"],
        "rationale": "10-year inflation expectations; term premium channel.",
    },
    "REAL_YIELD_10Y_BUND": {
        "transform": "level_bps",           # 10Y nominal - 10Y breakeven
        "bloomberg": "GDBR10 Index - EUIRSBE10 Index",
        "stream": ["A"],
        "rationale": "Decompose nominal yield into real rate + inflation comp; "
                     "helpful for understanding curve steepening drivers.",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 4: MACRO ACTIVITY AND SURPRISE  (both streams)
    # ─────────────────────────────────────────────────────────────────

    "EA_PMI_COMPOSITE": {
        "transform": "level",
        "bloomberg": "MPMIEZCO Index",
        "stream": ["A", "B"],
        "rationale": "Composite PMI is the single best high-frequency EA activity proxy. "
                     "Above 50 = expansion → curve steepens; below 50 = contraction → flattens.",
    },
    "EA_PMI_MANUFACTURING": {
        "transform": "level",
        "bloomberg": "MPMIEZMA Index",
        "stream": ["A", "B"],
        "rationale": "Manufacturing cycle; more volatile, captures global trade exposure.",
    },
    "EA_PMI_SERVICES": {
        "transform": "level",
        "bloomberg": "MPMIEZSE Index",
        "stream": ["A"],
        "rationale": "Domestic demand proxy; services inflation sticky component.",
    },
    "CESI_EUR": {
        "transform": "level",
        "bloomberg": "CESIEUR Index",
        "stream": ["A", "B"],
        "rationale": "ECB Citi Economic Surprise Index; captures unexpected data relative to consensus.",
    },
    "EA_GDP_YOY": {
        "transform": "level_pct",
        "bloomberg": "EUGDPEYS Index",
        "stream": ["A", "B"],
        "rationale": "Quarterly; proxy for regime. High-frequency PMI preferred for timing.",
        "risk": "Quarterly + revision cycle. Use advance estimate; flag revision dates.",
    },
    "EA_UNEMPLOYMENT": {
        "transform": "level_pct",
        "bloomberg": "UMRTEMU Index",
        "stream": ["A"],
        "rationale": "Labor market slack; influences ECB guidance and curve positioning.",
        "risk": "Monthly publication; ~4-week lag.",
    },
    "CREDIT_IMPULSE_EA": {
        "transform": "yoy_change",           # Change in credit-to-GDP ratio
        "bloomberg": "Derived from ECB SDW monetary/credit aggregates",
        "stream": ["A", "B"],
        "rationale": "Leading indicator of economic cycle; credit tightening precedes slowdown.",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 5: GLOBAL SPILLOVER  (both streams)
    # ─────────────────────────────────────────────────────────────────

    "UST_10Y": {
        "transform": "level_pct",
        "bloomberg": "USGG10YR Index",
        "stream": ["A", "B"],
        "rationale": "Global rate anchor; UST/Bund spread drives EURUSD and Bund flows.",
    },
    "UST_BUND_SPREAD_10Y": {
        "transform": "level_bps",
        "bloomberg": "USGG10YR Index - GDBR10 Index",
        "stream": ["A", "B"],
        "rationale": "Cross-market relative value; drives hot money flows into/out of Bunds.",
    },
    "EURUSD": {
        "transform": "frac_diff",            # Fractionally differenced (as in equity paper)
        "bloomberg": "EURUSD Curncy",
        "stream": ["A", "B"],
        "rationale": "EUR strength reflects risk sentiment, ECB vs Fed divergence.",
    },
    "VIX": {
        "transform": "level",
        "bloomberg": "VIX Index",
        "stream": ["A", "B"],
        "rationale": "Global risk-off → flight to Bund quality → curve flattening.",
    },
    "VSTOXX": {
        "transform": "level",
        "bloomberg": "V2X Index",
        "stream": ["A", "B"],
        "rationale": "Euro-specific volatility; more directly linked to EA sovereign risk.",
    },
    "MOVE_INDEX": {
        "transform": "level",
        "bloomberg": "MXWO Index",           # Merrill Lynch MOVE
        "stream": ["A"],
        "rationale": "US rates volatility; spills over to European term premium.",
    },
    "SRVIX": {
        "transform": "level",               # SRVIX = EUR swaption volatility index
        "bloomberg": "SRVIX Index",
        "stream": ["A"],
        "rationale": "EUR rates vol; directly prices uncertainty around ECB path.",
    },
    "GLOBAL_PMI": {
        "transform": "level",
        "bloomberg": "MPMIGMCO Index",
        "stream": ["A", "B"],
        "rationale": "Global growth proxy; weaker global PMI → Bund outperformance.",
    },
    "OIL_PRICE": {
        "transform": "frac_diff",
        "bloomberg": "CO1 Comdty",          # Brent crude
        "stream": ["A", "B"],
        "rationale": "Energy channel into HICP; oil rally → steepening via inflation expectations.",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 6: SOVEREIGN SPREAD DYNAMICS  (Stream B primarily)
    # ─────────────────────────────────────────────────────────────────

    "COUNTRY_SPREAD_10Y": {
        "transform": "level_bps",           # Per-country vs Bund 10Y
        "bloomberg": "e.g. GBTPGR10 Index - GDBR10 Index (Italy)",
        "stream": ["B"],
        "rationale": "Primary target proxy; level captures valuation regime. "
                     "Include lagged spread as a feature (momentum/reversion).",
        "risk": "Do NOT include contemporaneous spread as feature → look-ahead bias.",
    },
    "SPREAD_MOMENTUM_1M": {
        "transform": "1m_change_bps",
        "bloomberg": "Derived",
        "stream": ["B"],
        "rationale": "Spread momentum; spreads exhibit short-term persistence (Asness logic applies).",
    },
    "SPREAD_MOMENTUM_3M": {
        "transform": "3m_change_bps",
        "bloomberg": "Derived",
        "stream": ["B"],
        "rationale": "Medium-term spread momentum.",
    },
    "SPREAD_ZSCORE_12M": {
        "transform": "zscore_12m",          # (spread - 12m mean) / 12m std
        "bloomberg": "Derived",
        "stream": ["B"],
        "rationale": "Mean-reversion valuation signal; historically strong for BTP-Bund.",
    },
    "ASW_SPREAD": {
        "transform": "level_bps",           # Asset-swap spread (=credit component)
        "bloomberg": "e.g. GBTPASW10 Index",
        "stream": ["B"],
        "rationale": "Strips out duration component; cleaner credit spread signal.",
        "risk": "ASW conflates credit and EUR swap-Bund basis. Cross-currency basis adjustments needed.",
    },
    "CARRY_ROLL_DOWN": {
        "transform": "annualised_bps",      # Carry = yield × 1yr; roll = slope × modified_duration
        "bloomberg": "Derived from yield curve",
        "stream": ["B"],
        "rationale": "Carry and roll-down are key relative value signals in sovereign bonds. "
                     "High carry → spread compression over time, all else equal.",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 7: FISCAL / CREDIT FUNDAMENTALS  (Stream B)
    # ─────────────────────────────────────────────────────────────────

    "DEBT_TO_GDP": {
        "transform": "level_pct",
        "bloomberg": "Eurostat / Bloomberg MAADGB index family",
        "stream": ["B"],
        "rationale": "Structural credit risk; higher debt → wider spread at equivalent rating. "
                     "Annual publication; use last observation forward-filled.",
        "risk": "Annual frequency creates very stale signals at weekly granularity. "
                "Use as regime conditioning variable only (via QS transformation).",
    },
    "FISCAL_DEFICIT_TO_GDP": {
        "transform": "level_pct",
        "bloomberg": "Eurostat / Bloomberg",
        "stream": ["B"],
        "rationale": "Flow measure of fiscal stress; widening deficit → spread pressure.",
        "risk": "Same annual frequency limitation as Debt/GDP.",
    },
    "SOVEREIGN_CDS_5Y": {
        "transform": "level_bps",
        "bloomberg": "e.g. ITGV5YUSAC Index (Italy)",
        "stream": ["B"],
        "rationale": "Daily-frequency credit risk proxy; more timely than fiscal data.",
    },
    "CDS_SLOPE": {
        "transform": "level_bps",           # 5Y CDS - 1Y CDS
        "bloomberg": "Derived",
        "stream": ["B"],
        "rationale": "CDS term structure slope; steep slope → near-term stability but long-term risk.",
    },
    "MOODY_RATING_NUMERICAL": {
        "transform": "indicator",           # Rating mapped to 1–21 scale
        "bloomberg": "RATD Curncy (Bloomberg sovereign rating)",
        "stream": ["B"],
        "rationale": "Discrete event risk; rating downgrades have discontinuous spread impact. "
                     "More useful as a regime filter than continuous predictor.",
        "risk": "Encode as regime dummy: IG_stable / IG_negative_watch / near_junk / junk.",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 8: ECB PURCHASE PROGRAMS AND TPI  (Stream B specifically)
    # ─────────────────────────────────────────────────────────────────

    "ECB_APP_PURCHASES_COUNTRY": {
        "transform": "4w_rolling_sum_eur_bn",
        "bloomberg": "ECB SDW: SEC.W.U2.EUR.W3.DP.B1300.EUR",
        "stream": ["B"],
        "rationale": "Direct demand-side support; APP purchases suppress sovereign spreads. "
                     "Country-level deviations from capital key → spread signal.",
        "risk": "Weekly ECB publication with 1-week lag only → low publication lag risk.",
    },
    "PEPP_FLEXIBILITY_INDICATOR": {
        "transform": "indicator_0_1",       # 1 = PEPP active period
        "bloomberg": "Manual encoding: PEPP active Mar 2020 – Mar 2022",
        "stream": ["B"],
        "rationale": "PEPP had country flexibility to deviate from capital key → "
                     "explicit spread compression tool. Binary period indicator.",
    },
    "TPI_ACTIVATION_PROBABILITY": {
        "transform": "market_implied_proxy",  # No direct Bloomberg ticker
        "bloomberg": "Proxy: VSTOXX level + spread z-score composite",
        "stream": ["B"],
        "rationale": "TPI (Transmission Protection Instrument) activated when spreads "
                     "widen beyond 'disorderly' threshold. High probability → spread floor. "
                     "No market-observable variable; use composite proxy.",
        "risk": "This is a judgment variable. Define operationally or omit in Phase 1.",
    },
    "CAPITAL_KEY_DEVIATION": {
        "transform": "level_pct",           # Actual holdings vs capital key weight
        "bloomberg": "Derived from ECB SDW purchase data",
        "stream": ["B"],
        "rationale": "Deviation from capital key signals ECB favouring / neglecting a country. "
                     "Positive deviation (over-bought) → spread compression.",
    },
    "REDEMPTIONS_ROLLING_4W": {
        "transform": "eur_bn",             # PSPP reinvestment schedule
        "bloomberg": "ECB SDW + Bloomberg maturity schedule",
        "stream": ["B"],
        "rationale": "PSPP reinvestments create flow demand that compresses spreads. "
                     "Large upcoming maturities for country i → more ECB buying ahead.",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 9: LIQUIDITY AND MARKET MICROSTRUCTURE  (both streams)
    # ─────────────────────────────────────────────────────────────────

    "BUND_REPO_RATE": {
        "transform": "level_bps",
        "bloomberg": "EURR001W Index",       # ESTER proxy
        "stream": ["A", "B"],
        "rationale": "Bund specialness in repo reflects scarcity; distorts yield comparisons.",
    },
    "BUND_REPO_SPECIALNESS": {
        "transform": "level_bps",           # General collateral minus specific repo
        "bloomberg": "GC – SC spread from Eurex Repo",
        "stream": ["A", "B"],
        "rationale": "High specialness = Bund scarce = artificially rich → "
                     "mechanical flattening bias in 10Y-2Y as 2Y richens more.",
        "risk": "CRITICAL for Stream A: scarcity effect can dominate true slope signal. "
                "Include as control variable or use swap curve as alternative target.",
    },
    "BID_ASK_SPREAD_BTP": {
        "transform": "level_bps",
        "bloomberg": "MTS platform data or Bloomberg FXMM",
        "stream": ["B"],
        "rationale": "Liquidity proxy; widening bid-ask precedes spread widening in stress.",
    },
    "EUREX_BUND_OPEN_INTEREST": {
        "transform": "rolling_zscore_12m",
        "bloomberg": "RXA Comdty → OI field",
        "stream": ["A"],
        "rationale": "Speculative positioning in Bund futures; extreme OI → reversal risk.",
    },
    "COT_NET_POSITION": {
        "transform": "zscore_12m",
        "bloomberg": "CFTC COT reports via Bloomberg",
        "stream": ["A"],
        "rationale": "Asset manager vs leveraged fund net duration positioning. "
                     "Extreme positioning → contrarian signal (crowding).",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 10: POLITICAL RISK  (Stream B)
    # ─────────────────────────────────────────────────────────────────

    "EUREKA_INDEX": {
        "transform": "level",
        "bloomberg": "No direct ticker; use EPU Europe index or proxy",
        "stream": ["B"],
        "rationale": "Euro-area political uncertainty index; election risk, coalition fragility.",
    },
    "ELECTION_DUMMY": {
        "transform": "weeks_to_election",
        "bloomberg": "Manual encoding of election calendars",
        "stream": ["B"],
        "rationale": "Elections in IT/ES/FR/GR associated with spread volatility. "
                     "30-week window before election: indicator = 1.",
    },
    "POPULISM_POLL_SPREAD": {
        "transform": "pct_points",          # Euro-sceptic party poll gap vs pro-EU
        "bloomberg": "PollsAndElections.eu / manual data",
        "stream": ["B"],
        "rationale": "Tail risk: surge in anti-EU parties → redenomination risk premium.",
        "risk": "Data collection burden; proxy with 5Y CDS if unavailable.",
    },

    # ─────────────────────────────────────────────────────────────────
    # BLOCK 11: RELATIVE SPREAD INTERACTIONS  (Stream B; analogous to
    #           equity paper's 18 factor-specific relative metrics)
    # ─────────────────────────────────────────────────────────────────

    "CROSS_COUNTRY_SPREAD_ZSCORE_12M": {
        "transform": "zscore_12m",          # Per-pair: IT-ES, IT-PT, ES-PT, ...
        "bloomberg": "Derived",
        "stream": ["B"],
        "rationale": "Relative valuation between peripheral countries; "
                     "analogous to equity paper's zrel_i_vs_j_12 features.",
    },
    "SPREAD_RELVOL_12M": {
        "transform": "ratio",               # σ(country_spread) / σ(Bund_spread)
        "bloomberg": "Derived",
        "stream": ["B"],
        "rationale": "Volatility dispersion between spread series; "
                     "analogous to equity paper's relvol features.",
    },
    "SPREAD_CORR_12M": {
        "transform": "rolling_correlation",
        "bloomberg": "Derived",
        "stream": ["B"],
        "rationale": "Co-movement between country spreads; "
                     "low correlation = idiosyncratic driver dominant.",
    },
}


def get_features_by_stream(stream: str) -> dict:
    """Filter feature universe by stream."""
    return {k: v for k, v in FEATURE_UNIVERSE.items()
            if stream in v.get("stream", [])}


def get_high_risk_features() -> dict:
    """Return features with data leakage or staleness risks."""
    return {k: v for k, v in FEATURE_UNIVERSE.items()
            if "risk" in v and ("CRITICAL" in v["risk"] or "look-ahead" in v["risk"])}


if __name__ == "__main__":
    stream_a = get_features_by_stream("A")
    stream_b = get_features_by_stream("B")
    shared = {k for k in get_features_by_stream("A")
              if k in get_features_by_stream("B")}
    high_risk = get_high_risk_features()

    print(f"Stream A features: {len(stream_a)}")
    print(f"Stream B features: {len(stream_b)}")
    print(f"Shared features: {len(shared)}")
    print(f"\nHigh-risk features requiring attention:")
    for k, v in high_risk.items():
        print(f"  {k}: {v['risk']}")
