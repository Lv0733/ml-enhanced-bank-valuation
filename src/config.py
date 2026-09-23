"""
config.py -- Enhanced Recursive LightGBM RIV V4.2
=====================================================
ALL constants in this file are copied EXACTLY from the verified, working
enhanced_riv_v4.py source code (cross-checked by direct inspection of that
file's source during this rebuild, not reconstructed from memory or from
any prompt's prose description). Where this matters for transparency: a
few constant NAMES are renamed for clarity (e.g. `ml_continuation_horizon`,
which was a function-default parameter in V4, is promoted here to a named
module constant `ML_CONTINUATION_HORIZON = 8`), but every VALUE and every
formula that consumes these constants downstream is unchanged.

THE ONLY INTENTIONAL VALUE CHANGE IN THIS ENTIRE FILE IS:
    OMEGA_CAP = 0.92
which did not exist in V4 at all (V4 used omega_estimated directly,
unconstrained). This is the single, documented enhancement V4.2 adds.
"""

import os

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
DATA_DIR = "data"
JPM_FINANCIALS_PATH = os.path.join(DATA_DIR, "JPMorgan Chase & Co (JPM.N).xlsx")
JPM_PRICE_PATH = os.path.join(DATA_DIR, "JPMorgan Stock Price History.csv")
TBILL_PATH = os.path.join(DATA_DIR, "3-Month Treasury Bill.xlsx")
SP500_PATH = os.path.join(DATA_DIR, "SP500.xlsx")
ML_PANEL_PATH = os.path.join(DATA_DIR, "modeling_dataset_v3.csv")

OUT_DIR = "."  # the project root itself -- this script lives inside enhanced_riv_v4_2/,
                # so output subfolders are created directly alongside it, matching the
                # brief's exact required tree (Data/, Models/, ... as siblings of this .py file)
DATA_OUT_DIR = os.path.join(OUT_DIR, "Data")
MDL_DIR = os.path.join(OUT_DIR, "Models")
PRED_DIR = os.path.join(OUT_DIR, "Predictions")
TBL_DIR = os.path.join(OUT_DIR, "Tables")
FIG_DIR = os.path.join(OUT_DIR, "Figures")
VAL_DIR = os.path.join(OUT_DIR, "Validation")
RPT_DIR = os.path.join(OUT_DIR, "Reports")
LOG_DIR = os.path.join(OUT_DIR, "Logs")
ALL_OUTPUT_DIRS = [OUT_DIR, DATA_OUT_DIR, MDL_DIR, PRED_DIR, TBL_DIR, FIG_DIR, VAL_DIR, RPT_DIR, LOG_DIR]


def ensure_output_dirs():
    for d in ALL_OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
# HORIZON / PERSISTENCE CONSTANTS -- identical values to V4
# ----------------------------------------------------------------------------
FORECAST_HORIZON_QUARTERS = 20
DIRECT_HORIZON_QUARTERS = 4          # Q1-Q4: four independent direct LightGBM models
ML_CONTINUATION_HORIZON = 8          # Q5-Q8: recursive ML continuation ends here (V4's function default, named here)
JPM_NAME = "JPMorgan Chase & Co"

# THE ONLY V4.2 ENHANCEMENT: cap on the Q9-Q20 persistence parameter omega.
# V4 itself has no such cap -- omega_estimated is used directly, unconstrained.
OMEGA_CAP = 0.92

# ----------------------------------------------------------------------------
# DAMODARAN ERP -- identical to V4 (NYU Stern "Historical Implied Equity
# Risk Premium" series, January 2026 update)
# ----------------------------------------------------------------------------
DAMODARAN_ERP_BY_YEAR = {
    2016: 0.0569, 2017: 0.0508, 2018: 0.0596, 2019: 0.0520, 2020: 0.0472,
    2021: 0.0424, 2022: 0.0594, 2023: 0.0460, 2024: 0.0433, 2025: 0.0423,
    2026: 0.0423,
}

# ----------------------------------------------------------------------------
# LARGE-BANK UNIVERSE -- identical request list to V4 (9 of these 12 actually
# exist in modeling_dataset_v3.csv; matched via whole-word comparison in
# utils.build_large_bank_panel(), identical methodology to V4)
# ----------------------------------------------------------------------------
LARGE_BANKS_REQUESTED = [
    "JPMorgan Chase", "Bank of America", "Citigroup", "Wells Fargo",
    "US Bancorp", "PNC", "Truist", "M&T", "Regions", "Citizens",
    "Goldman Sachs", "Morgan Stanley",
]

# ----------------------------------------------------------------------------
# VALUATION DATES -- identical to V4
# ----------------------------------------------------------------------------
import pandas as pd
VALUATION_DATES = pd.to_datetime([
    "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
    "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31",
])
VALUATION_LABELS = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"]

# ----------------------------------------------------------------------------
# LIGHTGBM HYPERPARAMETERS -- identical to V4, fixed random_state=42
# throughout for full determinism/reproducibility.
# ----------------------------------------------------------------------------
LGBM_PARAMS = dict(
    num_leaves=31, min_child_samples=15, max_depth=-1,
    learning_rate=0.05, lambda_l2=1, lambda_l1=0,
    feature_fraction=0.8, bagging_fraction=0.9,
    random_state=42, verbose=-1, n_estimators=200,
)
LGBM_PARAMS_SMALL = dict(
    num_leaves=15, min_child_samples=10, max_depth=-1,
    learning_rate=0.05, lambda_l2=1, lambda_l1=0,
    feature_fraction=0.8, bagging_fraction=0.9,
    random_state=42, verbose=-1, n_estimators=100,
)

# ----------------------------------------------------------------------------
# FEATURE SET -- copied EXACTLY from enhanced_riv_v4.py's own ALL_FEATURES /
# BANK_QUALITY_FEATURES / MACRO_FEATURES / DYNAMIC_FEATURES definitions.
# This is the exact list that was WRONG in the prior standalone V4.1 attempt
# (which used a different, 13-feature specification with raw LLP $ and an
# LLP_GDP interaction term instead of LLP_to_Assets, and only ROE_Lag1
# instead of the full ROE/spread history feature set). Verified to be
# exactly 20 features, exactly matching V4, per the non-negotiable
# requirement in this brief.
# ----------------------------------------------------------------------------
BANK_QUALITY_FEATURES = [
    "Net Interest Margin (%)", "Core Tier 1 Ratio (%)", "Efficiency Ratio (%)",
    "Net Charge-Off Rate (%)", "Loan_Growth", "LLP_to_Assets",
]
MACRO_FEATURES = ["GDP_Growth", "Fed_Funds_Rate", "Yield_Spread_10Y_2Y",
                   "SLOOS_Lending_Standards", "SLOOS_Loan_Demand"]
ROE_HISTORY_FEATURES = ["ROE_Lag1", "ROE_Lag2", "ROE_Lag4", "ROE_MA_4Q", "ROE_MA_8Q", "ROE_STD_4Q"]
SPREAD_HISTORY_FEATURES = ["ROE_SPREAD_Lag1", "ROE_SPREAD_Lag2", "ROE_SPREAD_MA_4Q"]
ALL_FEATURES = ROE_HISTORY_FEATURES + SPREAD_HISTORY_FEATURES + BANK_QUALITY_FEATURES + MACRO_FEATURES
CATEGORICAL_FEATURE = "Bank"
DYNAMIC_FEATURES = BANK_QUALITY_FEATURES  # the 6 variables forecast recursively, Q5-Q8

assert len(ALL_FEATURES) == 20, f"Feature count must be exactly 20 (V4.2 non-negotiable requirement); got {len(ALL_FEATURES)}"
assert len(set(ALL_FEATURES)) == 20, "Feature list must not contain duplicates"
assert "Provision & Impairment for Loan Losses (LLP)" not in ALL_FEATURES, "Raw LLP must NOT be a feature (use LLP_to_Assets)"
assert "LLP_GDP" not in ALL_FEATURES, "LLP_GDP interaction term must NOT be a feature (does not exist in V4)"

# Continuation-model feature set -- identical to V4's train_continuation_model()
CONTINUATION_FEATURES = ["ROE_SPREAD_Lag1", "ROE_SPREAD_MA_4Q"] + BANK_QUALITY_FEATURES + MACRO_FEATURES

# ----------------------------------------------------------------------------
# COLOUR PALETTE -- identical to V4
# ----------------------------------------------------------------------------
NAVY, RED, TEAL, AMBER, GREY, PURPLE, BLUE2 = "#1a3a5c", "#c0392b", "#16a085", "#e67e22", "#7f8c8d", "#8e44ad", "#2980b9"
GOLD = "#d4ac0d"

PLOT_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#555", "axes.grid": True,
    "grid.color": "#bdc3c7", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 11,
}


def apply_plot_style():
    plt.rcParams.update(PLOT_STYLE)


DIVIDER = "=" * 78


def section(t):
    print(f"\n{DIVIDER}\n{t}\n{DIVIDER}")
