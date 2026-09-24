# Methodology

## 1. Research Design

This study develops a two-stage empirical framework that combines macroeconomic forecasting, machine learning, panel econometrics, and residual income valuation.

The framework consists of:

1. Forecasting bank-level Return on Equity (ROE)
2. Incorporating the ROE forecasts into a residual income valuation model

The objective is to examine whether macroeconomic and bank-specific information can improve forward-looking ROE forecasts and, subsequently, bank equity valuation.

---

## 2. Stage 1: ROE Forecasting

### 2.1 Dataset

The forecasting dataset contains quarterly observations for U.S. commercial banks covering 2017Q1 to 2025Q4.

The final forecasting sample contains:

- 20 banks
- 680 bank-quarter observations
- Training period: 2017Q1–2023Q4
- Test period: 2024Q1–2025Q4

The dependent variable is one-quarter-ahead Return on Equity (ROE).

### 2.2 Predictor Variables

The forecasting framework considers bank-specific financial variables and macroeconomic indicators.

Bank-level variables include measures related to:

- profitability
- net interest margins
- loan-loss provisions
- credit losses
- asset size
- operating efficiency
- loan growth
- capital and balance-sheet conditions

Macroeconomic variables capture broader economic and financial conditions.

The initial candidate feature set contained 51 variables. Feature selection and model design resulted in a final set of 14 predictors for the forecasting models, while machine-learning specifications additionally incorporated bank identity indicators.

---

## 3. Econometric Models

Three panel-econometric specifications are considered:

### Pooled OLS

A pooled ordinary least squares model estimates the relationship between ROE and the explanatory variables while pooling observations across banks.

### Fixed Effects

The fixed-effects specification controls for time-invariant bank-specific characteristics.

### Random Effects

The random-effects specification treats bank-specific effects as random variables.

These models provide interpretable econometric benchmarks against which the machine-learning models are evaluated.

---

## 4. Machine Learning Models

Nine machine-learning specifications are evaluated, including tree-based ensemble methods.

The main models include:

- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

The models are evaluated using a chronological train-test design rather than a random split in order to preserve the time-series structure of the financial data.

The selected LightGBM specification uses:

- `num_leaves = 63`
- `learning_rate = 0.03`
- `feature_fraction = 0.70`

The objective is to capture potentially nonlinear relationships between bank fundamentals, macroeconomic conditions, and future ROE.

---

## 5. Out-of-Sample Evaluation

Model performance is evaluated using the 2024Q1–2025Q4 test period.

The primary evaluation metrics are:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

Model comparisons are based on out-of-sample predictions rather than in-sample fit.

Statistical forecast comparison is conducted using the Diebold-Mariano (DM) test to examine whether differences in predictive accuracy are statistically significant.

---

## 6. Stage 2: Residual Income Valuation

The second stage incorporates the ROE forecasts into a residual income valuation (RIV) framework.

The analysis focuses on JPMorgan Chase and evaluates nine quarterly valuation dates from 2024Q1 to 2026Q1.

The valuation dataset contains:

- 342 bank-quarter observations
- 9 U.S. commercial banks
- 70 engineered candidate variables
- 20 valuation-specific features

The Stage 2 feature set is constructed separately from the Stage 1 forecasting framework because the objective of Stage 2 is valuation rather than general ROE prediction.

---

## 7. Residual Income Model

Residual income is defined as:

\[
RI_t = (ROE_t-r_q)B_{t-1}
\]

where:

- \(RI_t\) = residual income
- \(ROE_t\) = return on equity
- \(r_q\) = cost of equity
- \(B_{t-1}\) = beginning book equity

Equity value is estimated as:

\[
V_0 = B_0 + \sum_{t=1}^{T}\frac{RI_t}{(1+r_q)^t}
+\frac{TV_T}{(1+r_q)^T}
\]

The terminal value is calculated using a zero-growth assumption:

\[
TV_{20}=\frac{RI_{20}}{r_q}
\]

The cost of equity is estimated using the Capital Asset Pricing Model (CAPM).

---

## 8. Enhanced RIV Forecasting Framework

The enhanced valuation model replaces historical ROE extrapolation with machine-learning-based ROE forecasts.

The forecasting horizon is divided into three stages:

### Q1–Q4

Direct one-quarter-ahead LightGBM forecasts are generated.

### Q5–Q8

Forecasts are generated recursively, using previous predicted values as inputs where required.

### Q9–Q20

A capped geometric persistence approach is applied to extend the ROE-spread path toward the terminal period.

The persistence mechanism is designed to avoid unrealistic long-horizon extrapolation.

---

## 9. Traditional RIV Benchmark

A traditional RIV specification is constructed as a benchmark.

The traditional model uses historical ROE information to project future residual income rather than using the machine-learning forecasting framework.

Both the traditional and enhanced models use the same valuation framework and cost-of-equity methodology.

This allows the valuation comparison to focus on the effect of the different ROE forecasting approaches.

---

## 10. Valuation Evaluation

The estimated intrinsic values are compared with observed JPMorgan market prices at the nine valuation dates.

The evaluation metrics are:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

The market price is used as an empirical benchmark for evaluating valuation accuracy. It is not treated as an error-free measure of intrinsic value.

---

## 11. Reproducibility

The repository contains the research notebook, source code, model configuration, methodology documentation, and results documentation.

The raw LSEG-delivered financial dataset is not included because its distribution is subject to LSEG access and usage terms.

Researchers seeking to reproduce the analysis should obtain the relevant financial data through an appropriately authorized LSEG source.

Additional macroeconomic data are obtained from the Federal Reserve Bank of St. Louis (FRED), while equity risk-premium estimates are obtained from Aswath Damodaran's published data.
