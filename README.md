# Machine Learning Enhanced Residual Income Valuation for U.S. Banks
[![DOI](https://zenodo.org/badge/1383363734.svg)](https://doi.org/10.5281/zenodo.22945955)

A two-stage empirical framework combining macro-informed ROE forecasting,
machine learning, panel econometrics, and residual income valuation.

## Research Overview

This project investigates whether machine-learning-based forecasts of bank
return on equity (ROE) can improve residual income valuation for U.S.
commercial banks.

The framework consists of two stages:

1. One-quarter-ahead ROE forecasting across 20 U.S. commercial banks.
2. Integration of valuation-specific machine-learning forecasts into a
   clean-surplus-consistent Residual Income Valuation (RIV) framework for
   JPMorgan Chase & Co.

## Key Results

### Stage 1 — ROE Forecasting

- 20 U.S. commercial banks
- 680 bank-quarter observations
- Training period: 2017Q1–2023Q4
- Test period: 2024Q1–2025Q4
- 12 econometric and machine-learning specifications
- LightGBM test RMSE: 3.308
- LightGBM test R²: 0.405
- Pooled OLS test RMSE: 4.399

LightGBM produced the strongest point-estimate forecasting performance,
although Diebold–Mariano tests did not establish statistically significant
superiority over the main benchmark models.

### Stage 2 — Residual Income Valuation

A valuation-specific LightGBM forecasting architecture was incorporated
into a clean-surplus-consistent RIV framework and applied to JPMorgan
Chase & Co. across nine valuation dates from 2024Q1 to 2026Q1.

| Metric | Traditional RIV | Enhanced RIV |
|---|---:|---:|
| MAE | $115.55 | $59.98 |
| RMSE | $121.25 | $70.16 |
| MAPE | 44.07% | 21.91% |

Relative reductions:

- MAE: 48.1%
- RMSE: 42.1%
- MAPE: 50.3%

The valuation results are specific to JPMorgan and the period examined and
should not be interpreted as evidence of general machine-learning
superiority in equity valuation.

## Methodology

### Stage 1

- Panel econometric models
- Random Effects
- Fixed Effects
- Pooled OLS
- Linear Regression
- Ridge
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Chronological out-of-sample testing
- Diebold–Mariano forecast comparison

### Stage 2

- Residual Income Valuation
- Clean surplus accounting
- CAPM cost of equity
- Direct LightGBM forecasts for Q1–Q4
- Recursive forecasts for Q5–Q8
- Capped persistence for Q9–Q20
- Market-price benchmark comparison

## Repository Structure

```text
├── dissertation/
├── notebooks/
├── src/
├── data/
├── results/
├── docs/
├── requirements.txt
└── .gitignore
