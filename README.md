# Country Regression Model

This notebook explores which macroeconomic indicators best explain and predict a country's **Debt-to-GDP ratio**, using Estonia (`EST`) as the worked example, with a broader multi-country dataset available for extension.

## What the notebook does

### 1. Data setup
- Loads `training.csv` and `testing.csv` into pandas DataFrames.
- Splits the training data by `Country Code` into a dictionary of per-country DataFrames (`DataFrameDict`), so each country can be modeled independently.
- Estonia's slice (`DataFrameDict['EST']`) covers **2010–2020** (11 years) and is exported to `EstoniaData.csv`.

### 2. Single-variable OLS regressions (models 1–18)
For Estonia, the notebook fits a separate simple linear regression (`statsmodels.OLS`) of `Debt_GDP` against each candidate predictor, one at a time:

| # | Predictor | # | Predictor |
|---|---|---|---|
| 1 | Annual Inflation % | 10 | Population Growth |
| 2 | Current Account Balance (USD) | 11 | Working-Age Population |
| 3 | Export to GDP | 12 | Profit Tax to Commercial Profit *(skipped — missing data)* |
| 4 | FDI | 13 | Urban Proportion |
| 5 | Government Expenditure | 14 | Social Spending to GDP *(skipped — missing data)* |
| 6 | Import to GDP | 15 | Real Interest Rate *(skipped — missing data)* |
| 7 | Labour Force Participation | 16 | Tax Revenue to GDP |
| 8 | National GDP | 17 | Total Labour Force |
| 9 | Net Population Migration | 18 | Unemployment Rate |

For each model, the notebook:
1. Fits `Debt_GDP ~ constant + predictor`.
2. Generates in-sample predictions.
3. Computes the **absolute percentage error** per year and the **Mean Absolute Percentage Error (MAPE)** across all years.

### 3. MAPE comparison
All 15 usable MAPE scores are collected into a single `MAPEScore` DataFrame (saved to `EstoniaMAPE.csv`) to rank which single indicator best predicts debt on its own. **Unemployment Rate** had the lowest error (~0.26 MAPE) and **FDI / Total Labour Force** had the highest (~0.49 MAPE) among the fitted models.

### 4. Correlation matrix
A full pairwise correlation matrix is computed across all candidate indicators (including the ones with missing data) to check for multicollinearity and identify which variables move together with `Debt_GDP`.

### 5. Time-series forecasting (out-of-sample test)
Using an extended dataset (`fullFile.csv`, 2010–2023), the notebook builds simple **time-trend regressions** (`indicator ~ time`) trained on 2010–2020 and tested on held-out 2021–2023 data, for:
- **National GDP** — MAPE ≈ 0.13 (model under-forecasts the post-2020 GDP surge)
- **Total Labour Force** — MAPE ≈ 0.03
- **Urban Proportion** — MAPE ≈ 0.003 (very stable, near-linear trend)
- **Population Growth** — trend fit only (Estonia's population growth spikes sharply in 2022–2023, likely due to migration, and is poorly captured by a linear trend)

Each forecast includes an actual-vs-predicted line chart.

## Key takeaways
- No single indicator is a strong standalone predictor of Debt_GDP for Estonia — MAPEs cluster in the 0.26–0.49 range, indicating a multi-variable model would likely perform better than any univariate one.
- Structural/demographic variables (Urban Proportion, Total Labour Force, National GDP) forecast much more reliably over time than they explain debt levels — they trend smoothly, but that smooth trend doesn't map cleanly onto debt.
- Some fields (Profit Tax to Commercial Profit, Social Spending to GDP, Real Interest Rate) have missing data for Estonia and are intentionally commented out.
- Population Growth is the least forecastable series due to a sharp late-sample discontinuity, likely tied to a migration shock in 2022–2023.

## Requirements
```
pandas
numpy
matplotlib
statsmodels
```

## Suggested next steps
- Fit a multivariate regression combining the lowest-MAPE / least-correlated predictors (e.g., Unemployment Rate, Tax Revenue to GDP, Government Expenditure) rather than relying on univariate models.
- Extend the per-country loop (`DataFrameDict`) to run the same MAPE comparison across all countries, not just Estonia, to see if the same indicators generalize.
- Investigate the 2022–2023 discontinuity in Population Growth before trusting any forward forecast built on it.

