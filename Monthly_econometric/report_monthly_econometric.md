# European Natural Gas Data and Modelling for Forecasting TTF Prices

**Author:** G. Kurokawa
**Date:** July 2026
**Repo:** https://github.com/gskurokawa/Natural-gas/tree/main/Monthly_econometric

## 1. Executive Summary

This model uses a simple European natural gas market dataset, built entirely from free and public sources, to test how well such data can econometrically forecast month-ahead TTF prices. The final specification — a lagged, deseasonalized combination of price momentum, the change in European gas storage, and London heating-degree-days — shows a modest improvement over naive benchmarks (a random walk and a momentum-only model) in an out-of-sample walk-forward backtest, though the improvement is small and does not clear conventional statistical significance thresholds at any horizon tested. An LNG sendout variable was also tested but excluded on stationarity grounds (Section 6). This is not a production trading or investment tool, and should not be used as one.


## 2. Market Context

Understanding the European natural gas market is important context for interpreting TTF and other European gas prices. Some of the most important fundamental drivers of the market are underground storage levels, winter temperatures, and LNG availability. Data on these drivers is available at different frequencies and depths of history: some series are only available monthly but stretch back further, while others are available daily but have a shorter history. This study uses the monthly series throughout, prioritizing a longer, consistent history over higher-frequency granularity.

## 3. Dataset

| Variable | Source | Frequency | Period |
|---|---|---|---|
| TTF Price | Yahoo Finance | Monthly | 2015-01 to Present |
| EU+UK Gas storage | AGSI/GIE | Monthly | 2015-01 to Present |
| EU+UK LNG sendout | ALSI | Monthly | 2015-01 to Present |
| EU+UK HDDs | ECMWF | Monthly | 2015-01 to Present |
| EU+UK Gas pipeline transmission | ENTSOG | Monthly | 2015-01 to Present |
| Norwegian gas pipeline flows | ENTSOG | Monthly | 2015-01 to Present |
| EU+UK Power load | ENTSOE | Monthly | 2015-01 to Present |
| EU+UK Wind power generation | ENTSOE | Monthly | 2015-01 to Present |
| EU+UK Gas production | Eurostat | Monthly | 2015-01 to Present |
| EU+UK Gas consumption | Eurostat | Monthly | 2015-01 to Present |

## 4. Processing

- Determined how far back a coherent, gap-free time series could be constructed across all sources.
- Handled missing data via interpolation, external research, or separate modelling where necessary.
- Excluded non-European benchmark prices (JKM, Henry Hub) to isolate purely European fundamentals.
- Constructed a **storage change** variable as the month-over-month change in EU+UK average gas storage (bcm) — a flow measure of whether the market tightened or loosened that month.
- Transformed variables as appropriate (e.g. log-differenced) and tested for stationarity with ADF and KPSS (see Section 6 for full results and one variable this eliminated from the final model).
- Deseasonalized key predictors into **anomalies** using an expanding, out-of-sample-safe calendar-month climatology — each observation is compared only to prior years' same-month average, never to future data.
- Where individual supply/demand sub-components were collinear by construction (an accounting identity), a single composite or flow variable was used instead of the raw components together.
- Lagged all predictors relative to the return period they forecast, so every regressor is predetermined and not contaminated by same-period simultaneity.
- Split the data so the training set runs up to the end of 2024 and the test set covers the remainder of the dataset, through May 2026.

## 5. Methodology

- Variable selection was tied to market-fundamentals reasoning, including checking whether apparent effects reflect genuine surprise information or already-anticipated seasonal patterns, and whether they survive formal stationarity testing.
- Estimated a time series multiple regression on a relatively small sample (approx. 100–120 observations), with Newey-West HAC standard errors to account for residual autocorrelation and heteroskedasticity (checked via Durbin-Watson, residual autocorrelation, and Breusch-Pagan tests).
- Backtested with an expanding-window walk-forward procedure: starting from a minimum fit size, the model is refit at each step using only data available up to that point and used to forecast one month ahead, through to May 2026.
- Benchmarked against a random walk (no-change) and a momentum-only (AR(1)) model.
- Checked robustness across 1- to 3-month forecast horizons, with results treated as suggestive rather than confirmatory given the small sample and the known bias of overlapping-window regressions toward inflated apparent fit at longer horizons.

## 6. Results

### Stationarity checks

Every series used in the final model was tested for stationarity with both an Augmented Dickey-Fuller (ADF, null hypothesis: unit root) and a KPSS test (null hypothesis: stationary), on the training sample:

| Series | ADF t-stat | ADF 5% critical value | KPSS stat | KPSS 5% critical value | Verdict |
|---|---:|---:|---:|---:|---|
| dlog(TTF) | -7.81 | -2.86 | 0.078 | 0.463 | Stationary |
| Momentum anomaly | -8.08 | -2.86 | 0.131 | 0.463 | Stationary |
| Storage change anomaly | -2.46 | -2.86 | 0.071 | 0.463 | Conflicting (see note below) |
| London HDD anomaly | -3.08 | -2.86 | 0.138 | 0.463 | Stationary |

An LNG sendout variable was also tested (level, first difference, and deseasonalized anomaly) but failed both tests in every form tried, and was dropped rather than included below.

The storage change anomaly gets a conflicting verdict between the two tests; this is most likely a low-power issue with ADF in a modest sample rather than genuine non-stationarity, since it is economically implausible for the month-over-month change in a physically bounded, mean-reverting stock (gas storage cannot exceed capacity or go negative) to carry a genuine stochastic trend, and KPSS — whose null hypothesis is stationarity, and which comfortably fails to reject it here — is arguably the more informative test in this specific case.

### Model specification

We estimate the one-month-ahead change in log TTF price as a function of three lagged, seasonally-adjusted fundamentals:

$$
\Delta\log(TTF)_t = \beta_0 + \beta_1 \, \text{Momentum}_{t-1} + \beta_2 \, \text{StorageChange}_{t-1} + \beta_3 \, \text{HDD}_{t-1} + \varepsilon_t
$$

All three regressors are anomalies relative to an expanding, out-of-sample-safe calendar-month climatology, lagged one month so that every predictor is predetermined relative to the return it forecasts:

- **Momentum** — the deseasonalized lagged monthly TTF return itself
- **StorageChange** — the deseasonalized month-over-month change in EU+UK gas storage (bcm)
- **HDD** — the deseasonalized London heating-degree-days

The model is estimated by OLS over 2016-03 to 2024-12 (n = 106), with Newey–West HAC standard errors (Bartlett kernel) to account for both serial correlation and heteroskedasticity in the residuals.

### Coefficient estimates

| Variable | Coefficient | t-stat | p-value | Sig. |
|---|---:|---:|---:|:---:|
| Const | 0.0064 | 0.38 | 0.703 | |
| Momentum (t-1) | 0.2476 | 2.13 | 0.036 | \*\* |
| Storage change anomaly (t-1) | -0.0141 | -2.40 | 0.018 | \*\* |
| London HDD anomaly (t-1) | -0.0008 | -1.87 | 0.064 | \* |

\*\* p<0.05, \* p<0.10

**R² = 0.166, adjusted R² = 0.143** (n=106, 3 regressors + constant).

Momentum and storage change anomaly are significant at the 5% level; HDD anomaly is marginal (p=0.064). The constant is statistically indistinguishable from zero — once the fundamentals and momentum are accounted for, there is no significant leftover drift to explain. Storage change and HDD both carry a negative sign: a market that tightened less than usual (storage built up faster than normal, or it was warmer than normal) tends to see the TTF price fall slightly the following month. See Section 7 for why HDD's sign is negative despite intuitively "colder should mean higher prices."

### Diagnostics

| Test | Statistic | Interpretation |
|---|---|---|
| Durbin–Watson | 1.98 | No material residual autocorrelation (≈2.0 = ideal) |
| Residual lag-1 autocorrelation | 0.01 | Confirms residuals are close to white noise |
| Breusch–Pagan LM test | p = 0.324 | Fails to reject homoskedasticity — no material heteroskedasticity concern |

All three diagnostics are clean. Residual autocorrelation is essentially absent, and the Breusch-Pagan test comfortably fails to reject homoskedasticity. Newey-West HAC standard errors are still reported throughout for robustness, but this model's residuals are well-behaved by conventional OLS standards too.

### Out-of-sample performance

The model was evaluated with an expanding-window walk-forward backtest: starting from a minimum 60-month training window, the model is refit each month using only data available up to that point, and used to forecast the next month's return one step ahead. This produced 63 out-of-sample forecasts spanning 2021-03 to 2026-05, compared against a random-walk (no-change) benchmark, a momentum-only (AR(1)) benchmark, and the historical mean.

| Model | RMSE | MAE | Directional hit rate | OOS R² vs. random walk |
|---|---:|---:|---:|---:|
| Full model (fundamentals + momentum) | 0.196 | 0.146 | 60.3% | +0.042 |
| Momentum-only | 0.203 | 0.150 | 57.1% | -0.027 |
| Random walk (no-change) | 0.201 | 0.151 | n/a* | — |
| Historical mean | 0.201 | 0.153 | 49.2% | -0.007 |

\*Random walk always forecasts zero change, so a directional hit-rate is not defined for it.

The full model modestly outperforms all three benchmarks on RMSE, MAE, and directional accuracy, and is the only specification with a positive out-of-sample R² relative to the random walk. A paired test on the loss differential between the full model and the random walk gives a t-statistic of only 0.38, which does not clear conventional significance thresholds. Given the short out-of-sample window (63 months, covering essentially one full boom–bust gas-price cycle including the 2021–2023 European energy crisis), this should be read as weak, suggestive evidence at best that the fundamentals used here add real-time forecasting value beyond momentum alone.

### Actual vs. fitted

![Actual vs Fitted](actual_vs_fitted.png)

*Actual TTF price against the model's fitted price level, 2016-03 to 2026-05. Fitted price is reconstructed as the prior month's actual TTF price compounded by the model's predicted one-month log return. Coefficients are estimated once on the 2016-03–2024-12 training sample; the dotted line marks the boundary, after which the 2025-01–2026-05 segment applies the same fixed coefficients to genuinely new, out-of-sample data.*

Because a one-month-ahead return model necessarily produces small adjustments relative to the price level itself, the chart above is dominated visually by the carried-forward previous price (correlation between fitted and naive previous-price ≈ 0.99). The figure below isolates the model's actual contribution — its adjustment relative to a naive "no change" forecast — against what actually happened that month, which is a more direct read on model skill:

![Model Adjustment vs Actual Change](model_adjustment.png)

*The model's predicted deviation from a naive random-walk forecast (dashed) against the actual month-over-month price change (solid), in USD/mmbtu. The two series share the same direction (sign) in 61.8% of months across the full 2016-03–2026-05 sample.*

### Forecast horizon robustness (1 to 3 months ahead)

The model was re-estimated and re-backtested for 2- and 3-month-ahead horizons (predictors re-lagged accordingly):

| Horizon | adj. R² (in-sample) | Directional hit rate (OOS) | OOS R² vs. random walk | Loss-differential t-stat |
|---|---:|---:|---:|---:|
| 1 month | 0.143 | 60.3% | 0.042 | 0.38 |
| 2 months | 0.109 | 67.2% | 0.075 | 1.00 |
| 3 months | 0.129 | 61.0% | 0.110 | 1.27 |

Storage change anomaly is the only variable that remains significant at the 5% level across all three horizons (p=0.018, 0.019, 0.049); momentum and HDD are significant or marginal at H=1 but not at H=2/H=3. Out-of-sample R² relative to the random walk improves with horizon, but the loss-differential t-statistic does not reach conventional significance at any horizon tested. A placebo check using pure random noise in an identically-constructed overlapping regression shows R² rising mechanically with horizon too (the Boudoukh–Richardson–Whitelaw overlapping-regression bias), so the horizon results here should be read as **suggestive at best, not confirmatory** — broadly consistent with a modest, gradually-diffusing fundamentals effect, but not independent statistical proof of one.

## 7. Economic Interpretation

**Storage change** is the month-over-month change in EU+UK average gas storage levels (bcm) — a direct flow measure of whether the market tightened or loosened that month, distinct from the storage *level* itself (which was also tested and found to have little standalone explanatory power). A coefficient of -0.0141 is directionally intuitive: when storage built up faster than seasonal-normal (or drew down more slowly) last month, that is a bearish signal, and the TTF price tends to fall slightly the following month. It is also the most robust variable in this model, remaining significant at every forecast horizon tested (Section 6).

**London HDDs** (heating degree days) are a standard measure of how much colder than average the temperature was, used to proxy heating demand. Any European city's HDDs, or a combination, could have been used; London was chosen partly because CME lists London HDD derivatives, which could in principle be combined with this model or with ECMWF/other weather forecasts in future work. The coefficient (-0.0008) is, at first glance, counterintuitive: it says that when it is colder than normal in London, the TTF price tends to *fall* slightly the following month.

This apparent contradiction is resolved by looking at *when* each variable's effect shows up, not just its sign. Comparing each variable's contemporaneous correlation with TTF returns to its one-month-lagged correlation reveals two distinct patterns:

- **Storage change: continuation.** Its contemporaneous and lagged correlations with TTF returns are nearly identical (-0.31 vs -0.31). A bearish month stays bearish into the following month — the effect does not reverse, it persists. This is consistent with **sluggish price discovery**: storage figures (AGSI/GIE) are aggregated, reported-with-a-lag physical flow data that the market only fully digests over several weeks, not instantly. An unusually loose month keeps weighing on price into the next month as the market gradually absorbs the full scale of what built up.
- **HDD: reversal.** Here the sign flips (contemporaneous +0.18, lagged -0.08) — the signature of "priced in, then partially unwinds." This is plausible because temperature is a close-to-real-time, continuously observed and actively-traded input (forecasts update daily, weather derivatives exist), so a cold month is absorbed into price almost immediately; what is left over for the following month looks more like a partial correction of that overshoot than genuine residual under-pricing.

The general principle — that physical flow/balance data diffuses into price gradually, while weather is priced in close to instantly and then partially reverses — is a reasonable economic prior independent of this dataset, given how each type of data is generated and reported.

**Momentum**, the deseasonalized lagged TTF return, is included primarily as a control: it lets storage change and HDD be interpreted as adding value *beyond* what is already predictable from the price series' own recent behaviour, and it also helps absorb residual autocorrelation that would otherwise be left in the errors.

**The constant** is statistically indistinguishable from zero (p=0.703) — once storage change, HDD, and momentum are accounted for, there is no significant residual drift left for the model to explain.

## 8. Limitations

- The small number of observations (~100–120 monthly, effectively fewer given autocorrelation) limits the strength of any statistical inference throughout this analysis.
- Variables that are only available with a substantial reporting lag were excluded from the month-ahead model, since by the time they would be available, the forecast would effectively be "forecasting the past." Only variables that can plausibly be estimated or obtained close to real time were used.
- An LNG sendout variable was tested but excluded on stationarity grounds (Section 6) — a useful general caution that a non-stationary series trending alongside the dependent variable over one structural period can look like a genuine predictor when it is not.
- Combinations of more lagged variables were tried for several-months-ahead forecasting, which runs into the Boudoukh–Richardson–Whitelaw overlapping-regression bias discussed in Section 6: this bias reduces the effective number of independent observations, which is already limited in this dataset. The full dataset is nonetheless reported and retained for reference, in case it proves useful for other approaches.
- A raw supply–demand balance construction (production + LNG sendout + pipeline imports − pipeline exports − consumption) was also tested as an alternative to storage change and found broadly comparable in explanatory power; the choice between these related "market tightness" measures is not fully settled by this sample size and should be treated as a specification choice rather than a definitively established result.
- The model's out-of-sample directional hit rate (60.3% at H=1) is more modest than its error-based performance (RMSE/OOS R²) would suggest — it appears somewhat better at capturing the *size* of a plausible price move than reliably getting its *direction* right every month.
- A properly stationary treatment of LNG-related supply data (e.g. a capacity-utilization ratio rather than a raw sendout level) is a plausible direction for future work, given the ongoing structural growth in global LNG export capacity. An international crisis dummy, or a regime-dependent switching model (e.g. conditioned on the VIX or Brent crude), is another plausible extension not attempted here.

## 9. Repository Contents

- `ttf_model.py` — full pipeline: data loading, feature engineering, stationarity checks, HAC regression, diagnostics, walk-forward backtest, horizon robustness check, and chart generation.
- `NG_m_final.csv` — underlying dataset (see Section 3 for sources).
- `actual_vs_fitted.png`, `model_adjustment.png` — charts referenced in Section 6.
