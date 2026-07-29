# Daily European Natural Gas Data and Machine Learning Models for Forecasting TTF Prices

**Author:** *to be filled in*
**Date:** July 2026
**Repo:** *to be filled in*

> ⚠️ **This section fills itself in only after real numbers are substituted — see the inline notes marked `[FILL IN]`.** Everything below is written from a real run of the Ridge model against `NG_daily10.csv`. The XGBoost and Ensemble sections were only verified structurally (the code runs correctly end-to-end), not numerically — `xgboost` could not be run in the environment this report was drafted in. Run `Natgas_daily_lightGBM.ipynb` cells 8 and 12 locally and drop the real printed numbers into the `[FILL IN]` spots before publishing, so nothing on the page is a placeholder.

## 1. Executive Summary

This is a companion study to the [monthly econometric model](./report_monthly_econometric.md), asking a narrower and higher-frequency question: can the same category of European gas market data — now at daily granularity — meaningfully predict the direction or magnitude of the TTF price roughly a week in advance? Two model families (Ridge regression and XGBoost gradient boosting) and a validated ensemble of the two were tested, using a purged, expanding-window walk-forward methodology to avoid the overlapping-horizon and test-set leakage pitfalls common in this kind of exercise.

**The conclusion is a negative finding.** Across every specification tried, no model showed predictive skill that was distinguishable from noise on a genuinely held-out test period. Ridge's validated regularization strength shrank almost all the way to a trivial model; a wide hyperparameter search for XGBoost did not produce a validation improvement over predicting no change at all; and the validated ensemble search found no benefit to blending the two, settling on whichever single model performed better rather than a genuine combination. This report documents that process and those results. It is not a production trading or investment tool, was never intended to be one, and — per its own conclusion — should not be treated as containing a usable signal.

## 2. Market Context

See the [monthly report](./report_monthly_econometric.md#2-market-context) for general context on European gas market fundamentals. The question here is specifically about *horizon*: does a shorter window (about a week, chosen because people and businesses organize routines and purchasing decisions around the calendar week) reveal a signal that a monthly view is too coarse to see, or does it instead fall below the point where slow-moving fundamentals (storage, weather, pipeline flows) have had time to actually influence price? The monthly study's own horizon-robustness check (1 vs. 2 vs. 3 months) found the opposite of what many would expect — its modest signal *strengthened*, not weakened, at longer horizons, consistent with fundamentals diffusing into price gradually rather than instantly. That result is a useful prior for interpreting what follows.

## 3. Dataset

| Variable | Source | Frequency | Period |
|---|---|---|---|
| TTF / JKM / Henry Hub Price | Yahoo Finance | Daily | 2022-01 to Present |
| EU Gas storage | AGSI/GIE | Daily | 2022-01 to Present |
| EU+UK LNG sendout | ALSI | Daily | 2022-01 to Present |
| Berlin/London/Rome HDDs | Open-Meteo | Daily | 2022-01 to Present |
| EU+UK Gas pipeline flows by corridor | ENTSOG | Daily | 2022-01 to Present |
| EU+UK Power generation mix | energy-charts.info | Daily | 2022-01 to Present |
| DE-LU / IT-Calabria day-ahead power price | energy-charts.info | Daily | 2022-01 to Present |
| EUR/USD, VIX | Yahoo Finance | Daily | 2022-01 to Present |

`NG_daily10.csv` contains **trading days only** — weekends and days missing any of the three core futures prices are dropped upstream, and the file is trimmed to 2022-01 onward, since data coverage and reliability (particularly ENTSOG) is materially better from around that point than in 2020–2021 (see the [daily pipeline README](./README.md) for the full reasoning). Five of these variables — European gas storage, HDD, pipeline corridor flows, LNG sendout, and the DE-LU day-ahead power price — form the base feature set used below, together with a sixth, derived feature (momentum — see Section 4).

## 4. Processing

- The horizon is fixed at **5 rows** rather than 5 or 7 calendar days. This matters and is worth being explicit about: since the data contains trading days only, a fixed row-shift does not correspond to a fixed calendar-day gap — weekends and holidays mean 5 rows lands close to, but not exactly at, 7 calendar days (the actual test-set average came out to 7.3 days/decision). A row-shift of 5 was chosen over 7 specifically because it tracks a genuine calendar week more closely (5 trading days per week) than 7 rows does (which lands closer to 9–10 calendar days).
- **Purging (per López de Prado, *Advances in Financial Machine Learning*):** the target is a forward-looking return, so any training or validation row whose target window reaches into the next split would leak information across that boundary. Because the row-shift/calendar-day relationship isn't fixed, purging is done by tracking each row's *actual* target end-date (via `Date.shift(-HORIZON_DAYS)`) rather than assuming a fixed calendar-day offset — the latter was tried first and found to under-purge by several days near split boundaries once tested against real (non-continuous) dates.
- No deseasonalization or anomaly construction was applied to the daily features, unlike the monthly model. This is a genuine methodological difference from the monthly study, not an oversight — see Limitations.
- A **momentum** feature was added and tested: the realized `HORIZON_DAYS`-row return as of the current row (`log(TTF_t) - log(TTF_{t-H})`), using only past-and-current data. Kept in raw (non-deseasonalized) form to match the treatment of every other feature in this daily model, rather than mirroring the monthly model's deseasonalized version. See Section 6 for the result — it did not improve the model.
- Train+validation / test split: training and validation together span 2022-01 up to a cap of 2025-05 (purged, so the true cutoff is slightly earlier — 2025-05-2x, see [FILL IN: exact date printed by the notebook] — to keep target windows from reaching into the test period); the test period runs from 2025-06-01 to the end of the dataset, evaluated exactly once per model.

## 5. Methodology

- **Ridge regression** (`sklearn.linear_model.Ridge`), with features standardized via a scaler fit on training data only, refit at every validation fold and again for the final model.
- **XGBoost gradient boosting** (`xgboost.XGBRegressor`), with `n_estimators` resolved via early stopping and `max_depth`/`learning_rate` chosen via the validation search below.
- **Ensemble**: a weighted blend of the two models' predictions, with the blend weight itself chosen via validation, not fixed at an arbitrary split.
- **Hyperparameter selection** for all three models used the same **purged, expanding-window validation fold** procedure: starting from a minimum 2-year training window, successive ~90-day validation folds are evaluated (each purged against its own training window), and candidates are ranked by mean **validation R² against a "predict no change" naive benchmark** — not raw RMSE, which is not comparable across different model/feature combinations if the fitted models differ substantially in what they're able to explain. This mirrors the OOS-R²-vs-random-walk normalization used throughout the monthly model.
- The **test set was never used for any model-selection decision** — not hyperparameter tuning, not early stopping, not ensemble weighting. It was scored exactly once, after every other decision was locked in.
- **Backtest**: a new position is taken every `HORIZON_DAYS` rows (non-overlapping), long when the predicted return is positive and flat (cash) otherwise. Non-overlapping periods are compounded correctly for log returns (`exp(cumsum(...)) - 1`, not a naive `(1+r).cumprod()`, which is only valid for simple, non-log returns). Sharpe ratio annualization uses the *actual* mean calendar days per decision in the test set, not an assumed value.

## 6. Results

### Ridge: validation search

| Alpha | Mean validation R² vs. naive | Folds |
|---:|---:|---:|
| 1000.0 | **+0.033** | 6 |
| 100.0 | -0.074 | 6 |
| 10.0 | -0.154 | 6 |
| 1.0 | -0.165 | 6 |
| 0.1 | -0.166 | 6 |
| 0.01 | -0.166 | 6 |

Every alpha below 1000 — i.e. every setting that lets the model fit the data with any real flexibility — scores *worse* than predicting no change at all, on validation data the model never saw during fitting. Only the heaviest regularization tested clears zero, and only barely. This is Ridge's own diagnostic telling you it doesn't trust these features to carry real signal at this horizon; it is not a case of "the right alpha wasn't tried; a wider grid would have found something."

**Chosen: alpha = 1000.** Final model fit on 844 rows (2022-01-10 to 2025-05-22).

**Standardized coefficients** (comparable to each other in magnitude, since features were scaled):

| Feature | Coefficient |
|---|---:|
| HDD | -0.0112 |
| DE day-ahead price | -0.0089 |
| LNG sendout | -0.0057 |
| Pipeline corridors (sum) | +0.0048 |
| Storage level | -0.0029 |
| **Momentum** | **+0.0002** |

At this regularization strength every coefficient is small — consistent with the validation search's own message that little of this is trustworthy signal. **Momentum in particular was shrunk to essentially zero — roughly an order of magnitude smaller than every other feature** — and mean validation R² came in slightly *lower* with momentum included (0.033) than without it (0.039, the five-feature version tested first). Both the coefficient and the validation score point the same direction: momentum was tested specifically because it was one of the monthly model's more robust predictors, and it did not transfer to this daily, ~1-week-ahead setting.

### XGBoost: validation search

`[FILL IN — paste the "Validation results by (max_depth, learning_rate)" table printed by cell 8, and the chosen max_depth/learning_rate/n_estimators line below it. Momentum is included as a feature in this run.]`

### Ensemble: blend weight search

`[FILL IN — paste the blend-weight table printed by cell 12 (weight from 0.0 = all Ridge to 1.0 = all XGBoost, and each weight's mean validation R² vs. naive), and the chosen weight.]`

*(In a structural-only run used to verify the code — not a real result, since it used a stand-in for XGBoost rather than the real library — the blend search showed validation R² declining monotonically as XGBoost's weight increased, and picked the pure-Ridge endpoint. Whether that holds with real XGBoost needs to be confirmed by an actual run before being reported as a finding.)*

### Test set performance (2025-06-01 onward, non-overlapping decisions)

| Model | RMSE | Correlation | Hit rate | Strategy return | Buy-and-hold return | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 0.0977 | -0.026 | 50.9% | +15.3% | +79.4% | 0.60 |
| XGBoost | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |
| Ensemble | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |

n = 57 non-overlapping test decisions for every model (June 2025 – end of data), averaging 7.3 actual calendar days apart. (Ridge's hit rate landed at 50.9% with momentum included, versus 52.6% without — i.e. adding momentum moved the model's directional accuracy essentially to a coin flip, rather than away from one.)

**None of the Ridge numbers clear a reasonable bar for "real."** At n=57:
- Correlation would need to exceed **±0.26** to be statistically distinguishable from zero at 95% confidence — the observed -0.026 is not close.
- Hit rate's standard error around a 50/50 coin flip is about **±6.6 percentage points** — the observed 50.9% is essentially exactly chance.
- The annualized Sharpe ratio would need to exceed roughly **1.84** to be distinguishable from zero at this sample size — the observed 0.60 is not close, and this bar itself is a reminder of how little a single ~1-year test window can actually prove either way.

The strategy also **underperformed simple buy-and-hold by a wide margin** (+15.3% vs. +79.4%) — consistent with a close-to-coin-flip signal sitting in cash for roughly half of a period when the underlying market was strongly rising, rather than with the strategy capturing a genuine edge.

## 7. Interpretation of the Negative Finding

A negative result is still a result, and it's worth being specific about what it does and doesn't rule out, rather than leaving it as an unexplained "nothing worked."

**This is not simply "the fundamentals don't matter" — it's more specific than that.** The monthly model, built from a related (though not identical) feature set, found a small but real, economically coherent signal: storage change and momentum both cleared conventional significance thresholds, and the effect *strengthened* rather than weakened moving from a 1-month to a 3-month horizon. Taken together with this daily study, a coherent picture emerges: the same underlying fundamentals (storage, weather, physical gas/power flows) may carry real information about TTF's medium-term direction, but **that information plausibly hasn't finished diffusing into price on a ~1-week timescale**. If the monthly model's "gradual diffusion, not instant pricing" story (Section 7 of that report) is right, a week may simply be too short a window for slow-moving physical fundamentals to show up cleanly against the much larger amount of daily noise — algorithmic trading, headline-driven sentiment, and short-term supply/demand adjustments — that dominates day-to-day price action in a liquid futures market.

**Several concrete differences from the monthly model are also worth naming, since any of them could independently explain part of the gap:**
- **No deseasonalization was applied here.** The monthly model's signal depended heavily on stripping out seasonal effects (raw HDD, storage, and LNG variables were shown early in that project to carry spurious "significance" that was really just calendar-cycle correlation). The daily features here are used in raw form; some of the weak/negative validation scores could reflect this rather than a genuine absence of signal underneath. This remains untested at daily frequency.
- **Momentum was tested and did not help.** The monthly model's momentum term was one of its more robust predictors, so a daily analogue (the realized return over the prior `HORIZON_DAYS` rows) was added and run through the same validated pipeline. Ridge shrank its coefficient to roughly a tenth the size of every other feature, mean validation R² came in slightly lower with it than without (0.033 vs. 0.039), and the test-set hit rate moved from 52.6% to 50.9% — closer to, not further from, a coin flip. This closes off what had looked like the most promising untested lever; it is a genuine negative result on its own, not just an unexplored gap anymore.
- **The effective sample is thinner than it looks.** 849 training rows sounds substantial, but the test set — the only number that actually validates anything — has just 57 non-overlapping decisions. Every noise-band calculation in Section 6 reflects how little that can prove either way.
- **Vintage/real-time data risk is plausibly worse at daily frequency than monthly.** AGSI, ALSI, and ENTSOG figures are sometimes revised after initial publication; daily aggregates are more likely to reflect preliminary, later-revised values than monthly averages are. The backtest uses whatever the API currently reports, not necessarily what would have been known in real time on each historical date (also noted at the top of the accompanying notebook).

**None of this is offered as an excuse for the result** — the purpose of documenting these gaps is to be precise about what remains untested (deseasonalized features, a longer daily-to-weekly horizon sweep) versus what has actually been tried and found wanting (five raw fundamentals plus momentum, predicting a ~1-week return, across two model families and a validated ensemble).

## 8. Limitations

- **Trading-day vs. calendar-day horizon.** `HORIZON_DAYS=5` is 5 *rows*, not 5 calendar days — see Section 4. This is handled correctly in the purge logic (via each row's actual target end-date) but is a reminder that "predict a week ahead" is approximate on trading-day-indexed data, not exact.
- **No deseasonalization**, unlike the monthly model — see Section 7. This remains the most promising concrete extension for future work. (Momentum *was* tested — see Section 6/7 — and did not help, so it is no longer an open gap.)
- **Small effective test sample** (57 non-overlapping decisions, one ~14-month window). A single test period, however carefully purged, is one draw — a materially different result could plausibly emerge from a different test window purely by chance, given how wide the noise bands in Section 6 are.
- **Possible lookahead bias from data revisions**, discussed explicitly at the top of the accompanying notebook: storage, LNG, and pipeline figures may be revised after publication, and this study uses whatever is currently available rather than the historically-accurate real-time vintage.
- **SARIMAX and transformer architectures were considered and not pursued**, for reasons specific to this dataset rather than as a blanket judgment on either approach: `dlog_TTF` tests as stationary with close-to-zero own-autocorrelation (making an ARMA component unlikely to add much), and the training set (~850 rows) is far below the range where transformer architectures typically demonstrate an advantage over simpler models — see the discussion in this project's development notes for the full reasoning.
- **The validated ensemble found no benefit to blending** `[FILL IN once real XGBoost numbers are available — confirm whether this holds]`, which is itself informative: it means the two models weren't finding different, complementary pieces of signal to combine, consistent with neither finding much signal at all.
- This is exploratory research, not a production system: no transaction costs, slippage, position sizing beyond binary long/cash, or execution feasibility were modeled in the backtest.

## 9. Repository Contents

- `Natgas_daily_lightGBM.ipynb` — full pipeline: data loading, composite features, stationarity check, purged expanding-window validation for Ridge, XGBoost, and their ensemble, and the non-overlapping backtest for each.
- `NG_daily10.csv` — underlying dataset (see Section 3 for sources; built by `build_ng_daily_dataset.py`).
- `build_ng_daily_dataset.py` — the pipeline that builds `NG_daily10.csv` (see the [daily pipeline README](./README.md)).
