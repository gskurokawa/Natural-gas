# Daily European Natural Gas Data and Machine Learning Models for Forecasting TTF Prices

**Author:** G. Kurokawa
**Date:** July 2026
**Repo:** https://github.com/gskurokawa/Natural-gas/tree/main/Daily

## 1. Executive Summary

This is a companion study to the [monthly econometric model](./report_monthly_econometric.md), asking a narrower and higher-frequency question: can the same category of European gas market data — now at daily granularity — meaningfully predict the direction or magnitude of the TTF price roughly a week in advance? Two model families (Ridge regression and XGBoost gradient boosting) and a validated ensemble of the two were tested, using a purged, expanding-window walk-forward methodology to avoid the overlapping-horizon and test-set leakage pitfalls common in this kind of exercise.

**The conclusion is a negative finding.** Across every specification tried, no model showed predictive skill that was distinguishable from noise on a genuinely held-out test period. Ridge's validated regularization strength shrank almost all the way to a trivial model; XGBoost's hyperparameter search never found a single combination that beat predicting no change at all, out of 20 tested; and the validated ensemble search found only a negligible benefit to blending the two, landing within noise of pure Ridge rather than a meaningfully different combination. This report documents that process and those results. It is not a production trading or investment tool, was never intended to be one, and — per its own conclusion — should not be treated as containing a usable signal.

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
- Train+validation / test split: training and validation together span 2022-01 up to a cap of 2025-05 (purged, so the true cutoff is slightly earlier — 2025-05-22 in practice, three trading rows short of the 5/31 cap — to keep target windows from reaching into the test period); the test period runs from 2025-06-01 to the end of the dataset, evaluated exactly once per model.

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

| max_depth | learning_rate | Mean validation R² vs. naive | n_estimators (median) |
|---:|---:|---:|---:|
| 2 | 0.10 | **-0.017** | 22.5 |
| 2 | 0.05 | -0.029 | 5.0 |
| 2 | 0.03 | -0.031 | 11.5 |
| 2 | 0.01 | -0.033 | 35.0 |
| 3 | 0.01 | -0.049 | 18.5 |
| ... | ... | ... (15 more combinations, all more negative) | ... |
| 4 | 0.01 | -0.105 | 2.5 |

*(20 combinations tested in total, ranked by mean validation R²; only the best 5 and worst 1 shown — every single one scored below zero.)*

**Every one of the 20 (max_depth, learning_rate) combinations tested scored worse than predicting no change, on validation data.** Unlike Ridge, which found one setting (heavy regularization) that narrowly cleared zero, XGBoost's validation search found nothing that did — the "chosen" combination is simply the least-bad of twenty options that all underperformed the naive benchmark.

**Chosen: max_depth=2, learning_rate=0.1, n_estimators=22.** Final model fit on 844 rows (2022-01-10 to 2025-05-22).

**Feature importance:**

| Feature | Importance |
|---|---:|
| DE day-ahead price | 0.208 |
| Pipeline corridors (sum) | 0.201 |
| HDD | 0.191 |
| LNG sendout | 0.157 |
| Momentum | 0.151 |
| Storage level | 0.091 |

Unlike Ridge, which shrank momentum to roughly a tenth the size of its other coefficients, XGBoost's importance ranking treats momentum comparably to the other features — but this needs to be read alongside the validation result directly above it: XGBoost with momentum included still never beat the naive benchmark on validation data, so this importance ranking describes how the model *tried* to use its features, not evidence that any of them carried reliable signal.

### Ensemble: blend weight search

| Weight on XGBoost | Mean validation R² vs. naive |
|---:|---:|
| 0.1 | **+0.0330** |
| 0.0 (pure Ridge) | +0.0327 |
| 0.2 | +0.0305 |
| 0.3 | +0.0252 |
| 0.4 | +0.0172 |
| 0.5 | +0.0063 |
| 0.6 | -0.0073 |
| 0.7 | -0.0238 |
| 0.8 | -0.0430 |
| 0.9 | -0.0651 |
| 1.0 (pure XGBoost) | -0.0899 |

**Chosen weight: 0.1 on XGBoost, 0.9 on Ridge.** The margin over pure Ridge (weight 0.0) is +0.0003 — three ten-thousandths, on validation R² that itself only reaches +0.033 at its best. This is not meaningfully different from picking pure Ridge; the search technically preferred a small XGBoost contribution, but by a margin far smaller than any of the noise bands discussed in this report. The broader shape of the table is the more informative part: validation performance declines steadily as XGBoost's weight increases past about 0.4, consistent with XGBoost being the weaker of the two models by this metric (matching its validation search in the previous section, where it never beat the naive benchmark at all).

### Test set performance (2025-06-01 onward, non-overlapping decisions)

| Model | RMSE | Correlation | Hit rate | Strategy return | Buy-and-hold return | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 0.0977 | -0.026 | 50.9% | +15.3% | +79.4% | 0.60 |
| XGBoost | 0.0976 | +0.069 | 54.4% | +31.9% | +79.4% | 0.97 |
| Ensemble | 0.0975 | -0.011 | 52.6% | +15.8% | +79.4% | 0.62 |

n = 57 non-overlapping test decisions for every model (June 2025 – end of data), averaging 7.3 actual calendar days apart. (Ridge's hit rate landed at 50.9% with momentum included, versus 52.6% without — i.e. adding momentum moved the model's directional accuracy essentially to a coin flip, rather than away from one.)

**XGBoost's test numbers look the best of the three at a glance, and this is worth reading carefully rather than at face value.** Its correlation (+0.069) and hit rate (54.4%) are nominally better than Ridge's or the Ensemble's, and its Sharpe (0.97) is the highest. But recall from the previous section: **XGBoost's own validation search never found a single hyperparameter combination, out of 20 tested, that beat the naive "predict no change" benchmark.** Every validation score was negative. A model whose own validation process consistently says "this doesn't work," then producing a nominally better score on one ~14-month test window, is a textbook case of test-set noise rather than genuine skill revealing itself late — precisely the scenario the validation methodology in this report was built to guard against being fooled by. The correlation threshold for this sample size (below) makes this concrete rather than just an assertion.

**None of the three models' numbers clear a reasonable bar for "real."** At n=57:
- Correlation would need to exceed **±0.26** to be statistically distinguishable from zero at 95% confidence — the largest observed magnitude (XGBoost's +0.069) is not close, at barely a quarter of that threshold.
- Hit rate's standard error around a 50/50 coin flip is about **±6.6 percentage points** — every model's hit rate (50.9–54.4%) is within one standard error of pure chance.
- The annualized Sharpe ratio would need to exceed roughly **1.84** to be distinguishable from zero at this sample size — even XGBoost's 0.97, the highest of the three, is little more than half that bar.

Every model also **underperformed simple buy-and-hold by a wide margin** (+15–32% vs. +79.4%) — consistent with a close-to-coin-flip signal sitting in cash for roughly half of a period when the underlying market was strongly rising, rather than with any of the three capturing a genuine edge.

## 7. Interpretation of the Negative Finding

A negative result is still a result, and it's worth being specific about what it does and doesn't rule out, rather than leaving it as an unexplained "nothing worked."

**This is not simply "the fundamentals don't matter" — it's more specific than that.** The monthly model, built from a related (though not identical) feature set, found a small but real, economically coherent signal: storage change and momentum both cleared conventional significance thresholds, and the effect *strengthened* rather than weakened moving from a 1-month to a 3-month horizon. Taken together with this daily study, a coherent picture emerges: the same underlying fundamentals (storage, weather, physical gas/power flows) may carry real information about TTF's medium-term direction, but **that information plausibly hasn't finished diffusing into price on a ~1-week timescale**. If the monthly model's "gradual diffusion, not instant pricing" story (Section 7 of that report) is right, a week may simply be too short a window for slow-moving physical fundamentals to show up cleanly against the much larger amount of daily noise — algorithmic trading, headline-driven sentiment, and short-term supply/demand adjustments — that dominates day-to-day price action in a liquid futures market.

**Several concrete differences from the monthly model are also worth naming, since any of them could independently explain part of the gap:**
- **No deseasonalization was applied here.** The monthly model's signal depended heavily on stripping out seasonal effects (raw HDD, storage, and LNG variables were shown early in that project to carry spurious "significance" that was really just calendar-cycle correlation). The daily features here are used in raw form; some of the weak/negative validation scores could reflect this rather than a genuine absence of signal underneath. This remains untested at daily frequency.
- **Momentum was tested and did not help.** The monthly model's momentum term was one of its more robust predictors, so a daily analogue (the realized return over the prior `HORIZON_DAYS` rows) was added and run through the same validated pipeline. Ridge shrank its coefficient to roughly a tenth the size of every other feature, mean validation R² came in slightly lower with it than without (0.033 vs. 0.039), and the test-set hit rate moved from 52.6% to 50.9% — closer to, not further from, a coin flip. This closes off what had looked like the most promising untested lever; it is a genuine negative result on its own, not just an unexplored gap anymore.
- **The effective sample is thinner than it looks.** 844 training rows sounds substantial, but the test set — the only number that actually validates anything — has just 57 non-overlapping decisions. Every noise-band calculation in Section 6 reflects how little that can prove either way.
- **Vintage/real-time data risk is plausibly worse at daily frequency than monthly.** AGSI, ALSI, and ENTSOG figures are sometimes revised after initial publication; daily aggregates are more likely to reflect preliminary, later-revised values than monthly averages are. The backtest uses whatever the API currently reports, not necessarily what would have been known in real time on each historical date (also noted at the top of the accompanying notebook).

**None of this is offered as an excuse for the result** — the purpose of documenting these gaps is to be precise about what remains untested (deseasonalized features, a longer daily-to-weekly horizon sweep) versus what has actually been tried and found wanting (five raw fundamentals plus momentum, predicting a ~1-week return, across two model families and a validated ensemble).

## 8. Limitations

- **Trading-day vs. calendar-day horizon.** `HORIZON_DAYS=5` is 5 *rows*, not 5 calendar days — see Section 4. This is handled correctly in the purge logic (via each row's actual target end-date) but is a reminder that "predict a week ahead" is approximate on trading-day-indexed data, not exact.
- **No deseasonalization**, unlike the monthly model — see Section 7. This remains the most promising concrete extension for future work. (Momentum *was* tested — see Section 6/7 — and did not help, so it is no longer an open gap.)
- **Small effective test sample** (57 non-overlapping decisions, one ~14-month window). A single test period, however carefully purged, is one draw — a materially different result could plausibly emerge from a different test window purely by chance, given how wide the noise bands in Section 6 are.
- **Possible lookahead bias from data revisions**, discussed explicitly at the top of the accompanying notebook: storage, LNG, and pipeline figures may be revised after publication, and this study uses whatever is currently available rather than the historically-accurate real-time vintage.
- **SARIMAX and transformer architectures were considered and not pursued**, for reasons specific to this dataset rather than as a blanket judgment on either approach: `dlog_TTF` tests as stationary with close-to-zero own-autocorrelation (making an ARMA component unlikely to add much), and the training set (~850 rows) is far below the range where transformer architectures typically demonstrate an advantage over simpler models — see the discussion in this project's development notes for the full reasoning.
- **The validated ensemble found only a negligible benefit to blending** (weight 0.1 on XGBoost vs. 0.0, a validation-R² difference of 0.0003 — within noise), settling close to "use Ridge alone." This is itself informative: it means the two models weren't finding different, complementary pieces of signal to combine, consistent with neither finding much signal at all.
- This is exploratory research, not a production system: no transaction costs, slippage, position sizing beyond binary long/cash, or execution feasibility were modeled in the backtest.

## 9. Repository Contents

- `Natgas_daily_lightGBM.ipynb` — full pipeline: data loading, composite features, stationarity check, purged expanding-window validation for Ridge, XGBoost, and their ensemble, and the non-overlapping backtest for each.
- `NG_daily10.csv` — underlying dataset (see Section 3 for sources; built by `build_ng_daily_dataset.py`).
- `build_ng_daily_dataset.py` — the pipeline that builds `NG_daily10.csv` (see the [daily pipeline README](./README.md)).
