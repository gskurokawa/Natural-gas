# How Predictable is Month-Ahead European Gas (TTF) from European Fundamentals?

### A self-contained European model, from a deployable forecast to a perfect-foresight ceiling

*Prepared 1 August 2026. *

---

## Summary

We asked how much of the month-ahead change in European gas prices (TTF) a **self-contained European model** can explain, using only European and EU+UK fundamentals — gas balance and storage, the power-generation mix, Norwegian supply, and European weather. We attacked it from three angles of increasing generosity, and the picture that emerges is consistent and, in the end, decisive.

A strictly **deployable** forecast (every predictor known one month in advance) is close to a random walk: only storage change and price momentum carry durable signal, and even together they do not significantly beat "assume no change." Granting the model **perfect foresight of this month's fundamentals** raises in-sample explanatory power to ~0.42 and — for the only time in the study — significantly beats the random walk over the full sample; but roughly half of that is **simultaneity** (price driving fuel-switching and import pull, not the reverse), and it fails on recent calm-market data. Stripping the model down to fundamentals that price genuinely *cannot* cause — weather, plus storage — gives an honest, endogeneity-free **ceiling of ~0.21**, with all signs economically correct. The decisive result: **even with perfect foresight of those clean fundamentals, the model does not beat a random walk out of sample**, and its explanatory power collapses to ~0.05 in calm months.

The conclusion is therefore sharper than "fundamentals are hard to forecast." At a one-month horizon, the European fundamental-to-price link is **causally real but weak and noisy outside of crises** — so much so that an oracle on the fundamentals still would not have beaten a random walk in normal conditions. European gas at H=1 is close to unforecastable in calm markets, and only modestly explainable, mostly during dislocations.

---

## 1. Question and scope

The target throughout is the one-month log return of front-month TTF (USD/mmbtu), at a fixed **H = 1** horizon (chosen for degrees-of-freedom reasons; longer horizons induced severe overlapping-returns autocorrelation and added no reliable signal). The predictor universe is restricted to European / EU+UK fundamentals — no global prices (JKM, Henry Hub), no non-European weather, no global LNG trade, no financial variables — so that this measures what *Europe's own* fundamentals can do.

We ran the model in two modes:

- **Predetermined (deployable):** every predictor is lagged one month, so the forecast uses only information available at the forecast origin. This is an honest, real-world forecast.
- **Conditional (perfect foresight):** the same-month fundamentals enter *contemporaneously* — we assume we know (or could forecast) this month's storage, weather, and so on when predicting this month's price move. This is not a deployable forecast; it is an **explanatory ceiling** that separates "fundamentals don't matter" from "fundamentals matter but are themselves unforecastable." Momentum stays lagged in every mode (using the contemporaneous return would be circular).

## 2. Data

A monthly dataset spanning January 2015 to May 2026 (~137 observations). The European candidate predictors cover: EU+UK gas balance and storage (level and change, production, net pipeline imports, EU+UK LNG imports, net supply, and total / non-power / power-sector demand); the EU+UK power-generation mix (gas, coal, nuclear, hydro, residual load); Norwegian supply (production, supply reduction, planned and unplanned outages); and European weather (heating and cooling degree days, wind speed, solar irradiation, Nordic precipitation).

## 3. Methodology

The design is built to avoid the standard ways a small-sample forecasting study flatters itself:

**No look-ahead.** Every predictor is a *deseasonalized anomaly* — the raw series minus an expanding, prior-years-only calendar-month climatology — so the seasonal benchmark never uses future data. In deployable mode the anomalies are then lagged one month.

**Stationarity gating.** Candidates are screened with ADF and KPSS; series both tests agree are non-stationary are excluded. (This is why trending flow variables are entered as month-over-month *changes* rather than levels — the level anomalies of production, nuclear, LNG imports, etc. carry secular trends and fail the gate; their changes are stationary.)

**Univariate HAC screen, collinearity pruning, and training-only selection.** Each candidate is ranked by its Newey-West t-statistic; collinear pairs (|corr| > 0.70) are pruned; and the compact model is selected using data strictly before January 2025, so the out-of-sample test is never contaminated by hindsight.

**Honest out-of-sample evaluation.** An expanding-window walk-forward refits each month and forecasts the next, benchmarked against a random walk and the historical mean, with significance judged by a **Diebold-Mariano test using a Newey-West variance**. Results are reported over the full backtest and over the clean post-January-2025 slice.

**Regime split.** Coefficients are re-estimated in the 2021–22 crisis window versus calm, and (as a look-ahead-safe robustness check) in high- versus low-volatility months, with a joint Wald test for regime dependence.

**Endogeneity discipline (conditional mode).** When fundamentals enter contemporaneously, some are *price-responsive* — coal generation rises when gas is expensive (fuel switching), imports are pulled in by high prices — so their contemporaneous correlation with price partly runs price → fundamental. We therefore distinguish **truly exogenous** fundamentals (weather, which price cannot cause) from **endogenous / price-responsive** ones, and report a clean ceiling using only the former (plus storage, flagged as semi-endogenous).

## 4. Results

### 4.1 Stationarity

The weather, storage-change, momentum, and Norwegian series are stationary. The *level* anomalies of the trending flow/generation series (production, net piped imports, LNG imports, net supply, nuclear, hydro, gas burn, total/non-power demand) are non-stationary and are gated out; their **change** forms are stationary and used instead. Twenty-seven of thirty-six candidates survive the gate.

### 4.2 The predetermined (deployable) model

Only four predictors are individually significant at 5% in the training-window screen, and two of them are the model's core:

| Predictor | coef | t | p | univariate R² | sign correct? |
|---|---:|---:|---:|---:|:--:|
| Storage change (Δ, deseasonalized) | −0.0165 | −2.67 | 0.008 | 0.098 | yes |
| Momentum (own 1-month return) | +0.289 | +2.54 | 0.011 | 0.084 | yes |
| LNG imports change | −2.6e-5 | −2.12 | 0.034 | 0.029 | yes |
| Cooling degree days | +0.0017 | +2.10 | 0.036 | 0.045 | yes |

Everything else — residual load, net supply, production, pipeline flows, nuclear, hydro, coal, all four Norwegian series, wind, solar, precipitation, and heating-degree-days — is insignificant (p from 0.17 to 0.91). With ~27 candidates, one to two false positives at 5% are expected by chance, and the two "extra" finds (LNG imports and CDD) sit exactly in that p ≈ 0.03–0.04 band. Only storage (p = 0.008) and momentum (p = 0.011) clear a multiplicity-adjusted bar.

Out of sample, the selected model tracks the honest core and neither reliably beats a random walk (full-window OOS R² ≈ +6%, DM ≈ 0.6–0.8 — not significant), and on the clean post-2025 slice both have **negative** OOS R² and lose to the random walk. The regime split confirms why: the two extras are crisis artifacts. LNG-imports change is significant only in the 2021–22 window (crisis t = −3.2, calm t = −1.0, interaction t = −3.3; joint Wald p = 0.002), and cooling-degree-days is not robustly significant anywhere and flips sign across the volatility split. Storage and momentum, by contrast, are stable across regimes.

**Deployable verdict:** a self-contained European model delivers storage plus momentum and nothing else that survives scrutiny — and even that does not beat a random walk at H=1 on recent data.

### 4.3 Conditional model 1 — perfect foresight, all fundamentals (exogenous + endogenous)

Granting the model perfect foresight of *every* same-month fundamental (data-driven selection over the full contemporaneous pool) roughly doubles in-sample fit to **R² = 0.42**, and for the only time in the study the out-of-sample result clears the random walk at conventional significance:

| Model | RMSE | Hit rate | OOS R² vs RW | DM t vs RW |
|---|---:|---:|---:|---:|
| Honest core (lagged) | 0.195 | 60.3% | +5.7% | 0.58 |
| Conditional (perfect foresight, all fundamentals) | 0.170 | 66.7% | **+27.8%** | **2.36** |
| Random walk | 0.201 | — | — | — |

But two problems hollow out that headline. First, **endogeneity**: the two strongest contributors — contemporaneous coal generation (t = 4.4) and pipeline-imports change (t = 3.7) — have the *wrong* economic sign, because they reflect price driving the fundamental (coal-to-gas switching, import pull), not the reverse. Second, **regime concentration**: the fit is crisis-driven (crisis R² 0.59 vs calm 0.30; high-vol 0.43 vs low-vol 0.21), and on the clean post-2025 slice the model has OOS R² of **−26%**, losing to both the core and the random walk. So the impressive full-window number is a mixture of simultaneity and crisis-era co-movement, not durable, causal predictive power.

### 4.4 Conditional model 2 — the endogeneity-free ceiling (weather + storage)

To remove the simultaneity, we pre-specified (no data-driven selection) a model using only fundamentals price cannot cause — the weather block — plus storage (flagged semi-endogenous), all with perfect foresight. The in-sample decomposition:

| Model (training sample) | R² |
|---|---:|
| Momentum only (lagged) | 0.084 |
| Weather only (no momentum) | 0.115 |
| Weather + momentum | 0.171 |
| Weather + storage (no momentum) | 0.198 |
| **Weather + storage + momentum** | **0.214** |

The clean, causal ceiling is therefore about **0.21 — half the 0.42** from the unrestricted run, confirming that a full half of that number was simultaneity. Every coefficient is economically correctly signed (HDD +, CDD +, wind −, solar −, precipitation −, storage −, momentum +), which validates that fundamentals *do* causally move TTF — but only **storage change is individually significant** (t = −2.35, p = 0.019); the weather terms are all correctly signed yet individually weak (|t| ≈ 0.4–1.6). The signal, storage aside, is causally real but statistically thin.

The decisive result is out of sample. **Even with perfect foresight of these clean fundamentals, the model does not beat a random walk**, and it underperforms the parsimonious honest core:

| Model | Full-window OOS R² vs RW | DM t vs RW | Post-2025 OOS R² vs RW |
|---|---:|---:|---:|
| Honest core (lagged) | +5.7% | 0.58 | −15.2% |
| Weather + momentum (foresight) | −3.8% | −0.34 | −23.2% |
| Weather + storage + momentum (foresight) | −1.3% | −0.14 | −24.1% |
| Random walk | — | — | — |

The regime R² shows where the ceiling lives and where it vanishes: crisis 0.36 and high-vol 0.25, but **low-volatility months just 0.05**. In quiet markets, perfect foresight of weather and storage explains essentially nothing about the month-ahead move.

### 4.5 Conditional model 3 — the lean specification (a robustness check)

A natural objection to §4.4 is that its seven-variable model failed out of sample simply because it was over-parameterized — too many weak regressors, too much estimation noise. To rule that out, we ran a deliberately lean three-parameter version keeping only the drivers that carried the clean signal: momentum (lagged), plus contemporaneous storage change and HDD (perfect foresight).

Two results settle it. First, **perfect foresight of storage and HDD buys almost nothing over their lagged values.** The lean foresight model reaches in-sample R² = 0.167 — essentially identical to the honest all-lagged core's 0.173 (build-up: momentum alone 0.084 → +storage 0.142 → +HDD 0.167). Both fundamentals are now correctly signed and individually significant (storage change t = −2.34, p = 0.019; HDD t = +2.15, p = 0.032), so contemporaneous HDD is meaningfully sharper than the lagged version — yet the total explained variance does not rise, because storage and HDD are persistent month-to-month and last month's value already forecasts this month's well.

Second, and decisively, the lean perfect-foresight model **still does not beat a random walk** and is marginally worse than the honest core:

| Model | Full-window OOS R² vs RW | DM t vs RW | Post-2025 OOS R² vs RW |
|---|---:|---:|---:|
| Honest core (lagged) | +5.7% | 0.58 | −15.2% |
| Lean (foresight: momentum + storage + HDD) | +4.0% | 0.53 | −16.3% |
| Random walk | — | — | — |

with DM of the lean model versus the honest core of −0.24, and a regime R² that again collapses in quiet markets (low-volatility 0.023). So the seven-variable model's out-of-sample failure was **not** over-parameterization: even a tight three-parameter model with perfect foresight cannot beat "assume no change." The signal is genuinely weak, not merely diluted.

## 5. Discussion

Reading the three cuts together produces a layered and, ultimately, decisive picture.

The **deployable** model told us that a self-contained European forecast is close to a random walk: storage and momentum are the only durable signals, and they are not enough to win. The obvious next question — is that because fundamentals don't matter, or because we can't forecast them a month ahead? — is exactly what the **conditional** models were designed to answer.

The first conditional model appeared to answer emphatically in favour of fundamentals: perfect foresight doubled the fit and, uniquely, beat the random walk (DM 2.36). But that result does not survive inspection. Half of the explanatory power was **reverse causality** — high prices induce coal-to-gas switching and pull in imports, so those contemporaneous "predictors" are partly *consequences* of the price move — and what remained was **concentrated in the 2021–22 crisis** and failed outright on recent calm data. It is a cautionary example of how contemporaneous, endogenous regressors can manufacture an impressive but hollow number.

The clean ceiling settles the question. Once restricted to fundamentals price genuinely cannot cause, the honest upper bound is about a fifth of monthly variance, all correctly signed — so fundamentals are *causally real* — but carried almost entirely by storage, with weather correctly signed yet weak. And crucially, **that ceiling does not convert into out-of-sample skill even for an oracle**: a model that knows weather and storage perfectly still fails to beat a random walk, because the signal (outside storage, outside crises) is too weak to estimate reliably against the noise, and in calm months there is barely any signal at all (R² ≈ 0.05).

This reframes the study's core finding. The near-random-walk behaviour of month-ahead TTF is **not merely** a statement that fundamentals are hard to forecast. It is stronger: even granted perfect foresight of the clean, exogenous European fundamentals, one would **not** have beaten a random walk in normal market conditions. The monthly fundamental-to-price relationship is genuine but weak and noisy, and it becomes materially exploitable only during dislocations — precisely the episodes that are, themselves, the hardest to anticipate. There are thus two compounding barriers to a deployable European forecast: the fundamentals must be forecast (feasible for storage, much less so for weather even at monthly resolution), and the underlying link is too weak in calm markets to reward even perfect information.

## 6. Conclusion

A self-contained European model does not predict month-ahead TTF well, and the reason is now precisely characterised rather than merely asserted. Storage change and momentum are the only durable, deployable signals, and they do not beat a random walk out of sample on recent data. Fundamentals are causally real — with perfect foresight and clean, exogenous inputs the explanatory ceiling is ~0.21 and every sign is economically correct — but that ceiling is dominated by storage, concentrated in crisis/high-volatility regimes (≈0.05 in calm months), inflated to ~0.42 by simultaneity if endogenous fundamentals are admitted, and, decisively, does **not** translate into out-of-sample forecast skill even under perfect foresight. Month-ahead European gas is close to a random walk in calm conditions and only modestly explainable in stressed ones.

## 7. Limitations

The sample contains a single major crisis, which caps statistical power throughout (~137 monthly observations, and only ~17 genuinely out-of-sample months post-2025) and makes any regime-interaction relationship impossible to validate out of sample. The stationarity tests frequently hit the edge of their tabulated p-value ranges given short series. In conditional mode, storage is only semi-exogenous, so even the "clean" ceiling retains a little simultaneity; the weather block is the only fully exogenous element. One might worry that the out-of-sample underperformance of the seven-variable perfect-foresight model was merely a parsimony penalty — estimation noise on weakly-informative regressors — but the lean three-parameter robustness check (§4.5) rules this out: a tight model with perfect foresight of only storage and HDD still fails to beat a random walk (DM 0.53) and still collapses in calm markets (low-vol R² ≈ 0.02), so the failure is genuine signal weakness, not over-parameterization. Finally, the study is confined to H=1 and to linear OLS.

## 8. Next steps

The European forecasting question is answered thoroughly. Genuine incremental information for a globally traded commodity is most likely to lie in the **global** block deliberately excluded here — Asian and US price signals and the cross-market LNG-arbitrage relationship — which is the subject of the companion global and error-correction work. Within the European frame the question is now settled: the leaner perfect-foresight robustness check (§4.5) has confirmed that the ceiling's out-of-sample failure is signal weakness rather than over-parameterization, so no further European specification is likely to change the verdict.

---

*Artifacts: data pipeline `build_ng_dataset_monthly.py` → `NG_m_final.csv`. **There is now a single file to run — `European_monthly_gas_model.py`** — which reproduces the entire study: running it executes all four sub-models in sequence (the deployable model, the conditional all-fundamentals perfect-foresight model, the endogeneity-free weather-plus-storage ceiling, and the lean three-parameter robustness check).*