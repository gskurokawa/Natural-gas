# Global Gas Modelling — Consolidated Companion to the Umbrella TTF Study

*Consolidates the three global studies in the order they feed the umbrella paper's Section 5: global fundamentals (Compact and conditional models), feeding §5.1–5.2; cointegration and error correction, feeding §5.3; and VECM transmission and the regime split, feeding §5.4. Prepared 1 August 2026.*

---

> **Global fundamentals** · feeds umbrella paper §5.1–5.2 · script `Global_monthly_gas_model.py` (sections §5.1–§5.2)

# Does Global Information Improve a Month-Ahead European Gas (TTF) Forecast?

### A global model, from a deployable forecast to a perfect-foresight ceiling

*Prepared 1 August 2026. Companion to the European TTF study.*

---

## Summary

The European study asked how much of the month-ahead change in TTF a *self-contained European* model can explain, and found the answer to be "very little, and nothing that beats a random walk." This companion study asks the natural next question: does widening the information set to the **global** gas system — competing hub prices, world LNG supply and trade, and non-European weather — add anything a European-only model misses?

We attacked it the same three ways, in ascending order of generosity. A strictly **deployable** global forecast (every predictor lagged) adds nothing out of sample: the enriched global model does not beat the parsimonious European core or a random walk, and collapses on recent data. Granting **perfect foresight of this month's global fundamentals** — quantities only, never hub prices — raises in-sample fit by about four points over the European ceiling (0.21 → 0.25), but almost all of that comes from a single channel (Asian demand pull), and out of sample the global block does not just fail to help: it **actively degrades** the forecast, because the many weak global regressors inject estimation noise the one real signal cannot offset. Stripping the model to that single significant, correctly-signed channel — NE-Asia degree days, added leanly on top of the European core — confirms the point: it is a genuine in-sample driver (t ≈ 2.5) yet **still does not improve the out-of-sample forecast**, and explains essentially nothing in calm markets.

The conclusion is decisive and parallels Europe exactly. The one causally-real global link — cold in Asia pulling LNG cargoes east, away from Europe — is visible in-sample but too weak and noisy to be exploitable a month ahead. **No global information, even under perfect foresight, improves a month-ahead TTF forecast over the European core.** Whatever additional structure the global system contains is not a level-forecasting edge at H=1; if it lives anywhere, it lives in the *transmission* of shocks across hubs and in *volatility* — the subjects of the error-correction and GARCH work — not in the conditional mean of next month's price.

---

## 1. Question and scope

The target is unchanged: the one-month log return of front-month TTF (USD/mmbtu), at a fixed **H = 1** horizon. Where the European study confined the predictor universe to Europe's own fundamentals, this study *adds back* the global block that was deliberately held out — and asks whether it earns its place. The benchmark throughout is the established **European core** (momentum + storage change + heating-degree-days), so the headline question is direct: *does global information beat the self-contained European model out of sample?*

As in the European work, the model runs in two modes — **predetermined** (every predictor lagged one month; an honest, deployable forecast) and **conditional** (same-month fundamentals entered contemporaneously; a perfect-foresight explanatory ceiling that separates "global fundamentals don't matter" from "they matter but are unforecastable"). Momentum stays lagged in every mode.

One rule is specific to the global setting and load-bearing: **perfect foresight is granted to quantities, never to prices.** We assume we could forecast this month's storage, weather, LNG outages and trade flows, but *not* this month's JKM or Henry Hub. Contemporaneous hub prices co-move almost perfectly with TTF, so "knowing" them is near-circular — it would measure global price co-movement, not forecastability. JKM and Henry Hub therefore do not appear in the conditional ceiling at all.

## 2. Data

The European monthly dataset (January 2015 – May 2026, ~137 observations) is extended with a global block: competing hub prices and LNG-arbitrage spreads (JKM, Henry Hub, and the TTF–JKM and TTF–HH log spreads — used only in the deployable run, and lagged); non-European weather and supply risk (US and NE-Asia degree days, Atlantic hurricane energy, Gulf-of-Mexico storm days); global LNG supply and trade (world nameplate capacity, capacity offline from outages, and Asia/India LNG imports and Qatar-Australia-US / SE-Asia / Nigeria exports); and financial controls (VIX, USD-EUR FX). The global LNG capacity-offline series is the outage ledger built earlier in this project — a genuinely exogenous record of fires, strikes and war damage.

## 3. Methodology

The pipeline is identical to the European study — no-look-ahead deseasonalized anomalies (raw minus an expanding, prior-years-only calendar-month climatology), an ADF+KPSS stationarity gate, a univariate Newey-West HAC screen, collinearity pruning at |corr| > 0.70, model selection on the pre-2025 training window only, and an expanding-window walk-forward evaluated against a random walk and the historical mean with a Diebold-Mariano test using a Newey-West variance, plus a crisis / high-volatility regime split.

Two disciplines specific to the global conditional mode:

**Foresight for quantities, not prices** (as above): the conditional models exclude hub prices entirely and grant contemporaneous entry only to storage, weather, LNG outages and trade.

**Exogenous vs endogenous, pre-specified.** With ~19 global candidates and only ~17 clean out-of-sample months, data-driven selection would over-fit, so the global fundamentals are split a priori. **GLOBAL_EXO** holds fundamentals price genuinely cannot cause — global LNG capacity offline, US and NE-Asia degree days, Atlantic hurricane energy, Gulf storm days — a clean causal block. **GLOBAL_TRADE** holds the price-responsive flows — LNG import and export changes, nameplate-capacity change — which are endogenous (high TTF pulls cargoes to Europe), the global analogue of Europe's wrong-signed coal-switching and import-pull terms, and are reported separately so their contribution is never mistaken for forecast power.

## 4. Results

### 4.1 The deployable global model

With every predictor lagged, the disciplined pipeline does select global variables into the compact model, but they do not earn their keep out of sample. The enriched global specification tracks the European core almost exactly and fails to beat it: the Diebold-Mariano statistic for the global model versus the European core is essentially zero (≈ −0.01), and on the clean post-2025 slice the global model's out-of-sample R² versus a random walk is deeply negative (≈ −59%). The finding matches the European lesson at one remove: a wider, honestly-lagged information set does not improve the month-ahead forecast — it mostly adds noise. The obvious question — is that because the global fundamentals don't matter, or because we cannot forecast them a month ahead? — is what the conditional ceiling is built to answer.

### 4.2 The perfect-foresight ceiling

Granting perfect foresight of the global *quantities* lifts in-sample fit only modestly above the European ceiling, and the increment is concentrated almost entirely in one variable:

| Model (training sample, n = 106) | R² |
|---|---:|
| Momentum only (lagged) | 0.084 |
| + European clean ceiling (storage + weather, foresight) | 0.214 |
| + GLOBAL_EXO  [= global clean ceiling] | 0.254 |
| + GLOBAL_TRADE (endogenous) | 0.298 |
| *(ref)* honest European core (all lagged) | 0.173 |

The clean global fundamentals add about four points (0.214 → 0.254). The endogenous trade block adds another four — but, exactly as anticipated, that is simultaneity (price pulling cargoes), not forecast power. Inside the clean ceiling, the HAC fit shows **only one global variable is individually significant, and it is correctly signed**: NE-Asia degree days (+0.031, t = 2.21, p = 0.027) — the Asian demand-pull channel, whereby cold in Asia draws LNG east and away from Europe, lifting TTF. Storage change remains the one durable European driver (t = −2.47). The other four clean global fundamentals are insignificant, and two of them — global LNG capacity offline and US degree days — even come out *wrong*-signed (though insignificant, so noise).

Out of sample, the global block does not merely fail to help — it **hurts**:

| Model | Full-window OOS R² vs RW | DM t vs RW | Post-2025 OOS R² vs RW |
|---|---:|---:|---:|
| European core (lagged) | +5.7% | 0.58 | −15.2% |
| European ceiling (foresight) | −1.3% | −0.14 | −24.1% |
| Global clean ceiling (foresight) | −12.2% | −1.03 | −23.9% |
| Global full, + trade (foresight) | −25.2% | −1.69 | −91.6% |
| Random walk | — | — | — |

The global clean ceiling loses to the European ceiling, the honest European core, *and* the random walk (DM −1.08 versus the European ceiling; −1.27 versus the core). Adding the endogenous trade block makes it far worse. This is textbook overfitting under the bias-variance trade-off: a dozen regressors, eleven of them near-noise, inflate estimation variance that the single useful signal cannot offset, so the enriched model generalizes worse than the parsimonious core. (The in-sample regime R² for the global ceiling looks high even in calm months — around 0.34 — but that is largely mechanical: twelve regressors over-fit small subsamples. The out-of-sample column, where over-fitting is penalized, is the honest read.)

### 4.3 The lean check — isolating the Asian demand-pull channel

The ceiling result invites the same objection the European study answered with a lean model: perhaps the global block failed only because it was over-parameterized. To settle it, we isolated the single channel that carried real signal — NE-Asia degree days — and added it, leanly, on top of the lean European perfect-foresight model. This is a **deliberately generous test**: NE-Asia DD was chosen precisely because it was the one global variable significant in the full-sample fit, so selecting it is mildly post-hoc. That makes the reading asymmetric — if even this cherry-picked lean global model cannot beat the benchmarks, the verdict is decisive.

In-sample, the lean global model is genuine and clean. Every term is individually significant and correctly signed:

| Predictor | coef | t | p | sign correct? |
|---|---:|---:|---:|:--:|
| Storage change (now) | −0.0124 | −2.45 | 0.014 | yes |
| HDD (now) | +0.0003 | +2.04 | 0.041 | yes |
| NE-Asia degree days (now) | +0.0346 | +2.47 | 0.013 | yes |
| Momentum (lagged) | +0.145 | +1.40 | 0.160 | yes |

with the R² build-up momentum 0.084 → +storage 0.142 → +HDD [European lean] 0.167 → +NE-Asia DD [global lean] 0.194. The Asian channel adds a real, significant ~2.7 in-sample points.

But out of sample it **still does not help** — it slightly degrades the forecast:

| Model | Full-window OOS R² vs RW | DM t vs RW | Post-2025 OOS R² vs RW |
|---|---:|---:|---:|
| European core (lagged) | +5.7% | 0.58 | −15.2% |
| European lean (foresight) | +4.0% | 0.53 | −16.3% |
| Global lean (+ NE-Asia DD, foresight) | +2.2% | 0.27 | −16.7% |
| Random walk | — | — | — |

Adding the Asian channel *lowers* full-window OOS R² from 4.0% to 2.2%, with a Diebold-Mariano statistic of the global lean versus the European lean of **−0.58** (the channel hurts) and versus the honest core of **−0.40**. On the clean post-2025 slice everything loses to the random walk, and the low-volatility regime R² collapses to **0.025** — essentially identical to the European lean's 0.023. Even the single best, individually-significant, generously-selected global channel does not improve a month-ahead TTF forecast, and explains almost nothing in calm markets.

## 5. Discussion

Read together, the three cuts tell one story, and it is the European story extended to the world.

The **deployable** global model established that a wider, honestly-lagged information set does not beat the European core — the same near-random-walk verdict, now robust to adding hub prices, world LNG trade and non-European weather. The **conditional ceiling** then ruled out the most sympathetic explanation: it is not that we merely cannot forecast the global fundamentals. Handed perfect foresight of them, the global block *degrades* the forecast, because outside a single channel the global signal is too weak to survive the estimation noise its own breadth introduces. And the **lean check** removed the last hiding place — over-parameterization — by isolating that single channel: NE-Asia demand pull is a real, correctly-signed, individually-significant in-sample driver, and it *still* buys nothing out of sample.

That the one surviving global signal is Asian demand pull is economically satisfying rather than surprising. In a globally-integrated LNG market, the marginal cargo is contested between Europe and Asia, so a cold Asian winter genuinely tightens Europe's balance. The result is not that this mechanism is fictional — it is measurably present in-sample — but that at a monthly horizon it is too noisy, and too concentrated in stressed periods, to sharpen a point forecast of next month's price. The European finding was that fundamentals are causally real but weak at H=1; the global finding is that the *additional* fundamentals a global view brings are weaker still, to the point of being counter-productive when estimated.

The constructive implication is about *where* to look for global structure. If the world gas system adds anything at H=1, it is not in the conditional mean of the TTF return — three increasingly generous attempts agree on that. It is more plausibly in the **transmission** of shocks across hubs (the TTF–JKM–HH cointegration and its error-correction dynamics) and in **volatility** clustering. Those are questions about how shocks propagate and how risk evolves, not about the level of next month's price, and they are the natural home for the remaining global work.

## 6. Conclusion

Global information does not improve a month-ahead TTF forecast. A deployable global model does not beat the European core or a random walk; a perfect-foresight global ceiling adds only a few in-sample points over the European ceiling and actively degrades the out-of-sample forecast through estimation noise; and even the single causally-real global channel — Asian demand pull, individually significant in-sample — fails to help out of sample and vanishes in calm markets. The European core (storage plus momentum) is not improved by any global variable at H=1. Whatever exploitable structure the global gas system holds is not in the conditional mean of next month's price; it belongs to the transmission and volatility questions taken up elsewhere.

## 7. Limitations

The sample carries the same constraints as the European study — one major crisis, ~137 monthly observations, only ~17 genuinely out-of-sample months post-2025 — which caps power and makes regime-interaction relationships impossible to validate out of sample. Several global series are short or partly estimated: hub-price history and the arbitrage spreads are most reliable from the late 2010s, and the global outage and trade series involve estimation. The conditional ceiling grants foresight of quantities but excludes prices by design, so it is silent on how much contemporaneous hub co-movement "explains" TTF — deliberately, since that co-movement is not forecastable. The lean global test selected its one global regressor post-hoc, so its (already negative) verdict is generous, not conservative. Finally, the study is confined to H=1 and to linear OLS; the transmission and volatility structure that a global view is most likely to contain is, by construction, outside a single-equation conditional-mean model.

## 8. Next steps

The conditional-mean question — European and now global — is answered: at H=1, fundamentals are causally real but not a level-forecasting edge, and global information does not improve on Europe alone. The remaining global work targets the two channels this study points to rather than rules out. The **error-correction / VECM** strand examines how a shock to JKM or Henry Hub transmits into TTF and how fast the three cointegrated hubs re-converge — a transmission question, not a mean-forecast one; the earlier ARDL/ECM run already confirmed the hubs are cointegrated but that error-correction is too slow to help at H=1, and a VECM makes the cross-hub propagation explicit. The **GARCH** strand models TTF *volatility*, the natural home for the crisis-versus-calm regime distinction that recurs throughout both studies. Together those would complete the picture of a market whose *level* is near-unforecastable a month ahead but whose *shock transmission* and *risk* have real, estimable structure.

---

*Artifacts: data pipeline `build_ng_dataset_monthly.py` → `NG_m_final.csv`; deployable global model `ttf_global_model.py`; global perfect-foresight ceiling `ttf_global_conditional.py`; global lean robustness check `ttf_global_lean.py`. Companion: `European_TTF_model_report.md`.*


---

> **Cointegration and error correction (ARDL/ECM)** · feeds umbrella paper §5.3 · script `Global_monthly_gas_model.py` (section §5.3)

# Does the Global Price-Arbitrage Relationship Help Forecast Month-Ahead TTF?

### An error-correction (ARDL/ECM) study, from a deployable forecast to a perfect-foresight ceiling

*Prepared 1 August 2026. Third companion to the European and global TTF studies.*

---

## Summary

The European study found that Europe's own fundamentals barely move month-ahead TTF; the global study found that adding world fundamentals does not help and can hurt. Both, however, held back the one relationship most likely to bind gas prices together: **arbitrage across hubs**. In a globally-integrated LNG market, European (TTF), Asian (JKM) and US (Henry Hub) prices should share a long-run equilibrium, and when TTF drifts above or below that equilibrium the gap should predict a correction. This study tests exactly that, with an error-correction model (ECM), and gives it the same treatment the naive-OLS models received — a deployable forecast, a perfect-foresight ceiling, a lean cut and a regime split.

The long-run relationship is real and strong. TTF, JKM and Henry Hub are solidly cointegrated (Engle-Granger p = 0.000, Pesaran-Shin-Smith bounds F = 11.3, well above the 1% critical value), the estimated equilibrium vector is economically sensible and reasonably stable across the 2022 dislocation. **But the disequilibrium gap is not a month-ahead forecasting signal — and forcing it into the forecast actively hurts.** The speed of adjustment is statistically insignificant (α ≈ −0.14, p = 0.28); its point estimate implies a half-life of roughly four to five months, so at a one-month horizon only about a seventh of any gap closes, and that little is lost in the noise. In-sample, adding the error-correction term on top of perfect foresight of the fundamentals lifts R² by **0.001** (0.214 → 0.215). Out of sample it makes every forecast **significantly worse**: the Diebold-Mariano statistic for the error-correction model versus the no-error-correction benchmark is **−2.63** (ceiling) and **−2.09** (lean) — significant, in the wrong direction. With an insignificant adjustment coefficient, the estimated term chases noise and injects variance, so a model that "knows" the arbitrage gap forecasts worse than one that ignores it.

The verdict completes the trilogy and is the sharpest of the three. A genuine, strong, long-run price equilibrium exists across the world's gas hubs — but it reasserts itself far too slowly to be a month-ahead edge, and at H=1 the error-correction channel is not merely useless but counter-productive. Cointegration here is a fact about **transmission and convergence over many months**, not about the conditional mean of next month's price. The single durable predictor remains what it has been throughout: European storage. If the cross-hub relationship is worth modelling, it is as a *transmission* structure (the VECM question), not as a level forecast.

---

## 1. Question and scope

The target is unchanged — the one-month log return of front-month TTF (USD/mmbtu), at a fixed **H = 1** horizon. The distinctive object here is the **error-correction term (ECT)**: the gap between TTF and its long-run equilibrium with JKM and Henry Hub. The headline test is direct: *does the European core plus the error-correction term beat the core alone, out of sample?* — and, in the conditional mode, *does the error-correction term add anything on top of perfect foresight of the fundamentals?*

As in the naive-OLS studies, the model runs in two modes. **Predetermined (deployable):** every regressor is lagged, including the ECT (last month's disequilibrium), so the forecast is honest. **Conditional (perfect foresight):** the fundamentals enter contemporaneously — but, by design, **only the fundamentals**. Momentum stays lagged, and the ECT stays lagged as the honest price-arbitrage signal. We do *not* grant foresight of prices (JKM/HH), because contemporaneous hub prices co-move almost perfectly with TTF and "knowing" them is near-circular. This makes the ECM ceiling exactly the naive-OLS fundamentals ceiling **plus the lagged ECT**, which isolates the marginal contribution of the error-correction channel cleanly.

## 2. Data and the long-run relationship

The monthly dataset (January 2015 – May 2026, ~137 observations) supplies the three price series in logs. JKM is real, varying history back to 2015; a single four-month gap (January–April 2017) that the data pipeline had constant-filled at 7.86 is replaced with a log-linear interpolation rather than discarding 2015–2018, so the level and return series contain no flat patch or artificial cliff. The long-run relationship is estimated by OLS of log TTF on log JKM and log HH (Engle-Granger step one); the residual is the error-correction term. The short-run fundamentals — momentum, storage change, and European heating-degree-days — are the same European core used throughout, entered lagged (deployable) or contemporaneously (ceiling).

## 3. Methodology

The pipeline matches the naive-OLS studies — no-look-ahead deseasonalized anomalies, an expanding walk-forward that refits each month, a Diebold-Mariano test with a Newey-West variance against a random walk, and a crisis / high-volatility regime split — with three additions specific to the ECM. The long-run cointegrating vector is **re-estimated at every walk-forward step** on data up to the forecast origin, and the ECT is lagged, so no future information enters through the equilibrium. Cointegration is tested two ways (Engle-Granger ADF on the residual, and the Pesaran-Shin-Smith bounds test on the level relationship) and checked for stability across the 2022 break. Every model in the out-of-sample comparison is evaluated on one common window (minimum training 60 months), so the figures line up directly with the European and global reports.

## 4. Results

### 4.1 The long-run relationship is real and strong

The three hub prices are cointegrated, and the equilibrium is economically sensible:

> log TTF = −0.409 + 1.053 · log JKM + 0.151 · log HH

TTF tracks JKM close to one-for-one (the contestable LNG cargo ties Europe to Asia) with a smaller Henry Hub loading (the US export-netback anchor). The Engle-Granger residual is stationary (ADF stat −5.36, p = 0.000), and the Pesaran-Shin-Smith bounds test gives F = 11.3, above even the 1% upper I(1) bound (≈ 5.5) — both reject "no level relationship." The vector is reasonably stable across the 2022 dislocation (pre-crisis loadings b_JKM = 0.85, b_HH = 0.26 versus full-sample 1.05 and 0.15; the pre-crisis residual is also stationary, p = 0.000). So the premise of the model holds: there genuinely is a global gas-price equilibrium, and TTF genuinely deviates from and returns to it.

### 4.2 The deployable ECM — adjustment too slow to matter

The question is whether *deviations* from that equilibrium forecast next month's move. In the in-sample error-correction regression, they barely do. The speed of adjustment is the wrong quantity to be small, and it is: **α = −0.144 with p = 0.281** — correctly signed (a positive gap pulls TTF back down) but statistically indistinguishable from zero. Taken at face value the point estimate implies a half-life of roughly four to five months; at a one-month horizon that means only about 14% of any disequilibrium closes, and even that is not reliably estimated. Adding the ECT to the lagged European core lifts in-sample R² only from 0.173 to 0.183.

Out of sample, on the common window, the error-correction term does not help — it hurts:

| Model | Full-window OOS R² vs RW | DM t vs RW | Post-2025 OOS R² vs RW |
|---|---:|---:|---:|
| European core (lagged) | +5.7% | 0.58 | −15.2% |
| Deployable ECM (core + lagged ECT) | +2.5% | 0.27 | −15.5% |
| Random walk | — | — | — |

Adding the ECT *lowers* out-of-sample R² from +5.7% to +2.5% (Diebold-Mariano of the ECM versus the core = −1.07). The disequilibrium term, with an insignificant coefficient, adds estimation variance without adding signal.

### 4.3 The perfect-foresight ceiling — the error-correction term adds essentially nothing, and degrades the forecast significantly

Granting perfect foresight of the fundamentals and asking what the ECT contributes *on top* removes any excuse that the fundamentals were simply mis-measured. The in-sample decomposition is decisive:

| Model (training sample, n = 106) | R² |
|---|---:|
| Momentum only (lagged) | 0.084 |
| EU fundamentals ceiling (storage + weather, foresight) | 0.214 |
| + lagged ECT  [= ECM ceiling] | 0.215 |
| *(ref)* honest deployable ECM (all lagged) | 0.183 |
| *(ref)* honest EU core (all lagged) | 0.173 |

The error-correction term adds **0.001** of in-sample R² once the fundamentals are known. In the HAC fit of the ceiling, storage change remains the one significant driver (t = −2.25, p = 0.024) and the ECT is correctly signed but negligible (coefficient −0.049, t = −0.36, p = 0.720); the weather terms are all correctly signed but individually weak (cooling-degree-days is closest, p = 0.105).

Out of sample, the ECT does not merely fail to help on top of foresight — it makes the forecast **significantly worse**:

| Model | Full-window OOS R² vs RW | DM t vs RW | Post-2025 OOS R² vs RW |
|---|---:|---:|---:|
| European core (lagged) | +5.7% | 0.58 | −15.2% |
| EU fundamentals ceiling (foresight) | −1.3% | −0.14 | −24.1% |
| ECM ceiling (fundamentals foresight + ECT) | −5.6% | −0.58 | −25.2% |
| Random walk | — | — | — |

The decisive number is the head-to-head: **Diebold-Mariano of the ECM ceiling versus the pure fundamentals ceiling is −2.63** — the error-correction term degrades the forecast by a *statistically significant* margin. This is the strongest "adding information hurts" result across all three studies: not a weakly negative statistic, but a significant one in the wrong direction.

### 4.4 The lean cut and regime split — the same verdict, tightened

Isolating the effect in a tight four-parameter model rules out over-parameterization as the cause. The lean ECM (momentum lagged, storage and HDD with foresight, plus the lagged ECT) has correctly-signed, significant fundamentals (storage t = −2.25; HDD t = +2.02) and, again, an insignificant ECT (t = −0.29, p = 0.772). Out of sample it repeats the pattern exactly:

| Model | Full-window OOS R² vs RW | DM t vs RW | Post-2025 OOS R² vs RW |
|---|---:|---:|---:|
| European core (lagged) | +5.7% | 0.58 | −15.2% |
| European lean (foresight) | +4.0% | 0.53 | −16.3% |
| ECM lean (+ lagged ECT) | +1.3% | 0.17 | −17.5% |
| Random walk | — | — | — |

with **Diebold-Mariano of the ECM lean versus the European lean = −2.09** — again the error-correction term significantly hurts, even in the leanest specification. The regime split shows the familiar shape: the ECM ceiling explains 0.453 in crisis but only 0.056 in low-volatility months (the lean model 0.159 and 0.030), so whatever explanatory power exists lives in dislocations and vanishes in calm markets. The error-correction channel adds nothing to that story at any point.

## 5. Discussion

The ECM was the most promising of the three approaches on paper, because it rests on a relationship the data clearly contain: the world's gas hubs *are* cointegrated, strongly and stably. The disappointment is specific and instructive — the relationship is real but **slow**. Cointegration guarantees that TTF and its JKM/HH equilibrium reconverge eventually; it says nothing about the speed, and here the speed is the problem. An adjustment half-life of four to five months means the one-month horizon captures only a sliver of the correction, and that sliver is statistically indistinguishable from zero. The equilibrium is a gravitational pull that acts over quarters, not a force that moves next month's price.

Worse, forcing the slow signal into a fast forecast is actively harmful. Because α is insignificant, the walk-forward estimates its coefficient largely from noise, and the resulting ECT contribution is a noisy addition to the prediction — so the error-correction models lose to their no-ECT counterparts by a significant margin (DM −2.63 and −2.09). This is the cleanest illustration in the whole project of a general lesson: a variable can be genuinely, significantly related to the target in the *long run* and still degrade a *short-horizon* forecast, because relevance at one frequency does not transfer to another and an irrelevant-at-H=1 regressor imports only variance.

Read alongside the companions, the three studies converge on one conclusion from three directions. European fundamentals are causally real but weak at H=1 (study one). Global fundamentals add nothing and can hurt, even under perfect foresight (study two). And the global *price* relationship — the strongest structure of the three — is real but too slow to exploit a month ahead, and hurts when used (study three). The through-line is that month-ahead TTF is close to a random walk, its one durable predictor is European storage, and every richer structure that clearly exists in the data lives at horizons or in dimensions other than the conditional mean of next month's return.

## 6. Conclusion

The global price-arbitrage relationship does not help forecast month-ahead TTF, and using it hurts. TTF, JKM and Henry Hub are strongly and stably cointegrated, but the speed of adjustment is insignificant (α ≈ −0.14, ~4–5 month half-life), the error-correction term adds ~0.001 of in-sample R² over a perfect-foresight fundamentals model, and out of sample it degrades the forecast significantly (DM −2.63 for the ceiling, −2.09 for the lean, and negative deployably as well). Storage remains the sole durable driver, and low-volatility explanatory power collapses as everywhere else. The cross-hub equilibrium is a transmission-and-convergence phenomenon that plays out over many months, not a conditional-mean signal at H=1.

## 7. Limitations

The study shares the sample constraints of its companions — one major crisis, ~137 monthly observations, ~17 clean out-of-sample months — which limit power and make the (insignificant) speed of adjustment imprecisely estimated; a longer or calmer sample could in principle sharpen α, though its point estimate is already far too slow to matter at H=1. The Engle-Granger p-values are approximate because the ECT is a generated regressor. The long-run vector shifts somewhat across the 2022 break (b_JKM from 0.85 to 1.05), so the equilibrium is "reasonably" rather than perfectly stable, and a single-equation ECM treats TTF as the sole adjusting variable — an assumption the VECM relaxes. The conditional ceiling grants foresight of fundamentals but not prices by design, so it does not measure contemporaneous price co-movement (which is not forecastable anyway). Finally, the study is confined to H=1 and to linear, single-equation OLS.

## 8. Next steps

The single-equation, conditional-mean question is now answered from all three angles, and the ECM result points directly to the remaining work. The finding that the hubs are strongly cointegrated but adjust too slowly for H=1 is precisely a **transmission** result, and the natural way to develop it is a **VECM**, which treats TTF, JKM and Henry Hub symmetrically, estimates how a shock to any one hub propagates into the others over subsequent months, and identifies which hub does the adjusting. That is a question about the multi-month impulse response, not about next month's level, so it is the right home for a relationship this study has shown to be real but slow. The **GARCH** strand remains the home for volatility and the crisis-versus-calm regime distinction that recurs across all three reports. Together they would round out the picture of a market whose month-ahead *level* is near-unforecastable but whose *shock transmission* and *risk* have genuine, estimable structure.

---

*Artifacts: data pipeline `build_ng_dataset_monthly.py` → `NG_m_final.csv`; deployable ECM `ttf_ardl_ecm.py`; perfect-foresight ceiling / lean / regime `ttf_ecm_ceiling.py`. Companions: `European_TTF_model_report.md`, `Global_TTF_model_report.md`.*

---

> **VECM transmission and the regime split** · feeds umbrella paper §5.4 · script `Global_monthly_gas_model.py` (section §5.4)

# Who Leads the Global Gas Market? A VECM Transmission Study of TTF, JKM and Henry Hub

### Cointegration, error-correction, and impulse-response evidence on how price shocks flow between Europe, Asia and the US

*Prepared 1 August 2026. Fourth companion to the TTF modelling series.*

---

## Summary

The three earlier studies were forecasting horse-races, and all reached the same negative verdict: month-ahead TTF is close to a random walk, and neither European fundamentals, nor global fundamentals, nor the global price-arbitrage gap improves on that. This study asks a different, structural question — not *can we forecast the level* but *who leads whom*. It fits a vector error-correction model (VECM) to the three log prices (Europe's TTF, Asia's JKM, the US Henry Hub), treating them symmetrically, and traces how a shock to any one hub propagates into the others over subsequent months.

The answer is clear and, unlike the forecasting results, positive. The three hubs are strongly cointegrated (Johansen trace and maximum-eigenvalue tests both give rank 2 — a single common stochastic trend). Within that system **TTF is the price leader.** It is weakly exogenous — its error-correction loadings are statistically zero, so it adjusts to nothing — while JKM error-corrects toward it quickly (half-life ≈ 1 month) and Henry Hub is loosely and slowly tied. Granger causality points only one way: TTF causes JKM (p = 0.006), and nothing causes TTF. The variance decomposition is emphatic: **95–100% of TTF's forecast-error variance is its own shocks** at every horizon, while a TTF shock explains a large and growing share of JKM's path and a moderate share of Henry Hub's. Europe shocks the system; Asia and the US absorb it.

This inverts the assumption behind the single-equation error-correction model, which treated TTF as adjusting *toward* the global prices. Over this sample the causality runs the other way — Europe → world. A regime split (§4.6) shows this is **not** a crisis-era artefact, as we had initially conjectured: TTF already led Asia before 2021 (Granger TTF→JKM p = 0.004 pre-crisis, JKM→TTF p = 0.69, with JKM adjusting and TTF weakly exogenous). What the crisis changed was different, and arguably more interesting — it pulled the **United States** into the global equilibrium. Pre-crisis, Henry Hub was weakly exogenous and detached, running its own domestic shale market (cointegration rank 1, a TTF–JKM relation); crisis-on, HH became a significant adjuster and a second cointegrating relation formed (rank 2), as surging US LNG exports tied Henry Hub to the world price for the first time. So the durable structure is *Europe leads Asia*; the crisis addition is *the US joins the club of followers*.

The finding also explains *why* the forecasting studies failed on TTF specifically: TTF is the system's source of innovations, near-self-driven, so there is little external information to forecast it *from*. The exploitable structure in the global gas market is not in predicting TTF — it is in how TTF's moves transmit to everyone else, and that transmission plays out over months, consistent with the earlier finding that the error-correction channel is too slow to help at a one-month horizon.

---

## 1. Question and scope

The three prior studies fixed the horizon at H = 1 and asked what predicts the month-ahead TTF *return*. This one drops the single-equation, single-target frame. A VECM treats log TTF, log JKM and log Henry Hub as a joint system in which each variable can both respond to and cause the others, subject to the long-run equilibria that arbitrage imposes. The questions are structural: How many long-run relationships bind the three hubs? What are they? **Which hubs adjust back to equilibrium (the price *takers*) and which are weakly exogenous (the *drivers*)?** How does a shock to each hub transmit into the others over the following months, and how much of each hub's variance do the others' shocks explain? This is the transmission-and-convergence question the single-equation error-correction model assumed an answer to but could not test.

## 2. Data

The three front-month prices in logs, monthly, January 2015 – May 2026 (137 observations). JKM is real varying history from 2015; the single four-month gap (January–April 2017) that the data pipeline had constant-filled at 7.86 is log-linearly interpolated, as in the error-correction study, so no artificial flat patch or cliff enters the levels.

## 3. Methodology

Integration order is checked with augmented Dickey-Fuller tests on levels and first differences. The lag order of the levels VAR is chosen by information criteria. Cointegration rank is determined by the Johansen trace and maximum-eigenvalue tests (restricted-constant specification, matching the "constant in the cointegrating relation" deterministic choice of the VECM). The VECM is then estimated at the selected rank, yielding the cointegrating vectors (the equilibria, β) and the adjustment loadings (α), whose significance identifies which equations error-correct and which variables are weakly exogenous. The short-run dynamics — Granger causality, orthogonalized impulse responses and the forecast-error-variance decomposition — are read from an unrestricted levels VAR at the same lag, a standard and robust applied choice (the levels VAR nests the cointegration; impulse responses on cointegrated I(1) data are consistent).

**On the identification ordering.** The impulse responses and variance decomposition use a Cholesky ordering, which requires ranking the hubs from most to least exogenous. An initial run ordered them by prior assumption (US most insulated, Europe the price-taker, hence TTF last). That run's own weak-exogeneity and Granger results overturned the assumption — TTF error-corrects to nothing and is the sole Granger-cause — so the reported analysis places **TTF first**, an identification motivated by the data rather than by priors. Crucially, the cointegration rank, the equilibria, the adjustment loadings and the Granger tests are all ordering-invariant; only the impulse responses and variance shares depend on it, and the leadership conclusion is corroborated by the ordering-invariant evidence.

## 4. Results

### 4.1 Integration order and cointegration rank

TTF and JKM are I(1) (unit root in levels, stationary in first differences). Henry Hub is borderline — its level is marginally stationary (ADF p = 0.048), reflecting how range-bound US gas was across this sample — which, as noted below, softens its role. (The first difference of JKM also fails the ADF at 5%, p = 0.15; this is a small-sample power artifact of the enormous 2021–22 spike and collapse, not literal I(2) — monthly log price changes are economically stationary.)

The Johansen tests are decisive on rank. The trace statistic rejects rank 0 (67.1 versus a 5% critical value of 29.8) and rank ≤ 1 (20.3 versus 15.5) but not rank ≤ 2 (3.1 versus 3.8); the maximum-eigenvalue test agrees. **Cointegration rank is 2** — two independent stationary combinations among three prices, i.e. a single common stochastic trend driving the whole system. The three hubs are tied together tightly.

### 4.2 The equilibria

The two cointegrating relations describe TTF and JKM as moving nearly together, with Henry Hub a smaller, looser partner. Expressed with TTF as the reference, TTF and JKM track each other close to one-for-one (roughly TTF ≈ 1.2 · JKM in logs), while the US hub carries a much smaller elasticity (of order 0.3 per unit of TTF), consistent with the US being only partially coupled to the global LNG price through export-netback arbitrage. This is consistent with the single-equation error-correction vector estimated earlier (log TTF ≈ 1.05 · log JKM + 0.15 · log HH): the strong TTF–JKM link and the minor Henry Hub loading reappear. (The absolute normalization of β is a mathematical convention and carries no causal meaning; the causal content is in the adjustment loadings below, not in which variable the equilibrium is solved for.)

### 4.3 Who adjusts — the leadership result

The adjustment loadings α say which equation moves to close a disequilibrium. This is the heart of the study:

| Equation | Loading on relation 1 | Loading on relation 2 | Verdict |
|---|---:|---:|---|
| Δ log **TTF** | −0.021 (p = 0.86) | −0.002 (p = 0.99) | **weakly exogenous — the DRIVER** |
| Δ log **JKM** | +0.367 (p = 0.001) | −0.455 (p = 0.000) | error-corrects — fast ADJUSTER (half-life ≈ 1.1 mo) |
| Δ log **Henry Hub** | +0.178 (p = 0.051) | −0.125 (p = 0.23) | weak / slow adjuster (normalization-sensitive) |

TTF's loadings are statistically zero on both relations: **it does not adjust to anything** — it is weakly exogenous, the driver of the system. JKM adjusts hard and fast, closing on the order of 45% of a disequilibrium gap per month (half-life about one month). Henry Hub is a weak and slow adjuster: its status is normalization-sensitive in this sample (a clearly significant loading of −0.21, p = 0.000, appears under a different basis, but only borderline significance, p = 0.051, under this one), which is what one expects of a hub only loosely tied to the global price. A formal likelihood-ratio weak-exogeneity test would settle Henry Hub's status precisely; it does not affect the TTF-leadership conclusion, which is unambiguous.

### 4.4 Granger causality

The short-run causality is one-directional and points to the same conclusion. Of the six possible directions, only one is significant at 5%: **TTF → JKM** (F = 5.24, p = 0.006). TTF → Henry Hub is borderline (p = 0.055), and *nothing* Granger-causes TTF (JKM → TTF p = 0.47; Henry Hub → TTF p = 0.27). TTF is the sole information leader; past European prices help predict Asian prices, but no hub's past helps predict Europe's.

### 4.5 Transmission — impulse responses and variance decomposition

With TTF correctly ordered first, the orthogonalized dynamics quantify the leadership. The cumulative response of TTF to a one-standard-deviation shock is almost entirely to its *own* shock (rising to ≈ +2.9 in log terms by 18 months); its response to a JKM shock is essentially nil (near zero throughout, even turning slightly negative), and to a Henry Hub shock modest (≈ +0.66 by 18 months). So TTF barely responds to the other hubs even dynamically — the leadership is not a contemporaneous artifact of the ordering. In the other direction, a TTF shock propagates strongly into JKM (cumulatively +0.72 by 3 months, +1.95 by 12) and moderately into Henry Hub (+0.29 by 3 months, +0.77 by 12).

The forecast-error-variance decomposition of TTF makes the point starkly:

| Horizon (months) | Own (TTF) | JKM | Henry Hub |
|---:|---:|---:|---:|
| 1 | 100.0% | 0.0% | 0.0% |
| 3 | 98.4% | 0.3% | 1.3% |
| 6 | 96.9% | 0.2% | 2.9% |
| 12 | 95.4% | 0.2% | 4.4% |
| 18 | 94.8% | 0.2% | 5.0% |

TTF's variance is almost wholly self-generated — 95% or more at every horizon, with JKM contributing essentially nothing and Henry Hub creeping up only to ~5% at a year and a half. Europe's price is, within this system, exogenous: it is shocked from outside the three-hub system (by its own fundamentals and events) and transmits those shocks outward, while absorbing almost nothing back.

### 4.6 Regime split — is TTF's leadership crisis-born?

Because the full sample is dominated by the 2021–22 crisis, we re-estimated the system on two sub-samples split at the crisis onset: **pre-crisis** (< 2021-09, n = 80) and **crisis-on** (≥ 2021-09, n = 57), with the lag fixed for comparability. The result overturns our initial conjecture that leadership passed from Asia to Europe at the crisis. It did not — TTF already led Asia before 2021:

| | Pre-crisis (< 2021-09) | Crisis-on (≥ 2021-09) |
|---|---|---|
| Cointegration rank (trace) | 1 | 2 |
| TTF weakly exogenous (driver)? | yes (α p = 0.76) | yes (α p = 0.75, 0.82) |
| JKM adjusts to the system? | yes (α p = 0.000) | yes (α p = 0.048, half-life ≈ 0.8 mo) |
| Henry Hub adjusts? | **no** (α p = 0.14 — decoupled) | **yes** (α p = 0.046) |
| Granger TTF → JKM | **p = 0.004** | p = 0.58 |
| Granger JKM → TTF | p = 0.69 | p = 0.85 |
| TTF own-FEVD share (h = 12) | 91% | 83% |

Two things are stable across the break and one changes. Stable: **TTF is weakly exogenous in both eras** and Asia (JKM) adjusts to it in both; pre-crisis TTF even Granger-causes JKM strongly (p = 0.004) while JKM does not cause TTF (p = 0.69). So Europe-over-Asia leadership is a structural feature of 2015–2026, not a crisis creation. What changes is the **United States**. Pre-crisis, Henry Hub is weakly exogenous and detached — it runs its own domestic shale-gas market, and the system has cointegration rank 1 (essentially a TTF–JKM equilibrium with HH loosely attached, itself a second driver). Crisis-on, Henry Hub becomes a significant *adjuster* and a second cointegrating relation appears (rank 2): the surge in US LNG exports during the crisis tied Henry Hub into the global price for the first time. So the crisis did not crown Europe — Europe already led — it **inducted the US** into the global gas equilibrium.

One further change is that short-run Granger causality weakens crisis-on (TTF → JKM falls from p = 0.004 to p = 0.58). This is consistent with arbitrage tightening into a more *contemporaneous* linkage — cargoes are rerouted within the month, so month-lagged predictive causality fades even as the long-run cointegration strengthens (rank rises) — compounded by the low power of only 57 observations. The sub-sample magnitudes generally, and the crisis-on Granger and FEVD in particular, should be read as indicative given the small samples and asymptotic Johansen critical values.

## 5. Discussion

The transmission structure is the mirror image of the assumption embedded in the single-equation error-correction model, which placed TTF on the left-hand side as the variable adjusting toward a global equilibrium. The VECM, by letting the data decide, finds the reverse: TTF is weakly exogenous and leads, JKM follows quickly, Henry Hub follows weakly. The natural first interpretation — that Europe *became* the price-setter after 2021, when it lost Russian pipeline gas and had to outbid Asia for LNG — turns out to be too neat, and the regime split (§4.6) corrects it. Europe already led Asia before the crisis: pre-2021, TTF is weakly exogenous, JKM adjusts to it, and TTF Granger-causes JKM (p = 0.004) while the reverse is far from significant. Whatever made TTF the anchor of the Europe–Asia pair predates the crisis — plausibly Europe's role as the deep, liquid, storage-backed market whose forward curve the flexible LNG cargo is priced against.

What the crisis genuinely changed was the **integration of the United States**. Before 2021 the US hub sat outside the global relationship — weakly exogenous, running its own shale-driven domestic market, with the system carrying only one cointegrating relation. The 2021–22 surge in US LNG exports, drawn out by extraordinary European and Asian prices, tied Henry Hub into the world price: crisis-on, HH becomes an adjuster and a second cointegrating relation appears. The economically meaningful regime shift, then, is not a change of leader but an *enlargement of the system* — the arbitrage web that already bound Europe and Asia extended to the US. That is a more durable and more interesting structural change than the leadership flip we first hypothesised, and it survives the caveat that leadership itself is stable.

The study also closes a loop with the three forecasting failures. Those studies found month-ahead TTF near-unforecastable; the VECM explains why *for TTF specifically*. If TTF is the system's driver — weakly exogenous, 95%-plus self-driven, caused by nothing else in the system — then there is little external, cross-market information from which to forecast it. Its innovations are the primitives of the system, not the responses. Leadership and forecastability are, in fact, opposites here: being the leader means being the source of surprises, which is precisely what makes a series hard to predict. The corollary is constructive. The genuine, estimable structure in the global gas market is the *transmission* — TTF's moves are informative about where JKM (and, more weakly, Henry Hub) will go over subsequent months. That is a statement about Asian and US prices conditional on European ones, and it plays out over a horizon of months, exactly as the slow error-correction of the ARDL/ECM study implied.

A few threads temper the strength of the result. Henry Hub's borderline stationarity means one of the two cointegrating relations is partly a statement about the US hub's own mean-reversion, so the cleanest, most robust piece of the story is the TTF ↔ JKM pair: Europe leads Asia. And because the whole sample is one long crisis-and-aftermath, the magnitudes — a one-month JKM adjustment half-life, a 95%-plus own-variance share for TTF — should be read as characteristic of this era rather than as structural constants.

## 6. Conclusion

Europe's TTF is the leader of the global gas-price system over 2015–2026. The three hubs are strongly cointegrated (rank 2 on the full sample), but only JKM and, weakly, Henry Hub adjust; TTF is weakly exogenous, is Granger-caused by neither other hub while Granger-causing JKM, and accounts for 95–100% of its own forecast-error variance while transmitting its shocks strongly into Asia and moderately into the US. This inverts the single-equation error-correction model's premise and explains why TTF is hard to forecast: it is the system's source of shocks, not its follower. The exploitable structure lies in transmission — European prices lead Asian ones over subsequent months — not in a level forecast of TTF itself. The regime split refines the story: Europe's leadership over Asia is *not* crisis-born but predates 2021; what the crisis added was the integration of the United States, as booming LNG exports pulled Henry Hub out of its detached domestic market and into the global equilibrium (cointegration rank rising from 1 to 2, with HH turning from driver to adjuster).

## 7. Limitations

The sample is ~137 monthly observations spanning a single major crisis; the regime split that probes this (§4.6) leaves only 57 crisis-on observations, so its sub-sample magnitudes — the crisis-on Granger tests especially — are indicative rather than precise, and its Johansen critical values are asymptotic. Reassuringly, the *core* leadership finding (TTF weakly exogenous, JKM adjusting, TTF Granger-causing JKM) holds in the better-powered pre-crisis window too, so it is not an artefact of the crisis or of small n. Henry Hub is borderline stationary on the full sample (ADF p = 0.048), which contributes to the full-sample rank-2 finding and makes its adjustment status normalization-sensitive; the TTF ↔ JKM leadership is the robust core, the Henry Hub link the softer part. The impulse responses and variance decomposition depend on the Cholesky ordering, which is justified by the ordering-invariant weak-exogeneity and Granger evidence but remains an identifying assumption; the dynamic responses (TTF barely reacting to the others at any horizon) support it. A formal likelihood-ratio weak-exogeneity test would pin down Henry Hub's status. The lag order was set to a minimally dynamic two despite information criteria preferring one, to avoid a differences-free VECM. The analysis is linear.

## 8. Next steps

The regime split (§4.6) is now done, and it reshaped the story rather than merely confirming it: Europe's leadership over Asia predates the crisis, and the crisis-era change was the integration of the US. Two extensions remain. First, that US-integration finding is itself worth a closer look — a rolling or recursive estimate of Henry Hub's adjustment loading would date *when* the US joined the global equilibrium and whether it has stayed in as LNG export capacity keeps growing. Second, the remaining item from the original model plan is the **GARCH / volatility** strand: every study in this series has found explanatory power concentrated in stressed, high-volatility periods and near-absent in calm ones, which makes the conditional *variance* of TTF the natural final object — where the crisis-versus-calm distinction that recurs throughout is not a nuisance but the phenomenon of interest.

---

*Artifacts: data pipeline `build_ng_dataset_monthly.py` → `NG_m_final.csv`; VECM transmission `ttf_vecm.py` (+ `vecm_irf_TTF.png`, `vecm_fevd.png`); regime split `ttf_vecm_split.py`. Companions: `European_TTF_model_report.md`, `Global_TTF_model_report.md`, `ECM_TTF_model_report.md`.*