# The Volatility of European Gas (TTF): A GARCH Study

### Where the signal was hiding — modelling the conditional variance after four studies found the conditional mean near-unforecastable

*Prepared 1 August 2026. Fifth and final companion to the TTF modelling series. Toy / educational model — not a trading or investment tool.*

---

## Summary

Four earlier studies reached the same verdict about the *level* of month-ahead TTF: it is close to a random walk. European fundamentals barely move it, global fundamentals add nothing, the cross-hub error-correction gap is too slow to exploit, and even perfect foresight of clean fundamentals does not beat a random walk out of sample. But every one of those studies also noticed that whatever explanatory power did exist was concentrated in high-volatility, crisis months and vanished in calm ones. This study follows that clue to its natural home — the conditional **variance** — and finds a sharply different picture: TTF volatility is strongly structured, highly persistent, regime-concentrated, and, unlike the mean, genuinely (if modestly) forecastable.

Volatility clustering is unambiguous (ARCH-LM p = 0.0015). A parsimonious **GARCH(1,1) with Normal errors** is the best specification on BIC — the fatter-tailed Student-t and the asymmetric GJR/EGARCH variants do not earn their extra parameters — and it is well specified, absorbing the clustering completely (Ljung-Box on squared standardized residuals p = 0.99). Two features stand out. First, volatility is **integrated**: the ARCH and GARCH coefficients sum to 1.00 (α = 0.28, β = 0.72), so shocks to gas volatility are essentially permanent, and the reaction coefficient α ≈ 0.28 is very high — gas volatility both jumps hard and stays elevated. Second, there is **no significant asymmetry**: gas lacks the leverage sign equities have. The conditional volatility is enormous by cross-asset standards — averaging about 55% annualized, peaking near 129% in early 2023 — and roughly twice as high in crisis as in calm (97% vs 47%).

Two results tie the whole series together. Fundamentals that were nearly useless for the mean explain the variance far better (regression R² = 0.45 versus roughly 0.05 for the mean in calm markets), though that is driven mostly by the crisis regime, with the storage-surprise channel only marginally significant. And out of sample, GARCH beats a naive rolling-window variance forecast and edges out EWMA/RiskMetrics on the appropriate (QLIKE) loss — so the variance is forecastable in a way the mean never was. The series therefore closes on a clean two-part thesis: **month-ahead TTF's level is near-unforecastable, but its volatility is structured, persistent, regime-concentrated, and modestly forecastable — that is where the signal was all along.**

---

## 1. Question and scope

The four prior studies modelled the conditional mean of the monthly TTF return. This one models its conditional **variance**. The questions are: is there a volatility (ARCH) effect to model at all; which GARCH-family specification fits best and is it adequate; how persistent is volatility and is it asymmetric in the commodity ("inverse-leverage") direction; do fundamentals that failed to predict the mean help explain the variance; and — applying the same out-of-sample discipline used throughout — does a GARCH model actually forecast variance better than simple benchmarks? The target is the monthly log return of TTF (in percent), 2015–2026, and everything is deliberately kept at the same monthly frequency and sample as the rest of the series so the mean and variance results are directly comparable.

## 2. Data

Monthly log returns of front-month TTF, February 2015 to May 2026 (136 returns), expressed in percent (the GARCH optimizer is better conditioned on percent-scale data). For the fundamentals-and-variance regression, two deseasonalized "stress" measures are aligned to the returns: the absolute storage-change anomaly and the absolute heating-degree-day anomaly, each built with the same expanding, prior-years-only climatology used across the series.

## 3. Methodology

The volatility model is estimated by maximum likelihood with the `arch` package. We first test for an ARCH effect (Engle's LM test) to confirm there is clustering to model. We then compare five specifications — ARCH(1), GARCH(1,1), GJR-GARCH(1,1) (which adds a negative-shock asymmetry term), and EGARCH(1,1), under Normal and Student-t errors — on log-likelihood, AIC and BIC, and read off the volatility persistence (α + β, with half of the asymmetry term for GJR) and the implied shock half-life. The best model is checked for adequacy with Ljung-Box tests on the standardized residuals and their squares and a post-fit ARCH-LM test (all should be insignificant if the dynamics are captured). We then summarise the fitted conditional-volatility series overall and by regime (the 2021–23 calendar crisis, and a look-ahead-safe trailing high/low-volatility split), regress the log conditional volatility on fundamental stress plus a crisis dummy, and finally run an expanding-window walk-forward one-step-ahead variance forecast, scoring GARCH against EWMA (RiskMetrics, λ = 0.94) and a rolling-window variance by QLIKE and MSE loss against realized squared returns, with Diebold-Mariano tests.

## 4. Results

### 4.1 There is a strong ARCH effect

Engle's LM test rejects the null of no ARCH decisively: LM = 31.7 (p = 0.0015), F-form p = 0.0006. Volatility clustering is real — periods of large moves follow large moves. This alone is the first substantive contrast with the mean: where the return level showed almost no exploitable autocorrelation, the return *variance* is visibly structured.

### 4.2 Model selection — parsimony wins

| Specification | log-L | AIC | BIC | persistence | half-life |
|---|---:|---:|---:|---:|---:|
| ARCH(1) — Normal | −574.6 | 1155.1 | 1163.8 | 0.13 | 0.3 mo |
| **GARCH(1,1) — Normal** | **−556.4** | **1120.7** | **1132.4** | **1.00** | **∞** |
| GARCH(1,1) — Student-t | −555.4 | 1120.7 | 1135.3 | 1.00 | ∞ |
| GJR(1,1) — Student-t | −555.3 | 1122.6 | 1140.1 | 1.00 | ∞ |
| EGARCH(1,1) — Student-t | −555.4 | 1122.8 | 1140.3 | 0.95 | 14 mo |

ARCH(1) alone is far worse — it cannot represent persistence. Adding the GARCH term is a large improvement (log-likelihood jumps ~18 points). But beyond plain GARCH(1,1), nothing helps: the Student-t distribution barely raises the likelihood and loses on BIC to the extra parameter; the GJR asymmetry and the EGARCH form add nothing. On BIC the winner is the simplest adequate model, **GARCH(1,1) with Normal errors**.

### 4.3 The GARCH(1,1) — integrated, sharp-reacting, symmetric, adequate

The fitted variance equation is σ²ₜ = 10.58 + 0.277 ε²ₜ₋₁ + 0.723 σ²ₜ₋₁, with the mean return statistically zero (μ = 0.49, p = 0.68 — no drift, consistent with the random-walk-in-mean finding). Both ARCH and GARCH terms are significant (α = 0.277, p = 0.019; β = 0.723, p ≈ 0). Two features matter:

**Volatility is integrated (IGARCH).** α + β = 1.00 exactly, so a shock to volatility has an effectively permanent effect (infinite half-life). And α ≈ 0.28 is high for monthly data — gas volatility reacts *sharply* to a shock and then *keeps* the elevated level. Gas volatility is both jumpy and sticky.

**There is no leverage asymmetry.** Because GJR and EGARCH do not improve on plain GARCH, we cannot distinguish a differential effect of positive versus negative shocks. Gas lacks the pronounced negative-shock leverage of equities — plausible for a commodity where both supply scares (up) and demand collapses (down) can spike volatility.

The model is adequate. The standardized residuals show no remaining autocorrelation (Ljung-Box p = 0.15), and — the key check — no remaining volatility clustering: Ljung-Box on squared standardized residuals gives p = 0.99 and a post-fit ARCH-LM test p = 0.96. A two-parameter volatility model absorbs essentially all the structure, in sharp contrast to the many-predictor mean models that could not beat "no change."

### 4.4 The conditional volatility — huge and regime-concentrated

The fitted conditional volatility averages about **55% annualized**, ranges from a calm-market floor near 26% to a peak of **129% in February 2023** (the violent unwind of the crisis, as TTF collapsed from its highs), and splits cleanly by regime:

| Regime | mean conditional volatility (annualized) |
|---|---:|
| Crisis (2021-09 to 2023-07) | 97% |
| Calm | 47% |
| High-volatility months | 65% |
| Low-volatility months | 37% |

Crisis volatility is roughly double calm volatility. This is the quantitative form of the regime distinction that every earlier study kept running into as a nuisance in the mean: here it is the object of study, and it is large and clean.

### 4.5 Do fundamentals explain the variance?

Regressing the log conditional volatility on fundamental stress and a crisis dummy gives R² = 0.45 — an order of magnitude more than fundamentals explained of the *mean* in calm markets (~0.05). But the composition matters. The crisis dummy does almost all the work (coefficient +0.73, t = 8.7), and that is partly mechanical: "crisis" is by construction the period when volatility was high, so this is closer to a description than an explanation. The genuinely informative, non-tautological result is that the **storage-surprise magnitude is marginally associated with higher volatility** (coefficient +0.025, t = 1.81, p = 0.07, correct sign), while the heating-degree-day surprise is not (p = 0.65). So the storage channel that carried what little signal there was for the mean reappears, weakly, for the variance — larger storage surprises coincide with more volatile prices — while weather does not.

### 4.6 Out-of-sample variance forecasting

The decisive contrast with the mean is here. In an expanding walk-forward (64 one-step-ahead forecasts, February 2021 onward), GARCH is scored against EWMA and a rolling window on realized squared returns:

| Model | QLIKE | MSE |
|---|---:|---:|
| GARCH(1,1)-t | 6.931 | 407,910 |
| EWMA (λ = 0.94) | 6.981 | 394,513 |
| Rolling 12-month | 7.165 | 402,392 |

On **QLIKE** — the appropriate, statistically robust loss for variance forecasts — GARCH has the lowest loss: it beats the rolling window (Diebold-Mariano t = 1.62, significant at roughly the 10% level one-sided) and edges out EWMA (DM t = 0.54, not significant). On MSE the ranking flips slightly in EWMA's favour, but MSE is dominated by a few enormous crisis observations and is the less reliable variance loss. The honest read is that GARCH is competitive with EWMA (a famously tough benchmark) and better than a naive rolling window. That is a modest but real out-of-sample win — and it is more than *any* mean model achieved, none of which beat a random walk.

## 5. Discussion

Placed against the four mean studies, the volatility results complete a coherent picture of what is and is not knowable about month-ahead European gas. The **level** is close to a martingale: its best predictor is its current value, storage and momentum aside, and no enrichment — global prices, LNG trade, weather, perfect foresight, error-correction — reliably improves on that out of sample. The **variance** is the opposite: it clusters strongly, is captured almost completely by a two-parameter GARCH(1,1), is highly persistent, is twice as large in crisis as in calm, and can be forecast better than naive benchmarks. The signal that the mean studies kept glimpsing in high-volatility episodes was never really about the *direction* of the next move; it was about its *magnitude*. Predictability in this market lives in the second moment, not the first.

The integrated-volatility finding deserves a caution. A persistence of exactly 1.00 is a common GARCH result for energy, but it is also exactly what one expects when a single large regime shift is fitted as if it were stationary volatility: the Lamoureux–Lastrapes result shows that a neglected structural break or level shift in variance biases estimated persistence toward unity. Our sample contains one dominant volatility regime (the 2021–23 crisis and its unwind), so the "shocks are permanent" reading should be held loosely — some of that apparent permanence is the crisis being one big, slow level-shift in volatility rather than a sequence of ordinary shocks that never decay. The absence of measured asymmetry should likewise be read as "not detectable in 136 monthly observations" rather than "definitely absent."

The fundamentals-and-variance result is the constructive counterpart to the mean studies' negative ones. It is intuitive that storage stress should raise volatility: when inventories are far from normal, the market is closer to a binding constraint, and small news moves price more. That the effect is only marginally significant, and that the crisis dummy dominates, again says the structure is concentrated in stressed states. It also points to the natural way to make the volatility model genuinely useful rather than merely descriptive — letting fundamentals enter the variance equation directly (a GARCH-X), so that a forecastable storage or supply-stress state feeds a volatility forecast, rather than reading the crisis after the fact.

## 6. Conclusion

TTF volatility is strongly structured where TTF's level is not. A parsimonious, well-specified GARCH(1,1) captures pervasive volatility clustering; volatility is near-integrated (α + β = 1.00) and sharp-reacting (α ≈ 0.28), symmetric within the resolution of the data, enormous in level (≈55% annualized, ≈97% in crisis), and — uniquely in this series — modestly forecastable out of sample, beating a rolling window and matching EWMA on QLIKE. Fundamentals explain far more of the variance than of the mean, though mostly through the crisis regime, with storage stress a marginal genuine driver. The five studies together give a clean, defensible bottom line: **the month-ahead level of European gas is close to unforecastable, but its volatility is persistent, regime-driven, and predictable enough to model — the risk, not the direction, is the knowable quantity.**

## 7. Limitations

Monthly GARCH on 136 returns is data-hungry — volatility models are usually estimated on hundreds or thousands of higher-frequency observations — so all magnitudes are indicative and the finer distinctions (Student-t vs Normal, presence of asymmetry) are underpowered. The integrated-persistence estimate is likely inflated by the single dominant crisis acting as a structural break (Lamoureux–Lastrapes), so α + β = 1.00 should not be read literally as permanent shocks. The fundamentals-and-variance R² is inflated by the near-tautological crisis dummy; the storage effect, the non-circular part, is only marginal. The realized-variance proxy is the squared monthly return, a noisy target at this frequency (intramonth or daily realized variance would be far cleaner but is outside this dataset). Finally, the model is univariate and the fundamentals enter only post hoc; a GARCH-X or multivariate volatility model is the natural extension.

## 8. Closing — the series as a whole

This completes the programme laid out at the outset: a naive European mean model, a naive global mean model, an ARDL/ECM, a VECM (with a regime split), and now a GARCH volatility model, each estimated with the same no-look-ahead, honest-out-of-sample discipline and written up in a companion report. The through-line is consistent and, taken together, more informative than any single result. Month-ahead TTF is a near-random walk in the mean whose one durable fundamental is storage; the global block adds nothing to the mean even under perfect foresight; the world's gas hubs are cointegrated with Europe as the price leader and the US integrating after 2021; and the genuinely modelable, persistent, forecastable structure lives in the volatility. If the programme continues, the two natural extensions are a **GARCH-X** that lets forecastable storage/supply stress drive the volatility forecast directly, and a **multivariate volatility** model of the TTF–JKM–Henry Hub system to see whether the price-leadership found in the levels also holds for the transmission of *risk*.

---

*Artifacts: data pipeline `build_ng_dataset_monthly.py` → `NG_m_final.csv`; volatility model `ttf_garch.py`. Companions: `European_TTF_model_report.md`, `Global_TTF_model_report.md`, `ECM_TTF_model_report.md`, `VECM_TTF_transmission_report.md`; related `Literature_review_global_gas_markets.md`.*