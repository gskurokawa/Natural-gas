# Perfect Foresight and Week-Ahead Predictability of TTF Natural Gas Prices

Glen Kurokawa

*Prepared August 2026* 

## Abstract

This paper investigates how much of the Dutch Title Transfer Facility (TTF) natural gas price seven days ahead can be explained by supply–demand fundamentals known in advance. Granting perfect foresight of the independent variables yields an upper bound on predictability: if a model cannot forecast even when handed the future values of its inputs, then forecasting is difficult. Using daily data from January 2022 to July 2026 with an out-of-sample split at the end of February 2025, this study finds that TTF forecast models using European or global fundamentals do not beat random walks. The volatility is different. A heterogeneous-autoregressive model forecasts next week's variance better than a persistence benchmark. Further, adding the VIX as a regressor improves the forecast on the QLIKE loss. 

## 1. Introduction

Forecasting European gas prices has become much more important since the 2021–2022 energy crisis, but over the short term, they behave much like a random walk. This paper asks whether the fundamentals could forecast the price if they were known in advance. Supplying a model with the realised future values of storage, weather, supply outages, and other drivers turns the exercise into an investigation into the upper bound on fundamentals-based predictability.

For the price level, this study tests two information sets, a European fundamental set and a global one, across four model classes spanning linear distributed lag regressions, error correction, time-varying parameter state space models, and regime switching. The study then turns to volatility, asking whether volatility is forecastable seven days ahead. Predictive accuracy is assessed out of sample with the Diebold–Mariano test. 

The paper reviews the relevant literature (Section 2), sets out the data and common methodology (Section 3), and presents the level results under European (Section 4) and global (Section 5) fundamentals and the volatility results (Section 6), before a machine-learning robustness check (Section 7) and conclusion (Section 8).

## 2. Literature review

Baumeister, Huber, Lee and Ravazzolo (2024) show that statistically and economically meaningful results can be obtained using the monthly Henry Hub price. In a structural VAR of the German gas market, Nick and Thoenes (2014) identify storage levels, temperature-driven demand and supply disruptions as the principal drivers of weekly gas prices. A parallel machine-learning literature (e.g. Čeperić, Žiković and Čeperić, 2017) reports that nonlinear models coupled with feature-selection procedures can improve short-term accuracy using daily Henry Hub data, but the documented improvements are typically modest and vulnerable to overfitting. Linear and ARIMA-type specifications remain competitive benchmarks. 

The Dutch TTF has become the reference price for a globally integrated LNG market, so global conditions increasingly bear on it. Obadi and Korček (2020) obtain a high in-sample fit (R² ≈ 0.82) for month-ahead TTF using daily data, but the explanatory power is carried by German power and coal prices. However, the latter are co-determined with gas through power-sector switching. 

In contrast to the difficulty forecasting price levels, the volatility of natural gas prices is forecastable. Kristjanpoller (2024), using daily Henry Hub prices, confirms that natural gas realized volatility is predictable with heterogenous autoregressive (HAR) models and machine learning models while adding exogenous financial variables (EUR/USD, Brent, equity indices). Ergen and Rizvanoghlu (2016) and Liang, Xia, Lai and Wang (2022) using daily Henry Hub prices find that GARCH frameworks respond to storage levels as well as temperature deviations and extreme weather. Much of the forecastability is the autoregressive persistence of volatility itself, with fundamentals adding at the margin. 

## 3. Data and methodology

### 3.1 Data

The dataset comprises daily TTF front-month futures prices and a set of European and global fundamental variables spanning 3 January 2022 to 27 July 2026, constructed from free and public sources. The variables are: EU gas-in-storage (GIE AGSI+, TWh) and LNG sendout (GIE ALSI, GWh/day); heating and cooling degree days for major European cities (Open-Meteo); Norwegian unplanned-outage volumes (based on the author's research); EU and UK power-generation mix and day-ahead power prices (ENTSOE); and macro-financial and global-weather series including the VIX, the EUR/USD exchange rate, US and Northeast-Asia gas-weighted degree days, Atlantic accumulated cyclone energy, and a Gulf-of-Mexico storm indicator. The weather data was obtained from EWMWF. Observations are at business-day frequency, leaving roughly 1,140 daily observations. 

### 3.2 Study design and perfect foresight

Each exercise forecasts seven calendar days ahead. The study assumes perfect foresight: the fundamentals are assumed to be known at the time of the price forecast. It is therefore an upper bound on fundamentals-based predictability. The training set data is from early January 2022 to the end of February 2025, and the test set data is from 3 March 2025 to July 2026. The benchmark is the random walk (for the level) or its volatility analogue (for the second moment). Predictive accuracy is compared using the Diebold–Mariano (DM) test.

### 3.3 Stationarity

Augmented Dickey–Fuller (ADF) tests indicate that log TTF and the EUR/USD rate are I(1), while the remaining fundamentals are I(0). The level models are accordingly specified as an ARDL or ECM in levels,

$$\log \text{TTF}_t = c + \phi\,\log \text{TTF}_{t-7} + \beta' X_t + u_t,$$

where $X_t$ is the matrix of perfect-foresight fundamentals. The specification explains the stationary seven-day log return, $\log\text{TTF}_t - \log\text{TTF}_{t-7}$, with Newey–West HAC standard errors ($L=7$). 

### 3.4 Evaluation metrics

Price forecasts are compared using the root mean squared error (RMSE) of the price, a skill score $1-\text{MSE}(\text{model})/\text{MSE}(\text{random walk})$ computed on the log price (positive values beating the random walk), directional accuracy (the share of correctly signed seven-day moves), and the DM statistic. For volatility, the same skill score and DM statistic are reported on log realized variance, alongside the QLIKE loss, $V_a/V_h - \log(V_a/V_h) - 1$.

## 4. TTF price forecasting with perfect foresight of European fundamentals

### 4.1 Specifications

This part of the study assesses whether European supply–demand fundamentals with perfect seven-day foresight can improve on the random-walk forecast of the log TTF price. Two specifications of the ARDL mean equation of Section 3.3 are estimated on 757 observations (10 January 2022 – 28 February 2025) and evaluated on 337 (3 March 2025 – 27 July 2026). The first model is a parsimonious model with gas-in-storage and Norwegian unplanned outages (M1), and the second fuller model includes LNG sendout and heating degree days (M2) in addition. While the study's dataset includes JKM, Henry Hub and German day-ahead power prices, they are deliberately excluded from models, as they are co-determined with TTF.

**Table 1.** Out-of-sample level forecast accuracy, European fundamentals.

| Specification | RMSE | RW RMSE | MAE | Skill vs RW | Directional | DM |
|---|---:|---:|---:|---:|---:|---:|
| Random walk (benchmark) | 1.286 | 1.286 | 0.810 | +0.000 | — | +0.00 |
| M1: Storage + Norway | 1.275 | 1.286 | 0.814 | +0.025 | 0.52 | +0.71 |
| M2: all fundamentals | 1.276 | 1.286 | 0.812 | −0.004 | 0.56 | −0.13 |

*Notes.* RMSE and MAE are in USD/mmbtu on the price level. Skill and DM are defined in Section 3.4.

**Table 2.** Coefficient estimates, European fundamentals (training sample, Newey–West $L=7$).

| Regressor | M1 coef. | M1 $t$ | M2 coef. | M2 $t$ |
|---|---:|---:|---:|---:|
| Constant | +0.0959 | +1.45 | +0.1740 | +2.23 |
| Storage (TWh) | −0.000025 | −0.47 | −0.000044 | −0.85 |
| LNG sendout (GWh/d) | — | — | −0.000013 | −0.74 |
| Europe HDD | — | — | −0.000979 | −1.54 |
| Norway unplanned (mcm/d) | +0.000068 | +0.32 | +0.000040 | +0.20 |
| $\log \text{TTF}_{t-7}$ | +0.9717 | +52.86 | +0.9719 | +52.35 |

### 4.2 Findings

Neither specification in this part of the study beats the random walk to a statistically meaningful degree. M1 has a test-set RMSE of 1.275 USD/mmbtu against the benchmark's 1.286, which is a 2.5% reduction in mean-squared prediction error. The full model M2 is essentially indistinguishable from the benchmark (−0.4% skill). Directional accuracy is close to a coin toss (52% and 56%, respectively). Diebold–Mariano tests of predictive accuracy against random walks are insignificant (DM = 0.71 and −0.13, versus a one-sided 5% critical value of 1.65).

In both the Newey–West OLS and the ARMA estimation, the seven-day persistence term dominates, with an AR coefficient of 0.97–0.99 ($t \approx 53$) that is statistically indistinguishable from a unit root, while every exogenous fundamental is individually insignificant ($|t| < 1.6$). In short, even with perfect knowledge of future European fundamentals, TTF prices one week ahead are well described by their lagged value. The fundamentals add no predictive value.

## 5. Price forecasting with perfect foresight of global fundamentals

### 5.1 Specifications

This part of the study repeats the perfect-foresight exercise above but by augmenting the European fundamentals with several global fundamentals. The stationarity results of Section 3.3 (only log TTF and EUR/USD are I(1)) impact the error-correction specification. Two models are estimated, both targeting the seven-day price level. The first is an ARDL mean equation of Section 3.3 (M1). The second is an ARDL–ECM (Engle–Granger two-step) in which a long-term level regression defines an equilibrium error whose lag, with short-term driver changes, explains the seven-day change (M2). 

**Table 3.** Out-of-sample level forecast accuracy, global fundamentals.

| Model | RMSE | Skill vs RW | Directional | DM |
|---|---:|---:|---:|---:|
| Random walk (benchmark) | 1.286 | +0.000 | — | +0.00 |
| M1 ADL (global) | 1.365 | −0.148 | 0.50 | −1.18 |
| M2 ARDL–ECM | 1.339 | −0.105 | 0.51 | −0.70 |

*Notes.* RMSE is on the price level (USD/mmbtu); skill and DM as in Section 3.4. Negative DM values indicate the model is worse than the random walk.

### 5.2 Diagnostics

None of the global drivers is statistically significant in the ARDL mean equation. The only exogenous variable reaching the 5% level is the Gulf-storm flag ($-0.010$, $t=-2.20$), with US and NE-Asia degree days and the VIX marginal ($|t|\approx 1.3$–$1.6$) and LNG sendout, EUR/USD, and Atlantic ACE insignificant. The persistence term dominates ($\phi$ on the standardised anchor $=0.56$, $t=25.8$), though its weight is lower than in the European or bivariate models, where the fit was not competing with seven exogenous regressors. The ARDL–ECM error-correction speed is a negligible $-0.061$: with only log TTF (and EUR/USD) integrated of order one, there is no cointegrating relationship for the model to error-correct toward, so the specification is degenerate as anticipated. 

### 5.3 Findings

Both global fundamental models underperform the random walk out of sample. The Diebold-Mariano statistics are negative. The single-equation regressions (ADL, ARDL–ECM) lose 10–15% in mean-squared error to the random walk. This may be due to overfitting: granting the models perfect foresight of seven exogenous drivers gives them incentive to fit training-sample noise.

The exercise in this study is an upper bound as mentioned before. Because the fundamentals are supplied with perfect foresight, a genuine forecast that had to predict them first could only do worse. Even perfect foresight of fundamentals fails to beat a naïve no-change forecast. This is evidence for the random walk behaviour of TTF prices seven days ahead. Taken together with the European fundamentals model results, the finding is robust across information sets and models: neither regional nor global fundamentals lead to exploitable seven-day predictability of the TTF price level beyond simple persistence.

## 6. TTF price volatility foreacsting under perfect foresight of fundamentals

### 6.1 Specifications

The TTF price level was found essentially unforecastable above. This part of the study examines the second moment of TTF prices: the sum of squared daily log returns over the next five trading days (approximately seven calendar days), evaluated in log-variance space. Daily returns are scaled to percentage points for stability as the study uses maximum-likelihood estimation. The benchmark is a persistence forecast (RW-vol), which sets next week's variance equal to the current week's realized variance. The model is estimated from 770 observations through 28 February 2025 and evaluated on 346 observations (3 March 2025 – 20 July 2026). 

The perfect-foresight fundamentals are the VIX (the only variable with material marginal content over persistence, partial correlation +0.21), the Norwegian unplanned-outage shock (window maximum), and Europe heating degree days. Weather anomalies and storm indices were screened out as uninformative. Two GARCH models are estimated: a Gaussian GARCH(1,1) and a GARCH-X that adds the VIX to the conditional-variance equation. In addition, two realized-variance regressions are estimated: a HAR model of log next-week variance on log daily, weekly, and monthly realized variance, and a HAR-X that additionally includes the VIX.

**Table 4.** Out-of-sample volatility forecast accuracy.

| Model | RMSE | Skill vs RW-vol | QLIKE | Directional | DM |
|---|---:|---:|---:|---:|---:|
| RW-vol (persistence) | 1.138 | +0.000 | 1.040 | — | +0.00 |
| GARCH(1,1) | 1.183 | −0.080 | 0.545 | 0.56 | −0.70 |
| GARCH-X (VIX) | 1.387 | −0.484 | 0.661 | 0.55 | −2.84 |
| HAR (d+w+m) | 0.981 | +0.257 | 0.550 | 0.63 | +3.74 |
| **HAR-X (+VIX)** | 1.017 | +0.202 | **0.532** | 0.59 | +2.45 |
| HAR-X (+VIX+Nor+HDD) | 1.017 | +0.202 | 0.538 | 0.59 | +2.46 |

*Notes.* Skill is on log variance and QLIKE (lower is better) as in Section 3.4. Estimation diagnostics: the GARCH(1,1) is near-integrated ($\alpha+\beta=0.999$); in the GARCH-X the VIX enters the variance equation with a positive coefficient (+1.18) that is highly significant in-sample (likelihood ratio $2\Delta\ell = 13.34$ against a $\chi^2(1)$ 5% critical value of 3.84).

### 6.2 Findings

TTF price volatility is forecastable seven days ahead. The HAR model beats the persistence benchmark by a significant margin (skill +0.257, DM +3.74), and even the plain GARCH(1,1) roughly halves the RW-vol QLIKE. 

The VIX is a statistically significant driver inside the GARCH conditional variance ($2\Delta\ell = 13.34$), and augmenting the HAR with the forward-week VIX delivers the best out-of-sample QLIKE of any specification (0.532, versus 0.550 for the HAR and 0.545 for the GARCH(1,1)). The Diebold-Mariano test confirms HAR-X is significantly more accurate than persistence (DM +2.45). 

## 7. Machine-learning robustness check

### 7.1 Setup

This part of the study is simply a robustness check that the random-walk conclusion of TTF price level forecasting for European and global fundamentals is not an artefact of parametric or linear modelling. The check involves regularization and nonlinear tree. Since trees cannot extrapolate a trending price level, the target is the stationary log return, $r_{7,t}=\log\text{TTF}_t-\log\text{TTF}_{t-7}$. The random walk corresponds to $r_{7}=0$. The features are the perfect-foresight fundamentals dated at the target date together with the log-price anchor $\log\text{TTF}_{t-7}$ (which lets a learner exploit any mean reversion), standardised for the linear models. Hyperparameters are chosen on the training sample by a purged, embargoed time-series cross-validation (`TimeSeriesSplit` with seven calendar days ahead), minimising mean squared error. The study estimates a Ridge regression, a Lasso, and gradient-boosted trees (heavily constrained: depth two or three, subsampling, a minimum leaf size of twenty), alongside unregularized OLS and the random-walk benchmark. 

**Table 5.** Machine-learning robustness, seven-day level (global fundamentals + anchor).

| Model | RMSE | Skill vs RW | Directional | DM |
|---|---:|---:|---:|---:|
| Random walk (benchmark) | 1.275 | +0.000 | — | +0.00 |
| OLS | 1.359 | −0.159 | 0.51 | −1.28 |
| Ridge (CV $\alpha = 10^4$, grid maximum) | 1.279 | −0.009 | 0.48 | −0.82 |
| Lasso (0 of 8 features retained) | 1.279 | −0.008 | 0.48 | −0.80 |
| Gradient-boosted trees | 1.375 | −0.197 | 0.46 | −1.98 |

*Notes.* RMSE is on the price level (USD/mmbtu); skill, directional accuracy, and DM as defined in Section 3.4. Gradient boosting is the scikit-learn implementation (an XGBoost-equivalent). Sample: 787 training / 351 test observations; the modest difference from Sections 4–5 reflects feature-availability alignment across the combined predictor set.

### 7.2 Findings

No machine learning method beats the random walk for both the European and global data sets. Regularization discards the fundamentals almost entirely: Ridge cross-validation selects the largest penalty in the grid ($\alpha = 10^4$), shrinking every coefficient to near zero, and Lasso retains zero of the eight candidate features. Both therefore collapse onto the no-change forecast (skill −0.01). Gradient boosting predictably overfits, underperforming the random walk in the test set in every configuration (skill −0.20 on the global set).

## 8. Conclusions

This study considers limits of fundamentals-based short-term TTF price predictability by assuming perfect foresight of the fundamental variables. For the price level, the study finds for the European and global data sets, no specification beats a random walk out of sample. Because the fundamentals are supplied with perfect foresight, this is an upper bound: any other forecast would likely fare worse. Seven-day-ahead TTF price level forecasting is near-random-walk.

The machine-learning robustness check of Section 7 reinforces the TTF price level forecasting conclusion above. 

The volatility studies come to a different result. Next week's realized variance is forecastable beyond persistence. A HAR model beats the persistence benchmark. A global-uncertainty variable, the VIX, improves the forecast on the QLIKE loss, provided it is entered gently rather than compounded through a multi-step GARCH recursion. 

This study's findings are consistent with weak-form efficiency of the level combined with the well-documented predictability of volatility. This study confirms the literature. At weekly horizons, effort is better directed at forecasting risk than at forecasting price. 

## References

Baumeister, C., Huber, F., Lee, T. K., & Ravazzolo, F. (2024). Forecasting Natural Gas Prices in Real Time. *NBER Working Paper No. 33156*, National Bureau of Economic Research (forthcoming, *Journal of Applied Econometrics*).

Nick, S., & Thoenes, S. (2014). What drives natural gas prices? — A structural VAR approach. *Energy Economics*, 45, 517–527.

Čeperić, E., Žiković, S., & Čeperić, V. (2017). Short-term forecasting of natural gas prices using machine learning and feature selection algorithms. *Energy*, 140, 893–900.

Obadi, S. M., & Korček, M. (2020). Driving fundamentals of natural gas price in Europe. *International Journal of Energy Economics and Policy*, 10(6), 318–324.


Kristjanpoller, W. (2024). A hybrid econometrics and machine learning based modeling of realized volatility of natural gas. *Financial Innovation*, 10(1).

Ergen, I., & Rizvanoghlu, I. (2016). Asymmetric impacts of fundamentals on the natural gas futures volatility: An augmented GARCH approach. *Energy Economics*, 56, 64–74.

Liang, C., Xia, Z., Lai, X., & Wang, L. (2022). Natural gas volatility prediction: Fresh evidence from extreme weather and extended GARCH-MIDAS-ES model. *Energy Economics*, 116.
