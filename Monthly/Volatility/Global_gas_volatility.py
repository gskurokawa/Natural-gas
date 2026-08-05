
#%%
"""
ttf_garch.py

Phase 2, model (5): ARCH/GARCH volatility study for European gas (TTF).

Every earlier study in this series found the same thing about the CONDITIONAL
MEAN of month-ahead TTF: it is close to a random walk, with what little
explanatory power there is concentrated in high-volatility / crisis months and
near-absent in calm ones. This study turns to the CONDITIONAL VARIANCE, which is
where that regime structure actually lives. It asks:

  1. Is there an ARCH effect to model at all (ARCH-LM test)?
  2. Which volatility specification fits best -- ARCH(1), GARCH(1,1), GJR-GARCH
     (asymmetry / leverage), EGARCH -- and under Normal vs Student-t errors
     (gas returns are fat-tailed)?
  3. How PERSISTENT is volatility (alpha+beta and the shock half-life)? Gas vol
     is famously near-integrated.
  4. Is the asymmetry the "inverse leverage" of commodities -- do POSITIVE price
     shocks (supply scares) raise volatility more than negative ones (unlike
     equities)? (sign of the GJR/EGARCH asymmetry term.)
  5. Do fundamentals that were useless for the MEAN (storage, HDD) help explain
     the VARIANCE? (post-hoc regression of conditional vol on fundamental stress.)
  6. Does the GARCH model actually FORECAST variance better than simple
     benchmarks out of sample -- EWMA (RiskMetrics) and a rolling window --
     judged by QLIKE and MSE loss against realized squared returns, with a
     Diebold-Mariano test? (the same OOS discipline used throughout.)

Returns are monthly log returns of TTF x100 (percent), as the arch optimizer
prefers percent-scale data. Monthly GARCH on ~136 returns is data-hungry, so
read magnitudes as indicative and lean on the qualitative structure.

Requirements: arch (pip install arch), statsmodels, numpy, pandas, scipy
"""
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
import statsmodels.api as sm

DATA_PATH = "NG_m_final.csv"
SAMPLE_START = "2015-01-01"
CRISIS_START, CRISIS_END = "2021-09-01", "2023-07-01"
OOS_MIN_TRAIN = 72        # 6 yrs before the first OOS variance forecast
LB_LAGS = 12
EWMA_LAMBDA = 0.94        # RiskMetrics; also report a rolling-window benchmark
ROLL_WIN = 12


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["month"] = df["Date"].dt.month
    df = df.loc[df["Date"] >= SAMPLE_START].reset_index(drop=True)
    return df


def expanding_seasonal_mean(series, month):
    out = pd.Series(index=series.index, dtype=float)
    tmp = pd.DataFrame({"val": series, "month": month})
    for _, grp in tmp.groupby("month"):
        out.loc[grp.index] = grp["val"].expanding().mean().shift(1)
    return out


def build_returns(df):
    """Monthly log return of TTF in percent, plus aligned fundamental stress."""
    r = 100.0 * np.log(df["TTF(USD/mmbtu)"]).diff()
    stor_chg = df["EU+UK_av_storage(bcm)"].diff()
    stor_anom = stor_chg - expanding_seasonal_mean(stor_chg, df["month"])
    hdd_anom = df["Europe_HDD"] - expanding_seasonal_mean(df["Europe_HDD"], df["month"])
    out = pd.DataFrame({
        "Date": df["Date"], "r": r,
        "abs_stor_anom": stor_anom.abs(),
        "abs_hdd_anom": hdd_anom.abs(),
    }).iloc[1:].reset_index(drop=True)      # drop first (NaN return)
    return out


# --------------------------------------------------------------------------- #
# Spec helpers
# --------------------------------------------------------------------------- #

SPECS = [
    ("ARCH(1)-N",      dict(vol="ARCH",   p=1, o=0, q=0, dist="normal")),
    ("GARCH(1,1)-N",   dict(vol="GARCH",  p=1, o=0, q=1, dist="normal")),
    ("GARCH(1,1)-t",   dict(vol="GARCH",  p=1, o=0, q=1, dist="t")),
    ("GJR(1,1)-t",     dict(vol="GARCH",  p=1, o=1, q=1, dist="t")),
    ("EGARCH(1,1)-t",  dict(vol="EGARCH", p=1, o=1, q=1, dist="t")),
]


def fit_spec(r, vol="GARCH", p=1, o=0, q=1, dist="normal", mean="Constant"):
    am = arch_model(r, mean=mean, vol=vol, p=p, o=o, q=q, dist=dist)
    return am.fit(disp="off")


def persistence(res, vol, p=1, o=0, q=1):
    pr = res.params
    b = sum(pr.get("beta[%d]" % i, 0.0) for i in range(1, q + 1))
    if vol == "EGARCH":
        return b                                        # beta is the persistence in EGARCH
    a = sum(pr.get("alpha[%d]" % i, 0.0) for i in range(1, p + 1))
    g = sum(pr.get("gamma[%d]" % i, 0.0) for i in range(1, o + 1))
    return a + b + g / 2.0                               # +gamma/2: shock is negative w.p. 0.5


def half_life(pers):
    return np.log(0.5) / np.log(pers) if 0 < pers < 1 else np.inf


# --------------------------------------------------------------------------- #
# OOS variance-forecast benchmarks + losses
# --------------------------------------------------------------------------- #

def ewma_pred_var(r, lam=EWMA_LAMBDA, init_win=24):
    """One-step-ahead EWMA variance: pred[t] uses info through t-1."""
    r = np.asarray(r, float)
    n = len(r)
    pred = np.full(n, np.nan)
    s2 = np.nanvar(r[:init_win])
    for t in range(1, n):
        s2 = lam * s2 + (1 - lam) * r[t - 1] ** 2
        pred[t] = s2
    return pred


def rolling_pred_var(r, win=ROLL_WIN):
    r = pd.Series(np.asarray(r, float))
    return r.rolling(win).var().shift(1).values     # pred[t] = var of r[t-win..t-1]


def qlike(realized, pred):
    m = (~np.isnan(pred)) & (pred > 0)
    rz, pz = realized[m], pred[m]
    return np.mean(np.log(pz) + rz / pz)            # realized = r^2 here


def mse(realized, pred):
    m = ~np.isnan(pred)
    return np.mean((pred[m] - realized[m]) ** 2)


def dm_tstat(loss_worse, loss_better):
    d = np.asarray(loss_worse, float) - np.asarray(loss_better, float)
    d = d[~np.isnan(d)]
    n = len(d)
    return d.mean() / (d.std(ddof=1) / np.sqrt(n)) if d.std(ddof=1) > 0 else np.nan


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    df = load_data()
    dat = build_returns(df)
    r = dat["r"]
    print("TTF GARCH volatility study.  Sample %s..%s  (%d monthly returns, percent).\n"
          % (dat["Date"].min().date(), dat["Date"].max().date(), len(r)))

    # ---- 1. ARCH-LM test: is there volatility clustering to model? ----
    print("=" * 70)
    print("1. ARCH-LM test on returns (H0: no ARCH effect)")
    print("=" * 70)
    lm, lmp, f, fp = het_arch(r - r.mean(), nlags=LB_LAGS)
    print("   LM stat=%.2f  p=%.4f   F=%.2f  p=%.4f  -> %s"
          % (lm, lmp, f, fp, "ARCH effect present (GARCH warranted)" if lmp < 0.05
             else "no significant ARCH effect"))

    # ---- 2. Model comparison ----
    print("\n" + "=" * 70)
    print("2. Volatility-model comparison (mean=Constant)")
    print("=" * 70)
    print("   %-15s %9s %9s %9s %11s %9s" % ("Spec", "logL", "AIC", "BIC", "persist", "half-life"))
    fits = {}
    for name, kw in SPECS:
        try:
            res = fit_spec(r, **kw)
            fits[name] = (res, kw)
            pers = persistence(res, kw["vol"], kw["p"], kw["o"], kw["q"])
            hl = half_life(pers)
            print("   %-15s %9.2f %9.2f %9.2f %11.3f %9s"
                  % (name, res.loglikelihood, res.aic, res.bic, pers,
                     ("%.1f mo" % hl) if np.isfinite(hl) else "inf"))
        except Exception as e:
            print("   %-15s  [fit failed: %s]" % (name, type(e).__name__))
    best_name = min(fits, key=lambda k: fits[k][0].bic)
    print("\n   Best by BIC: %s" % best_name)

    # ---- 3. Best model: parameters, asymmetry, residual diagnostics ----
    best, bkw = fits[best_name]
    print("\n" + "=" * 70)
    print("3. Best model (%s) -- parameters & diagnostics" % best_name)
    print("=" * 70)
    print(best.summary())
    # asymmetry sign (if present): gamma
    if "gamma[1]" in best.params.index:
        g = best.params["gamma[1]"]; gp = best.pvalues["gamma[1]"]
        if bkw["vol"] == "EGARCH":
            tell = ("positive shocks raise vol MORE (inverse leverage, commodity-like)"
                    if g > 0 else "negative shocks raise vol more (equity-like leverage)")
        else:  # GJR: gamma multiplies negative-shock term; gamma<0 => positive shocks raise vol more
            tell = ("negative shocks raise vol more (equity-like leverage)"
                    if g > 0 else "positive shocks raise vol MORE (inverse leverage, commodity-like)")
        print("\n   Asymmetry gamma=%+.4f (p=%.3f) -> %s" % (g, gp, tell))
    # standardized-residual adequacy
    z = pd.Series(np.asarray(best.std_resid, float)).dropna()
    lb = acorr_ljungbox(z, lags=[LB_LAGS], return_df=True)
    lb2 = acorr_ljungbox(z ** 2, lags=[LB_LAGS], return_df=True)
    arch_after = het_arch(z, nlags=LB_LAGS)
    print("\n   Standardized-residual diagnostics (want all INSIGNIFICANT):")
    print("     Ljung-Box(z, %d):   stat=%.2f  p=%.3f" % (LB_LAGS, lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0]))
    print("     Ljung-Box(z^2, %d): stat=%.2f  p=%.3f" % (LB_LAGS, lb2["lb_stat"].iloc[0], lb2["lb_pvalue"].iloc[0]))
    print("     ARCH-LM(z, %d):     p=%.3f  -> %s" % (LB_LAGS, arch_after[1],
          "clustering absorbed" if arch_after[1] >= 0.05 else "residual ARCH remains"))

    # ---- 4. Conditional volatility: level and regimes ----
    cv = np.asarray(best.conditional_volatility, float)      # monthly %, same length as r
    dat = dat.copy(); dat["cv"] = cv
    ann = cv * np.sqrt(12)
    peak_i = int(np.nanargmax(cv))
    print("\n" + "=" * 70)
    print("4. Conditional volatility (annualized = monthly x sqrt(12))")
    print("=" * 70)
    print("   mean annualized vol: %.1f%%   min: %.1f%%   peak: %.1f%% (%s)"
          % (np.nanmean(ann), np.nanmin(ann), np.nanmax(ann), dat["Date"].iloc[peak_i].date()))
    dd = dat.copy()
    dd["crisis"] = ((dd["Date"] >= CRISIS_START) & (dd["Date"] < CRISIS_END)).astype(int)
    vol6 = dd["r"].rolling(6).std().shift(1)
    dd["highvol"] = (vol6 > vol6.expanding().median().shift(1)).astype(float)
    for col, hi, lo in [("crisis", "crisis", "calm"), ("highvol", "high_vol", "low_vol")]:
        for val, lab in [(1, hi), (0, lo)]:
            s = dd.loc[dd[col] == val, "cv"]
            if len(s) >= 3:
                print("   %-9s n=%3d  mean cond vol (annualized): %.1f%%"
                      % (lab, len(s), np.sqrt(12) * s.mean()))

    # ---- 5. Do fundamentals explain the VARIANCE? ----
    print("\n" + "=" * 70)
    print("5. Do fundamentals (useless for the mean) explain the VARIANCE?")
    print("=" * 70)
    reg = dat.dropna(subset=["cv", "abs_stor_anom", "abs_hdd_anom"]).copy()
    reg["crisis"] = ((reg["Date"] >= CRISIS_START) & (reg["Date"] < CRISIS_END)).astype(float)
    X = sm.add_constant(reg[["abs_stor_anom", "abs_hdd_anom", "crisis"]])
    ols = sm.OLS(np.log(reg["cv"]), X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    print("   OLS: log(conditional vol) ~ |storage anom| + |HDD anom| + crisis dummy")
    for c in ["abs_stor_anom", "abs_hdd_anom", "crisis", "const"]:
        print("     %-14s coef=%+.4f  t=%+.2f  p=%.3f" % (c, ols.params[c], ols.tvalues[c], ols.pvalues[c]))
    print("   R2=%.3f  (n=%d)" % (ols.rsquared, int(ols.nobs)))

    # ---- 6. OOS variance forecast: GARCH vs EWMA vs rolling ----
    print("\n" + "=" * 70)
    print("6. Out-of-sample 1-step variance forecast (QLIKE & MSE; lower=better)")
    print("=" * 70)
    rv = r.values
    realized = rv ** 2                                   # realized variance proxy
    ewma = ewma_pred_var(rv)
    roll = rolling_pred_var(rv)
    garch = np.full(len(rv), np.nan)
    for t in range(OOS_MIN_TRAIN, len(rv)):
        try:
            res_t = fit_spec(pd.Series(rv[:t]), vol="GARCH", p=1, o=0, q=1, dist="t")
            fc = res_t.forecast(horizon=1, reindex=False)
            garch[t] = np.asarray(fc.variance.values, float)[-1, 0]
        except Exception:
            garch[t] = np.nan
    idx = slice(OOS_MIN_TRAIN, len(rv))
    R, G, E, Ro = realized[idx], garch[idx], ewma[idx], roll[idx]
    n_oos = np.sum(~np.isnan(G))
    print("   OOS window: %s .. %s  (n=%d)"
          % (dat["Date"].iloc[OOS_MIN_TRAIN].date(), dat["Date"].iloc[-1].date(), int(n_oos)))
    print("   %-14s %10s %10s" % ("Model", "QLIKE", "MSE"))
    for name, P in [("GARCH(1,1)-t", G), ("EWMA(0.94)", E), ("rolling-%d" % ROLL_WIN, Ro)]:
        print("   %-14s %10.4f %10.1f" % (name, qlike(R, P), mse(R, P)))
    # DM on QLIKE differentials vs GARCH (positive => GARCH better)
    def qlike_vec(realized, pred):
        out = np.full(len(pred), np.nan); m = (~np.isnan(pred)) & (pred > 0)
        out[m] = np.log(pred[m]) + realized[m] / pred[m]; return out
    lg, le, lr = qlike_vec(R, G), qlike_vec(R, E), qlike_vec(R, Ro)
    print("   DM t, EWMA vs GARCH  (positive => GARCH better): %.2f" % dm_tstat(le, lg))
    print("   DM t, rolling vs GARCH (positive => GARCH better): %.2f" % dm_tstat(lr, lg))

    print("\nNOTE: monthly GARCH on ~136 obs is data-hungry; magnitudes indicative.")
    print("Conditional-vol series is best.conditional_volatility if you want to plot it.")


if __name__ == "__main__":
    main()

# %%