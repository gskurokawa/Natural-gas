#%%

#%%
"""
ttf_european_exogenous.py

Endogeneity-free explanatory ceiling for month-ahead (H=1) TTF.

The perfect-foresight run (ttf_european_conditional.py) hit in-sample R2 ~0.42,
but its top contributors -- contemporaneous coal generation and pipeline imports
-- had the WRONG sign, i.e. they reflected price DRIVING the fundamental
(coal-to-gas switching, import pull) rather than the reverse. That inflates the
ceiling with simultaneity.

This script removes that problem two ways:
  1. PRE-SPECIFIED model (no data-driven selection), so nothing can chase a
     wrong-signed, price-responsive variable.
  2. Restricted to fundamentals price genuinely CANNOT cause:
       - Weather: HDD, CDD, wind, solar, Nordic precipitation  (truly exogenous)
       - Storage change: included but SEMI-endogenous (injection/withdrawal
         responds to price), so reported separately from the pure-weather ceiling.
All fundamentals enter CONTEMPORANEOUSLY (perfect month-ahead foresight, weather
at the monthly-average level). momentum stays LAGGED (predetermined control).

Reported:
  - an R2 decomposition (momentum -> +weather -> +storage), so you can see the
    clean weather-only ceiling and what storage adds;
  - the HAC coefficient signs (are the exogenous drivers correctly signed?);
  - walk-forward OOS vs the honest predetermined core and a random walk, full
    sample and the clean post-2025 slice;
  - a regime split (crisis / high-vol vs calm).

This is still a CEILING (perfect foresight), not a deployable forecast, but it is
the honest, causal upper bound: weather is exogenous, so its explanatory power is
not contaminated by price feeding back into the regressor.

European variables only. H=1. Toy / educational model.

Requirements: pandas, numpy, statsmodels, scipy
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = "NG_m_final.csv"
TRAIN_END = "2025-01-01"
MIN_TRAIN_OBS = 60
HORIZON = 1
CRISIS_START, CRISIS_END = "2021-09-01", "2023-07-01"


# --------------------------------------------------------------------------- #
# Data + anomaly builders
# --------------------------------------------------------------------------- #

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["month"] = df["Date"].dt.month
    df["TTF"] = df["TTF(USD/mmbtu)"]
    return df


def expanding_seasonal_mean(series, month):
    out = pd.Series(index=series.index, dtype=float)
    tmp = pd.DataFrame({"val": series, "month": month})
    for _, grp in tmp.groupby("month"):
        out.loc[grp.index] = grp["val"].expanding().mean().shift(1)
    return out


def level_anom(df, col):
    return df[col] - expanding_seasonal_mean(df[col], df["month"])


def change_anom(df, col):
    ch = df[col].diff()
    return ch - expanding_seasonal_mean(ch, df["month"])


def dlog_anom(df, col):
    dl = np.log(df[col]).diff()
    return dl - expanding_seasonal_mean(dl, df["month"])


# Truly-exogenous weather (price cannot cause these); storage is semi-endogenous.
WEATHER = ["hdd_anom", "cdd_anom", "wind_anom", "solar_anom", "precip_anom"]
EXO_WEATHER = ["momentum"] + WEATHER                    # + lagged momentum control
EXO_FULL = EXO_WEATHER + ["storage_change_anom"]        # add contemporaneous storage
CORE_LAG = ["momentum", "storage_change_lag", "hdd_lag"]   # honest predetermined benchmark

SIGN = {"hdd_anom": +1, "cdd_anom": +1, "wind_anom": -1, "solar_anom": -1,
        "precip_anom": -1, "storage_change_anom": -1, "momentum": +1}


def build_features(df, horizon=HORIZON):
    out = df[["Date", "month", "TTF"]].copy()
    out["y"] = np.log(df["TTF"]).diff(horizon)
    out["momentum"] = dlog_anom(df, "TTF").shift(horizon)          # predetermined
    # contemporaneous exogenous weather (perfect foresight)
    out["hdd_anom"] = level_anom(df, "Europe_HDD")
    out["cdd_anom"] = level_anom(df, "Europe_CDD")
    out["wind_anom"] = level_anom(df, "EU_wind_speed")
    out["solar_anom"] = level_anom(df, "EU_solar")
    out["precip_anom"] = level_anom(df, "Nordic_precip")
    # contemporaneous storage (semi-endogenous)
    out["storage_change_anom"] = change_anom(df, "EU+UK_av_storage(bcm)")
    # lagged core benchmark terms
    out["storage_change_lag"] = change_anom(df, "EU+UK_av_storage(bcm)").shift(horizon)
    out["hdd_lag"] = level_anom(df, "Europe_HDD").shift(horizon)
    return out


def hac_lags(n, extra=0):
    return int(np.floor(4 * (n / 100) ** (2 / 9))) + extra


def fit_hac(feat, cols, train_end=TRAIN_END):
    d = feat.loc[feat["Date"] < train_end, ["y"] + cols].dropna()
    return sm.OLS(d["y"], sm.add_constant(d[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags(len(d)), "use_correction": True})


def r2(feat, cols, train_end=TRAIN_END):
    d = feat.loc[feat["Date"] < train_end, ["y"] + cols].dropna()
    return sm.OLS(d["y"], sm.add_constant(d[cols])).fit().rsquared, len(d)


# --------------------------------------------------------------------------- #
# Walk-forward / DM / summarize
# --------------------------------------------------------------------------- #

def walk_forward(feat, model_cols, min_train=MIN_TRAIN_OBS):
    all_cols = sorted({c for cols in model_cols.values() for c in cols})
    data = feat[["Date", "y"] + all_cols].dropna().reset_index(drop=True)
    rows = []
    for i in range(min_train, len(data)):
        train, test = data.iloc[:i], data.iloc[i]
        rec = {"Date": test["Date"], "actual": test["y"],
               "pred_random_walk": 0.0, "pred_hist_mean": train["y"].mean()}
        for name, cols in model_cols.items():
            fit = sm.OLS(train["y"], sm.add_constant(train[cols])).fit()
            Xte = sm.add_constant(test[cols].to_frame().T, has_constant="add")
            rec[f"pred_{name}"] = fit.predict(Xte).iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def dm_tstat(actual, pred_worse, pred_better, h=HORIZON):
    a = np.asarray(actual, float)
    d = (a - np.asarray(pred_worse, float)) ** 2 - (a - np.asarray(pred_better, float)) ** 2
    d = d[~np.isnan(d)]
    n = len(d); dbar = d.mean(); lag = max(h - 1, 0)
    var = ((d - dbar) ** 2).mean()
    for l in range(1, lag + 1):
        var += 2 * (1 - l / (lag + 1)) * ((d[l:] - dbar) * (d[:-l] - dbar)).mean()
    return dbar / np.sqrt(var / n) if var > 0 else np.nan


def summarize(res, model_names):
    rw = res["pred_random_walk"]; rw_err = res["actual"] - rw
    rows = []
    for name in model_names + ["random_walk", "hist_mean"]:
        pred = res[f"pred_{name}"]; err = res["actual"] - pred
        rows.append({"Model": name, "RMSE": np.sqrt((err ** 2).mean()),
                     "Hit rate": np.nan if name == "random_walk" else (np.sign(res["actual"]) == np.sign(pred)).mean(),
                     "OOS R2 vs RW": np.nan if name == "random_walk" else 1 - (err ** 2).sum() / (rw_err ** 2).sum(),
                     "DM t vs RW": np.nan if name == "random_walk" else dm_tstat(res["actual"], rw, pred),
                     "n": len(res)})
    return pd.DataFrame(rows)


def regime_r2(feat, cols):
    f = feat.copy()
    f["crisis"] = ((f["Date"] >= CRISIS_START) & (f["Date"] < CRISIS_END)).astype(int)
    vol = f["y"].rolling(6).std().shift(1)
    f["highvol"] = (vol > vol.expanding().median().shift(1)).astype(float)
    print("\nRegime R2 for the weather+storage ceiling model:")
    for rc, hi, lo in [("crisis", "crisis", "calm"), ("highvol", "high_vol", "low_vol")]:
        for val, lab in [(1, hi), (0, lo)]:
            d = f.loc[f[rc] == val, ["y"] + cols].dropna()
            if len(d) >= len(cols) + 5:
                rr = sm.OLS(d["y"], sm.add_constant(d[cols])).fit().rsquared
                print("  %-9s n=%3d  R2=%.3f" % (lab, len(d), rr))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    df = load_data()
    feat = build_features(df, horizon=HORIZON)

    print("ENDOGENEITY-FREE CEILING -- pre-specified exogenous model, perfect foresight.")
    print("Weather is truly exogenous; storage is semi-endogenous (flagged).\n")

    print("=" * 70)
    print("In-sample R2 decomposition (train %s..%s)" % (feat["Date"].min().date(), TRAIN_END[:7]))
    print("=" * 70)
    decomp = [
        ("momentum only (lagged)",            ["momentum"]),
        ("weather only (no momentum)",        WEATHER),
        ("weather + momentum",                EXO_WEATHER),
        ("weather + storage (no momentum)",   WEATHER + ["storage_change_anom"]),
        ("weather + storage + momentum",      EXO_FULL),
    ]
    for label, cols in decomp:
        rr, n = r2(feat, cols)
        print("  %-34s R2=%.3f   (n=%d)" % (label, rr, n))
    print("\n  (Compare: the earlier UNRESTRICTED perfect-foresight model hit R2~0.42, but")
    print("   leaned on wrong-signed endogenous vars. This is the clean causal ceiling.)")

    print("\n" + "=" * 70)
    print("HAC fit -- weather + storage + momentum (are the signs right?)")
    print("=" * 70)
    m = fit_hac(feat, EXO_FULL)
    print(m.summary())
    print("\nSign check (coef sign vs economic prior):")
    for c in EXO_FULL:
        ok = np.sign(m.params[c]) == SIGN[c]
        print("  %-22s coef=%+.5f  t=%+.2f  p=%.3f  sign_ok=%s"
              % (c, m.params[c], m.tvalues[c], m.pvalues[c], ok))

    print("\n" + "=" * 70)
    print("Out-of-sample walk-forward: honest core vs exogenous-foresight models")
    print("=" * 70)
    res = walk_forward(feat, {"core_lag": CORE_LAG, "exo_weather": EXO_WEATHER, "exo_full": EXO_FULL})
    print("\n-- Full backtest (%s to %s) --" % (res["Date"].min().date(), res["Date"].max().date()))
    print(summarize(res, ["core_lag", "exo_weather", "exo_full"]).to_string(index=False))
    clean = res.loc[res["Date"] >= TRAIN_END]
    if len(clean) >= 6:
        print("\n-- Clean post-%s slice --" % TRAIN_END[:7])
        print(summarize(clean, ["core_lag", "exo_weather", "exo_full"]).to_string(index=False))
    print("\nDM t, exo_full vs honest core (positive => foresight helps): %.2f"
          % dm_tstat(res["actual"], res["pred_core_lag"], res["pred_exo_full"]))

    regime_r2(feat, EXO_FULL)
    print("\nNOTE: perfect-foresight ceiling; weather exogenous (clean), storage semi-endogenous.")


if __name__ == "__main__":
    main()

# %%