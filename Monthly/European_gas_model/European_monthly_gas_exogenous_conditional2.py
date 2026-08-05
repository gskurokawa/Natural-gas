#%%
"""
ttf_european_lean.py

Robustness check on the endogeneity-free ceiling: a LEAN three-variable
perfect-foresight model, to separate "the fundamental signal is genuinely weak"
from "the 7-variable ceiling model was over-parameterized".

Model (H=1):
    Dlog TTF_t = c + b1*momentum + b2*storage_change_now + b3*HDD_now + e_t

  - momentum          : own 1-month return anomaly, LAGGED (predetermined control)
  - storage_change_now: contemporaneous storage-change anomaly (perfect foresight)
  - HDD_now           : contemporaneous heating-degree-day anomaly (perfect foresight)

These are the only two fundamentals that carried real weight in the clean ceiling
(storage was the sole individually-significant driver; HDD the strongest weather
term). Keeping just three parameters minimizes estimation noise, so if even this
lean, perfect-foresight model fails to beat a random walk out of sample, the
conclusion is "the monthly fundamental-price link is too weak", not "too many
parameters".

Benchmarks: the honest predetermined core (all lagged), random walk, hist mean.
Still a perfect-foresight CEILING, not a deployable forecast. European only. H=1.

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


LEAN = ["momentum", "storage_change_now", "hdd_now"]
CORE_LAG = ["momentum", "storage_change_lag", "hdd_lag"]
SIGN = {"momentum": +1, "storage_change_now": -1, "hdd_now": +1}


def build_features(df, horizon=HORIZON):
    out = df[["Date", "month", "TTF"]].copy()
    out["y"] = np.log(df["TTF"]).diff(horizon)
    out["momentum"] = dlog_anom(df, "TTF").shift(horizon)                 # predetermined
    out["storage_change_now"] = change_anom(df, "EU+UK_av_storage(bcm)")  # contemporaneous
    out["hdd_now"] = level_anom(df, "Europe_HDD")                         # contemporaneous
    out["storage_change_lag"] = change_anom(df, "EU+UK_av_storage(bcm)").shift(horizon)
    out["hdd_lag"] = level_anom(df, "Europe_HDD").shift(horizon)
    return out


def hac_lags(n):
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def fit_hac(feat, cols, train_end=TRAIN_END):
    d = feat.loc[feat["Date"] < train_end, ["y"] + cols].dropna()
    return sm.OLS(d["y"], sm.add_constant(d[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags(len(d)), "use_correction": True})


def r2(feat, cols, train_end=TRAIN_END):
    d = feat.loc[feat["Date"] < train_end, ["y"] + cols].dropna()
    return sm.OLS(d["y"], sm.add_constant(d[cols])).fit().rsquared, len(d)


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


def main():
    df = load_data()
    feat = build_features(df)

    print("LEAN perfect-foresight model (3 params): momentum(lag) + storage_now + HDD_now\n")

    print("=" * 66)
    print("In-sample R2 build-up (train %s..%s)" % (feat["Date"].min().date(), TRAIN_END[:7]))
    print("=" * 66)
    for label, cols in [("momentum only", ["momentum"]),
                        ("+ storage (now)", ["momentum", "storage_change_now"]),
                        ("+ HDD (now)  [= LEAN]", LEAN),
                        ("honest core (all lagged)", CORE_LAG)]:
        rr, n = r2(feat, cols)
        print("  %-28s R2=%.3f  (n=%d)" % (label, rr, n))

    print("\n" + "=" * 66)
    print("HAC fit -- LEAN model (signs & significance)")
    print("=" * 66)
    m = fit_hac(feat, LEAN)
    print(m.summary())
    for c in LEAN:
        print("  %-20s coef=%+.5f  t=%+.2f  p=%.3f  sign_ok=%s"
              % (c, m.params[c], m.tvalues[c], m.pvalues[c], np.sign(m.params[c]) == SIGN[c]))

    print("\n" + "=" * 66)
    print("Out-of-sample walk-forward: LEAN vs honest core vs random walk")
    print("=" * 66)
    res = walk_forward(feat, {"core_lag": CORE_LAG, "lean": LEAN})
    print("\n-- Full backtest (%s to %s) --" % (res["Date"].min().date(), res["Date"].max().date()))
    print(summarize(res, ["core_lag", "lean"]).to_string(index=False))
    clean = res.loc[res["Date"] >= TRAIN_END]
    if len(clean) >= 6:
        print("\n-- Clean post-%s slice --" % TRAIN_END[:7])
        print(summarize(clean, ["core_lag", "lean"]).to_string(index=False))
    print("\nDM t, LEAN vs honest core (positive => foresight helps): %.2f"
          % dm_tstat(res["actual"], res["pred_core_lag"], res["pred_lean"]))

    # regime R2
    f = feat.copy()
    f["crisis"] = ((f["Date"] >= CRISIS_START) & (f["Date"] < CRISIS_END)).astype(int)
    vol = f["y"].rolling(6).std().shift(1)
    f["highvol"] = (vol > vol.expanding().median().shift(1)).astype(float)
    print("\nRegime R2 (LEAN model):")
    for rc, hi, lo in [("crisis", "crisis", "calm"), ("highvol", "high_vol", "low_vol")]:
        for val, lab in [(1, hi), (0, lo)]:
            d = f.loc[f[rc] == val, ["y"] + LEAN].dropna()
            if len(d) >= len(LEAN) + 5:
                rr = sm.OLS(d["y"], sm.add_constant(d[LEAN])).fit().rsquared
                print("  %-9s n=%3d  R2=%.3f" % (lab, len(d), rr))
    print("\nNOTE: perfect-foresight ceiling; storage semi-endogenous, HDD exogenous.")


if __name__ == "__main__":
    main()

# %%