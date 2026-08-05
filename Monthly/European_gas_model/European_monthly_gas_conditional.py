#%%

"""
ttf_european_conditional.py

CONDITIONAL ("perfect-foresight") European model for month-ahead (H=1) TTF.

This is the companion to ttf_european_model.py. There, every predictor was
LAGGED one month, so it was a strict, deployable-today forecast. Here we instead
give the model PERFECT FORESIGHT of the same-month European fundamentals -- we
assume we know (or can forecast) this month's storage, weather, LNG, generation
mix, and Norwegian supply when predicting this month's TTF move. Weather is
treated as forecastable at the MONTHLY-AVERAGE level (not daily), per the brief.

What this is and is NOT
-----------------------
It is an EXPLANATORY CEILING, not a deployable forecast: it answers "how much of
TTF could European fundamentals explain IF we forecast them perfectly?", which
separates "fundamentals don't matter" from "fundamentals matter but are
themselves unforecastable a month out". It is NOT a real forecast, because (a)
in practice you'd plug in fundamentals *forecasts*, not actuals, and (b)
same-month storage/LNG and price are jointly determined, so contemporaneous
coefficients are correlations (endogenous), not clean predictive effects.

Design
------
  - Target: y_t = one-month log return of TTF (same as before).
  - momentum: LAGGED one month (predetermined) -- the only price term; using the
    contemporaneous return would just be the target itself.
  - all fundamentals: CONTEMPORANEOUS (perfect foresight of month t).
  - Deseasonalization still uses an expanding, PRIOR-YEARS-ONLY climatology and
    selection is still TRAINING-ONLY, so the only look-ahead is the intended
    foresight of this month's fundamentals -- nothing else leaks.

Benchmarks in the walk-forward:
  - core_lag  : the honest predetermined core (momentum + lagged storage + lagged HDD)
  - conditional: the selected perfect-foresight model
  - random walk / historical mean

European variables only (no JKM/HH/global). Horizon fixed at H=1.
Toy / educational model, not a trading tool.

Requirements: pandas, numpy, statsmodels, scipy
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

DATA_PATH = "NG_m_final.csv"
TRAIN_END = "2025-01-01"
MIN_TRAIN_OBS = 60
HORIZON = 1
N_SELECT = 6
CORR_PRUNE_THRESH = 0.70
SCREEN_P_MAX = 0.10
CRISIS_START, CRISIS_END = "2021-09-01", "2023-07-01"


# --------------------------------------------------------------------------- #
# 1. Data + anomaly builders
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


TRANSFORMS = {"level_anom": level_anom, "change_anom": change_anom, "dlog_anom": dlog_anom}

# (name, source column, transform, expected sign, note). momentum is lagged; all
# fundamentals enter CONTEMPORANEOUSLY (perfect foresight of month t).
CANDIDATES = [
    ("momentum",              "TTF",                       "dlog_anom",  +1, "own 1-month log return (LAGGED)"),
    ("storage_change_anom",   "EU+UK_av_storage(bcm)",     "change_anom", -1, "MoM storage change (draw = bullish)"),
    ("storage_level_anom",    "EU+UK_av_storage(bcm)",     "level_anom",  -1, "storage level"),
    ("production_anom",       "EU+UK Production(bcm)",      "level_anom",  -1, "indigenous production"),
    ("net_piped_anom",        "EU+UK Net_piped(bcm)",      "level_anom",  -1, "net pipeline imports"),
    ("lng_imports_anom",      "EU+UK LNG imports",         "level_anom",  -1, "EU+UK LNG imports (supply)"),
    ("net_supply_anom",       "EU+UK Net_supply",          "level_anom",  -1, "total net supply"),
    ("total_demand_anom",     "EU+UK Total(bcm)",          "level_anom",  +1, "total gas demand"),
    ("nonpower_demand_anom",  "EU+UK Non_power(bcm)",       "level_anom",  +1, "non-power gas demand"),
    ("power_gas_demand_anom", "EU+UK Electricity(bcm)",    "level_anom",  +1, "gas-for-power demand (bcm)"),
    ("residual_load_anom",    "EU+UK Residual load",       "level_anom",  +1, "thermal power demand"),
    ("coal_gen_anom",         "EU+UK Coal",                "level_anom",  -1, "coal generation"),
    ("nuclear_gen_anom",      "EU+UK Nuclear",             "level_anom",  -1, "nuclear generation"),
    ("hydro_gen_anom",        "EU+UK Hydro_gen",           "level_anom",  -1, "hydro generation"),
    ("gas_burn_anom",         "EU+UK Fossil gas",          "level_anom",  +1, "gas-fired generation (endogenous)"),
    ("production_chg_anom",     "EU+UK Production(bcm)",    "change_anom", -1, "change in indigenous production"),
    ("net_piped_chg_anom",      "EU+UK Net_piped(bcm)",    "change_anom", -1, "change in net pipeline imports"),
    ("lng_imports_chg_anom",    "EU+UK LNG imports",       "change_anom", -1, "change in EU+UK LNG imports"),
    ("net_supply_chg_anom",     "EU+UK Net_supply",        "change_anom", -1, "change in total net supply"),
    ("total_demand_chg_anom",   "EU+UK Total(bcm)",        "change_anom", +1, "change in total gas demand"),
    ("nonpower_demand_chg_anom","EU+UK Non_power(bcm)",     "change_anom", +1, "change in non-power gas demand"),
    ("power_gas_demand_chg_anom","EU+UK Electricity(bcm)",  "change_anom", +1, "change in gas-for-power demand"),
    ("residual_load_chg_anom",  "EU+UK Residual load",     "change_anom", +1, "change in thermal power demand"),
    ("coal_gen_chg_anom",       "EU+UK Coal",              "change_anom", -1, "change in coal generation"),
    ("nuclear_gen_chg_anom",    "EU+UK Nuclear",           "change_anom", -1, "change in nuclear generation"),
    ("hydro_gen_chg_anom",      "EU+UK Hydro_gen",         "change_anom", -1, "change in hydro generation"),
    ("gas_burn_chg_anom",       "EU+UK Fossil gas",        "change_anom", +1, "change in gas-fired generation (endogenous)"),
    ("norway_prod_anom",      "Norway_gas_prod",           "level_anom",  -1, "Norwegian gas production"),
    ("norway_supplyred_anom", "Norway_supply_red",         "level_anom",  +1, "Norway supply reduction"),
    ("norway_planned_anom",   "Norway_planned_outage",     "level_anom",  +1, "Norway planned outages"),
    ("norway_unplanned_anom", "Norway_unplanned_outage",   "level_anom",  +1, "Norway unplanned outages"),
    ("hdd_anom",              "Europe_HDD",                "level_anom",  +1, "heating degree days (monthly)"),
    ("cdd_anom",              "Europe_CDD",                "level_anom",  +1, "cooling degree days (monthly)"),
    ("wind_anom",             "EU_wind_speed",             "level_anom",  -1, "wind speed (more wind = less gas)"),
    ("solar_anom",            "EU_solar",                  "level_anom",  -1, "solar irradiation"),
    ("precip_anom",           "Nordic_precip",             "level_anom",  -1, "Nordic precipitation (hydro inflows)"),
]
CANDIDATE_NAMES = [c[0] for c in CANDIDATES]
CANDIDATE_META = {c[0]: {"col": c[1], "transform": c[2], "sign": c[3], "note": c[4]} for c in CANDIDATES}

# Honest predetermined benchmark (everything lagged) -- the deployable core.
CORE_LAG = ["momentum", "storage_change_lag", "hdd_lag"]


def build_candidates(df, horizon=HORIZON):
    out = df[["Date", "month", "TTF"]].copy()
    out["y"] = np.log(df["TTF"]).diff(horizon)
    for name in CANDIDATE_NAMES:
        m = CANDIDATE_META[name]
        s = TRANSFORMS[m["transform"]](df, m["col"])
        shift = horizon if name == "momentum" else 0     # momentum lagged; fundamentals contemporaneous
        out[name] = s.shift(shift)
    # lagged core terms for the honest benchmark
    out["storage_change_lag"] = change_anom(df, "EU+UK_av_storage(bcm)").shift(horizon)
    out["hdd_lag"] = level_anom(df, "Europe_HDD").shift(horizon)
    return out


def hac_lags(n, extra=0):
    return int(np.floor(4 * (n / 100) ** (2 / 9))) + extra


# --------------------------------------------------------------------------- #
# 2. Stationarity gate / screen / prune / select
# --------------------------------------------------------------------------- #

def check_stationarity(series, name=""):
    s = series.dropna()
    adf_p = adfuller(s, autolag="AIC")[1]
    kpss_p = kpss(s, regression="c", nlags="auto")[1]
    stationary = (adf_p < 0.05) and (kpss_p > 0.05)
    borderline = (adf_p < 0.05) or (kpss_p > 0.05)
    verdict = "stationary" if stationary else ("conflicting" if borderline else "non-stationary")
    return {"name": name, "adf_p": adf_p, "kpss_p": kpss_p, "verdict": verdict, "usable": borderline}


def stationarity_gate(feat, names, train_end=TRAIN_END):
    train = feat.loc[feat["Date"] < train_end]
    rep = pd.DataFrame([check_stationarity(train[nm], nm) for nm in names])
    return rep, rep.loc[rep["usable"], "name"].tolist()


def univariate_screen(feat, names, train_end=TRAIN_END):
    train = feat.loc[feat["Date"] < train_end]
    rows = []
    for nm in names:
        d = train[["y", nm]].dropna()
        if len(d) < 30:
            continue
        m = sm.OLS(d["y"], sm.add_constant(d[[nm]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags(len(d)), "use_correction": True})
        rows.append({"name": nm, "coef": m.params[nm], "t": m.tvalues[nm], "p": m.pvalues[nm],
                     "R2": m.rsquared, "exp_sign": CANDIDATE_META[nm]["sign"],
                     "sign_ok": np.sign(m.params[nm]) == CANDIDATE_META[nm]["sign"],
                     "note": CANDIDATE_META[nm]["note"]})
    out = pd.DataFrame(rows)
    return out.reindex(out["t"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def prune_collinear(feat, screen, train_end=TRAIN_END, thresh=CORR_PRUNE_THRESH):
    train = feat.loc[feat["Date"] < train_end]
    ranked = screen["name"].tolist()
    corr = train[ranked].corr().abs()
    kept, dropped = [], {}
    for nm in ranked:
        clash = next((k for k in kept if corr.loc[nm, k] > thresh), None)
        (kept.append(nm) if clash is None else dropped.setdefault(nm, clash))
    return kept, dropped


def select_model(screen, kept_pool, n_select=N_SELECT, p_max=SCREEN_P_MAX):
    ordered = [nm for nm in screen["name"].tolist() if nm in kept_pool]
    sig = set(screen.loc[screen["p"] < p_max, "name"])
    chosen = ["momentum"] if "momentum" in kept_pool else []
    for nm in ordered:
        if len(chosen) >= n_select:
            break
        if nm == "momentum" or nm not in sig:
            continue
        chosen.append(nm)
    return chosen


# --------------------------------------------------------------------------- #
# 3. Fit / walk-forward / DM
# --------------------------------------------------------------------------- #

def fit_hac(feat, cols, train_end=TRAIN_END):
    train = feat.loc[feat["Date"] < train_end, ["y"] + cols].dropna()
    m = sm.OLS(train["y"], sm.add_constant(train[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags(len(train)), "use_correction": True})
    return m, train


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


# --------------------------------------------------------------------------- #
# 4. Regime split
# --------------------------------------------------------------------------- #

def add_regime(feat):
    f = feat.copy()
    f["crisis"] = ((f["Date"] >= CRISIS_START) & (f["Date"] < CRISIS_END)).astype(int)
    vol = f["y"].rolling(6).std().shift(1)
    f["highvol"] = (vol > vol.expanding().median().shift(1)).astype(float)
    return f


def _sub_fit(feat, cols, mask):
    d = feat.loc[mask, ["y"] + cols].dropna()
    if len(d) < len(cols) + 5:
        return None
    return sm.OLS(d["y"], sm.add_constant(d[cols])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags(len(d)), "use_correction": True})


def regime_report(feat, cols):
    feat = add_regime(feat)
    print("\n" + "=" * 78)
    print("Regime split -- selected conditional model: %s" % cols)
    print("=" * 78)
    for rc, hi, lo in [("crisis", "crisis", "calm"), ("highvol", "high_vol", "low_vol")]:
        tag = ("Calendar crisis" if rc == "crisis" else "Trailing-volatility (look-ahead-safe)")
        print("\n--- %s ---" % tag)
        for val, lab in [(1, hi), (0, lo)]:
            m = _sub_fit(feat, cols, feat[rc] == val)
            print("  %-9s: %s" % (lab, "n<min" if m is None else "n=%d  R2=%.3f" % (int(m.nobs), m.rsquared)))


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #

def main():
    df = load_data()
    feat = build_candidates(df, horizon=HORIZON)

    print("CONDITIONAL / PERFECT-FORESIGHT model -- fundamentals enter contemporaneously.")
    print("This is an EXPLANATORY CEILING, not a deployable forecast (see docstring).\n")

    print("=" * 78 + "\nStationarity gate\n" + "=" * 78)
    rep, usable = stationarity_gate(feat, CANDIDATE_NAMES)
    print(rep.to_string(index=False))
    print(f"\nUsable: {len(usable)} of {len(CANDIDATE_NAMES)}")

    print("\n" + "=" * 78 + "\nUnivariate HAC screen (contemporaneous) -- ranked by |t|\n" + "=" * 78)
    screen = univariate_screen(feat, usable)
    print(screen[["name", "coef", "t", "p", "R2", "exp_sign", "sign_ok"]].to_string(index=False))

    kept, dropped = prune_collinear(feat, screen)
    if dropped:
        print("\nCollinearity pruning (|corr| > %.2f):" % CORR_PRUNE_THRESH)
        for d, k in dropped.items():
            print(f"  {d:26} -> {k}")

    selected = select_model(screen, kept)
    print("\nSelected conditional model:", selected)

    print("\n" + "=" * 78 + "\nIn-sample HAC fit -- selected conditional model\n" + "=" * 78)
    m, _ = fit_hac(feat, selected)
    print(m.summary())

    print("\n" + "=" * 78)
    print("Out-of-sample walk-forward (H=1): honest core (lagged) vs conditional (foresight)")
    print("=" * 78)
    res = walk_forward(feat, {"core_lag": CORE_LAG, "conditional": selected})
    print("\n-- Full backtest (%s to %s) --" % (res["Date"].min().date(), res["Date"].max().date()))
    print(summarize(res, ["core_lag", "conditional"]).to_string(index=False))

    clean = res.loc[res["Date"] >= TRAIN_END]
    if len(clean) >= 6:
        print("\n-- Clean post-%s slice --" % TRAIN_END[:7])
        print(summarize(clean, ["core_lag", "conditional"]).to_string(index=False))

    dm = dm_tstat(res["actual"], res["pred_core_lag"], res["pred_conditional"])
    print("\nDM t-stat, conditional vs honest core (positive => foresight lowers MSE): %.2f" % dm)
    print("\nNOTE: 'conditional' uses perfect foresight of same-month fundamentals -- it is an")
    print("upper bound on fundamental explanatory power, not an achievable live forecast.")

    regime_report(feat, selected)


if __name__ == "__main__":
    main()

# %%