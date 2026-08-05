#%%
"""
ttf_european_model.py

Exploratory H=1 study: how much of month-ahead TTF can a SELF-CONTAINED
European gas model explain?

Candidate predictors are restricted to European / EU+UK fundamentals only:
  - EU+UK gas balance & storage (storage level & change, production, net piped
    imports, EU+UK LNG imports, net supply, total/non-power/electricity demand)
  - EU+UK power generation mix (gas, coal, nuclear, hydro, residual load)
  - Norway supply (production, planned/unplanned outages, supply reduction)
  - European weather (HDD, CDD, wind speed, solar, Nordic precipitation)
plus the target's own 1-month momentum.

NOT used (deliberately, for the "self-contained Europe" test): JKM / Henry Hub
prices, USD-EUR / VIX, non-European weather (US/NE-Asia GWDD, Atlantic ACE,
Gulf storm days), and all global LNG capacity / trade series (incl. global LNG
capacity offline). Those are for a later, wider model.

Methodology mirrors ttf_model.py and is designed to avoid fooling ourselves:
  * Every predictor is a deseasonalized anomaly vs an EXPANDING, prior-years-only
    calendar-month climatology, then lagged 1 month, so it is predetermined and
    uses no future information. Trending supply/demand/generation series enter
    BOTH as level anomalies and as month-over-month CHANGE anomalies; the level
    versions are typically non-stationary (secular trends) and get gated out,
    while the change versions are stationary and reach the screen.
  * Candidates are stationarity-gated (ADF + KPSS).
  * A univariate HAC screen ranks individual predictive content.
  * Collinear candidates are pruned (|corr| > CORR_PRUNE_THRESH).
  * The compact model is SELECTED ON THE TRAINING WINDOW ONLY, then evaluated
    out-of-sample with an expanding walk-forward and a Diebold-Mariano test
    (Newey-West) vs a random walk. Metrics are reported over the full backtest
    and over the clean post-TRAIN_END slice.

Horizon is fixed at H = 1 month.

Toy / educational model, not a trading or investment tool.

Requirements: pandas, numpy, statsmodels, scipy
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

DATA_PATH = "NG_m_final.csv"
TRAIN_END = "2025-01-01"     # selection & in-sample fit use data strictly before this
MIN_TRAIN_OBS = 60           # months of history before the walk-forward begins
HORIZON = 1                  # months ahead (H=1 only, per the current experiment)

N_SELECT = 6                 # max predictors in the compact European model (incl. momentum)
CORR_PRUNE_THRESH = 0.70     # drop the less-significant of any candidate pair above this |corr|
SCREEN_P_MAX = 0.10          # a candidate must clear this univariate HAC p-value to be selectable

# The always-excluded global block, kept here only for documentation / assertion.
GLOBAL_COLS_EXCLUDED = [
    "JKM(USD/mmbtu)", "HH(USD/mmbtu)", "USD-EUR_FX", "VIX",
    "US_GWDD", "NE_Asia_GWDD", "Atlantic_ACE", "Gulf_storm_days",
    "US_GWDD_anomaly", "NE_Asia_GWDD_anomaly", "Atlantic_ACE_anomaly",
    "Global LNG nameplate capacity", "Global LNG capacity offline",
    "CH+JP+KR LNG imports", "EG LNG imports", "IN LNG imports",
    "QA+AU+US LNG exports", "ID+MY+BN LNG exports", "NG LNG exports",
]


# --------------------------------------------------------------------------- #
# 1. Data + no-look-ahead anomaly builders
# --------------------------------------------------------------------------- #

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["month"] = df["Date"].dt.month
    df["TTF"] = df["TTF(USD/mmbtu)"]
    return df


def expanding_seasonal_mean(series: pd.Series, month: pd.Series) -> pd.Series:
    """Mean of all PRIOR years' values for the same calendar month (current excluded)."""
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

# Candidate European predictors: (name, source column, transform, expected sign, note)
# Expected sign is the hypothesized effect on the forward TTF return; it is used only
# for reporting/interpretation, never to force selection.
CANDIDATES = [
    ("momentum",              "TTF",                       "dlog_anom",  +1, "own 1-month log return"),
    # --- EU+UK storage & gas balance ---
    ("storage_change_anom",   "EU+UK_av_storage(bcm)",     "change_anom", -1, "MoM storage change (build = bearish)"),
    ("storage_level_anom",    "EU+UK_av_storage(bcm)",     "level_anom",  -1, "storage level"),
    ("production_anom",       "EU+UK Production(bcm)",      "level_anom",  -1, "indigenous production"),
    ("net_piped_anom",        "EU+UK Net_piped(bcm)",      "level_anom",  -1, "net pipeline imports"),
    ("lng_imports_anom",      "EU+UK LNG imports",         "level_anom",  -1, "EU+UK LNG imports (supply)"),
    ("net_supply_anom",       "EU+UK Net_supply",          "level_anom",  -1, "total net supply"),
    ("total_demand_anom",     "EU+UK Total(bcm)",          "level_anom",  +1, "total gas demand"),
    ("nonpower_demand_anom",  "EU+UK Non_power(bcm)",       "level_anom",  +1, "non-power gas demand"),
    ("power_gas_demand_anom", "EU+UK Electricity(bcm)",    "level_anom",  +1, "gas-for-power demand (bcm)"),
    # --- EU+UK power generation mix (substitutes for gas in the stack) ---
    ("residual_load_anom",    "EU+UK Residual load",       "level_anom",  +1, "thermal power demand (load - wind/solar)"),
    ("coal_gen_anom",         "EU+UK Coal",                "level_anom",  -1, "coal generation (gas substitute)"),
    ("nuclear_gen_anom",      "EU+UK Nuclear",             "level_anom",  -1, "nuclear generation"),
    ("hydro_gen_anom",        "EU+UK Hydro_gen",           "level_anom",  -1, "hydro generation"),
    ("gas_burn_anom",         "EU+UK Fossil gas",          "level_anom",  +1, "gas-fired generation (demand; note: endogenous)"),
    # --- EU+UK supply / demand / generation as MONTH-OVER-MONTH CHANGES ---
    # Stationary variants of the trending level series above: a supply/demand
    # shock is a change, not a level, and diffing removes the secular trend that
    # made the level anomalies non-stationary (so these actually reach the screen).
    ("production_chg_anom",     "EU+UK Production(bcm)",   "change_anom", -1, "change in indigenous production"),
    ("net_piped_chg_anom",      "EU+UK Net_piped(bcm)",   "change_anom", -1, "change in net pipeline imports"),
    ("lng_imports_chg_anom",    "EU+UK LNG imports",      "change_anom", -1, "change in EU+UK LNG imports"),
    ("net_supply_chg_anom",     "EU+UK Net_supply",       "change_anom", -1, "change in total net supply"),
    ("total_demand_chg_anom",   "EU+UK Total(bcm)",       "change_anom", +1, "change in total gas demand"),
    ("nonpower_demand_chg_anom","EU+UK Non_power(bcm)",    "change_anom", +1, "change in non-power gas demand"),
    ("power_gas_demand_chg_anom","EU+UK Electricity(bcm)", "change_anom", +1, "change in gas-for-power demand (bcm)"),
    ("residual_load_chg_anom",  "EU+UK Residual load",    "change_anom", +1, "change in thermal power demand"),
    ("coal_gen_chg_anom",       "EU+UK Coal",             "change_anom", -1, "change in coal generation"),
    ("nuclear_gen_chg_anom",    "EU+UK Nuclear",          "change_anom", -1, "change in nuclear generation"),
    ("hydro_gen_chg_anom",      "EU+UK Hydro_gen",        "change_anom", -1, "change in hydro generation"),
    ("gas_burn_chg_anom",       "EU+UK Fossil gas",       "change_anom", +1, "change in gas-fired generation (endogenous)"),
    # --- Norway supply ---
    ("norway_prod_anom",      "Norway_gas_prod",           "level_anom",  -1, "Norwegian gas production"),
    ("norway_supplyred_anom", "Norway_supply_red",         "level_anom",  +1, "Norway supply reduction"),
    ("norway_planned_anom",   "Norway_planned_outage",     "level_anom",  +1, "Norway planned outages"),
    ("norway_unplanned_anom", "Norway_unplanned_outage",   "level_anom",  +1, "Norway unplanned outages"),
    # --- European weather ---
    ("hdd_anom",              "Europe_HDD",                "level_anom",  +1, "heating degree days"),
    ("cdd_anom",              "Europe_CDD",                "level_anom",  +1, "cooling degree days"),
    ("wind_anom",             "EU_wind_speed",             "level_anom",  -1, "wind speed (more wind = less gas)"),
    ("solar_anom",            "EU_solar",                  "level_anom",  -1, "solar irradiation"),
    ("precip_anom",           "Nordic_precip",             "level_anom",  -1, "Nordic precipitation (hydro inflows)"),
]

CANDIDATE_NAMES = [c[0] for c in CANDIDATES]
CANDIDATE_META = {c[0]: {"col": c[1], "transform": c[2], "sign": c[3], "note": c[4]} for c in CANDIDATES}
CORE_MODEL = ["momentum", "storage_change_anom", "hdd_anom"]   # the ttf_model.py baseline


def build_candidates(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Target y = H-month forward log TTF return; every candidate is its lagged anomaly."""
    out = df[["Date", "month", "TTF"]].copy()
    out["y"] = np.log(df["TTF"]).diff(horizon)
    for name in CANDIDATE_NAMES:
        m = CANDIDATE_META[name]
        series = TRANSFORMS[m["transform"]](df, m["col"])
        out[name] = series.shift(horizon)
    return out


def hac_lags(n: int, extra: int = 0) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9))) + extra


# --------------------------------------------------------------------------- #
# 2. Stationarity gate
# --------------------------------------------------------------------------- #

def check_stationarity(series: pd.Series, name: str = "") -> dict:
    s = series.dropna()
    adf_p = adfuller(s, autolag="AIC")[1]
    kpss_p = kpss(s, regression="c", nlags="auto")[1]
    stationary = (adf_p < 0.05) and (kpss_p > 0.05)
    borderline = (adf_p < 0.05) or (kpss_p > 0.05)   # at least one test says stationary
    verdict = "stationary" if stationary else ("conflicting" if borderline else "non-stationary")
    return {"name": name, "adf_p": adf_p, "kpss_p": kpss_p, "verdict": verdict, "usable": borderline}


def stationarity_gate(feat: pd.DataFrame, names, train_end: str = TRAIN_END):
    """Return (report_df, usable_names). A candidate is usable if not clearly non-stationary."""
    train = feat.loc[feat["Date"] < train_end]
    rows = [check_stationarity(train[nm], nm) for nm in names]
    rep = pd.DataFrame(rows)
    usable = rep.loc[rep["usable"], "name"].tolist()
    return rep, usable


# --------------------------------------------------------------------------- #
# 3. Univariate HAC screen + collinearity pruning + selection
# --------------------------------------------------------------------------- #

def univariate_screen(feat: pd.DataFrame, names, train_end: str = TRAIN_END) -> pd.DataFrame:
    """One-predictor HAC regressions on the training window; ranked by |t|."""
    train = feat.loc[feat["Date"] < train_end]
    rows = []
    for nm in names:
        d = train[["y", nm]].dropna()
        if len(d) < 30:
            continue
        X = sm.add_constant(d[[nm]])
        m = sm.OLS(d["y"], X).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": hac_lags(len(d)), "use_correction": True})
        rows.append({
            "name": nm, "coef": m.params[nm], "t": m.tvalues[nm], "p": m.pvalues[nm],
            "R2": m.rsquared, "exp_sign": CANDIDATE_META[nm]["sign"],
            "sign_ok": np.sign(m.params[nm]) == CANDIDATE_META[nm]["sign"],
            "n": len(d), "note": CANDIDATE_META[nm]["note"],
        })
    out = pd.DataFrame(rows)
    return out.reindex(out["t"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def prune_collinear(feat: pd.DataFrame, screen: pd.DataFrame, train_end: str = TRAIN_END,
                    thresh: float = CORR_PRUNE_THRESH):
    """Greedily drop the lower-|t| member of any candidate pair correlated above `thresh`."""
    train = feat.loc[feat["Date"] < train_end]
    ranked = screen["name"].tolist()               # already sorted by |t| desc
    corr = train[ranked].corr().abs()
    kept, dropped = [], {}
    for nm in ranked:
        clash = next((k for k in kept if corr.loc[nm, k] > thresh), None)
        if clash is None:
            kept.append(nm)
        else:
            dropped[nm] = clash
    return kept, dropped


def select_model(screen: pd.DataFrame, kept_pool, n_select: int = N_SELECT,
                 p_max: float = SCREEN_P_MAX):
    """
    Compact European model: momentum as anchor, then the strongest surviving
    candidates (by |t|) that clear p_max, up to n_select total. Selection uses
    training-window statistics only.
    """
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
# 4. In-sample fit, walk-forward backtest, Diebold-Mariano
# --------------------------------------------------------------------------- #

def fit_hac(feat: pd.DataFrame, cols, train_end: str = TRAIN_END):
    train = feat.loc[feat["Date"] < train_end, ["Date", "y"] + cols].dropna()
    X = sm.add_constant(train[cols])
    m = sm.OLS(train["y"], X).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": hac_lags(len(train)), "use_correction": True})
    return m, train


def walk_forward(feat: pd.DataFrame, model_cols: dict, min_train: int = MIN_TRAIN_OBS) -> pd.DataFrame:
    """
    Expanding-window one-step-ahead backtest for several models at once.

    `model_cols` maps model name -> list of feature columns. Coefficients are
    refit each step (feature SET is fixed, having been chosen on the training
    window only). Random walk (0) and historical mean are added automatically.
    """
    all_cols = sorted({c for cols in model_cols.values() for c in cols})
    data = feat[["Date", "y"] + all_cols].dropna().reset_index(drop=True)
    rows = []
    for i in range(min_train, len(data)):
        train, test = data.iloc[:i], data.iloc[i]
        rec = {"Date": test["Date"], "actual": test["y"],
               "pred_random_walk": 0.0, "pred_hist_mean": train["y"].mean()}
        for name, cols in model_cols.items():
            Xtr = sm.add_constant(train[cols])
            fit = sm.OLS(train["y"], Xtr).fit()
            Xte = sm.add_constant(test[cols].to_frame().T, has_constant="add")
            rec[f"pred_{name}"] = fit.predict(Xte).iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def dm_tstat(actual, pred_worse, pred_better, h: int = HORIZON) -> float:
    """
    Diebold-Mariano statistic for squared-error loss with a Newey-West variance
    (h-1 lags). Positive => `pred_better` has significantly lower MSE.
    """
    a = np.asarray(actual, float)
    d = (a - np.asarray(pred_worse, float)) ** 2 - (a - np.asarray(pred_better, float)) ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    lag = max(h - 1, 0)
    var = ((d - dbar) ** 2).mean()
    for l in range(1, lag + 1):
        cov = ((d[l:] - dbar) * (d[:-l] - dbar)).mean()
        var += 2 * (1 - l / (lag + 1)) * cov
    return dbar / np.sqrt(var / n) if var > 0 else np.nan


def summarize(res: pd.DataFrame, model_names, label: str = "") -> pd.DataFrame:
    rw = res["pred_random_walk"]
    rw_err = res["actual"] - rw
    rows = []
    for name in model_names + ["random_walk", "hist_mean"]:
        pred = res[f"pred_{name}"]
        err = res["actual"] - pred
        rmse = np.sqrt((err ** 2).mean())
        hit = np.nan if name == "random_walk" else (np.sign(res["actual"]) == np.sign(pred)).mean()
        oos_r2 = np.nan if name == "random_walk" else 1 - (err ** 2).sum() / (rw_err ** 2).sum()
        dm = np.nan if name == "random_walk" else dm_tstat(res["actual"], rw, pred)
        rows.append({"Model": name, "RMSE": rmse, "Hit rate": hit,
                     "OOS R2 vs RW": oos_r2, "DM t vs RW": dm, "n": len(res)})
    out = pd.DataFrame(rows)
    if label:
        out.attrs["label"] = label
    return out


# --------------------------------------------------------------------------- #
# 4b. Regime-split diagnostic (is the signal only a 2021-22 crisis artifact?)
# --------------------------------------------------------------------------- #

# Calendar crisis window: acute run-up through post-peak normalization. This is
# the "was it just the energy crisis?" test, so the window is set to the event.
CRISIS_START = "2021-09-01"
CRISIS_END = "2023-07-01"   # crisis regime = [START, END); everything else = calm


def add_regime(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Add two regime flags:
      crisis  -- calendar dummy for the 2021-22 energy crisis window.
      highvol -- LOOK-AHEAD-SAFE: trailing 6-month realized vol of y, shifted 1,
                 above its own expanding median (also shifted). A data-driven,
                 no-peeking proxy for "turbulent" months.
    """
    f = feat.copy()
    f["crisis"] = ((f["Date"] >= CRISIS_START) & (f["Date"] < CRISIS_END)).astype(int)
    vol = f["y"].rolling(6).std().shift(1)
    med = vol.expanding().median().shift(1)
    f["highvol"] = (vol > med).astype(float)
    return f


def _subsample_fit(feat, cols, mask):
    d = feat.loc[mask, ["y"] + cols].dropna()
    if len(d) < len(cols) + 5:
        return None, d
    X = sm.add_constant(d[cols])
    m = sm.OLS(d["y"], X).fit(cov_type="HAC",
                              cov_kwds={"maxlags": hac_lags(len(d)), "use_correction": True})
    return m, d


def regime_subsample(feat, cols, regime_col, hi_label, lo_label):
    """Side-by-side HAC coefficients (coef, t) in the two regimes, plus n and R2."""
    tab = {c: {} for c in ["const"] + cols}
    meta = {}
    for val, lab in [(1, hi_label), (0, lo_label)]:
        m, d = _subsample_fit(feat, cols, feat[regime_col] == val)
        meta[lab] = {"n": len(d), "R2": (m.rsquared if m is not None else np.nan)}
        for c in ["const"] + cols:
            tab[c][f"coef[{lab}]"] = (m.params[c] if m is not None else np.nan)
            tab[c][f"t[{lab}]"] = (m.tvalues[c] if m is not None else np.nan)
    df = pd.DataFrame(tab).T[[f"coef[{hi_label}]", f"t[{hi_label}]",
                              f"coef[{lo_label}]", f"t[{lo_label}]"]]
    return df.round(4), meta


def regime_interaction(feat, cols, regime_col):
    """
    Pooled HAC regression y ~ const + regime + X + X:regime, with a joint Wald
    test that ALL interaction terms are zero (H0: coefficients don't differ by
    regime). A small joint p => the model's coefficients are regime-dependent.
    Also returns the per-predictor interaction t-stats.
    """
    d = feat[["y", regime_col] + cols].dropna().copy()
    X = pd.DataFrame({"regime": d[regime_col].astype(float)}, index=d.index)
    inter = []
    for c in cols:
        X[c] = d[c].values
        ic = f"{c}_Xreg"
        X[ic] = (d[c] * d[regime_col]).values
        inter.append(ic)
    X = sm.add_constant(X)
    m = sm.OLS(d["y"], X).fit(cov_type="HAC",
                              cov_kwds={"maxlags": hac_lags(len(d)), "use_correction": True})
    names = list(X.columns)
    R = np.zeros((len(inter), len(names)))
    for i, ic in enumerate(inter):
        R[i, names.index(ic)] = 1.0
    w = m.wald_test(R, use_f=True)
    inter_t = {c: m.tvalues[f"{c}_Xreg"] for c in cols}
    return m, inter_t, float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue)), len(d)


def regime_report(feat, cols):
    feat = add_regime(feat)

    print("\n" + "=" * 78)
    print("Regime split -- selected European model: %s" % cols)
    print("=" * 78)

    for regime_col, hi, lo in [("crisis", "crisis", "calm"),
                               ("highvol", "high_vol", "low_vol")]:
        tag = ("Calendar crisis %s..%s" % (CRISIS_START[:7], CRISIS_END[:7])
               if regime_col == "crisis"
               else "Trailing-volatility (look-ahead-safe)")
        print("\n--- %s ---" % tag)
        sub, meta = regime_subsample(feat, cols, regime_col, hi, lo)
        for lab, mm in meta.items():
            print(f"  {lab:9}: n={mm['n']:3d}   R2={mm['R2']:.3f}")
        print(sub.to_string())
        m, inter_t, wstat, wp, npool = regime_interaction(feat, cols, regime_col)
        print("  interaction t-stats (coef differs in %s vs %s):" % (hi, lo))
        for c, t in inter_t.items():
            print(f"    {c:24} t = {t:+.2f}")
        print("  Joint Wald test (all interactions = 0):  F = %.2f,  p = %.3f   [n=%d]"
              % (wstat, wp, npool))
        print("  => %s" % ("coefficients ARE regime-dependent (signal concentrated in one regime)"
                           if wp < 0.10 else
                           "no significant regime dependence detected"))


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #

def main():
    df = load_data()

    # Guardrail: make sure no global column ever sneaks into a candidate.
    used_cols = {CANDIDATE_META[n]["col"] for n in CANDIDATE_NAMES}
    leaked = used_cols.intersection(GLOBAL_COLS_EXCLUDED)
    assert not leaked, f"global column(s) leaked into candidates: {leaked}"

    feat = build_candidates(df, horizon=HORIZON)

    print("=" * 78)
    print("Stationarity gate (training sample, H=1 candidates)")
    print("=" * 78)
    rep, usable = stationarity_gate(feat, CANDIDATE_NAMES)
    print(rep.to_string(index=False))
    print(f"\nUsable (not clearly non-stationary): {len(usable)} of {len(CANDIDATE_NAMES)}")

    print("\n" + "=" * 78)
    print("Univariate HAC screen (training sample) -- ranked by |t|")
    print("=" * 78)
    screen = univariate_screen(feat, usable)
    show = screen[["name", "coef", "t", "p", "R2", "exp_sign", "sign_ok", "note"]]
    print(show.to_string(index=False))

    kept_pool, dropped = prune_collinear(feat, screen)
    if dropped:
        print("\nCollinearity pruning (|corr| > %.2f), dropped -> kept-instead:" % CORR_PRUNE_THRESH)
        for d, k in dropped.items():
            print(f"  {d:22} -> {k}")

    selected = select_model(screen, kept_pool)
    print("\nSelected compact European model:", selected)

    # In-sample fit of the selected European model + the core baseline.
    print("\n" + "=" * 78)
    print("In-sample HAC fit -- selected European model")
    print("=" * 78)
    eu_model, eu_train = fit_hac(feat, selected)
    print(eu_model.summary())

    print("\n" + "=" * 78)
    print("Out-of-sample walk-forward (H=1): core vs European vs benchmarks")
    print("=" * 78)
    model_cols = {"core": CORE_MODEL, "european": selected}
    res = walk_forward(feat, model_cols)

    full = summarize(res, ["core", "european"])
    print("\n-- Full backtest window (%s to %s) --"
          % (res["Date"].min().date(), res["Date"].max().date()))
    print(full.to_string(index=False))

    clean = res.loc[res["Date"] >= TRAIN_END]
    if len(clean) >= 6:
        post = summarize(clean, ["core", "european"])
        print("\n-- Clean post-%s slice (selection never saw this) --" % TRAIN_END[:7])
        print(post.to_string(index=False))
        print("\n(DM t vs RW > ~1.65 one-sided ~ 10%%, > ~1.96 ~ 5%%. "
              "Small n post-%s, so treat as indicative.)" % TRAIN_END[:7])
    else:
        print("\n(Too few post-%s observations for a separate clean slice.)" % TRAIN_END[:7])

    # Does the European block add anything beyond the core, out of sample?
    dm_eu_vs_core = dm_tstat(res["actual"], res["pred_core"], res["pred_european"])
    print("\nDM t-stat, European vs core (positive => European lowers MSE): %.2f" % dm_eu_vs_core)

    # Regime-split diagnostic: does the selected model's signal live only in the
    # 2021-22 crisis / high-volatility months? (descriptive, full-sample)
    regime_report(feat, selected)


if __name__ == "__main__":
    main()
# %%
