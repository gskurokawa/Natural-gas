#%%
"""
Gas_monthly_ML.py

Consolidated MACHINE-LEARNING / regularisation robustness checks for the
month-ahead (H=1) TTF study -- the umbrella paper's Section 7. It merges the two
scikit-learn scripts into one file and runs them in sequence:

  §7.3  RIDGE / ELASTIC-NET   [was ttf_ridge.py]
        L2 (and, as a secondary lens, elastic-net) shrinkage over the WHOLE
        candidate pool, with no manual selection -- does regularising the full
        pool find forecastable signal the hand-rolled OLS selection missed, or is
        the signal genuinely weak? The selected penalty lambda itself measures
        how much signal there is (lambda -> infinity collapses to the mean/RW).

  §7.4  PCR / PLS             [was ttf_pcr_standalone.py]
        Is there a low-dimensional LATENT-FACTOR structure in the ~55 collinear
        predictors that forecasts even when no individual predictor does?
          - PCR: PCA on standardized predictors, regress y on first k PCs (unsupervised).
          - PLS: components chosen to maximise covariance with y (supervised; fairer).

Both share ONE feature pipeline (defined once below): no-look-ahead deseasonalized
anomalies over the SAME four specifications -- the 2x2 of the headline results:
  1. EU deployable      -- all European predictors, LAGGED
  2. Global deployable  -- all EU + global predictors, LAGGED
  3. EU ceiling         -- EU fundamentals CONTEMPORANEOUS (perfect foresight), momentum lagged
  4. Global ceiling     -- EU + global FUNDAMENTALS contemporaneous; PRICES (JKM/HH/
                           spreads/VIX/FX) and momentum LAGGED (foresight is for quantities,
                           never prices).

Discipline (both models): z-score standardisation fit on the TRAINING window only
(inside a pipeline); the hyperparameter (ridge lambda / #components) chosen by
TimeSeriesSplit CV, re-selected each step of an expanding walk-forward (no
hyperparameter look-ahead); benchmarks are the random walk (0), the historical
mean, and the honest OLS core (momentum + storage_change + HDD, all lagged);
Diebold-Mariano tests vs the random walk and vs the core.

Run:  python Gas_monthly_ML.py   (runs the ridge check, then the PCR/PLS check)

Requirements: scikit-learn, numpy, pandas   (NO statsmodels needed)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV, LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# --------------------------------------------------------------------------- #
# Shared configuration
# --------------------------------------------------------------------------- #

DATA_PATH = "NG_m_final.csv"
SAMPLE_START = "2015-01-01"
TRAIN_END = "2025-01-01"
MIN_TRAIN = 60
HORIZON = 1
CV_SPLITS = 5

# Ridge / elastic-net grids
ALPHAS = np.logspace(-2, 5, 40)          # ridge lambda grid
L1_RATIOS = [0.1, 0.5, 0.9]              # elastic-net mix grid
# PCR / PLS grid
K_GRID = [1, 2, 3, 5, 8, 12]

SPECS = [("eu_deploy", "EU deployable (all lagged)"),
         ("global_deploy", "Global deployable (all lagged)"),
         ("eu_ceiling", "EU ceiling (fundamentals foreseen)"),
         ("global_ceiling", "Global ceiling (quantities foreseen, prices lagged)")]


# --------------------------------------------------------------------------- #
# Shared: data + no-look-ahead anomaly builders
# --------------------------------------------------------------------------- #

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["month"] = df["Date"].dt.month
    df["TTF"] = df["TTF(USD/mmbtu)"]
    # JKM: replace the 4-month (Jan-Apr 2017) constant fill at 7.86 with interpolation
    if "JKM(USD/mmbtu)" in df.columns:
        jkm = df["JKM(USD/mmbtu)"].astype(float).copy()
        jkm[np.isclose(jkm, 7.86)] = np.nan
        df["JKM(USD/mmbtu)"] = np.exp(np.log(jkm).interpolate(limit_direction="both"))
        df["ttf_jkm_logspread"] = np.log(df["TTF(USD/mmbtu)"]) - np.log(df["JKM(USD/mmbtu)"])
        df["ttf_hh_logspread"] = np.log(df["TTF(USD/mmbtu)"]) - np.log(df["HH(USD/mmbtu)"])
    return df.loc[df["Date"] >= SAMPLE_START].reset_index(drop=True)


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

# (name, source col, transform, block)  -- block in {"Own","EU","Global"}
CANDIDATES = [
    ("momentum",                 "TTF",                        "dlog_anom",  "Own"),
    # ---- European block ----
    ("storage_change_anom",      "EU+UK_av_storage(bcm)",      "change_anom", "EU"),
    ("storage_level_anom",       "EU+UK_av_storage(bcm)",      "level_anom",  "EU"),
    ("production_anom",          "EU+UK Production(bcm)",       "level_anom",  "EU"),
    ("net_piped_anom",           "EU+UK Net_piped(bcm)",       "level_anom",  "EU"),
    ("lng_imports_anom",         "EU+UK LNG imports",          "level_anom",  "EU"),
    ("net_supply_anom",          "EU+UK Net_supply",           "level_anom",  "EU"),
    ("total_demand_anom",        "EU+UK Total(bcm)",           "level_anom",  "EU"),
    ("nonpower_demand_anom",     "EU+UK Non_power(bcm)",        "level_anom",  "EU"),
    ("power_gas_demand_anom",    "EU+UK Electricity(bcm)",     "level_anom",  "EU"),
    ("residual_load_anom",       "EU+UK Residual load",        "level_anom",  "EU"),
    ("coal_gen_anom",            "EU+UK Coal",                 "level_anom",  "EU"),
    ("nuclear_gen_anom",         "EU+UK Nuclear",              "level_anom",  "EU"),
    ("hydro_gen_anom",           "EU+UK Hydro_gen",            "level_anom",  "EU"),
    ("gas_burn_anom",            "EU+UK Fossil gas",           "level_anom",  "EU"),
    ("production_chg_anom",      "EU+UK Production(bcm)",       "change_anom", "EU"),
    ("net_piped_chg_anom",       "EU+UK Net_piped(bcm)",       "change_anom", "EU"),
    ("lng_imports_chg_anom",     "EU+UK LNG imports",          "change_anom", "EU"),
    ("net_supply_chg_anom",      "EU+UK Net_supply",           "change_anom", "EU"),
    ("total_demand_chg_anom",    "EU+UK Total(bcm)",           "change_anom", "EU"),
    ("nonpower_demand_chg_anom", "EU+UK Non_power(bcm)",        "change_anom", "EU"),
    ("power_gas_demand_chg_anom","EU+UK Electricity(bcm)",     "change_anom", "EU"),
    ("residual_load_chg_anom",   "EU+UK Residual load",        "change_anom", "EU"),
    ("coal_gen_chg_anom",        "EU+UK Coal",                 "change_anom", "EU"),
    ("nuclear_gen_chg_anom",     "EU+UK Nuclear",              "change_anom", "EU"),
    ("hydro_gen_chg_anom",       "EU+UK Hydro_gen",            "change_anom", "EU"),
    ("gas_burn_chg_anom",        "EU+UK Fossil gas",           "change_anom", "EU"),
    ("norway_prod_anom",         "Norway_gas_prod",            "level_anom",  "EU"),
    ("norway_supplyred_anom",    "Norway_supply_red",          "level_anom",  "EU"),
    ("norway_planned_anom",      "Norway_planned_outage",      "level_anom",  "EU"),
    ("norway_unplanned_anom",    "Norway_unplanned_outage",    "level_anom",  "EU"),
    ("hdd_anom",                 "Europe_HDD",                 "level_anom",  "EU"),
    ("cdd_anom",                 "Europe_CDD",                 "level_anom",  "EU"),
    ("wind_anom",                "EU_wind_speed",              "level_anom",  "EU"),
    ("solar_anom",               "EU_solar",                   "level_anom",  "EU"),
    ("precip_anom",              "Nordic_precip",              "level_anom",  "EU"),
    # ---- Global block ----
    ("jkm_mom",                  "JKM(USD/mmbtu)",             "dlog_anom",   "Global"),
    ("hh_mom",                   "HH(USD/mmbtu)",              "dlog_anom",   "Global"),
    ("ttf_jkm_spread_anom",      "ttf_jkm_logspread",          "level_anom",  "Global"),
    ("ttf_hh_spread_anom",       "ttf_hh_logspread",           "level_anom",  "Global"),
    ("ttf_jkm_spread_chg_anom",  "ttf_jkm_logspread",          "change_anom", "Global"),
    ("us_gwdd_anom",             "US_GWDD",                    "level_anom",  "Global"),
    ("neasia_gwdd_anom",         "NE_Asia_GWDD",               "level_anom",  "Global"),
    ("atlantic_ace_anom",        "Atlantic_ACE",               "level_anom",  "Global"),
    ("gulf_storm_anom",          "Gulf_storm_days",            "level_anom",  "Global"),
    ("global_lng_offline_anom",  "Global LNG capacity offline","level_anom",  "Global"),
    ("global_lng_offline_chg_anom","Global LNG capacity offline","change_anom","Global"),
    ("global_lng_capacity_chg_anom","Global LNG nameplate capacity","change_anom","Global"),
    ("asia_imports_chg_anom",    "CH+JP+KR LNG imports",       "change_anom", "Global"),
    ("india_imports_chg_anom",   "IN LNG imports",             "change_anom", "Global"),
    ("qaus_exports_chg_anom",    "QA+AU+US LNG exports",       "change_anom", "Global"),
    ("seasia_exports_chg_anom",  "ID+MY+BN LNG exports",       "change_anom", "Global"),
    ("nigeria_exports_chg_anom", "NG LNG exports",             "change_anom", "Global"),
    ("vix_anom",                 "VIX",                        "level_anom",  "Global"),
    ("fx_ret_anom",              "USD-EUR_FX",                 "dlog_anom",   "Global"),
]

# Prices / financial / own-return: NEVER given foresight (always lagged, even in ceilings)
ALWAYS_LAG = {"momentum", "jkm_mom", "hh_mom", "ttf_jkm_spread_anom",
              "ttf_hh_spread_anom", "ttf_jkm_spread_chg_anom", "vix_anom", "fx_ret_anom"}

CORE = ["momentum__lag", "storage_change_anom__lag", "hdd_anom__lag"]


def build_features(df, horizon=HORIZON):
    """No-look-ahead anomalies; every predictor gets a __lag column and (unless a
    price/financial/own-return term) a contemporaneous __now column."""
    pieces = {"Date": df["Date"], "y": np.log(df["TTF"]).diff(horizon)}
    for name, col, tr, block in CANDIDATES:
        if col not in df.columns:
            continue
        anom = TRANSFORMS[tr](df, col)
        pieces[name + "__lag"] = anom.shift(horizon)          # predetermined
        if name not in ALWAYS_LAG:
            pieces[name + "__now"] = anom                      # contemporaneous (foresight)
    return pd.DataFrame(pieces)          # built at once -> no fragmentation warning


def spec_cols(which):
    eu = [n for n, c, t, b in CANDIDATES if b in ("Own", "EU")]
    alln = [n for n, c, t, b in CANDIDATES]
    if which == "eu_deploy":
        return [n + "__lag" for n in eu]
    if which == "global_deploy":
        return [n + "__lag" for n in alln]
    if which == "eu_ceiling":
        return [(n + "__lag" if n in ALWAYS_LAG else n + "__now") for n in eu]
    if which == "global_ceiling":
        return [(n + "__lag" if n in ALWAYS_LAG else n + "__now") for n in alln]
    raise ValueError(which)


def dm_tstat(loss_worse, loss_better):
    d = np.asarray(loss_worse, float) - np.asarray(loss_better, float)
    d = d[~np.isnan(d)]
    n = len(d); s = d.std(ddof=1)
    return d.mean() / (s / np.sqrt(n)) if s > 0 else np.nan


def summarize(res, names):
    rw_err = res["actual"] - res["pred_rw"]
    out = []
    for nm in names:
        pred = res["pred_" + nm]; err = res["actual"] - pred
        out.append({
            "Model": nm,
            "RMSE": np.sqrt((err ** 2).mean()),
            "Hit%": np.nan if nm == "rw" else 100 * (np.sign(res["actual"]) == np.sign(pred)).mean(),
            "OOS_R2_vs_RW": np.nan if nm == "rw" else 1 - (err ** 2).sum() / (rw_err ** 2).sum(),
            "DM_vs_RW": np.nan if nm == "rw" else dm_tstat((res["actual"] - res["pred_rw"]) ** 2, err ** 2),
        })
    return pd.DataFrame(out)


# =========================================================================== #
# §7.3  RIDGE / ELASTIC-NET   (was ttf_ridge.py)
# =========================================================================== #

def ridge_walk_forward(feat, cols, min_train=MIN_TRAIN):
    keep = ["Date", "y"] + sorted(set(cols) | set(CORE))
    keep = [c for c in keep if c in feat.columns]
    core = [c for c in CORE if c in feat.columns]
    data = feat[keep].dropna().reset_index(drop=True)
    rows = []
    for i in range(min_train, len(data)):
        tr, te = data.iloc[:i], data.iloc[i:i + 1]
        ytr = tr["y"].values
        Xtr, Xte = tr[cols].values, te[cols].values
        nsp = int(min(CV_SPLITS, max(2, i // 12)))
        tscv = TimeSeriesSplit(n_splits=nsp)
        rid = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=tscv)).fit(Xtr, ytr)
        en = make_pipeline(StandardScaler(),
                           ElasticNetCV(l1_ratio=L1_RATIOS, alphas=ALPHAS, cv=tscv, max_iter=5000)
                           ).fit(Xtr, ytr)
        rec = {"Date": te["Date"].values[0], "actual": te["y"].values[0],
               "pred_rw": 0.0, "pred_hist": ytr.mean(),
               "pred_ridge": rid.predict(Xte)[0], "pred_enet": en.predict(Xte)[0],
               "alpha_ridge": rid.named_steps["ridgecv"].alpha_}
        if core:
            lr = LinearRegression().fit(tr[core].values, ytr)
            rec["pred_core"] = lr.predict(te[core].values)[0]
        else:
            rec["pred_core"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows), data


def ridge_top_coefs(feat, cols, k=6):
    """Full-training-window standardized ridge coefs + elastic-net sparsity."""
    d = feat[["y"] + cols].loc[feat["Date"] < TRAIN_END].dropna()
    X, y = d[cols].values, d["y"].values
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    rid = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=tscv)).fit(X, y)
    en = make_pipeline(StandardScaler(),
                       ElasticNetCV(l1_ratio=L1_RATIOS, alphas=ALPHAS, cv=tscv, max_iter=5000)).fit(X, y)
    rc = pd.Series(rid.named_steps["ridgecv"].coef_, index=cols).sort_values(key=np.abs, ascending=False)
    ec = pd.Series(en.named_steps["elasticnetcv"].coef_, index=cols)
    nz = ec[ec.abs() > 1e-8].sort_values(key=np.abs, ascending=False)
    return rid.named_steps["ridgecv"].alpha_, rc.head(k), len(nz), nz.head(k)


def run_ridge():
    print("\n\n" + "#" * 78)
    print("# §7.3  RIDGE / ELASTIC-NET robustness check")
    print("#" * 78)
    df = load_data()
    feat = build_features(df)
    print("Ridge / elastic-net robustness check.  Sample %s..%s (n=%d rows before dropna)."
          % (feat["Date"].min().date(), feat["Date"].max().date(), len(feat)))
    print("lambda by TimeSeriesSplit CV; z-score on train only; benchmarks RW / hist-mean / OLS core.\n")

    summary_rows = []
    for key, label in SPECS:
        cols = [c for c in spec_cols(key) if c in feat.columns]
        print("=" * 78)
        print("%s   [%d predictors]" % (label, len(cols)))
        print("=" * 78)
        res, data = ridge_walk_forward(feat, cols)
        print("OOS window %s..%s  (n=%d)"
              % (pd.Timestamp(res["Date"].min()).date(), pd.Timestamp(res["Date"].max()).date(), len(res)))
        tab = summarize(res, ["ridge", "enet", "core", "hist", "rw"])
        print(tab.to_string(index=False, float_format=lambda x: "%.3f" % x))
        print("DM ridge vs core (positive => ridge better): %.2f"
              % dm_tstat((res["actual"] - res["pred_core"]) ** 2, (res["actual"] - res["pred_ridge"]) ** 2))
        amed = np.nanmedian(res["alpha_ridge"])
        print("Selected ridge lambda: median=%.3g  (min=%.3g, max=%.3g)  -- large => little signal"
              % (amed, np.nanmin(res["alpha_ridge"]), np.nanmax(res["alpha_ridge"])))
        alpha_ft, rc, n_nz, ec = ridge_top_coefs(feat, cols)
        print("Full-train ridge, top standardized coefficients:")
        for nm, v in rc.items():
            print("   %-30s %+.4f" % (nm, v))
        print("Elastic-net keeps %d/%d predictors; top nonzero:" % (n_nz, len(cols)))
        for nm, v in ec.items():
            print("   %-30s %+.4f" % (nm, v))
        r = tab.set_index("Model")
        summary_rows.append({"spec": label,
                             "ridge_OOS_R2": r.loc["ridge", "OOS_R2_vs_RW"],
                             "ridge_DM_RW": r.loc["ridge", "DM_vs_RW"],
                             "core_OOS_R2": r.loc["core", "OOS_R2_vs_RW"],
                             "lambda_med": amed})
        print()

    print("=" * 78)
    print("CROSS-SPEC SUMMARY (does regularizing the full pool beat RW / the hand-picked core?)")
    print("=" * 78)
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\nRead: ridge OOS R2 vs RW near/below 0 and a large lambda => the full pool holds no")
    print("more month-ahead signal than the parsimonious core; confirms weak signal, not crude selection.")


# =========================================================================== #
# §7.4  PCR / PLS   (was ttf_pcr_standalone.py)
# =========================================================================== #

def _cv(i):
    return TimeSeriesSplit(n_splits=int(min(CV_SPLITS, max(2, i // 12))))


def _fit_pcr(X, y, tscv, kmax):
    grid = [k for k in K_GRID if k <= kmax]
    pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA()), ("lr", LinearRegression())])
    gs = GridSearchCV(pipe, {"pca__n_components": grid}, cv=tscv,
                      scoring="neg_mean_squared_error").fit(X, y)
    return gs.best_estimator_, gs.best_params_["pca__n_components"]


def _fit_pls(X, y, tscv, kmax):
    grid = [k for k in K_GRID if k <= kmax]
    gs = GridSearchCV(PLSRegression(scale=True), {"n_components": grid}, cv=tscv,
                      scoring="neg_mean_squared_error").fit(X, y)
    return gs.best_estimator_, gs.best_params_["n_components"]


def pcr_walk_forward(feat, cols, min_train=MIN_TRAIN):
    keep = ["Date", "y"] + sorted(set(cols) | set(CORE))
    keep = [c for c in keep if c in feat.columns]
    core = [c for c in CORE if c in feat.columns]
    data = feat[keep].dropna().reset_index(drop=True)
    rows = []
    for i in range(min_train, len(data)):
        tr, te = data.iloc[:i], data.iloc[i:i + 1]
        ytr = tr["y"].values
        Xtr, Xte = tr[cols].values, te[cols].values
        tscv = _cv(i)
        kmax = min(len(cols), i - 2)
        pcr, k_pcr = _fit_pcr(Xtr, ytr, tscv, kmax)
        pls, k_pls = _fit_pls(Xtr, ytr, tscv, kmax)
        rec = {"Date": te["Date"].values[0], "actual": te["y"].values[0],
               "pred_rw": 0.0, "pred_hist": ytr.mean(),
               "pred_pcr": float(np.ravel(pcr.predict(Xte))[0]),
               "pred_pls": float(np.ravel(pls.predict(Xte))[0]),
               "k_pcr": k_pcr, "k_pls": k_pls}
        rec["pred_core"] = (LinearRegression().fit(tr[core].values, ytr)
                            .predict(te[core].values)[0]) if core else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def pc_variance(feat, cols, k=5):
    d = feat[["y"] + cols].loc[feat["Date"] < TRAIN_END].dropna()
    Xs = StandardScaler().fit_transform(d[cols].values)
    pca = PCA().fit(Xs)
    return np.cumsum(pca.explained_variance_ratio_)[:k]


def run_pcr():
    print("\n\n" + "#" * 78)
    print("# §7.4  PCR / PLS dimension-reduction check")
    print("#" * 78)
    df = load_data()
    feat = build_features(df)
    print("PCR / PLS dimension-reduction check.  Sample %s..%s.  k by TimeSeriesSplit CV.\n"
          % (feat["Date"].min().date(), feat["Date"].max().date()))
    summary = []
    for key, label in SPECS:
        cols = [c for c in spec_cols(key) if c in feat.columns]
        print("=" * 78)
        print("%s   [%d predictors]" % (label, len(cols)))
        print("=" * 78)
        res = pcr_walk_forward(feat, cols)
        print("OOS window %s..%s  (n=%d)"
              % (pd.Timestamp(res["Date"].min()).date(), pd.Timestamp(res["Date"].max()).date(), len(res)))
        tab = summarize(res, ["pcr", "pls", "core", "hist", "rw"])
        print(tab.to_string(index=False, float_format=lambda x: "%.3f" % x))
        print("DM PCR vs core (positive => PCR better): %.2f"
              % dm_tstat((res["actual"] - res["pred_core"]) ** 2, (res["actual"] - res["pred_pcr"]) ** 2))
        print("DM PLS vs core (positive => PLS better): %.2f"
              % dm_tstat((res["actual"] - res["pred_core"]) ** 2, (res["actual"] - res["pred_pls"]) ** 2))
        print("Selected components: PCR median k=%.0f (range %d-%d);  PLS median k=%.0f (range %d-%d)"
              % (np.median(res["k_pcr"]), res["k_pcr"].min(), res["k_pcr"].max(),
                 np.median(res["k_pls"]), res["k_pls"].min(), res["k_pls"].max()))
        cv = pc_variance(feat, cols)
        print("Cumulative PC variance explained (train): " + ", ".join(
              "PC1-%d=%.0f%%" % (j + 1, 100 * v) for j, v in enumerate(cv)))
        r = tab.set_index("Model")
        summary.append({"spec": label, "PCR_OOS_R2": r.loc["pcr", "OOS_R2_vs_RW"],
                        "PLS_OOS_R2": r.loc["pls", "OOS_R2_vs_RW"],
                        "core_OOS_R2": r.loc["core", "OOS_R2_vs_RW"],
                        "PLS_DM_RW": r.loc["pls", "DM_vs_RW"]})
        print()
    print("=" * 78)
    print("CROSS-SPEC SUMMARY (do latent factors beat RW / the hand-picked core?)")
    print("=" * 78)
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\nRead: PCR/PLS OOS R2 near/below the core and few selected components => no")
    print("exploitable low-dimensional factor structure beyond what storage+momentum already give.")


# =========================================================================== #
# Master entry point
# =========================================================================== #

def main():
    run_ridge()   # §7.3
    run_pcr()     # §7.4


if __name__ == "__main__":
    main()
# %%
