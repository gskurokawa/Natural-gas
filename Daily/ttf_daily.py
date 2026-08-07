"""
ttf_paper_code.py
=============================================================================
Unified replication code for:

  "Perfect Foresight and the Short-Horizon Predictability of TTF Natural Gas
   Prices: Evidence on Level and Volatility"

Running this file reproduces every empirical result in the paper from the
single input NG_daily10.csv (built separately by build_ng_daily_dataset.py):

  Section 3.3  Stationarity (Augmented Dickey-Fuller screen)
  Section 4    Level under European fundamentals   -> Tables 1 and 2
  Section 5    Level under global fundamentals      -> Table 3 (+ diagnostics)
  Section 6    Volatility                           -> Table 4
  Section 7    Machine-learning robustness          -> Table 5

Design common to all studies: 7 calendar-day horizon, fundamentals supplied
under perfect foresight (dated at the target date / target week), train through
2025-02-28 and test thereafter, random-walk (or RW-vol) benchmark, Diebold-
Mariano tests. The mean equation is an autoregressive distributed-lag / error-
correction form in levels (d=0); log TTF is I(1) and its 7-day lag makes the
regression balanced. GARCH models are estimated by scipy MLE; the ML section
uses scikit-learn (gradient boosting stands in for XGBoost).

Dependencies: numpy, pandas, scipy, scikit-learn.
Usage:  python ttf_paper_code.py        # expects NG_daily10.csv in the cwd
=============================================================================
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

warnings.filterwarnings("ignore")

DATA_PATH = "NG_daily10.csv"
H = 7
SPLIT_DATE = pd.Timestamp("2025-02-28")


# =========================================================================== #
# Shared helpers
# =========================================================================== #
def load_df():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    df["logTTF"] = np.log(df["TTF(USD/mmbtu)"])
    return df


def lag_cal(df_index, s, days=H):
    """Value `days` calendar days earlier, aligned to df_index."""
    o = s.reindex(df_index - pd.Timedelta(days=days))
    o.index = df_index
    return o


def dm_stat(err_bench, err_model, L=H):
    """Diebold-Mariano on squared-error differential (>0 => model beats bench)."""
    dd = err_bench ** 2 - err_model ** 2
    n = len(dd); m = dd.mean(); s = np.var(dd)
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * np.mean((dd[l:] - m) * (dd[:-l] - m))
    return float(m / np.sqrt(s / n)) if s > 0 else 0.0


def level_scores(logp_hat, logp_act, anchor):
    """RMSE (price), skill vs RW, directional accuracy, DM. Level studies."""
    rmse = lambda u, v: float(np.sqrt(np.mean((np.exp(u) - np.exp(v)) ** 2)))
    e_m, e_rw = logp_hat - logp_act, anchor - logp_act
    skill = float(1 - np.mean(e_m ** 2) / np.mean(e_rw ** 2))
    dir_acc = float(np.mean(np.sign(logp_hat - anchor) == np.sign(logp_act - anchor)))
    return {"rmse": rmse(logp_hat, logp_act), "rmse_rw": rmse(anchor, logp_act),
            "skill": skill, "dir": dir_acc, "dm": dm_stat(e_rw, e_m)}


def adf_t(y, p=5):
    """Augmented Dickey-Fuller t-statistic on the level coefficient (constant, no trend)."""
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    dy = np.diff(y); n = len(dy)
    rows = n - p
    ylag = y[p:n]
    X = [np.ones(rows), ylag] + [dy[p - i:n - i] for i in range(1, p + 1)]
    X = np.column_stack(X); Y = dy[p:]
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ b; s2 = resid @ resid / (rows - X.shape[1])
    se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    return b[1] / se[1]


# =========================================================================== #
# Section 3.3  Stationarity screen
# =========================================================================== #
def stationarity_screen(df):
    print("\n" + "=" * 70)
    print("Section 3.3  Stationarity (ADF, constant, 5% critical value = -2.86)")
    print("=" * 70)
    series = {
        "log TTF": df["logTTF"], "Storage": df["Storage(TWh)"],
        "LNG sendout": df["LNG_sendout(GWh/d)"], "Europe HDD": df["Europe_HDD"],
        "Norway outages": df["Unplanned_norway(mcm/d)"], "US GWDD": df["US_GWDD"],
        "NE-Asia GWDD": df["NE_Asia_GWDD"], "VIX": df["VIX"], "EUR/USD": df["USD-EUR"],
        "Atlantic ACE": df["Atlantic_ACE"], "Gulf storm": df["gulf_storm_present"],
        "DE power price": df["DE_price(EUR/MWh)"],
    }
    print(f"{'series':16s} {'ADF t':>8s}   integration order")
    print("-" * 46)
    for nm, s in series.items():
        t = adf_t(s.values)
        print(f"{nm:16s} {t:8.2f}   {'I(0) stationary' if t < -2.86 else 'I(1) unit root'}")


# =========================================================================== #
# Section 4  Level under European fundamentals  (Tables 1 & 2)
# =========================================================================== #
EU_COLS = {"TTF(USD/mmbtu)": "TTF", "Storage(TWh)": "Storage",
           "LNG_sendout(GWh/d)": "LNG", "Europe_HDD": "HDD",
           "Unplanned_norway(mcm/d)": "Norway"}
EU_SPECS = {"Random walk (benchmark)": [], "M1: Storage+Norway": ["Storage", "Norway"],
            "M2: all fundamentals": ["Storage", "LNG", "HDD", "Norway"]}


def _eu_ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(X1, y, rcond=None); return b


def _eu_nw(X, y, b, L=H):
    X1 = np.column_stack([np.ones(len(X)), X]); e = y - X1 @ b
    S = (X1 * e[:, None]).T @ (X1 * e[:, None])
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = (X1[l:] * e[l:, None]).T @ (X1[:-l] * e[:-l, None]); S += w * (G + G.T)
    XtXi = np.linalg.inv(X1.T @ X1); return np.sqrt(np.diag(XtXi @ S @ XtXi))


def european_study(df):
    d = df[["logTTF"] + list(EU_COLS)].rename(columns=EU_COLS)
    d = d.assign(anchor=lag_cal(d.index, d["logTTF"])).dropna(
        subset=["logTTF", "anchor"] + list(EU_COLS.values()))
    trm = d.index <= SPLIT_DATE
    tr, te = d[trm], d[~trm]
    print("\n" + "=" * 70)
    print(f"Section 4  European fundamentals | train {len(tr)}  test {len(te)}")
    print("=" * 70)
    coeffs = {}
    print(f"\nTable 1.  {'model':24s} {'RMSE':>7s} {'skill':>8s} {'dir':>6s} {'DM':>7s}")
    print("-" * 60)
    for name, cols in EU_SPECS.items():
        if not cols:
            r = level_scores(te["anchor"].values, te["logTTF"].values, te["anchor"].values)
            print(f"{'':10s}{name:24s} {r['rmse_rw']:7.3f} {0.0:+8.3f} {'  -  ':>6s} {0.0:+7.2f}")
            continue
        Xtr = np.column_stack([tr[cols].values, tr["anchor"].values])
        Xte = np.column_stack([te[cols].values, te["anchor"].values])
        b = _eu_ols(Xtr, tr["logTTF"].values)
        se = _eu_nw(Xtr, tr["logTTF"].values, b)
        yhat = b[0] + Xte @ b[1:]
        r = level_scores(yhat, te["logTTF"].values, te["anchor"].values)
        print(f"{'':10s}{name:24s} {r['rmse']:7.3f} {r['skill']:+8.3f} {r['dir']:>6.2f} {r['dm']:+7.2f}")
        coeffs[name] = (["const"] + cols + ["anchor"], b, b / se)
    print("\nTable 2.  Coefficients (Newey-West L=7):")
    for name, (labs, b, t) in coeffs.items():
        print(f"  {name}")
        for lab, bb, tt in zip(labs, b, t):
            print(f"    {lab:12s} {bb:+.6f}  t={tt:+.2f}")


# =========================================================================== #
# Section 5  Level under global fundamentals  (Table 3 + diagnostics)
# =========================================================================== #
GLOBAL = {"US_GWDD": "US_GWDD", "NEAsia_GWDD": "NE_Asia_GWDD", "LNG": "LNG_sendout(GWh/d)",
          "VIX": "VIX", "USDEUR": "USD-EUR", "ACE": "Atlantic_ACE", "GulfStorm": "gulf_storm_present"}


def _gl_build(df):
    d = pd.DataFrame({"logTTF": df["logTTF"]})
    for k, c in GLOBAL.items():
        d[k] = df[c]
    d = d.ffill()
    d["anchor"] = lag_cal(d.index, d["logTTF"])
    return d.dropna()


def _gl_design(d):
    cols = list(GLOBAL); trm = np.asarray(d.index <= SPLIT_DATE)
    mu, sd = d.loc[d.index <= SPLIT_DATE, cols].mean(), d.loc[d.index <= SPLIT_DATE, cols].std()
    am, asd = d.loc[d.index <= SPLIT_DATE, "anchor"].mean(), d.loc[d.index <= SPLIT_DATE, "anchor"].std()
    Z = ((d[cols] - mu) / sd).values
    Za = ((d["anchor"] - am) / asd).values.reshape(-1, 1)
    X = np.column_stack([np.ones(len(d)), Z, Za])
    return X, d["logTTF"].values, d["anchor"].values, trm, cols


def _gl_ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); return b


def _gl_nw(X, y, b, L=H):
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None]); S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X); return np.sqrt(np.diag(XtXi @ S @ XtXi))


def _gl_scores(logp_hat, y_te, anchor_te):
    return level_scores(logp_hat, y_te, anchor_te)


def _gl_origin(dates):
    idx = pd.Series(np.arange(len(dates)), index=dates); out = np.full(len(dates), -1)
    for i, dt in enumerate(dates):
        j = idx.get(dt - pd.Timedelta(days=H), None)
        out[i] = int(j) if j is not None else (i - 5 if i >= 5 else -1)
    return out


def _gl_sarimax(X, y, anchor, trm, cols):
    b = _gl_ols(X[trm], y[trm]); se = _gl_nw(X[trm], y[trm], b)
    yhat = X[~trm] @ b
    r = _gl_scores(yhat, y[~trm], anchor[~trm])
    r["coef"] = list(zip(["const"] + cols + ["anchor"], b, b / se))
    return r


def _gl_ardl(d):
    cols = list(GLOBAL); trm0 = d.index <= SPLIT_DATE
    mu, sd = d.loc[trm0, cols].mean(), d.loc[trm0, cols].std()
    Z = (d[cols] - mu) / sd
    dd = pd.DataFrame({"logTTF": d["logTTF"], "anchor": d["anchor"]}, index=d.index)
    dd[cols] = Z
    blr = _gl_ols(np.column_stack([np.ones(trm0.sum()), Z[trm0].values]), d["logTTF"][trm0].values)
    dd["ECT"] = d["logTTF"].values - (blr[0] + Z.values @ blr[1:])
    dd["ECT_o"] = lag_cal(dd.index, dd["ECT"])
    for c in cols:
        dd["d7_" + c] = dd[c] - lag_cal(dd.index, dd[c])
    dd["d7"] = dd["logTTF"] - dd["anchor"]
    dd = dd.dropna(subset=["ECT_o", "d7"] + ["d7_" + c for c in cols])
    trm = np.asarray(dd.index <= SPLIT_DATE)
    Xd = np.column_stack([np.ones(len(dd)), dd["ECT_o"].values, dd[["d7_" + c for c in cols]].values])
    bec = _gl_ols(Xd[trm], dd["d7"].values[trm])
    yhat = dd["anchor"].values[~trm] + Xd[~trm] @ bec
    r = _gl_scores(yhat, dd["logTTF"].values[~trm], dd["anchor"].values[~trm])
    r["ect_speed"] = float(bec[1])
    return r


def _gl_tvp(X, y, anchor, trm, dates):
    k = X.shape[1]

    def kalman(q, keep=False):
        b = _gl_ols(X[trm], y[trm]); res = y[trm] - X[trm] @ b
        Hobs = res @ res / (trm.sum() - k)
        P = np.eye(k) * 10.0; Q = np.eye(k) * q * Hobs
        betas = np.zeros((len(dates), k)); ll = 0.0
        for t in range(len(dates)):
            Pp = P + Q; xt = X[t]
            v = y[t] - xt @ b; F = xt @ Pp @ xt + Hobs
            if trm[t]:
                ll += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
            K = Pp @ xt / F; b = b + K * v; P = Pp - np.outer(K, xt @ Pp)
            betas[t] = b
        return (betas, q) if keep else ll

    qs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    qbest = qs[int(np.argmax([kalman(q) for q in qs]))]
    betas, _ = kalman(qbest, keep=True)
    origin = _gl_origin(dates)
    yhat = np.array([X[t] @ betas[origin[t]] if origin[t] >= 0 else np.nan for t in range(len(dates))])
    te = (~trm) & np.isfinite(yhat)
    r = _gl_scores(yhat[te], y[te], anchor[te]); r["q"] = qbest
    return r


def _gl_ms(X, y, anchor, trm, dates, iters=100):
    Xtr, ytr = X[trm], y[trm]; n = len(ytr); k = X.shape[1]
    npdf = lambda v, s: np.exp(-0.5 * (v / s) ** 2) / (s * np.sqrt(2 * np.pi))
    b0 = _gl_ols(Xtr, ytr); s0 = (ytr - Xtr @ b0).std()
    beta = [b0.copy(), b0.copy()]; sig = [0.6 * s0, 1.8 * s0]
    P = np.array([[0.97, 0.03], [0.06, 0.94]]); pi0 = np.array([0.5, 0.5])
    for _ in range(iters):
        dens = np.column_stack([npdf(ytr - Xtr @ beta[s], sig[s]) for s in range(2)])
        filt = np.zeros((n, 2)); pred = np.zeros((n, 2)); prev = pi0
        for t in range(n):
            pr = prev @ P if t > 0 else pi0
            num = pr * dens[t]; filt[t] = num / (num.sum() + 1e-300); pred[t] = pr; prev = filt[t]
        sm = np.zeros((n, 2)); sm[-1] = filt[-1]; xi = np.zeros((n, 2, 2))
        for t in range(n - 2, -1, -1):
            for i in range(2):
                for j in range(2):
                    xi[t + 1, i, j] = filt[t, i] * P[i, j] * sm[t + 1, j] / (pred[t + 1, j] + 1e-300)
            sm[t] = xi[t + 1].sum(axis=1)
        for s in range(2):
            w = sm[:, s]
            beta[s] = np.linalg.solve(Xtr.T @ (w[:, None] * Xtr), Xtr.T @ (w * ytr))
            res = ytr - Xtr @ beta[s]; sig[s] = np.sqrt((w * res * res).sum() / w.sum())
        num = xi[1:].sum(axis=0); P = num / num.sum(axis=1, keepdims=True); pi0 = sm[0]
    dens = np.column_stack([npdf(y - X @ beta[s], sig[s]) for s in range(2)])
    filt = np.zeros((len(dates), 2)); prev = pi0
    for t in range(len(dates)):
        pr = prev @ P if t > 0 else pi0
        num = pr * dens[t]; filt[t] = num / (num.sum() + 1e-300); prev = filt[t]
    origin = _gl_origin(dates); Ph = np.linalg.matrix_power(P, 5)
    yhat = np.full(len(dates), np.nan)
    for t in range(len(dates)):
        if origin[t] >= 0:
            prob = filt[origin[t]] @ Ph
            yhat[t] = prob[0] * (X[t] @ beta[0]) + prob[1] * (X[t] @ beta[1])
    te = (~trm) & np.isfinite(yhat)
    r = _gl_scores(yhat[te], y[te], anchor[te])
    r["sig"] = sorted(sig); r["P_diag"] = (float(P[0, 0]), float(P[1, 1]))
    return r


def global_study(df):
    d = _gl_build(df)
    X, y, anchor, trm, cols = _gl_design(d)
    dates = d.index
    m1 = _gl_sarimax(X, y, anchor, trm, cols)
    m2 = _gl_ardl(d)
    m3 = _gl_tvp(X, y, anchor, trm, dates)
    m4 = _gl_ms(X, y, anchor, trm, dates)
    print("\n" + "=" * 70)
    print(f"Section 5  Global fundamentals | train {trm.sum()}  test {(~trm).sum()}")
    print("=" * 70)
    print(f"\nTable 3.  {'model':22s} {'RMSE':>7s} {'skill':>8s} {'dir':>6s} {'DM':>7s}")
    print("-" * 56)
    rw = {"rmse": m1["rmse_rw"], "skill": 0.0, "dir": float("nan"), "dm": 0.0}
    for name, r in [("Random walk", rw), ("M1 ADL (global)", m1), ("M2 ARDL-ECM", m2),
                    ("M3 TVP / Kalman", m3), ("M4 Markov-switching", m4)]:
        ds = "  -  " if np.isnan(r["dir"]) else f"{r['dir']:.2f}"
        print(f"{'':10s}{name:22s} {r['rmse']:7.3f} {r['skill']:+8.3f} {ds:>6s} {r['dm']:+7.2f}")
    print("\nDiagnostics:")
    print("  M1 coefficients (Newey-West L=7):")
    for nm, b, t in m1["coef"]:
        print(f"    {nm:12s} {b:+.5f}  t={t:+.2f}")
    print(f"  M2 ARDL-ECM error-correction speed: {m2['ect_speed']:+.4f}")
    print(f"  M3 TVP signal-to-noise q = {m3['q']:g}")
    print(f"  M4 Markov regimes: sigma_low={m4['sig'][0]:.3f}, sigma_high={m4['sig'][1]:.3f}; "
          f"P_diag={m4['P_diag'][0]:.2f},{m4['P_diag'][1]:.2f}")


# =========================================================================== #
# Section 6  Volatility  (Table 4)
# =========================================================================== #
WEEK = 5
SCALE = 100.0


def _v_nll(theta, ret, Xm=None):
    w, a, b = theta[:3]; g = theta[3:]
    if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    n = len(ret); s2 = np.empty(n); s2[0] = np.var(ret[:60]); ll = 0.0
    for t in range(1, n):
        ex = 0.0 if Xm is None else g @ Xm[t]
        s2[t] = max(w + a * ret[t - 1] ** 2 + b * s2[t - 1] + ex, 1e-8)
        ll += 0.5 * (np.log(2 * np.pi * s2[t]) + ret[t] ** 2 / s2[t])
    return ll


def _v_fit(ret, Xm=None, k=0):
    v = np.var(ret); best = None
    for a0, b0 in [(0.08, 0.9), (0.05, 0.93), (0.1, 0.85)]:
        x0 = [v * (1 - a0 - b0), a0, b0] + [0.0] * k
        res = minimize(_v_nll, x0, args=(ret, Xm), method="L-BFGS-B",
                       bounds=[(1e-8, None), (0, 1), (0, 1)] + [(None, None)] * k)
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


def _v_filter(theta, ret, Xm=None):
    w, a, b = theta[:3]; g = theta[3:]; n = len(ret); s2 = np.empty(n); s2[0] = np.var(ret[:60])
    for t in range(1, n):
        ex = 0.0 if Xm is None else g @ Xm[t]
        s2[t] = max(w + a * ret[t - 1] ** 2 + b * s2[t - 1] + ex, 1e-8)
    return s2


def _v_week(theta, s2o, Xf=None):
    w, a, b = theta[:3]; g = theta[3:]; ab = a + b; e = s2o; tot = 0.0
    for k in range(WEEK):
        ex = 0.0 if Xf is None else g @ Xf[k]
        e = w + ex + ab * e; tot += e
    return tot


def _v_ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(X1, y, rcond=None); return b


def _v_scores(lyhat, lyact, lybench):
    rmse = float(np.sqrt(np.mean((lyhat - lyact) ** 2)))
    skill = float(1 - np.mean((lyhat - lyact) ** 2) / np.mean((lybench - lyact) ** 2))
    Vh, Va = np.exp(lyhat), np.exp(lyact)
    qlike = float(np.mean(Va / Vh - np.log(Va / Vh) - 1))
    dir_acc = float(np.mean(np.sign(lyhat - lybench) == np.sign(lyact - lybench)))
    return {"rmse": rmse, "skill": skill, "qlike": qlike, "dir": dir_acc,
            "dm": dm_stat(lybench - lyact, lyhat - lyact, L=WEEK + 2)}


def volatility_study(df):
    d = df.copy()
    d["r"] = np.log(d["TTF(USD/mmbtu)"]).diff() * SCALE
    r2 = d["r"] ** 2; v = r2.values
    fwd = np.full(len(v), np.nan)
    for i in range(len(v) - WEEK):
        fwd[i] = np.nansum(v[i + 1:i + 1 + WEEK])
    d["RVar_fwd"] = fwd; d["RVar_d"] = r2
    d["RVar_w"] = r2.rolling(WEEK).sum(); d["RVar_m"] = r2.rolling(22).mean() * WEEK
    for a, b in [("RVar_fwd", "lVar_fwd"), ("RVar_d", "lVar_d"), ("RVar_w", "lVar_w"), ("RVar_m", "lVar_m")]:
        d[b] = np.log(d[a].clip(lower=1e-8))

    def fagg(col, how="mean"):
        s = d[col].values; out = np.full(len(s), np.nan)
        for i in range(len(s) - WEEK):
            wk = s[i + 1:i + 1 + WEEK]; out[i] = np.nanmean(wk) if how == "mean" else np.nanmax(wk)
        return out

    d["VIXf"] = fagg("VIX"); d["NORf"] = fagg("Unplanned_norway(mcm/d)", "max"); d["HDDf"] = fagg("Europe_HDD")
    dates = d.index; ret = d["r"].fillna(0).values
    tr_end = int(np.searchsorted(dates.values, np.datetime64(SPLIT_DATE), "right"))
    g11, ll11 = _v_fit(ret[:tr_end])
    vix = d["VIX"].values
    vixz = np.nan_to_num((vix - np.nanmean(vix[:tr_end])) / np.nanstd(vix[:tr_end])).reshape(-1, 1)
    gx, llx = _v_fit(ret[:tr_end], vixz[:tr_end], k=1)
    s11 = _v_filter(g11, ret); sx = _v_filter(gx, ret, vixz)
    fc11 = np.full(len(dates), np.nan); fcx = np.full(len(dates), np.nan)
    for t in range(len(dates) - WEEK):
        fc11[t] = _v_week(g11, s11[t]); fcx[t] = _v_week(gx, sx[t], [vixz[t + 1 + k] for k in range(WEEK)])
    d["fc11"] = fc11; d["fcx"] = fcx
    use = d.dropna(subset=["lVar_fwd", "lVar_d", "lVar_w", "lVar_m", "VIXf", "NORf", "HDDf", "fc11", "fcx"]).copy()
    trm = use.index <= SPLIT_DATE; tr, te = use[trm], use[~trm]
    a = te["lVar_fwd"].values; bench = te["lVar_w"].values

    def reg(cols):
        b = _v_ols(tr[cols].values, tr["lVar_fwd"].values)
        return b[0] + te[cols].values @ b[1:]

    rows = [("RW-vol (persistence)", _v_scores(bench, a, bench)),
            ("GARCH(1,1)", _v_scores(np.log(te["fc11"].clip(lower=1e-8).values), a, bench)),
            ("GARCH-X (VIX)", _v_scores(np.log(te["fcx"].clip(lower=1e-8).values), a, bench)),
            ("HAR (d+w+m)", _v_scores(reg(["lVar_d", "lVar_w", "lVar_m"]), a, bench)),
            ("HAR-X (+VIX)", _v_scores(reg(["lVar_d", "lVar_w", "lVar_m", "VIXf"]), a, bench)),
            ("HAR-X (+VIX+Nor+HDD)", _v_scores(reg(["lVar_d", "lVar_w", "lVar_m", "VIXf", "NORf", "HDDf"]), a, bench))]
    print("\n" + "=" * 70)
    print(f"Section 6  Volatility | train {len(tr)}  test {len(te)}")
    print("=" * 70)
    print(f"  GARCH(1,1) a+b={g11[1] + g11[2]:.3f}  GARCH-X VIX coef={gx[3]:+.3f}  "
          f"LR 2dLL={2 * (ll11 - llx):.2f} (chi2(1) 5% = 3.84)")
    print(f"\nTable 4.  {'model':22s} {'RMSE':>6s} {'skill':>8s} {'QLIKE':>7s} {'dir':>6s} {'DM':>7s}")
    print("-" * 62)
    for name, r in rows:
        ds = "  -  " if name.startswith("RW") else f"{r['dir']:.2f}"
        print(f"{'':6s}{name:22s} {r['rmse']:6.3f} {r['skill']:+8.3f} {r['qlike']:7.3f} {ds:>6s} {r['dm']:+7.2f}")


# =========================================================================== #
# Section 7  Machine-learning robustness  (Table 5)
# =========================================================================== #
ML_EU = ["Storage(TWh)", "LNG_sendout(GWh/d)", "Europe_HDD", "Unplanned_norway(mcm/d)"]
ML_GL = ["US_GWDD", "NE_Asia_GWDD", "LNG_sendout(GWh/d)", "VIX", "USD-EUR", "Atlantic_ACE", "gulf_storm_present"]


def _ml_eval(name, r7hat, te, rows):
    anc = te["anchor"].values; act = te["logTTF"].values; rw = anc; ph = anc + r7hat
    rmse = lambda u, v: float(np.sqrt(np.mean((np.exp(u) - np.exp(v)) ** 2)))
    skill = 1 - np.mean((ph - act) ** 2) / np.mean((rw - act) ** 2)
    dacc = float(np.mean(np.sign(r7hat) == np.sign(te["r7"].values)))
    rows.append((name, rmse(ph, act), float(skill), dacc, dm_stat(rw - act, ph - act)))


def _ml_run(df, feats, label):
    cols = feats + ["anchor"]
    d = df.dropna(subset=["r7", "anchor", "logTTF"] + cols).copy()
    trm = d.index <= SPLIT_DATE; tr, te = d[trm], d[~trm]
    Xtr, ytr = tr[cols].values, tr["r7"].values; Xte = te[cols].values
    tscv = TimeSeriesSplit(n_splits=5, gap=5)
    rows = []
    _ml_eval("Random walk", np.zeros(len(te)), te, rows)
    _ml_eval("OLS", make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, ytr).predict(Xte), te, rows)
    rg = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(-2, 4, 13)},
                      cv=tscv, scoring="neg_mean_squared_error").fit(Xtr, ytr)
    _ml_eval(f"Ridge (a={rg.best_params_['ridge__alpha']:.0f})", rg.predict(Xte), te, rows)
    ls = GridSearchCV(make_pipeline(StandardScaler(), Lasso(max_iter=5000)), {"lasso__alpha": np.logspace(-4, 0, 12)},
                      cv=tscv, scoring="neg_mean_squared_error").fit(Xtr, ytr)
    nz = int(np.sum(ls.best_estimator_.named_steps["lasso"].coef_ != 0))
    _ml_eval(f"Lasso ({nz} feats kept)", ls.predict(Xte), te, rows)
    gb = GridSearchCV(GradientBoostingRegressor(subsample=0.8, min_samples_leaf=20, random_state=0),
                      {"max_depth": [2, 3], "n_estimators": [100, 300], "learning_rate": [0.03]},
                      cv=tscv, scoring="neg_mean_squared_error").fit(Xtr, ytr)
    _ml_eval("Gradient-boosted trees", gb.predict(Xte), te, rows)
    print(f"\n  {label}: {len(feats)} fundamentals + anchor | train {len(tr)} test {len(te)}")
    print(f"  {'model':24s} {'RMSE':>7s} {'skill':>8s} {'dir':>6s} {'DM':>7s}")
    print("  " + "-" * 54)
    for nm, rmse, sk, dr, dm in rows:
        ds = "  -  " if nm == "Random walk" else f"{dr:.2f}"
        print(f"  {nm:24s} {rmse:7.3f} {sk:+8.3f} {ds:>6s} {dm:+7.2f}")


def ml_study(df):
    d = df.copy()
    d["anchor"] = lag_cal(d.index, d["logTTF"]); d["r7"] = d["logTTF"] - d["anchor"]
    d = d.ffill()
    print("\n" + "=" * 70)
    print("Section 7  Machine-learning robustness (Table 5)")
    print("=" * 70)
    _ml_run(d, ML_GL, "GLOBAL")
    _ml_run(d, ML_EU, "EUROPEAN")
    _ml_run(d, sorted(set(ML_EU + ML_GL)), "COMBINED")


# =========================================================================== #
def main():
    df = load_df()
    stationarity_screen(df)
    european_study(df)
    global_study(df)
    volatility_study(df)
    ml_study(df)
    print("\nAll tables reproduced from", DATA_PATH)


if __name__ == "__main__":
    main()