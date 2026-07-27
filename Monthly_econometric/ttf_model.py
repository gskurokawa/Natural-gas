"""
ttf_model.py

An econometric model for forecasting month-ahead TTF (European natural
gas) prices from European supply, demand, and weather fundamentals.

Pipeline:
  1. Load monthly European gas market data.
  2. Engineer deseasonalized ("anomaly") predictors: momentum, storage
     change, and London heating-degree-days -- each compared only to its
     own expanding, prior-years-only calendar-month average, so no
     predictor ever uses future information.
  3. Stationarity-check every candidate series (ADF + KPSS). An LNG
     sendout variable was tested during development and excluded: both
     tests agreed it (and its deseasonalized anomaly) were non-stationary,
     and a stationary growth-rate version of it turned out to be
     statistically insignificant. It is kept out of the final model here;
     see `check_stationarity()` / `stationarity_report()` below if you
     want to re-run that check yourself, or re-add a revised LNG variable.
  4. Fit OLS with Newey-West HAC standard errors.
  5. Run regression diagnostics: Durbin-Watson, residual autocorrelation,
     Breusch-Pagan heteroskedasticity test.
  6. Backtest out-of-sample with an expanding-window walk-forward loop,
     benchmarked against a random walk, a momentum-only model, and the
     historical mean.
  7. Repeat steps 2-6 for 1-, 2-, and 3-month-ahead horizons.
  8. Produce the two summary charts used in the accompanying report.

This is a toy / educational model, not a trading or investment tool.

Requirements: pandas, numpy, statsmodels, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "NG_m_final.csv"
TRAIN_END = "2025-01-01"        # training window: everything before this date
MIN_TRAIN_OBS = 60              # minimum months of history before the walk-forward backtest starts


# --------------------------------------------------------------------------- #
# 1. Data loading and feature engineering
# --------------------------------------------------------------------------- #

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset and add the base date/month columns."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["month"] = df["Date"].dt.month
    df["TTF"] = df["TTF(USD/mmbtu)"]
    return df


def expanding_seasonal_mean(series: pd.Series, month: pd.Series) -> pd.Series:
    """
    Out-of-sample-safe calendar-month climatology.

    For each row, returns the mean of all PRIOR years' values for the same
    calendar month (the current observation is excluded). This is what makes
    the resulting anomaly safe to use as a predictor: at any point in time,
    it only reflects information that was actually available up to that
    point, never future data.
    """
    out = pd.Series(index=series.index, dtype=float)
    tmp = pd.DataFrame({"val": series, "month": month})
    for _, grp in tmp.groupby("month"):
        clim = grp["val"].expanding().mean().shift(1)
        out.loc[grp.index] = clim
    return out


def build_features(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Build the target and the three deseasonalized, horizon-lagged predictors
    for a given forecast horizon H (in months).

    Target:      H-month forward log return of TTF, dlogH_TTF(t) = log(TTF_t) - log(TTF_{t-H})
    Predictors (all anomalies, lagged H months so they are predetermined
    relative to the return they forecast):
      - momentum         : deseasonalized H-month TTF return itself
      - storage_change   : deseasonalized month-over-month change in EU+UK storage (bcm)
      - HDD               : deseasonalized London heating-degree-days

    Note: an LNG sendout variable was deliberately left out -- see module
    docstring and check_stationarity() below.
    """
    out = df[["Date", "month", "TTF"]].copy()

    dlogH = np.log(df["TTF"]).diff(horizon)
    dlogH_anom = dlogH - expanding_seasonal_mean(dlogH, df["month"])
    out["y"] = dlogH
    out["momentum"] = dlogH_anom.shift(horizon)

    storage_change = df["Av_storage(bcm)"].diff()
    out["storage_change_anom"] = (
        storage_change - expanding_seasonal_mean(storage_change, df["month"])
    ).shift(horizon)

    out["HDD_anom"] = (
        df["London_HDD"] - expanding_seasonal_mean(df["London_HDD"], df["month"])
    ).shift(horizon)

    return out


FEATURE_COLS = ["momentum", "storage_change_anom", "HDD_anom"]


# --------------------------------------------------------------------------- #
# 2. Stationarity checks
# --------------------------------------------------------------------------- #

def check_stationarity(series: pd.Series, name: str = "") -> dict:
    """
    Run ADF (H0: unit root) and KPSS (H0: stationary) on a series and
    return both test statistics plus a simple combined verdict.

    A series is flagged 'stationary' only if ADF rejects its null AND KPSS
    fails to reject its null (i.e. both tests agree). Anything else is
    flagged for manual review -- as with storage_change_anom in this model,
    a conflicting verdict is not automatically disqualifying, especially
    for a variable that is economically implausible as a genuine unit-root
    process (e.g. the change in a physically bounded stock).
    """
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")

    adf_reject_unit_root = adf_p < 0.05
    kpss_fail_reject_stationary = kpss_p > 0.05
    verdict = (
        "stationary" if (adf_reject_unit_root and kpss_fail_reject_stationary)
        else "non-stationary" if (not adf_reject_unit_root and not kpss_fail_reject_stationary)
        else "conflicting -- review manually"
    )
    return {
        "name": name, "adf_stat": adf_stat, "adf_p": adf_p,
        "kpss_stat": kpss_stat, "kpss_p": kpss_p, "verdict": verdict,
    }


def stationarity_report(df: pd.DataFrame, train_end: str = TRAIN_END) -> pd.DataFrame:
    """Run check_stationarity() over every series considered for this model."""
    train = df.loc[df["Date"] < train_end].copy()

    log_lng = np.log(train["LNG_sendout(bcm)"])
    lng_anom = log_lng - expanding_seasonal_mean(log_lng, train["month"])
    dlog_lng = log_lng.diff()
    dlog_lng_anom = dlog_lng - expanding_seasonal_mean(dlog_lng, train["month"])

    dlog_ttf = np.log(train["TTF"]).diff()
    dlog_ttf_anom = dlog_ttf - expanding_seasonal_mean(dlog_ttf, train["month"])

    storage_change = train["Av_storage(bcm)"].diff()
    storage_change_anom = storage_change - expanding_seasonal_mean(storage_change, train["month"])

    hdd_anom = train["London_HDD"] - expanding_seasonal_mean(train["London_HDD"], train["month"])

    series_to_check = {
        "dlog(TTF)": dlog_ttf,
        "momentum anomaly": dlog_ttf_anom,
        "storage_change_anom": storage_change_anom,
        "HDD_anom": hdd_anom,
        "log_LNG (level, excluded)": log_lng,
        "LNG_anom (excluded)": lng_anom,
        "dlog_LNG_anom (stationary but insignificant, excluded)": dlog_lng_anom,
    }
    return pd.DataFrame([check_stationarity(s, name) for name, s in series_to_check.items()])


# --------------------------------------------------------------------------- #
# 3. Estimation and diagnostics
# --------------------------------------------------------------------------- #

def fit_hac_model(feat: pd.DataFrame, train_end: str = TRAIN_END, extra_hac_lags: int = 0):
    """Fit OLS with Newey-West HAC standard errors on the training window."""
    train = feat.loc[feat["Date"] < train_end, ["Date", "y"] + FEATURE_COLS].dropna()
    y = train["y"]
    X = sm.add_constant(train[FEATURE_COLS])

    n = len(train)
    maxlags = int(np.floor(4 * (n / 100) ** (2 / 9))) + extra_hac_lags
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": True})
    return model, train


def run_diagnostics(model, train: pd.DataFrame) -> dict:
    """Durbin-Watson, residual lag-1 autocorrelation, and Breusch-Pagan test."""
    resid = model.resid
    dw = durbin_watson(resid)
    resid_ar1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]

    X = sm.add_constant(train[FEATURE_COLS])
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, X)

    return {
        "durbin_watson": dw,
        "resid_ar1": resid_ar1,
        "breusch_pagan_lm": bp_lm,
        "breusch_pagan_p": bp_p,
    }


# --------------------------------------------------------------------------- #
# 4. Walk-forward out-of-sample backtest
# --------------------------------------------------------------------------- #

def walk_forward_backtest(feat: pd.DataFrame, min_train: int = MIN_TRAIN_OBS) -> pd.DataFrame:
    """
    Expanding-window, one-step-ahead backtest.

    At each step, refit the full model (and a momentum-only benchmark) using
    only data available up to that point, then forecast the next observation.
    Also records a random-walk (zero-change) and historical-mean benchmark.
    """
    data = feat[["Date", "y"] + FEATURE_COLS].dropna().reset_index(drop=True)
    rows = []

    for i in range(min_train, len(data)):
        train, test = data.iloc[:i], data.iloc[i]

        X_train = sm.add_constant(train[FEATURE_COLS])
        full_fit = sm.OLS(train["y"], X_train).fit()
        X_test = sm.add_constant(test[FEATURE_COLS].to_frame().T, has_constant="add")
        pred_full = full_fit.predict(X_test).iloc[0]

        X_train_mom = sm.add_constant(train[["momentum"]])
        mom_fit = sm.OLS(train["y"], X_train_mom).fit()
        X_test_mom = sm.add_constant(test[["momentum"]].to_frame().T, has_constant="add")
        pred_mom = mom_fit.predict(X_test_mom).iloc[0]

        rows.append({
            "Date": test["Date"],
            "actual": test["y"],
            "pred_full": pred_full,
            "pred_momentum": pred_mom,
            "pred_random_walk": 0.0,
            "pred_hist_mean": train["y"].mean(),
        })

    return pd.DataFrame(rows)


def backtest_summary(res: pd.DataFrame) -> pd.DataFrame:
    """RMSE, MAE, directional hit rate, and OOS R^2 vs. random walk for each model."""
    rows = []
    rw_err = res["actual"] - res["pred_random_walk"]
    for label, col in [
        ("Full model", "pred_full"),
        ("Momentum-only", "pred_momentum"),
        ("Random walk", "pred_random_walk"),
        ("Historical mean", "pred_hist_mean"),
    ]:
        err = res["actual"] - res[col]
        rmse = np.sqrt((err ** 2).mean())
        mae = err.abs().mean()
        hit = np.nan if label == "Random walk" else (np.sign(res["actual"]) == np.sign(res[col])).mean()
        oos_r2 = 1 - (err ** 2).sum() / (rw_err ** 2).sum() if label != "Random walk" else np.nan
        rows.append({"Model": label, "RMSE": rmse, "MAE": mae, "Hit rate": hit, "OOS R2 vs RW": oos_r2})

    summary = pd.DataFrame(rows)

    full_err = res["actual"] - res["pred_full"]
    loss_diff = rw_err ** 2 - full_err ** 2
    t_stat = loss_diff.mean() / (loss_diff.std(ddof=1) / np.sqrt(len(loss_diff)))
    summary.attrs["loss_diff_tstat_full_vs_rw"] = t_stat
    return summary


# --------------------------------------------------------------------------- #
# 5. Plots
# --------------------------------------------------------------------------- #

def plot_actual_vs_fitted(df: pd.DataFrame, feat: pd.DataFrame, model, train_end: str = TRAIN_END,
                           outpath: str = "actual_vs_fitted.png"):
    """Reconstruct the fitted price level and plot it against the actual price."""
    full = feat[["Date", "TTF"] + FEATURE_COLS].dropna().reset_index(drop=True)
    X_full = sm.add_constant(full[FEATURE_COLS], has_constant="add")
    full["pred_dlog"] = model.predict(X_full)

    prev_price = df[["Date", "TTF"]].rename(columns={"TTF": "TTF_prev"})
    prev_price = prev_price.assign(Date=prev_price["Date"] + pd.offsets.MonthBegin(1))
    full = full.merge(prev_price, on="Date", how="left")
    full["fitted_TTF"] = full["TTF_prev"] * np.exp(full["pred_dlog"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(full["Date"], full["TTF"], label="Actual TTF", color="#1f4e79", linewidth=1.8)
    ax.plot(full["Date"], full["fitted_TTF"], label="Fitted TTF", color="#c0392b",
            linewidth=1.6, linestyle="--")
    split = pd.Timestamp(train_end)
    ax.axvline(split, color="grey", linewidth=1, linestyle=":")
    ax.text(split, ax.get_ylim()[1] * 0.97, "  in-sample | out-of-sample", fontsize=8.5, color="grey", va="top")
    ax.set_title("Actual vs. Fitted TTF Price ($/mmbtu)", fontsize=12)
    ax.set_ylabel("TTF (USD/mmbtu)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    return full


def plot_model_adjustment(full: pd.DataFrame, train_end: str = TRAIN_END,
                           outpath: str = "model_adjustment.png"):
    """Isolate and plot the model's adjustment over a naive random-walk forecast."""
    full = full.copy()
    full["model_adjustment"] = full["fitted_TTF"] - full["TTF_prev"]
    full["actual_change"] = full["TTF"] - full["TTF_prev"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(full["Date"], full["actual_change"], label="Actual month-over-month change",
            color="#1f4e79", linewidth=1.6)
    ax.plot(full["Date"], full["model_adjustment"], label="Model's adjustment vs. naive forecast",
            color="#c0392b", linewidth=1.6, linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.8)
    split = pd.Timestamp(train_end)
    ax.axvline(split, color="grey", linewidth=1, linestyle=":")
    ax.set_title("Model's Price Adjustment vs. Actual Price Change (USD/mmbtu)", fontsize=12)
    ax.set_ylabel("USD/mmbtu")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Main
# --------------------------------------------------------------------------- #

def main():
    df = load_data()

    print("=" * 70)
    print("Stationarity checks (training sample)")
    print("=" * 70)
    print(stationarity_report(df).to_string(index=False))

    print("\n" + "=" * 70)
    print("Horizon comparison (H = 1, 2, 3 months ahead)")
    print("=" * 70)
    for H in (1, 2, 3):
        feat = build_features(df, horizon=H)
        model, train = fit_hac_model(feat, extra_hac_lags=H - 1)
        diag = run_diagnostics(model, train)
        res = walk_forward_backtest(feat)
        summary = backtest_summary(res)

        print(f"\n--- H = {H} month(s) ---")
        print(model.summary())
        print(f"Durbin-Watson: {diag['durbin_watson']:.3f}   "
              f"Residual AR(1): {diag['resid_ar1']:.3f}   "
              f"Breusch-Pagan LM: {diag['breusch_pagan_lm']:.2f} (p={diag['breusch_pagan_p']:.3f})")
        print(summary.to_string(index=False))
        print(f"Loss-differential t-stat (full model vs. random walk): "
              f"{summary.attrs['loss_diff_tstat_full_vs_rw']:.2f}")

        if H == 1:
            full = plot_actual_vs_fitted(df, feat, model)
            plot_model_adjustment(full)
            print("\nSaved actual_vs_fitted.png and model_adjustment.png")


if __name__ == "__main__":
    main()
