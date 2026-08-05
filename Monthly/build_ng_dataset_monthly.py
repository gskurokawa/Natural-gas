#%%

"""
build_ng_dataset.py

Builds the monthly European natural gas dataset (NG_m_final.csv) used by
the TTF forecasting model in this repository.

Combines, on a common monthly Date column:
  - TTF / Henry Hub / JKM futures prices (Yahoo Finance) + EUR/USD FX
  - EU gas storage levels (GIE AGSI+ API)
  - EU+UK LNG sendout (GIE ALSI API)
  - Berlin / London / Rome heating & cooling degree days (Open-Meteo)
  - EUR/USD and VIX (Yahoo Finance)
  - EU+UK power generation mix (energy-charts.info API)
  - EU gas production / consumption balance (Eurostat API)
  - EU+UK cross-border pipeline flows (local CSVs, 2021+, plus a
    user-supplied legacy file for 2015-2020)
  - Supplementary hand-compiled figures (local Excel workbook)

Required external inputs not produced by this script (see README.md):
  - GIE_API_KEY, ALSI_API_KEY environment variables (in a local .env file)
  - TTF_prices_Jan15-Sep17.csv           (manually compiled early TTF history)
  - NG_europe_pipelines_2015-2020.csv    (manually compiled legacy pipeline
                                           flows, same corridor_* column
                                           schema as the detailed CSVs below)
  - europe_pipeline_{imports,exports}_detailed_<year>.csv, 2021-2026
  - European_natural_gas_-_SUPPLEMENTARY_DATA.xlsx
    (sheets used: "Transform", "CH+RS")

Usage:
    python build_ng_dataset.py

Each remote data source is cached to CACHE_DIR after its first successful
fetch, so re-running the script does not re-hit any API unless the
corresponding cache file is deleted or use_cache=False is passed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

load_dotenv()
GIE_API_KEY = os.getenv("GIE_API_KEY")
ALSI_API_KEY = os.getenv("ALSI_API_KEY")

START_DATE = "2015-01-01"
END_DATE = pd.Timestamp.now().strftime("%Y-%m-%d")

CACHE_DIR = Path("data/cache")
INPUT_DIR = Path("data/inputs")   # user-supplied local files (see docstring)
OUTPUT_PATH = Path("NG_m_final.csv")

REQUEST_HEADERS_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Unit-conversion / fallback constants (kept named and documented rather than
# left as unexplained magic numbers in the original script)
MWH_PER_MMBTU = 3.412              # standard higher-heating-value conversion, used to
                                    # convert TTF from EUR/MWh to USD/mmbtu
AGSI_TWH_PER_BCM = 10.5            # AGSI reports storage in TWh; ~10.5 TWh per bcm
PIPELINE_GWH_PER_BCM = 10_500      # ENTSOG-style pipeline flows, GWh -> bcm
JKM_EARLY_HISTORY_FALLBACK = 7.86  # USD/mmbtu; JKM futures don't exist before
                                    # 2019, so early months are backfilled with
                                    # a researched approximate level
LNG_SENDOUT_EARLY_FALLBACK_GWH_D = 1200  # ALSI has no reliable data for the
                                    # first two months (Jan-Feb 2015);
                                    # approximate fallback based on research
HDD_BASE_TEMP_C = 18.0
CDD_BASE_TEMP_C = 22.0

WEATHER_CITIES = {
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Rome": {"lat": 41.9028, "lon": 12.4964},
}

PIPELINE_CORRIDOR_COLS = [
    "corridor_BY", "corridor_DE", "corridor_DZ", "corridor_LY", "corridor_MA",
    "corridor_NO", "corridor_RU-Baltic", "corridor_TR", "corridor_UA", "corridor_ES",
]


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #

def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.csv"


def _cached(name: str, fetch_fn, use_cache: bool = True) -> pd.DataFrame:
    """Load `name` from CACHE_DIR if present, otherwise call fetch_fn() and cache the result."""
    path = _cache_path(name)
    if use_cache and path.exists():
        print(f"[cache] loading {name} from {path}")
        return pd.read_csv(path, parse_dates=["Date"])
    df = fetch_fn()
    df.to_csv(path, index=False)
    return df


def _to_month_start(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    if getattr(df[date_col].dt, "tz", None) is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)
    return df


def _resample_monthly(df: pd.DataFrame, date_col: str, how: str = "mean") -> pd.DataFrame:
    """Resample a daily/irregular series to month-start frequency."""
    df = df.set_index(date_col)
    df = df.resample("MS").mean() if how == "mean" else df.resample("MS").sum()
    return df.reset_index().rename(columns={date_col: "Date"})


# --------------------------------------------------------------------------- #
# 1. Futures prices (TTF, Henry Hub, JKM) + implied TTF in USD/mmbtu
# --------------------------------------------------------------------------- #

def fetch_futures_prices(use_cache: bool = True) -> pd.DataFrame:
    """
    Monthly TTF / Henry Hub / JKM prices in USD/mmbtu.

    TTF trades in EUR/MWh on Yahoo Finance; it's converted to USD/mmbtu using
    the daily EUR/USD rate pulled in the same request and the standard
    3.412 mmbtu-per-MWh conversion. JKM has no listed futures before 2019,
    so early months are backfilled with a researched approximate level
    rather than left blank.
    """
    def _fetch() -> pd.DataFrame:
        tickers = "NG=F TTF=F JKM=F UNG BOIL KOLD EURUSD=X"
        raw = yf.download(tickers, period="max", group_by="ticker")
        raw = raw.drop(["Open", "High", "Low", "Volume"], level=1, axis=1)
        raw.columns = raw.columns.droplevel(1)
        raw = raw.reset_index()
        raw = raw.rename(columns={
            "NG=F": "HH", "JKM=F": "JKM", "TTF=F": "TTF(EUR)", "EURUSD=X": "EURUSD",
            "BOIL": "US_ETF2x", "KOLD": "US_iETF2x", "UNG": "US_ETF",  # unused leveraged ETFs
        })
        raw["TTF"] = raw["TTF(EUR)"] * raw["EURUSD"] / MWH_PER_MMBTU
        raw = raw.drop(columns=["US_ETF2x", "US_iETF2x", "US_ETF", "TTF(EUR)"])
        raw = raw[raw["Date"] >= START_DATE].sort_values("Date")
        monthly = _resample_monthly(_to_month_start(raw), "Date", how="mean")

        # Backfill Jan15-Sep17 TTF from a separately hand-compiled history file,
        # since Yahoo's TTF series doesn't extend that far back. Matched by Date
        # (not row position) so it's robust to the file's exact row count --
        # the source file also has some trailing blank rows, dropped here.
        early_ttf_path = INPUT_DIR / "TTF_prices_Jan15-Sep17.csv"
        early = pd.read_csv(early_ttf_path).dropna(subset=["TTF"])
        early["Date"] = pd.to_datetime(early["Date"], dayfirst=True)
        early = early.rename(columns={"TTF": "TTF_eur"})

        lookup = monthly[["Date", "EURUSD"]].merge(early, on="Date", how="left")
        early_ttf_usd = lookup["TTF_eur"] * lookup["EURUSD"] / MWH_PER_MMBTU
        monthly["TTF"] = monthly["TTF"].fillna(early_ttf_usd)

        monthly = monthly.drop(columns=["EURUSD"])
        monthly["JKM"] = monthly["JKM"].fillna(JKM_EARLY_HISTORY_FALLBACK)
        monthly = monthly.rename(columns={
            "TTF": "TTF(USD/mmbtu)", "JKM": "JKM(USD/mmbtu)", "HH": "HH(USD/mmbtu)",
        })
        return monthly

    return _cached("futures_prices", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 2. EU gas storage (GIE AGSI+)
# --------------------------------------------------------------------------- #

def fetch_agsi_storage(use_cache: bool = True) -> pd.DataFrame:
    """
    EU average storage level (bcm), sampled on each calendar month's first day.

    Deliberately NOT averaged over the month: AGSI reports a daily snapshot,
    and this keeps the same "level as of month start" semantics as the
    original pipeline (a left-join onto month-start dates naturally does this).
    """
    def _fetch() -> pd.DataFrame:
        headers = {"x-key": GIE_API_KEY, **REQUEST_HEADERS_UA}
        all_rows, page = [], 1
        print(f"Fetching AGSI storage data from {START_DATE}...")
        while True:
            url = f"https://agsi.gie.eu/api?type=eu&from={START_DATE}&to={END_DATE}&size=300&page={page}"
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"  stopped at page {page}: HTTP {resp.status_code}")
                break
            payload = resp.json()
            rows = payload.get("data", [])
            if not rows:
                break
            all_rows.extend(rows)
            if page >= payload.get("last_page", 1):
                break
            page += 1
            time.sleep(0.2)

        df = pd.DataFrame(all_rows)
        df["gasDayStart"] = pd.to_datetime(df["gasDayStart"])
        df["gasInStorage"] = pd.to_numeric(df["gasInStorage"], errors="coerce")
        df = df.sort_values("gasDayStart")[["gasDayStart", "gasInStorage"]]
        df = df.rename(columns={"gasDayStart": "Date", "gasInStorage": "Av_storage(bcm)"})
        df["Av_storage(bcm)"] = df["Av_storage(bcm)"] / AGSI_TWH_PER_BCM
        return _to_month_start(df)

    return _cached("agsi_storage", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 3. EU+UK LNG sendout (GIE ALSI)
# --------------------------------------------------------------------------- #

def fetch_alsi_lng(use_cache: bool = True) -> pd.DataFrame:
    """
    Monthly average LNG sendout (GWh/day), from daily ALSI data fetched year by year.

    Interior gaps are linearly interpolated here. The leading gap (ALSI has no
    reliable data for Jan-Feb 2015) is NOT filled here -- it's only visible
    once this is merged against the full date range, so that fallback is
    applied in build_dataset() after the merge, matching the original
    pipeline's order of operations.
    """
    def _fetch() -> pd.DataFrame:
        headers = {"x-key": ALSI_API_KEY, **REQUEST_HEADERS_UA}
        all_rows = []
        print("Fetching ALSI LNG sendout data (yearly chunks)...")
        for year in range(2015, pd.Timestamp.now().year + 1):
            start = f"{year}-01-01"
            end = f"{year}-12-31" if year < pd.Timestamp.now().year else END_DATE
            url = f"https://alsi.gie.eu/api?type=eu&from={start}&to={end}&size=366"
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                rows = resp.json().get("data", [])
                all_rows.extend(rows)
                print(f"  {year}: {len(rows)} rows")
            else:
                print(f"  {year}: failed (HTTP {resp.status_code})")
            time.sleep(0.3)

        df = pd.DataFrame(all_rows)
        df["gasDayStart"] = pd.to_datetime(df["gasDayStart"])
        df = df[df["gasDayStart"] >= START_DATE].sort_values("gasDayStart")
        df["LNG_sendout(GWh/d)"] = pd.to_numeric(df.get("sendOut"), errors="coerce")
        df = df.rename(columns={"gasDayStart": "Date"})[["Date", "LNG_sendout(GWh/d)"]]
        monthly = _resample_monthly(_to_month_start(df), "Date", how="mean")
        monthly["LNG_sendout(GWh/d)"] = monthly["LNG_sendout(GWh/d)"].interpolate(method="linear")
        return monthly

    return _cached("alsi_lng", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 4. Heating / cooling degree days (Open-Meteo)
# --------------------------------------------------------------------------- #

def fetch_hdd_cdd(use_cache: bool = True) -> pd.DataFrame:
    """Monthly HDD/CDD for Berlin, London, and Rome, pivoted to one column per city."""
    def _fetch() -> pd.DataFrame:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        city_frames = []
        print(f"Fetching Open-Meteo daily temperatures from {START_DATE}...")
        for city, coords in WEATHER_CITIES.items():
            params = {
                "latitude": coords["lat"], "longitude": coords["lon"],
                "start_date": START_DATE, "end_date": END_DATE,
                "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"],
                "timezone": "auto",
            }
            resp = requests.get(base_url, params=params)
            if resp.status_code != 200:
                print(f"  {city}: failed (HTTP {resp.status_code})")
                continue
            daily = resp.json().get("daily", {})
            city_frames.append(pd.DataFrame({
                "date": pd.to_datetime(daily.get("time")),
                "city": city,
                "temp_mean": daily.get("temperature_2m_mean"),
            }))
            print(f"  {city}: ok")

        combined = pd.concat(city_frames, ignore_index=True)
        combined["HDD"] = (HDD_BASE_TEMP_C - combined["temp_mean"]).clip(lower=0)
        combined["CDD"] = (combined["temp_mean"] - CDD_BASE_TEMP_C).clip(lower=0)

        monthly = (
            combined.set_index("date")
            .groupby("city")[["HDD", "CDD"]]
            .resample("MS").sum()
            .reset_index()
        )
        hdd = monthly.pivot(index="date", columns="city", values="HDD")
        hdd.columns = [f"{c}_HDD" for c in hdd.columns]
        cdd = monthly.pivot(index="date", columns="city", values="CDD")
        cdd.columns = [f"{c}_CDD" for c in cdd.columns]
        out = hdd.join(cdd).reset_index().rename(columns={"date": "Date"})

        # Consolidate data (keep Date so this frame can be merged on it)
        out['Europe_HDD'] = out['London_HDD'] + out['Berlin_HDD'] + out['Rome_HDD']
        out['Europe_CDD'] = out['London_CDD'] + out['Berlin_CDD'] + out['Rome_CDD']
        out = out[['Date', 'Europe_HDD', 'Europe_CDD']]
        return out

    return _cached("hdd_cdd", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 5. FX and VIX (Yahoo Finance)
# --------------------------------------------------------------------------- #

def fetch_fx(use_cache: bool = True) -> pd.DataFrame:
    """Monthly average EUR/USD spot rate."""
    def _fetch() -> pd.DataFrame:
        df = yf.Ticker("EURUSD=X").history(start=START_DATE, interval="1d").reset_index()
        df = _to_month_start(df)[["Date", "Close"]].rename(columns={"Close": "USD-EUR"})
        return _resample_monthly(df, "Date", how="mean")

    return _cached("fx", _fetch, use_cache)


def fetch_vix(use_cache: bool = True) -> pd.DataFrame:
    """Monthly average VIX."""
    def _fetch() -> pd.DataFrame:
        df = yf.Ticker("^VIX").history(start=START_DATE, interval="1d").reset_index()
        # NB: uses vix's own Date column here -- the original script accidentally
        # reused the FX section's `df['Date']`, which only avoided crashing because
        # of a specific interactive-cell execution order. Fixed here.
        df = _to_month_start(df)[["Date", "Close"]].rename(columns={"Close": "VIX"})
        return _resample_monthly(df, "Date", how="mean")

    return _cached("vix", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 6. EU+UK power generation mix (energy-charts.info)
# --------------------------------------------------------------------------- #

def fetch_power_generation(use_cache: bool = True) -> pd.DataFrame:
    """Monthly EU+UK generation by source (MW), summed across the two regions."""
    def _fetch_country(country_code: str) -> pd.DataFrame:
        url = "https://api.energy-charts.info/public_power"
        resp = requests.get(url, params={"country": country_code, "start": START_DATE, "end": END_DATE})
        if resp.status_code != 200:
            raise RuntimeError(f"energy-charts fetch failed for {country_code}: HTTP {resp.status_code}")
        payload = resp.json()
        idx = pd.to_datetime(payload["unix_seconds"], unit="s")
        series = {item["name"]: item["data"] for item in payload["production_types"]}
        return pd.DataFrame(series, index=idx)

    def _fetch() -> pd.DataFrame:
        print("Fetching energy-charts.info generation mix (EU + UK)...")
        combined = _fetch_country("eu").add(_fetch_country("uk"), fill_value=0)
        daily = combined.resample("D").mean()
        monthly = daily.resample("MS").mean().reset_index().rename(columns={"index": "Date"})
        return monthly.drop(columns=["Battery", "Battery Consumption"], errors="ignore")

    return _cached("power_generation", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 7. EU gas production / consumption balance (Eurostat)
# --------------------------------------------------------------------------- #

def fetch_eurostat_gas_balance(use_cache: bool = True) -> pd.DataFrame:
    """Monthly EU-27 gas production, transformation, and final consumption (MIO_M3)."""
    def _fetch() -> pd.DataFrame:
        url = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_cb_gasm"
        params = {
            "format": "JSON", "lang": "en", "geo": "EU27_2020", "siec": "G3000", "unit": "MIO_M3",
            "nrg_bal": ["PRO", "TI_EGC", "FC_IND", "FC_O_S"], "sinceTimePeriod": "2015-01",
        }
        print("Fetching Eurostat gas balance (nrg_cb_gasm)...")
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"Eurostat fetch failed: HTTP {resp.status_code}")
        payload = resp.json()

        nrg_bal_index = payload["dimension"]["nrg_bal"]["category"]["index"]
        time_index = payload["dimension"]["time"]["category"]["index"]
        rev_nrg_bal = {v: k for k, v in nrg_bal_index.items()}
        rev_time = {v: k for k, v in time_index.items()}
        n_times = len(time_index)

        rows = []
        for pos_str, val in payload.get("value", {}).items():
            pos = int(pos_str)
            time_code, nrg_code = rev_time.get(pos % n_times), rev_nrg_bal.get(pos // n_times)
            if time_code and nrg_code:
                rows.append({"Date": time_code, "nrg_bal": nrg_code, "Value": val})

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"] + "-01")
        df = df[df["Date"] >= START_DATE]
        pivot = df.pivot(index="Date", columns="nrg_bal", values="Value").reset_index()
        return pivot.rename(columns={
            "PRO": "Indigenous_Production(MIO_M3)",
            "TI_EGC": "Transformation_Electricity_Heat(MIO_M3)",
            "FC_IND": "Final_Consumption_Industry(MIO_M3)",
            "FC_O_S": "Final_Consumption_Other(MIO_M3)",
        })

    return _cached("eurostat_gas_balance", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 8. Local inputs: supplementary workbook and pipeline flow CSVs
# --------------------------------------------------------------------------- #

def load_supplementary_excel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the hand-compiled supplementary workbook.

    Returns (transform_df, ch_rs_df):
      transform_df -- EU+UK production/consumption breakdown & storage/loss adjustments
      ch_rs_df     -- Switzerland + Serbia net export adjustment
    """
    path = INPUT_DIR / "European natural gas - SUPPLEMENTARY DATA.xlsx"

    transform = pd.read_excel(path, sheet_name="Transform")
    transform["Date"] = pd.to_datetime(transform["Date"] + "-01").dt.tz_localize(None)
    transform["Stock_change(bcm)"] = transform["Stock_change(bcm)"].interpolate(method="linear")
    transform["Distribution_loss(bcm)"] = transform["Distribution_loss(bcm)"].interpolate(method="linear")

    ch_rs = pd.read_excel(path, sheet_name="CH+RS")
    ch_rs["Date"] = pd.to_datetime(ch_rs["Date"] + "-01").dt.tz_localize(None)

    return transform, ch_rs


def load_pipeline_flows() -> pd.DataFrame:
    """
    Monthly EU+UK cross-border pipeline flows (bcm) by corridor, 2015-present.

    Combines the detailed yearly CSVs (2021+) with a user-supplied legacy file
    for 2015-2020 that must share the same corridor_* column naming.
    """
    years = range(2021, pd.Timestamp.now().year + 1)

    imports = pd.concat(
        [pd.read_csv(INPUT_DIR / f"europe_pipeline_imports_detailed_{y}.csv") for y in years],
        ignore_index=True,
    )
    imports.columns = imports.columns.str.strip()
    imports = imports.rename(columns={"date": "Date"})
    imports["Date"] = pd.to_datetime(imports["Date"]).dt.tz_localize(None)
    imports = imports.set_index("Date").resample("MS").sum().reset_index()

    exports = pd.concat(
        [pd.read_csv(INPUT_DIR / f"europe_pipeline_exports_detailed_{y}.csv") for y in years],
        ignore_index=True,
    )
    exports.columns = exports.columns.str.strip()
    exports = exports.rename(columns={"date": "Date"})
    # exports files mix date formats across years; 'mixed' + dayfirst matches
    # what the source files actually contain
    exports["Date"] = pd.to_datetime(exports["Date"], format="mixed", dayfirst=True).dt.tz_localize(None)
    exports = exports.set_index("Date").resample("MS").sum().reset_index()

    recent = pd.merge(imports, exports, on="Date", how="left")
    recent["Date"] = pd.to_datetime(recent["Date"]).dt.strftime("%Y-%m-%d")
    recent = recent[["Date"] + [c for c in PIPELINE_CORRIDOR_COLS if c in recent.columns]]

    legacy_path = INPUT_DIR / "NG_europe_pipelines_2015-2020.csv"
    legacy = pd.read_csv(legacy_path, encoding="utf-8-sig")
    legacy["Date"] = pd.to_datetime(legacy["Date"], format="%b-%y").dt.strftime("%Y-%m-%d")

    combined = pd.concat([legacy, recent], axis=0, ignore_index=True)
    combined.columns = combined.columns.str.replace("corridor", "bcm", regex=False)
    for corridor in [c.replace("corridor_", "") for c in PIPELINE_CORRIDOR_COLS]:
        col = f"bcm_{corridor}"
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce") / PIPELINE_GWH_PER_BCM

    combined["Date"] = pd.to_datetime(combined["Date"])

    # Temporary retrieve Swiss and Balkan trade again for the net piped calculation
    path = INPUT_DIR / "European natural gas - SUPPLEMENTARY DATA.xlsx"
    ch_rs = pd.read_excel(path, sheet_name="CH+RS")
    ch_rs["Date"] = pd.to_datetime(ch_rs["Date"] + "-01").dt.tz_localize(None)
    combined = pd.merge(combined, ch_rs, on='Date', how='left')

    pipeline_corridors = [c.replace("corridor_", "") for c in PIPELINE_CORRIDOR_COLS]
    # NB: this includes 'ES' (the Spain -> Morocco export corridor) alongside the
    # import corridors, matching the original pipeline's construction. It's added
    # here and then subtracted again below in Net_supply, so it nets to zero --
    # kept for fidelity with the original rather than "simplified" away.

    combined["Net_piped(bcm)"] = sum(combined[f"bcm_{c}"] for c in pipeline_corridors if f"bcm_{c}" in combined.columns)
    combined['Net_piped(bcm)'] = combined['Net_piped(bcm)'] - combined["CH_RS_net_exports"] - combined["bcm_ES"]
    combined = combined[['Date', 'Net_piped(bcm)']]

    return combined

# Norway to Europe piped gas capacity outages
def Norway_outages() -> pd.DataFrame:
    path = INPUT_DIR / "Norway_outages.xlsx"
    df = pd.read_excel(path, sheet_name='Monthly 2015-2026')
    df = df.iloc[:,:6]
    df["Date"] = pd.to_datetime(df["Month"] + "-01").dt.tz_localize(None)
    df = df.drop(columns=['Month'])
    return df

# Global LNG export terminal capacities and outages (outages are all surprises)
def LNG_export_capacity() -> pd.DataFrame:
    path = INPUT_DIR / "LNG_export_capacity.xlsx"

    # header=1 -> Excel row 2 ("Month", "MonthKey", "Global Total", <terminals>) becomes
    # the columns and the data starts at Jan 2010, so no manual header-row juggling is
    # needed. The sheets have a few footnote rows at the bottom, so we keep only rows
    # whose Month parses as a real date (robust to months being added/removed later).
    df_LNG = pd.read_excel(path,
                           sheet_name='Monthly capacity (Mtpa)', header=1)
    df_LNG['Date'] = pd.to_datetime(df_LNG['Month'], errors='coerce')
    df_LNG = df_LNG.dropna(subset=['Date'])[['Date', 'Global Total']]
    df_LNG = df_LNG.rename(columns={'Global Total': 'Global LNG nameplate capacity'})

    df_LNG_outages = pd.read_excel(path,
                                   sheet_name='Monthly outages (Mtpa)', header=1)
    df_LNG_outages['Date'] = pd.to_datetime(df_LNG_outages['Month'], errors='coerce')
    df_LNG_outages = df_LNG_outages.dropna(subset=['Date'])[['Date', 'Total offline']]
    df_LNG_outages = df_LNG_outages.rename(columns={'Total offline': 'Global LNG capacity offline'})

    df_LNG['Date'] = df_LNG['Date'].dt.strftime("%Y-%m-%d")
    df_LNG_outages['Date'] = df_LNG_outages['Date'].dt.strftime("%Y-%m-%d")

    df_LNG = pd.merge(df_LNG, df_LNG_outages, on='Date', how='left')
    return df_LNG

# Global LNG imports
def Global_LNG_trade() -> pd.DataFrame:
    path = INPUT_DIR / "Global JODI LNG trade.xlsx"

    df_imports = pd.read_excel(path, sheet_name='JODI imports')
    df_imports['Date'] = pd.to_datetime(df_imports['Date'])
    df_imports = df_imports[['Date', 'CH+JP+KR LNG imports', 'EU+UK LNG imports', 'EG LNG imports', 'IN LNG imports']]   

    df_exports = pd.read_excel(path, sheet_name='JODI exports')
    df_exports['Date'] = pd.to_datetime(df_exports['Date'])
    df_exports = df_exports[['Date', 'QA+AU+US LNG exports', 'ID+MY+BN LNG exports', 'NG LNG exports']]   

    df_LNG_trade = pd.merge(df_imports, df_exports, on='Date', how='left')
    return df_LNG_trade

# Global weather derivatives
def weather_derivatives() -> pd.DataFrame:
    path = INPUT_DIR / "Global weather derivatives.xlsx"
    df = pd.read_excel(path, sheet_name = 'Monthly 2015-present')
    df["Date"] = pd.to_datetime(df["month"] + "-01").dt.tz_localize(None)
    df = df.drop(columns=['month'])
    return df

# --------------------------------------------------------------------------- #
# 9. Derived variables
# --------------------------------------------------------------------------- #

def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add supply-balance and geopolitical-risk-tier variables computed from
    columns already present in `df`.
    """
    df = df.copy()
    df["Days"] = df["Date"].dt.days_in_month
    df["LNG_sendout(bcm)"] = df["LNG_sendout(GWh/d)"] * df["Days"] / PIPELINE_GWH_PER_BCM
    df = df.drop(columns=["LNG_sendout(GWh/d)"])

    df["Production(bcm)"] = pd.to_numeric(df["Production(bcm)"], errors="coerce")
    df["Net_supply"] = (
        df["LNG_sendout(bcm)"] + df["Production(bcm)"] + df["Net_piped(bcm)"]
        - df["Distribution_loss(bcm)"]
    )

    df["Total_adj(bcm)"] = pd.to_numeric(df["Total_adj(bcm)"], errors="coerce")
    df["Calc_storage_change"] = df["Net_supply"] - df["Total_adj(bcm)"]
    df["Act_storage_change"] = -df["Av_storage(bcm)"].diff(-1)
    df["Diff_storage_changes"] = df["Act_storage_change"] - df["Calc_storage_change"]

    df = df.drop(columns=['Days'])
    return df


# --------------------------------------------------------------------------- #
# 9b. Feature consolidation (dimensionality reduction)
# --------------------------------------------------------------------------- #

def consolidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the collinear EU+UK power-generation block into a compact set of
    economically-meaningful features, aggregate minor / split fuels, tidy the
    weather and Norway column names, and drop unused columns.

    Applied at the very end of build_dataset(), after add_derived_variables(),
    so nothing upstream depends on the columns removed here.
    """
    df = df.copy()

    # Hydro -> one precipitation-driven generation series. Run-of-river + reservoir
    # is the gas-displacing output; the two pumped-storage columns are a ~net-zero
    # storage cycle (consumption is demand, not supply), so they are dropped.
    df['Hydro_gen'] = df['Hydro Run-of-River'] + df['Hydro water reservoir']

    # Aggregate minor fuels into the existing 'Others'.
    df['Others'] = (df['Others'] + df['Biomass'] + df['Fossil peat']
                    + df['Geothermal'] + df['Other renewables'] + df['Waste'])

    # Combine the coal fuels.
    df['Coal'] = (df['Fossil brown coal / lignite'] + df['Fossil coal-derived gas']
                  + df['Fossil hard coal'])

    # Combine the oil fuels (name reused).
    df['Fossil oil'] = df['Fossil oil'] + df['Fossil oil shale']

    # Drop the raw power-block columns now folded away or superseded by Residual load.
    df = df.drop(columns=[
        # hydro components (now Hydro_gen) + the pumped-storage cycle
        'Hydro Run-of-River', 'Hydro water reservoir',
        'Hydro pumped storage', 'Hydro pumped storage consumption',
        # renewables + load already embedded in Residual load
        'Wind offshore', 'Wind onshore', 'Solar', 'Load',
        'Renewable share of generation', 'Renewable share of load',
        # minor fuels folded into Others
        'Biomass', 'Fossil peat', 'Geothermal', 'Other renewables', 'Waste',
        # coal fuels folded into Coal
        'Fossil brown coal / lignite', 'Fossil coal-derived gas', 'Fossil hard coal',
        # oil fuel folded into Fossil oil
        'Fossil oil shale',
        # not carried forward (from the earlier consolidation plan)
        'Cross border electricity trading',
    ], errors='ignore')

    # Tidy weather / climate column names.
    df = df.rename(columns={
        'eu_wind_100m_ms': 'EU_wind_speed',
        'us_gwdd_F': 'US_GWDD',
        'neasia_gwdd_C': 'NE_Asia_GWDD',
        'eu_solar_MJ': 'EU_solar',
        'nordic_precip_mm': 'Nordic_precip',
        'atl_ace': 'Atlantic_ACE',
        'gulf_storm_days': 'Gulf_storm_days',
        'eu_wind_100m_anom': 'EU_wind_speed_anomaly',
        'us_gwdd_anom': 'US_GWDD_anomaly',
        'neasia_gwdd_anom': 'NE_Asia_GWDD_anomaly',
        'eu_solar_anom': 'EU_solar_anomaly',
        'nordic_precip_anom': 'Nordic_precip_anomaly',
        'atl_anom': 'Atlantic_ACE_anomaly',
    })

    # Tidy Norway column names.
    df = df.rename(columns={
        'NCS gas prod (mcm/d)': 'Norway_gas_prod',
        'Supply reduction (mcm/d)': 'Norway_supply_red',
        'Planned outage est (mcm/d)': 'Norway_planned_outage',
        'Unplanned outage est (mcm/d)': 'Norway_unplanned_outage',
    })

    # Drop unused columns.
    df = df.drop(columns=['CH_RS_net_exports', 'Total outage est (mcm/d)'],
                 errors='ignore')

    return df


# --------------------------------------------------------------------------- #
# 9c. Final column names and order (output schema)
# --------------------------------------------------------------------------- #
#
# The output schema is declared here as data, not code: COLUMN_RENAMES lists only
# the columns whose names change, and OUTPUT_COLUMN_ORDER is the exact left-to-right
# order of the final CSV (excluding 'Date', which is always kept first). Keeping
# both as named constants means the intended schema lives in one obvious place, and
# rename_and_reorder() validates against it so an upstream add/remove fails loudly
# instead of silently dropping or misordering a column.

COLUMN_RENAMES = {
    'Av_storage(bcm)':        'EU+UK_av_storage(bcm)',
    'USD-EUR':                'USD-EUR_FX',
    'Fossil gas':             'EU+UK Fossil gas',
    'Fossil oil':             'EU+UK Fossil oil',
    'Nuclear':                'EU+UK Nuclear',
    'Others':                 'EU+UK Others',
    'Residual load':          'EU+UK Residual load',
    'Production(bcm)':        'EU+UK Production(bcm)',
    'Electricity(bcm)':      'EU+UK Electricity(bcm)',
    'Total(bcm)':            'EU+UK Total(bcm)',
    'Non_power(bcm)':        'EU+UK Non_power(bcm)',
    'Stock_change(bcm)':     'EU+UK Stock_change(bcm)',
    'Distribution_loss(bcm)':'EU+UK Distribution_loss(bcm)',
    'Total_adj(bcm)':        'EU+UK Total_adj(bcm)',
    'Net_piped(bcm)':        'EU+UK Net_piped(bcm)',
    'LNG_sendout(bcm)':      'EU+UK LNG_sendout(bcm)',
    'Net_supply':            'EU+UK Net_supply',
    'Calc_storage_change':   'EU+UK Calc_storage_change',
    'Act_storage_change':    'EU+UK Act_storage_change',
    'Diff_storage_changes':  'EU+UK Diff_storage_changes',
    'Hydro_gen':             'EU+UK Hydro_gen',
    'Coal':                  'EU+UK Coal',
}

# Final left-to-right order (post-rename), excluding the leading 'Date' column.
OUTPUT_COLUMN_ORDER = [
    # Prices / macro
    'JKM(USD/mmbtu)', 'HH(USD/mmbtu)', 'TTF(USD/mmbtu)', 'USD-EUR_FX', 'VIX',
    # EU+UK power generation by source
    'EU+UK Fossil gas', 'EU+UK Fossil oil', 'EU+UK Coal', 'EU+UK Hydro_gen',
    'EU+UK Nuclear', 'EU+UK Others', 'EU+UK Residual load',
    # EU+UK gas balance
    'EU+UK Production(bcm)', 'EU+UK Electricity(bcm)', 'EU+UK Total(bcm)',
    'EU+UK Non_power(bcm)', 'EU+UK Stock_change(bcm)', 'EU+UK Distribution_loss(bcm)',
    'EU+UK Total_adj(bcm)', 'EU+UK Net_piped(bcm)', 'EU+UK LNG_sendout(bcm)',
    'EU+UK Net_supply', 'EU+UK_av_storage(bcm)', 'EU+UK Calc_storage_change',
    'EU+UK Act_storage_change', 'EU+UK Diff_storage_changes',
    # Norway supply
    'Norway_gas_prod', 'Norway_supply_red', 'Norway_planned_outage',
    'Norway_unplanned_outage',
    # Weather / climate
    'Europe_HDD', 'Europe_CDD', 'EU_wind_speed', 'US_GWDD', 'NE_Asia_GWDD',
    'EU_solar', 'Nordic_precip', 'Atlantic_ACE', 'Gulf_storm_days',
    'EU_wind_speed_anomaly', 'US_GWDD_anomaly', 'NE_Asia_GWDD_anomaly',
    'EU_solar_anomaly', 'Nordic_precip_anomaly', 'Atlantic_ACE_anomaly',
    # Global LNG capacity / trade
    'Global LNG nameplate capacity', 'Global LNG capacity offline',
    'CH+JP+KR LNG imports', 'EU+UK LNG imports', 'EG LNG imports', 'IN LNG imports',
    'QA+AU+US LNG exports', 'ID+MY+BN LNG exports', 'NG LNG exports',
]


def rename_and_reorder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply COLUMN_RENAMES, then reorder to ['Date'] + OUTPUT_COLUMN_ORDER.

    Raises KeyError if the post-rename frame doesn't match the declared schema
    exactly (missing or unexpected columns), so schema drift is caught here
    rather than producing a silently truncated or misordered CSV.
    """
    df = df.rename(columns=COLUMN_RENAMES)
    ordered = ['Date'] + OUTPUT_COLUMN_ORDER

    missing = [c for c in ordered if c not in df.columns]
    extra = [c for c in df.columns if c not in ordered]
    if missing or extra:
        raise KeyError(
            "rename_and_reorder: output schema mismatch.\n"
            f"  missing (expected, not found): {missing}\n"
            f"  extra (found, not expected):   {extra}"
        )

    return df[ordered]


# --------------------------------------------------------------------------- #
# 10. Assembly
# --------------------------------------------------------------------------- #

def build_dataset(use_cache: bool = True) -> pd.DataFrame:
    prices = fetch_futures_prices(use_cache)
    storage = fetch_agsi_storage(use_cache)
    lng = fetch_alsi_lng(use_cache)
    hdd_cdd = fetch_hdd_cdd(use_cache)
    fx = fetch_fx(use_cache)
    vix = fetch_vix(use_cache)
    generation = fetch_power_generation(use_cache)
    eurostat = fetch_eurostat_gas_balance(use_cache)
    transform, ch_rs = load_supplementary_excel()
    pipelines = load_pipeline_flows()
    LNG_capacity = LNG_export_capacity()
    LNG_trade = Global_LNG_trade() 
    weather = weather_derivatives()
    norway = Norway_outages()

    def _merge(left: pd.DataFrame, right: pd.DataFrame, how: str) -> pd.DataFrame:
        left["Date"] = pd.to_datetime(left["Date"]).dt.tz_localize(None)
        right["Date"] = pd.to_datetime(right["Date"]).dt.tz_localize(None)
        return pd.merge(left, right, on="Date", how=how)

    # Join types below are preserved exactly as in the original pipeline --
    # most are 'left', but LNG and the HDD/CDD weather merge are 'outer'
    # (so either source can extend the date range), and CH+RS is 'inner'
    # (so the final dataset's date range is bounded by whatever the CH+RS
    # sheet covers). These aren't arbitrary and shouldn't be "tidied" to
    # all-left without checking the row count doesn't change.
    df = prices
    df = _merge(df, storage, how="left")
    df = _merge(df, lng, how="outer")
    
    # ALSI has no reliable data for the first two months of the sample
    # (Jan-Feb 2015); the interior gaps were already interpolated in
    # fetch_alsi_lng, but this leading gap only appears once merged against
    # the full date range, so it's patched here -- matching where the
    # original pipeline did this. Sorting first (defensive, not in the
    # original) guarantees rows 0/1 really are Jan/Feb 2015 regardless of
    # how the outer merge ordered things.
    df = df.sort_values("Date").reset_index(drop=True)
    df.loc[0:1, "LNG_sendout(GWh/d)"] = LNG_SENDOUT_EARLY_FALLBACK_GWH_D

    df = _merge(df, hdd_cdd, how="outer")   # Europe_HDD / Europe_CDD (was being dropped)
    df = _merge(df, weather, how="outer")   # weather derivatives
    df = _merge(df, fx, how="left")
    df = _merge(df, vix, how="left")
    df = _merge(df, generation, how="left")
    df = _merge(df, eurostat, how="left")
    df = _merge(df, transform, how="left")
    df = _merge(df, pipelines, how="left")
    df = _merge(df, ch_rs, how="inner")
    df = _merge(df, LNG_capacity, how='left')
    df = _merge(df, LNG_trade, how='left')
    df = _merge(df, norway, how='left')

    df = df.drop(columns=["Final_Consumption_Industry(MIO_M3)"], errors="ignore")
    df = add_derived_variables(df)
    df = consolidate_features(df)
    df = df.sort_values("Date").reset_index(drop=True)
    return rename_and_reorder(df)

def main():
    df = build_dataset(use_cache=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH} ({len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()})")


if __name__ == "__main__":
    main()

    
# %%
