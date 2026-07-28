"""
build_ng_daily_dataset.py

Builds a DAILY European natural gas dataset (NG_daily10.csv): a
higher-frequency companion to the monthly dataset built by
build_ng_dataset.py, covering TTF/Henry Hub/JKM prices, EU gas storage,
LNG sendout, EU pipeline flows by corridor, HDD/CDD for three cities,
FX, VIX, EU+UK power generation, and DE-LU / IT-Calabria day-ahead power
prices.

Required external inputs:
  - GIE_API_KEY, ALSI_API_KEY environment variables (in a local .env file)

Everything else is fetched automatically, including the ENTSOG pipeline
flow data -- the original workflow this was built from fetched that one
year at a time by hand-editing a `year` variable and re-running the cell;
here it's a loop over ENTSOG_YEARS instead. Note that the yearly
`europe_pipeline_{imports,exports}_detailed_<year>.csv` files this
produces are the same files consumed by build_ng_dataset.py (the monthly
pipeline) as local inputs -- this script is where those actually come from.

Data quality note: the final output is deliberately restricted to
FINAL_START_DATE (2022-01-01) onward, even though most sources are fetched
from 2020-01-01. Coverage and reliability across several sources (notably
ENTSOG pipeline data) is materially better from around that point, so the
earlier ~2 years are fetched (useful for context/backfilling other work)
but trimmed from this dataset's final output rather than left in as
partially-missing rows.

Usage:
    python build_ng_daily_dataset.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
GIE_API_KEY = os.getenv("GIE_API_KEY")
ALSI_API_KEY = os.getenv("ALSI_API_KEY")

FETCH_START_DATE = "2020-01-01"     # most sources are pulled from this far back...
CORE_PRICE_FILTER_START = "2021-01-01"  # ...core prices (JKM/HH/TTF) must all be present from here...
FINAL_START_DATE = "2022-01-01"     # ...and the final saved dataset only keeps this onward (see module docstring)
END_DATE = pd.Timestamp.now().strftime("%Y-%m-%d")
ENTSOG_YEARS = range(2021, pd.Timestamp.now().year + 1)

CACHE_DIR = Path("cache_daily")
OUTPUT_PATH = Path("NG_daily10.csv")

REQUEST_HEADERS_UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

MWH_PER_MMBTU = 3.412  # EUR/MWh -> USD/mmbtu conversion for TTF

WEATHER_CITIES = {
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Rome": {"lat": 41.9028, "lon": 12.4964},
}
HDD_BASE_TEMP_C = 18.0
CDD_BASE_TEMP_C = 22.0

POWER_PRICE_ZONES = {"DE-LU": "DE-LU", "IT-Calabria": "IT-Calabria"}

# ENTSOG point -> corridor mappings. Kept exactly as in the original: note
# Russia appears as two separate corridors (RU-DE via Nordstream, and
# RU-Baltic via several smaller points) rather than one combined corridor.
ENTSOG_IMPORT_CORRIDORS = {
    "NO": ["Dornum / NETRA (OGE)", "Emden (EPT1) (OGE)", "Emden (EPT1) (GTS)",
           "Emden (EPT1) (GUD)", "Zeebrugge ZPT", "Dunkerque", "Easington", "St. Fergus"],
    "DE": ["Greifswald / NEL", "Greifswald / OPAL"],
    "TR": ["Strandzha 2 (BG) / Malkoclar (TR)", "Kipi (TR) / Kipi (GR)", "Kipoi"],
    "BY": ["Kondratki", "Kotlovka", "Tieterowka", "Wysokoje"],
    "UA": ["Beregdaróc 1400 (HU) - Beregovo (UA) (UA>HU)", "GCP GAZ-SYSTEM/UA TSO",
           "Isaccea (RO) - Orlovka (UA)", "Isaccea (RO) - Orlovka (UA) II",
           "Uzhgorod (UA) - Velké Kapušany (SK)", "Uzhhorod (UA) - Velké Kapušany (SK)",
           "VIP Bereg (HU) / VIP Bereg (UA)", "VIP Mediesu Aurit - Isaccea (RO-UA)"],
    "LY": ["Gela"],
    "DZ": ["Mazara del Vallo", "Almería"],
    "RU-DE": ["Nordstream"],
    "RU-Baltic": ["Luhamaa", "Misso Izborsk", "Narva", "Värska", "Imatra"],
    "MA": ["Tarifa"],
}
ENTSOG_EXPORT_CORRIDORS = {"ES": ["Tarifa"]}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.csv"


def _cached(name: str, fetch_fn, use_cache: bool = True) -> pd.DataFrame:
    path = _cache_path(name)
    if use_cache and path.exists():
        print(f"[cache] loading {name} from {path}")
        return pd.read_csv(path, parse_dates=["Date"])
    df = fetch_fn()
    df.to_csv(path, index=False)
    return df


def _to_naive_datetime(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    if getattr(df[date_col].dt, "tz", None) is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)
    return df


# --------------------------------------------------------------------------- #
# 1. Futures prices (TTF, Henry Hub, JKM), daily
# --------------------------------------------------------------------------- #

def fetch_futures_prices(use_cache: bool = True) -> pd.DataFrame:
    """Daily TTF / Henry Hub / JKM prices in USD/mmbtu, from FETCH_START_DATE onward."""
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
        raw = raw.drop(columns=["US_ETF2x", "US_iETF2x", "US_ETF", "EURUSD", "TTF(EUR)"])
        raw = _to_naive_datetime(raw)
        raw = raw[raw["Date"] >= FETCH_START_DATE].sort_values("Date")
        return raw.rename(columns={
            "TTF": "TTF(USD/mmbtu)", "JKM": "JKM(USD/mmbtu)", "HH": "HH(USD/mmbtu)",
        })

    return _cached("futures_prices_daily", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 2. EU gas storage (GIE AGSI+), daily
# --------------------------------------------------------------------------- #

def fetch_agsi_storage(use_cache: bool = True) -> pd.DataFrame:
    """Daily EU storage level (TWh)."""
    def _fetch() -> pd.DataFrame:
        headers = {"x-key": GIE_API_KEY, **REQUEST_HEADERS_UA}
        all_rows, page = [], 1
        print(f"Fetching AGSI storage data from {FETCH_START_DATE}...")
        while True:
            url = f"https://agsi.gie.eu/api?type=eu&from={FETCH_START_DATE}&to={END_DATE}&size=300&page={page}"
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
        df = df.sort_values("gasDayStart")
        df = df.rename(columns={"gasDayStart": "Date", "gasInStorage": "Storage(TWh)"})[["Date", "Storage(TWh)"]]
        return _to_naive_datetime(df)

    return _cached("agsi_storage_daily", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 3. EU+UK LNG sendout (GIE ALSI), daily
# --------------------------------------------------------------------------- #

def fetch_alsi_lng(use_cache: bool = True) -> pd.DataFrame:
    """Daily LNG sendout (GWh/day), fetched year by year."""
    def _fetch() -> pd.DataFrame:
        headers = {"x-key": ALSI_API_KEY, **REQUEST_HEADERS_UA}
        all_rows = []
        print("Fetching ALSI LNG sendout data (yearly chunks)...")
        for year in range(2020, pd.Timestamp.now().year + 1):
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
        df = df[df["gasDayStart"] >= FETCH_START_DATE].sort_values("gasDayStart")
        df["LNG_sendout(GWh/d)"] = pd.to_numeric(df.get("sendOut"), errors="coerce")
        df = df.rename(columns={"gasDayStart": "Date"})[["Date", "LNG_sendout(GWh/d)"]]
        return _to_naive_datetime(df)

    return _cached("alsi_lng_daily", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 4. ENTSOG pipeline flows by corridor, daily
# --------------------------------------------------------------------------- #

def _fetch_entsog_year(year: int, direction: str, corridors: dict[str, list[str]]) -> pd.DataFrame:
    """
    One year of ENTSOG physical flow data for one direction ('entry' or
    'exit'), classified into corridors and individual entry/exit points.
    Fetched one calendar month at a time, since ENTSOG's API doesn't
    reliably return a full year in a single request.
    """
    api_url = "https://transparency.entsog.eu/api/v1/operationaldatas"
    months = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="MS")
    monthly_dfs = []

    print(f"  {direction} {year}...")
    for start_month in months:
        start_str = start_month.strftime("%Y-%m-%d")
        end_str = (start_month + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        params = {"from": start_str, "to": end_str, "indicator": "Physical Flow",
                  "directionKey": direction, "limit": -1}
        try:
            resp = requests.get(api_url, params=params, timeout=45)
            if resp.status_code != 200:
                print(f"    failed {start_str}: HTTP {resp.status_code}")
                continue
            raw = resp.json().get("operationaldatas", [])
        except Exception as e:
            print(f"    error on {start_str}: {e}")
            continue

        records = []
        for row in raw:
            point_label, value = row.get("pointLabel"), row.get("value")
            if value is None:
                continue
            for corridor, points in corridors.items():
                if point_label in points:
                    records.append({
                        "date": pd.to_datetime(row.get("periodFrom")).date(),
                        "corridor": corridor,
                        "pointLabel": point_label,
                        "value_gwh": pd.to_numeric(value, errors="coerce") / 1_000_000,
                    })
        if records:
            monthly_dfs.append(pd.DataFrame(records))
        time.sleep(1)

    if not monthly_dfs:
        return pd.DataFrame(columns=["date"])

    flows = pd.concat(monthly_dfs, ignore_index=True)

    point_matrix = (
        flows.groupby(["date", "pointLabel"])["value_gwh"].sum()
        .reset_index().pivot(index="date", columns="pointLabel", values="value_gwh")
        .fillna(0).add_prefix("point_")
    )
    corridor_matrix = (
        flows.groupby(["date", "corridor"])["value_gwh"].sum()
        .reset_index().pivot(index="date", columns="corridor", values="value_gwh")
        .fillna(0).add_prefix("corridor_")
    )
    return pd.concat([corridor_matrix, point_matrix], axis=1).fillna(0).reset_index()


def fetch_entsog_flows(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily EU+UK pipeline entries (imports) and exits (exports) by corridor,
    for every year in ENTSOG_YEARS.

    Also saves one CSV per year per direction
    (europe_pipeline_{imports,exports}_detailed_<year>.csv) -- these are
    the same files build_ng_dataset.py (the monthly pipeline) reads as
    local inputs, so they're written to the working directory rather than
    just the cache folder.
    """
    imports_frames, exports_frames = [], []
    print("Fetching ENTSOG pipeline flows...")
    for year in ENTSOG_YEARS:
        imports_path = Path(f"europe_pipeline_imports_detailed_{year}.csv")
        if use_cache and imports_path.exists():
            imp = pd.read_csv(imports_path)
        else:
            imp = _fetch_entsog_year(year, "entry", ENTSOG_IMPORT_CORRIDORS)
            imp.to_csv(imports_path, index=False)
        imports_frames.append(imp)

        exports_path = Path(f"europe_pipeline_exports_detailed_{year}.csv")
        if use_cache and exports_path.exists():
            exp = pd.read_csv(exports_path)
        else:
            exp = _fetch_entsog_year(year, "exit", ENTSOG_EXPORT_CORRIDORS)
            exp.to_csv(exports_path, index=False)
        exports_frames.append(exp)

    imports = pd.concat(imports_frames, ignore_index=True)
    imports.columns = imports.columns.str.strip()
    imports = imports.rename(columns={"date": "Date"})
    # a few points (e.g. Tarifa) appear in both imports and exports corridor
    # mappings -- prefix point-level columns by direction so they never collide
    imports = imports.rename(columns={c: c.replace("point_", "point_import_")
                                       for c in imports.columns if c.startswith("point_")})
    # format='mixed' + dayfirst=True: pre-existing local CSVs (from earlier
    # runs / the original workflow) mix date formats across rows/years, so a
    # plain pd.to_datetime() can misfire on ambiguous entries like "13/01/2021"
    imports["Date"] = pd.to_datetime(imports["Date"], format="mixed", dayfirst=True)

    exports = pd.concat(exports_frames, ignore_index=True)
    exports.columns = exports.columns.str.strip()
    exports = exports.rename(columns={"date": "Date"})
    exports = exports.rename(columns={c: c.replace("point_", "point_export_")
                                       for c in exports.columns if c.startswith("point_")})
    exports["Date"] = pd.to_datetime(exports["Date"], format="mixed", dayfirst=True)

    return _to_naive_datetime(imports), _to_naive_datetime(exports)


# --------------------------------------------------------------------------- #
# 5. Heating / cooling degree days (Open-Meteo), daily
# --------------------------------------------------------------------------- #

def fetch_hdd_cdd(use_cache: bool = True) -> pd.DataFrame:
    """Daily HDD/CDD for Berlin, London, and Rome, pivoted to one column per city."""
    def _fetch() -> pd.DataFrame:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        city_frames = []
        print(f"Fetching Open-Meteo daily temperatures from {FETCH_START_DATE}...")
        for city, coords in WEATHER_CITIES.items():
            params = {
                "latitude": coords["lat"], "longitude": coords["lon"],
                "start_date": FETCH_START_DATE, "end_date": END_DATE,
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

        hdd = combined.pivot(index="date", columns="city", values="HDD")
        hdd.columns = [f"{c}_HDD" for c in hdd.columns]
        cdd = combined.pivot(index="date", columns="city", values="CDD")
        cdd.columns = [f"{c}_CDD" for c in cdd.columns]
        out = hdd.join(cdd).reset_index().rename(columns={"date": "Date"})
        return _to_naive_datetime(out)

    return _cached("hdd_cdd_daily", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 6. FX and VIX (Yahoo Finance), daily
# --------------------------------------------------------------------------- #

def fetch_fx(use_cache: bool = True) -> pd.DataFrame:
    """Daily EUR/USD spot rate."""
    def _fetch() -> pd.DataFrame:
        df = yf.Ticker("EURUSD=X").history(start=FETCH_START_DATE, interval="1d").reset_index()
        df = _to_naive_datetime(df)[["Date", "Close"]].rename(columns={"Close": "USD-EUR"})
        return df

    return _cached("fx_daily", _fetch, use_cache)


def fetch_vix(use_cache: bool = True) -> pd.DataFrame:
    """Daily VIX."""
    def _fetch() -> pd.DataFrame:
        df = yf.Ticker("^VIX").history(start=FETCH_START_DATE, interval="1d").reset_index()
        # NB: uses vix's own Date column here -- the original script accidentally
        # reused the FX section's `df['Date']` at this point. Since neither series
        # was resampled in the daily version, the two date ranges are close enough
        # that this likely didn't cause visible problems the way it did in the
        # monthly version (see that repo's README) -- but it's still a copy-paste
        # bug, not intentional, and is fixed here.
        df = _to_naive_datetime(df)[["Date", "Close"]].rename(columns={"Close": "VIX"})
        return df

    return _cached("vix_daily", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 7. EU+UK power generation mix (energy-charts.info), daily
# --------------------------------------------------------------------------- #

def fetch_power_generation(use_cache: bool = True) -> pd.DataFrame:
    """Daily EU+UK generation by source (MW, averaged from hourly), summed across the two regions."""
    def _fetch_country(country_code: str) -> pd.DataFrame:
        url = "https://api.energy-charts.info/public_power"
        resp = requests.get(url, params={"country": country_code, "start": FETCH_START_DATE, "end": END_DATE})
        if resp.status_code != 200:
            raise RuntimeError(f"energy-charts fetch failed for {country_code}: HTTP {resp.status_code}")
        payload = resp.json()
        idx = pd.to_datetime(payload["unix_seconds"], unit="s")
        series = {item["name"]: item["data"] for item in payload["production_types"]}
        return pd.DataFrame(series, index=idx)

    def _fetch() -> pd.DataFrame:
        print("Fetching energy-charts.info generation mix (EU + UK)...")
        combined = _fetch_country("eu").add(_fetch_country("uk"), fill_value=0)
        daily = combined.resample("D").mean().reset_index().rename(columns={"index": "Date"})
        return _to_naive_datetime(daily)

    return _cached("power_generation_daily", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 8. DE-LU / IT-Calabria day-ahead power prices (energy-charts.info), daily
# --------------------------------------------------------------------------- #

def fetch_power_prices(use_cache: bool = True) -> pd.DataFrame:
    """Daily average day-ahead power price (EUR/MWh) for each zone in POWER_PRICE_ZONES."""
    def _fetch() -> pd.DataFrame:
        url = "https://api.energy-charts.info/price"
        all_frames = []
        print("Fetching energy-charts.info day-ahead prices...")
        for zone_name, bzn_code in POWER_PRICE_ZONES.items():
            resp = requests.get(url, params={"bzn": bzn_code, "start": FETCH_START_DATE, "end": END_DATE})
            if resp.status_code != 200 or "unix_seconds" not in resp.json():
                print(f"  {zone_name}: failed")
                continue
            payload = resp.json()
            all_frames.append(pd.DataFrame({
                "Timestamp": pd.to_datetime(payload["unix_seconds"], unit="s"),
                "Zone": zone_name,
                "Price_EUR_MWh": payload["price"],
            }))
            print(f"  {zone_name}: ok")

        prices = pd.concat(all_frames, ignore_index=True)
        prices["Date"] = prices["Timestamp"].dt.date
        daily = prices.groupby(["Date", "Zone"])["Price_EUR_MWh"].mean().reset_index()
        wide = daily.pivot(index="Date", columns="Zone", values="Price_EUR_MWh").reset_index()
        wide = wide.rename(columns={"DE-LU": "DE_price(EUR/MWh)", "IT-Calabria": "IT_price(EUR/MWh)"})
        return _to_naive_datetime(wide)

    return _cached("power_prices_daily", _fetch, use_cache)


# --------------------------------------------------------------------------- #
# 9. Assembly
# --------------------------------------------------------------------------- #

def build_dataset(use_cache: bool = True) -> pd.DataFrame:
    prices = fetch_futures_prices(use_cache)
    storage = fetch_agsi_storage(use_cache)
    lng = fetch_alsi_lng(use_cache)
    imports, exports = fetch_entsog_flows(use_cache)
    weather = fetch_hdd_cdd(use_cache)
    fx = fetch_fx(use_cache)
    vix = fetch_vix(use_cache)
    generation = fetch_power_generation(use_cache)
    power_prices = fetch_power_prices(use_cache)

    def _merge(left: pd.DataFrame, right: pd.DataFrame, how: str) -> pd.DataFrame:
        left = _to_naive_datetime(left)
        right = _to_naive_datetime(right)
        return pd.merge(left, right, on="Date", how=how)

    # Join types preserved from the original: storage/LNG/pipeline/weather
    # are outer joins (so any of these sources can extend the date range),
    # FX/VIX/generation/prices are left joins.
    df = prices
    df = _merge(df, storage, how="outer")
    df = _merge(df, lng, how="outer")
    df = _merge(df, imports, how="outer")
    df = _merge(df, exports, how="outer")
    df = _merge(df, weather, how="outer")
    df = _merge(df, fx, how="left")
    df = _merge(df, vix, how="left")
    df = _merge(df, generation, how="left")
    df = _merge(df, power_prices, how="left")

    df = df[df["Date"] >= CORE_PRICE_FILTER_START]
    # Keep only days where all three core price series are present (drops
    # weekends/holidays and any day missing one of the three futures prices)
    df = df.dropna(subset=["JKM(USD/mmbtu)", "HH(USD/mmbtu)", "TTF(USD/mmbtu)"])

    # Interior gaps in LNG sendout are interpolated linearly
    df["LNG_sendout(GWh/d)"] = df["LNG_sendout(GWh/d)"].interpolate(method="linear")

    # Battery/Battery Consumption aren't meaningful gas-market variables;
    # IT day-ahead price and the ES export corridor total aren't used downstream
    df = df.drop(columns=["Battery", "Battery Consumption", "IT_price(EUR/MWh)", "corridor_ES"],
                 errors="ignore")

    # Final trim to FINAL_START_DATE -- see module docstring for why.
    df = df[df["Date"] >= FINAL_START_DATE]

    # Individual entry/exit point columns are dropped -- only the
    # corridor-level totals are kept in the final output
    df = df.drop(columns=[c for c in df.columns if "point" in c])

    # corridor_MA (Morocco) has some known small gaps that are genuinely
    # zero-flow rather than missing data, unlike the other corridors
    df["corridor_MA"] = df["corridor_MA"].fillna(0)

    return df.sort_values("Date").reset_index(drop=True)


def main():
    df = build_dataset(use_cache=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH} ({len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()})")


if __name__ == "__main__":
    main()
