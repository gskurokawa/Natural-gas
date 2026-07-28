# European Natural Gas Daily Dataset

Builds a daily European natural gas dataset from public sources: TTF/Henry Hub/JKM prices, EU gas storage, LNG sendout, EU pipeline flows by corridor, heating/cooling degree days for Berlin/London/Rome, FX, VIX, EU+UK power generation, and DE-LU / IT-Calabria day-ahead power prices.

## `build_ng_daily_dataset.py`

Fetches each source, merges everything onto a common `Date` column, and saves the result. It also fetches EU pipeline flow data from ENTSOG one calendar year at a time, saving each year to its own CSV (`europe_pipeline_imports_detailed_<year>.csv` / `europe_pipeline_exports_detailed_<year>.csv`) as it goes.

Most sources are fetched from 2020-01-01 onward, but the final saved file only keeps 2022-01-01 onward: data coverage and reliability (particularly for the ENTSOG pipeline data) is noticeably better from around that point, so the earlier ~2 years are fetched but trimmed from the final output rather than kept as partially-missing rows.

Each remote source is cached to `cache_daily/` after its first successful fetch, so re-running the script doesn't re-hit the same API repeatedly. Delete the relevant file in `cache_daily/` (or pass `use_cache=False`) to force a refresh of that source.

## `NG_daily10.csv`

The finished daily dataset, January 2022 to present.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GIE_API_KEY and ALSI_API_KEY
```

Free API keys are available at [agsi.gie.eu](https://agsi.gie.eu) and [alsi.gie.eu](https://alsi.gie.eu).

## Running it

```bash
python build_ng_daily_dataset.py
```
