# European Natural Gas Monthly Dataset — Build Pipeline

`build_ng_dataset.py` builds `NG_m_final.csv`, the monthly dataset used by the [TTF forecasting model](./report_monthly_econometric.md) in this repository, from a mix of public APIs and a few manually-compiled local files.

This is a cleaned-up rewrite of the original working script (`Natgas_monthly_data.py`, kept in this repo for reference). The rewrite is functionally equivalent wherever possible — see **Notes on this rewrite** below for the handful of places where behavior was deliberately preserved despite looking odd, and the one place a genuine bug was fixed.

## What it builds

One row per month, January 2015 to present, combining:

| Source | Data | Access |
|---|---|---|
| Yahoo Finance | TTF, Henry Hub, JKM futures; EUR/USD; VIX | `yfinance`, no key needed |
| GIE AGSI+ | EU gas storage levels | API key required |
| GIE ALSI | EU+UK LNG sendout | API key required |
| Open-Meteo | Daily temperatures → HDD/CDD (Berlin, London, Rome) | no key needed |
| energy-charts.info | EU+UK power generation mix | no key needed |
| Eurostat | EU gas production / consumption balance | no key needed |
| Local files | Cross-border pipeline flows, supplementary figures, early TTF history | see below |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GIE_API_KEY and ALSI_API_KEY
```

Get free API keys at [agsi.gie.eu](https://agsi.gie.eu) and [alsi.gie.eu](https://alsi.gie.eu).

## Required local input files

Place these in `data/inputs/` before running:

- `TTF_prices_Jan15-Sep17.csv` — manually compiled TTF history predating Yahoo Finance's series. Must contain a `TTF` column covering exactly the leading months missing from the Yahoo Finance data, in chronological order.
- `NG_europe_pipelines_2015-2020.csv` — manually compiled legacy pipeline flow data. Must use the same `corridor_*` column naming as the detailed yearly CSVs below (e.g. `corridor_NO`, `corridor_RU-Baltic`), plus a `Date` column in `%b-%y` format (e.g. `Jan-15`).
- `europe_pipeline_imports_detailed_<year>.csv` and `europe_pipeline_exports_detailed_<year>.csv`, for 2021–2026 — detailed daily pipeline flow data by corridor.
- `European_natural_gas_-_SUPPLEMENTARY_DATA.xlsx` — supplementary workbook. Sheets used: `Transform` (production/consumption breakdown, storage & distribution-loss adjustments) and `CH+RS` (Switzerland + Serbia net export adjustment).

None of these are produced by any public API used here; they're either hand-researched or licensed/derived data the original author compiled separately.

## Running it

```bash
python build_ng_dataset.py
```

Output: `NG_m_final.csv` in the working directory.

Each remote source is cached to `data/cache/` after its first successful fetch. Re-running the script re-uses the cache rather than re-hitting any API; delete the relevant file in `data/cache/` (or pass `use_cache=False` to the individual `fetch_*` functions) to force a refresh.

## Notes on this rewrite

A few behaviors in the original script look inconsistent at first glance but are preserved deliberately, because changing them would change the output:

- **Most merges are left joins, but LNG and the HDD/CDD weather merge are outer joins, and the CH+RS merge is an inner join.** The inner join in particular constrains the final dataset's date range to whatever the CH+RS sheet covers — this is not an oversight, it's reproduced exactly.
- **The `Piped` variable includes the `ES` (Spain→Morocco export) corridor alongside the import corridors**, and `bcm_ES` is then subtracted again separately when computing `Net_supply` — these cancel out, which looks redundant but is exactly what the original computed.
- **One genuine bug was fixed**: the original VIX-fetching section accidentally reused the FX section's `df['Date']` column instead of the VIX data's own date column (`vix['Date']`). This only avoided causing a length-mismatch error because of how the original was run interactively, cell-by-cell, out of strict top-to-bottom order — a linear script (which is what this is now) needs the corrected version to run at all. If you compare outputs against an old `NG_m_final.csv` and see different VIX values, this is why.
- **The early-TTF backfill is matched by Date, not row position.** The original filled in early TTF history by lining up two DataFrames positionally, which happened to work but was fragile. `TTF_prices_Jan15-Sep17.csv` also has some trailing blank rows, which are dropped before use.

## Validation

Network access and the two hand-compiled input files weren't both available during development, so this couldn't be run fully end-to-end against live APIs. It has been validated in two ways:
1. **Structural dry run**: every network call stubbed with synthetic-but-correctly-shaped data, while exercising the real local-file loaders and merge/derived-variable logic against your actual `TTF_prices_Jan15-Sep17.csv` and `NG_europe_pipelines_2015-2020.csv`. This produced exactly 137 rows spanning 2015-01 to 2026-05 — matching your existing `NG_m_final.csv` row-for-row — with no unexpected nulls.
2. Manual line-by-line comparison against the original script's logic (see **Notes on this rewrite** above).

**Before deleting or fully trusting this over the original**, run it for real and diff the output against your existing `NG_m_final.csv`.

## Files in this repo

- `build_ng_dataset.py` — builds `NG_m_final.csv` from scratch.
- `plot_storage_diagnostics.py` — loads `NG_m_final.csv` and reproduces the calculated-vs-actual storage change diagnostic charts from the end of the original script.
- `Natgas_monthly_data.py` — original script, kept for reference.
- `ttf_model.py` — the forecasting model that consumes `NG_m_final.csv`.
- `report_monthly_econometric.md` — model write-up and results.
