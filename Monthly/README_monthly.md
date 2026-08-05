# European Natural Gas Monthly Dataset & TTF Forecasting Model

This repository builds a monthly European natural gas dataset from public sources, and uses it to test whether European supply/demand/weather fundamentals help forecast month-ahead TTF price moves.

## Pipeline: raw data → dataset

### `build_ng_dataset.py`
Builds `NG_m_final.csv` from a mix of public APIs and local input files. For each source (TTF/Henry Hub/JKM futures, EUR/USD, VIX, EU gas storage, LNG sendout, city-level temperatures/degree-days, EU+UK power generation mix, Eurostat gas balance, pipeline flows, supplementary figures), it fetches or loads the data, resamples it to monthly frequency, and merges everything onto a common `Date` column. It also computes a handful of derived variables (pipeline totals, an implied supply/demand balance, and calculated vs. actual storage change) before saving the final file. Each remote source is cached locally after its first fetch, so re-running the script doesn't re-hit the same API repeatedly.

### Local input files (`data/inputs/`)
A few of the pipeline's inputs come from local files rather than an API, either because they're hand-compiled history or licensed/derived data:

| File | Contents |
|---|---|
| `TTF_prices_Jan15-Sep17.csv` | Hand-compiled TTF price history for the period before Yahoo Finance's own TTF futures series begins. |
| `NG_europe_pipelines_2015-2020.csv` | Monthly cross-border pipeline flow volumes by corridor (Norway, Russia, Algeria, etc.) for 2015–2020. |
| `europe_pipeline_imports_detailed_<year>.csv` (2021–2026) | Daily pipeline import volumes by corridor and entry point. |
| `europe_pipeline_exports_detailed_<year>.csv` (2021–2026) | Daily pipeline export volumes for the Spain→Morocco corridor. |
| `European_natural_gas_-_SUPPLEMENTARY_DATA.xlsx` | Supplementary workbook. The `Transform` sheet holds the EU+UK production/consumption breakdown and storage/distribution-loss adjustments; the `CH+RS` sheet holds Switzerland + Serbia net export adjustments. |

### `NG_m_final.csv`
The finished monthly dataset, January 2015 to present — one row per month, combining TTF/Henry Hub/JKM prices, EU gas storage, LNG sendout, heating/cooling degree days for Berlin/London/Rome, FX and VIX, EU+UK power generation by source, EU gas production and consumption, pipeline flows by corridor, and the derived supply/demand balance variables. This is the file both the model and the diagnostic charts are built from.

## Data quality check

### `plot_storage_diagnostics.py`
Loads `NG_m_final.csv` and produces four charts comparing two independently-derived storage change series: one *calculated* purely from the collected supply and demand variables, and one *actual*, taken directly from AGSI-reported storage levels. Since these two series share no common inputs, how closely they agree is a check on the accuracy of the underlying data collection. The four charts show: the two series plotted together over time; the gap between them over time (which narrows substantially from around Jan 2022 onward); their correlation across the full sample; and their correlation restricted to Jan 2022 onward, which is noticeably tighter.

## Model

### `ttf_model.py`
Loads `NG_m_final.csv` and fits a small regression forecasting the one-month-ahead change in log TTF price from three lagged, deseasonalized fundamentals: price momentum, the change in EU gas storage, and London heating degree days. It runs the regression with Newey-West HAC standard errors, checks the residuals (autocorrelation, heteroskedasticity), backtests the model out-of-sample with an expanding-window walk-forward procedure against a random walk and a momentum-only benchmark, repeats the whole exercise at 2- and 3-month horizons, and produces the two chart images below.

### `actual_vs_fitted.png`
The model's fitted TTF price level against the actual TTF price, 2016 to present. Fitted price is reconstructed by compounding the prior month's actual price with the model's predicted one-month return. A marker separates the in-sample training period from the out-of-sample period.

### `model_adjustment.png`
Isolates the model's actual contribution: its predicted deviation from a naive "no change" forecast, plotted against what the price actually did that month. This is a more direct read on the model's skill than the price-level chart above, since a one-month-ahead return model's adjustments are necessarily small relative to the price level itself.

### `report_monthly_econometric.md`
The full write-up: market context, data sources, processing steps, methodology, stationarity checks, regression results, diagnostics, out-of-sample performance, the two charts above, forecast-horizon robustness, economic interpretation of each variable, and limitations.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GIE_API_KEY and ALSI_API_KEY
```

Free API keys are available at [agsi.gie.eu](https://agsi.gie.eu) and [alsi.gie.eu](https://alsi.gie.eu).

### `requirements.txt`
Python package dependencies for the whole repository: `pandas` and `numpy` for data handling, `requests` and `yfinance` for the API/data pulls, `python-dotenv` for loading API keys from `.env`, `openpyxl` for reading the supplementary Excel workbook, `statsmodels` for the regression and its diagnostics, and `plotly` for the diagnostic charts.

## Running it

```bash
python build_ng_dataset.py           # data/inputs/*.csv, *.xlsx  ->  NG_m_final.csv
python plot_storage_diagnostics.py   # NG_m_final.csv  ->  data quality charts
python ttf_model.py                  # NG_m_final.csv  ->  regression results + actual_vs_fitted.png, model_adjustment.png
```
