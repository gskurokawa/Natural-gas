#%% Load libraries

import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots


#%% Load API keys

load_dotenv()

# Retrieve the API key safely from environment variables
GIE_API_KEY = os.getenv("GIE_API_KEY")
ALSI_API_KEY = os.getenv("ALSI_API_KEY")


#%% Load natural gas prices

gas_tickers = "NG=F TTF=F JKM=F UNG BOIL KOLD EURUSD=X"
data = yf.download(gas_tickers, period="max", group_by="ticker") # default: daily data
data = data.drop(['Open', 'High', 'Low', 'Volume'], level=1, axis=1)
data.columns = data.columns.droplevel(1)
data = data.reset_index()

data = data.rename(columns={'NG=F':'HH', 'BOIL': 'US_ETF2x', 'KOLD': 'US_iETF2x', 'JKM=F':'JKM', 'TTF=F': 'TTF(EUR)', 'UNG': 'US_ETF'})
data['TTF'] = data['TTF(EUR)'] * data['EURUSD=X'] / 3.412
data = data.drop(columns=['US_ETF2x', 'US_iETF2x', 'US_ETF', 'EURUSD=X', 'TTF(EUR)'])
data = data[data['Date'] >= '2018-01-01']
print(data)

data_melt = data.melt(id_vars='Date', var_name='Series', value_name='Prices')
print(data_melt)

x = yf.Ticker("TTF=F")
print(f"Currency: {x.info.get('currency')}")  # Output: EUR
x = yf.Ticker("NG=F")
print(f"Currency: {x.info.get('currency')}")  # Output: USD
x = yf.Ticker("JKM=F")
print(f"Currency: {x.info.get('currency')}")  # Output: None; assume USD

#%% Create new file and add prices to it

data = data[data['Date'] >= '2020-01-01']
data = data.rename(columns={'TTF':'TTF(USD/mmbtu)', 'JKM':'JKM(USD/mmbtu)', 'HH':'HH(USD/mmbtu)'})

output_filename = "NG_daily0.csv"
data.to_csv(output_filename, index=False)

#%% Graphing natural gas prices as a check

print(data_melt.info())

data = data[data['Date'] >= '2020-01-01']
data_melt = data.melt(id_vars='Date', var_name='Series', value_name='Prices')
data_melt['Prices'] = data_melt['Prices'].interpolate(method='linear', limit_direction='both')
print(data_melt)

fig = px.line(
    data_melt, 
    x='Date', 
    y='Prices',
    title='Daily gas prices',
    color='Series',
    labels={'Date': 'Date', 'Prices': 'Price'},
    markers=False  # Adds clickable dots to each month's data point
)

fig.update_traces(hovertemplate='Date: %{x}<br>Price Index: %{y:.2f}')
fig.update_traces(line=dict(width=1.2))
fig.update_layout(
    hovermode='x unified',
    xaxis_tickangle=-45,
    template='plotly_white'  # Gives a clean, modern background grid
)

fig.show()

#%% Load GIE storage and sendout data

API_KEY = GIE_API_KEY
headers = {
    "x-key": API_KEY,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# 1. Establish your target timeframe constraints
start_date = "2020-01-01"
end_date = "2026-07-20"  # Plugs into the current date framework

all_data = []
current_page = 1

print(f"Extracting historical gas storage from {start_date} down to 2026...")

while True:
    # Query with fixed, safe chunks using size=300 while advancing the page count
    url = f"https://agsi.gie.eu/api?type=eu&from={start_date}&to={end_date}&size=300&page={current_page}"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Stopped at page {current_page} with error status: {response.status_code}")
        break
        
    res_json = response.json()
    data_list = res_json.get('data', [])
    
    # If a page returns completely empty, the loop has successfully collected everything
    if not data_list:
        break
        
    all_data.extend(data_list)
    print(f"Collected page {current_page}...")
    
    # Safety valve: Stop if we've cleared the server's maximum page calculation
    last_page = res_json.get('last_page', 1)
    if current_page >= last_page:
        break
        
    current_page += 1
    
    # Pause for 200ms between calls to avoid hitting their firewalls
    time.sleep(0.2)

# 2. Build and clean the complete historical DataFrame
storage_df = pd.DataFrame(all_data)
storage_df['gasDayStart'] = pd.to_datetime(storage_df['gasDayStart'])
storage_df['gasInStorage'] = pd.to_numeric(storage_df['gasInStorage'])
storage_df['full'] = pd.to_numeric(storage_df['full'])

# Sort chronologically and index by date
storage_df = storage_df.sort_values('gasDayStart')
storage_df.set_index('gasDayStart', inplace=True)

print("\nHistory compilation successful!")
print(f"Total entries downloaded: {len(storage_df)} rows")
print(f"Data range: {storage_df.index.min().strftime('%Y-%m-%d')} to {storage_df.index.max().strftime('%Y-%m-%d')}")

storage_df = storage_df.reset_index()
storage_df = storage_df.sort_values('gasDayStart', ascending=True)
storage_df = storage_df[['gasDayStart', 'gasDayEnd', 'gasInStorage']]
storage_df = storage_df.drop(columns=['gasDayEnd'])
storage_df = storage_df.rename(columns={'gasDayStart': 'Date', 'gasInStorage': 'Storage(TWh)'})
print(storage_df)

x = pd.merge(data, storage_df, on='Date', how='outer')
x.to_csv('NG_daily1.csv', index=False)
         
#%% Load ALSI LNG sendout data

API_KEY = ALSI_API_KEY
headers = {
    "x-key": API_KEY,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# 1. Define yearly intervals to bypass the server's loop block
years = list(range(2020, 2027))
all_lng_data = []

print("Extracting full multi-year history via yearly chunking...")

for year in years:
    start = f"{year}-01-01"
    end = f"{year}-12-31" if year < 2026 else "2026-07-20"
    
    # Grab the whole year in a single page request (size=366 covers leap years)
    url = f"https://alsi.gie.eu/api?type=eu&from={start}&to={end}&size=366"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data_list = response.json().get('data', [])
        all_lng_data.extend(data_list)
        print(f"-> Successfully pulled {year} ({len(data_list)} rows)")
    else:
        print(f"-> Failed to pull {year}: Status {response.status_code}")
        
    time.sleep(0.3) # Keep the server happy

# 2. Build and structure the full historical DataFrame
lng_df = pd.DataFrame(all_lng_data)
lng_df['gasDayStart'] = pd.to_datetime(lng_df['gasDayStart'])
lng_df = lng_df.sort_values('gasDayStart', ascending=True)
lng_df = lng_df[lng_df['gasDayStart'] >= '2020-01-01']
print(lng_df)

# 3. Apply the dynamic column check we built earlier
if 'sendOut' in lng_df.columns:
    lng_df['lngSendout_GWh'] = pd.to_numeric(lng_df['sendOut'])
else:
    lng_df['lngSendout_GWh'] = 0.0

if 'lngInventory' in lng_df.columns:
    lng_df['lngInventory_m3'] = pd.to_numeric(lng_df['lngInventory'])
else:
    lng_df['lngInventory_m3'] = pd.NA

lng_df = lng_df.sort_values('gasDayStart')
lng_df.set_index('gasDayStart', inplace=True)

print("\n--- Download Complete! ---")
print(f"Data now spans from: {lng_df.index.min().strftime('%Y-%m-%d')} to {lng_df.index.max().strftime('%Y-%m-%d')}")
print(f"Total entries: {len(lng_df)} rows")
lng_df = lng_df.reset_index()
print(lng_df)

lng = lng_df[['gasDayStart', 'sendOut']]
lng = lng.rename(columns={'gasDayStart':'Date', 'sendOut': 'LNG_sendout(GWh/d)'})
print(lng)

x = pd.merge(x, lng, on='Date', how='outer')
print(x)
x.to_csv('NG_daily2.csv', index=False)


#%% Load Entsog gas pipeline entries

API_URL = "https://transparency.entsog.eu/api/v1/operationaldatas"


# ==========================================================
# POINT MAPPING TABLE
# ==========================================================

corridors = {

    "NO": [
        "Dornum / NETRA (OGE)", 
        "Emden (EPT1) (OGE)", 
        "Emden (EPT1) (GTS)", 
        "Emden (EPT1) (GUD)", 
        "Zeebrugge ZPT", 
        "Dunkerque", 
        "Easington", 
        "St. Fergus"
    ],

    "TR": [
        "Strandzha 2 (BG) / Malkoclar (TR)",
        "Kipi (TR) / Kipi (GR)",
        "Kipoi"
    ],

    "BY": [
        "Konratki",
        "Kotlovka",
        "Tieterowka",
        "Wysokoje"
    ],

    "UA": [
        "Beregdaróc 1400 (HU) - Beregovo (UA) (UA>HU)",
        "GCP GAZ-SYSTEM/UA TSO",
        "Isaccea (RO) - Orlovka (UA)",
        "Isaccea (RO) - Orlovka (UA) II",
        "Uzhgorod (UA) - Velké Kapušany (SK)",
        "Uzhhorod (UA) - Velké Kapušany (SK)",
        "VIP Bereg (HU) / VIP Bereg (UA)",
        "VIP Mediesu Aurit - Isaccea (RO-UA)",
    ],

    "LY": [
        "Gela",

    ],

    "DZ": [
        "Mazara del Vallo",
        "Almería"
    ],

    "RU-DE": [
        "Nordstream"
    ],

    "RU-Baltic": [
        "Luhamaa",
        "Misso Izborsk",
        "Narva",
        "Värska",
        "Imatra"
    ],

    "MA": [
        "Tarifa"
    ],

}

# PARAMETERS

year = 2020

months = pd.date_range(
    start=f"{year}-01-01",
    end=f"{year}-12-31",
    freq="MS"
)

all_monthly_dfs = []

print(f"Downloading {year} data...")


# DOWNLOAD LOOP

for start_month in months:

    start_str = start_month.strftime("%Y-%m-%d")
    end_str = (
        start_month + pd.offsets.MonthEnd(0)
    ).strftime("%Y-%m-%d")


    params = {
        "from": start_str,
        "to": end_str,
        "indicator": "Physical Flow",
        "directionKey": "entry",
        "limit": -1
    }


    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=45
        )


        if response.status_code != 200:
            print(
                f"Failed {start_str}: {response.status_code}"
            )
            continue


        raw_data = (
            response.json()
            .get("operationaldatas", [])
        )


        month_records = []


        # Classify points into corridors

        for row in raw_data:

            point_label = row.get("pointLabel")
            value = row.get("value")


            if value is None:
                continue


            for corridor, points in corridors.items():

                if point_label in points:

                    month_records.append({

                        "date": pd.to_datetime(
                            row.get("periodFrom")
                        ).date(),

                        "corridor": corridor,

                        "pointLabel": point_label,

                        "value_gwh": (
                            pd.to_numeric(
                                value,
                                errors="coerce"
                            ) / 1_000_000
                        )
                    })


        if month_records:
            all_monthly_dfs.append(
                pd.DataFrame(month_records)
            )


        print(start_str, "done")


    except Exception as e:

        print(
            f"Error on {start_str}: {e}"
        )


    time.sleep(1)


# COMBINE RESULTS

if all_monthly_dfs:

    flows = pd.concat(
        all_monthly_dfs,
        ignore_index=True
    )


    # 1. Detailed point flows matrix (Individual points as columns)

    detailed = (
        flows
        .groupby(
            [
                "date",
                "pointLabel"
            ]
        )["value_gwh"]
        .sum()
        .reset_index()
    )

    point_matrix = (
        detailed
        .pivot(
            index="date",
            columns="pointLabel",
            values="value_gwh"
        )
        .fillna(0)
    )

    # Prefix individual point columns to distinguish them clearly
    point_matrix = point_matrix.add_prefix("point_")


    # 2. Corridor totals matrix (Summed to NO, TR, BY, etc.)

    corridor_daily = (
        flows
        .groupby(
            [
                "date",
                "corridor"
            ]
        )["value_gwh"]
        .sum()
        .reset_index()
    )

    corridor_matrix = (
        corridor_daily
        .pivot(
            index="date",
            columns="corridor",
            values="value_gwh"
        )
        .fillna(0)
    )

    # Prefix corridor columns to distinguish them clearly
    corridor_matrix = corridor_matrix.add_prefix("corridor_")


    # Combine both individual points and corridor totals into one master dataframe

    master_matrix = pd.concat([corridor_matrix, point_matrix], axis=1).fillna(0)


    output_filename = (
        f"europe_pipeline_imports_detailed_{year}.csv"
    )

    master_matrix.to_csv(
        output_filename
    )


    print("\nSUCCESS")
    print(output_filename)
    print(master_matrix.head())

else:

    print("No data compiled")

#%% Load Entsog gas pipeline exits

API_URL = "https://transparency.entsog.eu/api/v1/operationaldatas"


# POINT MAPPING TABLE

corridors = {

    "ES": [
        "Tarifa"
    ],
}


# PARAMETERS

year = 2026

months = pd.date_range(
    start=f"{year}-01-01",
    end=f"{year}-12-31",
    freq="MS"
)

all_monthly_dfs = []

print(f"Downloading {year} data...")


# DOWNLOAD LOOP

for start_month in months:

    start_str = start_month.strftime("%Y-%m-%d")
    end_str = (
        start_month + pd.offsets.MonthEnd(0)
    ).strftime("%Y-%m-%d")


    params = {
        "from": start_str,
        "to": end_str,
        "indicator": "Physical Flow",
        "directionKey": "exit",
        "limit": -1
    }


    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=45
        )


        if response.status_code != 200:
            print(
                f"Failed {start_str}: {response.status_code}"
            )
            continue


        raw_data = (
            response.json()
            .get("operationaldatas", [])
        )


        month_records = []


        # Classify points into corridors

        for row in raw_data:

            point_label = row.get("pointLabel")
            value = row.get("value")


            if value is None:
                continue


            for corridor, points in corridors.items():

                if point_label in points:

                    month_records.append({

                        "date": pd.to_datetime(
                            row.get("periodFrom")
                        ).date(),

                        "corridor": corridor,

                        "pointLabel": point_label,

                        "value_gwh": (
                            pd.to_numeric(
                                value,
                                errors="coerce"
                            ) / 1_000_000
                        )
                    })


        if month_records:
            all_monthly_dfs.append(
                pd.DataFrame(month_records)
            )


        print(start_str, "done")


    except Exception as e:

        print(
            f"Error on {start_str}: {e}"
        )


    time.sleep(1)



# COMBINE RESULTS

if all_monthly_dfs:

    flows = pd.concat(
        all_monthly_dfs,
        ignore_index=True
    )


    # 1. Detailed point flows matrix (Individual points as columns)

    detailed = (
        flows
        .groupby(
            [
                "date",
                "pointLabel"
            ]
        )["value_gwh"]
        .sum()
        .reset_index()
    )

    point_matrix = (
        detailed
        .pivot(
            index="date",
            columns="pointLabel",
            values="value_gwh"
        )
        .fillna(0)
    )

    # Prefix individual point columns to distinguish them clearly
    point_matrix = point_matrix.add_prefix("point_")


    # 2. Corridor totals matrix (Summed to NO, TR, BY, etc.)

    corridor_daily = (
        flows
        .groupby(
            [
                "date",
                "corridor"
            ]
        )["value_gwh"]
        .sum()
        .reset_index()
    )

    corridor_matrix = (
        corridor_daily
        .pivot(
            index="date",
            columns="corridor",
            values="value_gwh"
        )
        .fillna(0)
    )

    # Prefix corridor columns to distinguish them clearly
    corridor_matrix = corridor_matrix.add_prefix("corridor_")


    # Combine both individual points and corridor totals into one master dataframe

    master_matrix = pd.concat([corridor_matrix, point_matrix], axis=1).fillna(0)


    output_filename = (
        f"europe_pipeline_imports_detailed_{year}.csv"
    )

    master_matrix.to_csv(
        output_filename
    )


    print("\nSUCCESS")
    print(output_filename)
    print(master_matrix.head())


else:

    print("No data compiled")

#%% Add Entsog data to data file

z1 = pd.read_csv('europe_pipeline_imports_detailed_2021.csv')
z2 = pd.read_csv('europe_pipeline_imports_detailed_2022.csv')
z3 = pd.read_csv('europe_pipeline_imports_detailed_2023.csv')
z4 = pd.read_csv('europe_pipeline_imports_detailed_2024.csv')
z5 = pd.read_csv('europe_pipeline_imports_detailed_2025.csv')
z6 = pd.read_csv('europe_pipeline_imports_detailed_2026.csv')

z1.columns = z1.columns.str.strip()
z2.columns = z2.columns.str.strip()
z3.columns = z3.columns.str.strip()
z4.columns = z4.columns.str.strip()
z5.columns = z5.columns.str.strip()
z6.columns = z6.columns.str.strip()

z = pd.concat([z1, z2, z3, z4, z5, z6], axis=0, ignore_index=True)
z = z.rename(columns={'date':'Date'})
print(z)


y = pd.read_csv('NG_daily2.csv')

y = pd.merge(y, z, on='Date', how='outer')
y.to_csv('NG_daily3.csv', index=False)


#%% Load temperature data

# 1. SETUP LOCATIONS & PARAMETERS

locations = {
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Rome": {"lat": 41.9028, "lon": 12.4964},
}
start_date = "2020-01-01"

# Use today's date dynamically to cover up to the present
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
base_url = "https://archive-api.open-meteo.com/v1/archive"
all_city_dfs = []
print(
    f"Fetching historical daily weather data from Open-Meteo ({start_date} to {end_date})..."
)

# 2. FETCH DATA PER LOCATION FROM OPEN-METEO ARCHIVE API

for city, coords in locations.items():
  params = {
      "latitude": coords["lat"],
      "longitude": coords["lon"],
      "start_date": start_date,
      "end_date": end_date,
      "daily": [
          "temperature_2m_max",
          "temperature_2m_min",
          "temperature_2m_mean",
      ],
      "timezone": "auto",
  }
  response = requests.get(base_url, params=params)

  if response.status_code == 200:
    data = response.json()
    daily_data = data.get("daily", {})
    df_city = pd.DataFrame({
        "date": pd.to_datetime(daily_data.get("time")),
        "city": city,
        "temp_max": daily_data.get("temperature_2m_max"),
        "temp_min": daily_data.get("temperature_2m_min"),
        "temp_mean": daily_data.get("temperature_2m_mean"),
    })
    all_city_dfs.append(df_city)
    print(f"Successfully downloaded data for {city}")
  else:
    print(f"Failed to fetch data for {city}: {response.status_code} - {response.text}")

# 3. COMBINE, CALCULATE CDD & HDD, AND EXPORT

if all_city_dfs:
  combined_df = pd.concat(all_city_dfs, ignore_index=True)

  # Standard Energy Baselines:
  # HDD Base: 18°C (Heating required when mean temp drops below 18°C)
  # CDD Base: 22°C (Cooling required when mean temp exceeds 22°C)
  combined_df["HDD"] = (18.0 - combined_df["temp_mean"]).clip(lower=0)
  combined_df["CDD"] = (combined_df["temp_mean"] - 22.0).clip(lower=0)

  # Reorder and sort
  final_output = combined_df[

      ["date", "city", "temp_mean", "temp_max", "temp_min", "HDD", "CDD"]

  ].sort_values(["date", "city"])
  output_filename = "cdd_hdd_open_meteo_daily.csv"
  final_output.to_csv(output_filename, index=False)
  print(f"\nSUCCESS! Saved daily data to {output_filename}")
  print(final_output.head(10))
else:
  print("No data retrieved.") 

daily_meteo = pd.read_csv('cdd_hdd_open_meteo_daily.csv')
x = daily_meteo.pivot(index='date',columns='city', values='HDD')
x = x.rename(columns={'Berlin':'Berlin_HDD', 'London': 'London_HDD', 'Rome': 'Rome_HDD'})
x = x.reset_index()
y = daily_meteo.pivot(index='date',columns='city', values='CDD')
y = y.rename(columns={'Berlin':'Berlin_CDD', 'London': 'London_CDD', 'Rome': 'Rome_CDD'})
y = y.reset_index()
z = pd.merge(x, y, on='date', how='outer')
z = z.rename(columns={'date':'Date'})

x = pd.read_csv('NG_daily3.csv')
x = pd.merge(x, z, on='Date', how='outer')
x.to_csv('NG_daily4.csv', index=False)

#%% Load FX data

import yfinance as yf

# Download daily EUR/USD data
eurusd = yf.Ticker("EURUSD=X")
df = eurusd.history(start='2020-01-01', interval="1d")
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.date
df = df[['Date', 'Close']]
df = df.rename(columns={'Close': 'USD-EUR'})
print(df)

x = pd.read_csv('NG_daily4.csv')
x['Date'] = pd.to_datetime(x['Date'])
df['Date'] = pd.to_datetime(df['Date'])
x = pd.merge(x, df, on='Date', how='left')
x.to_csv('NG_daily5.csv', index=False)


#%% Load VIX data

import yfinance as yf

# Download daily VIX history
vix = yf.Ticker("^VIX")
vix = vix.history(start='2020-01-01', interval="1d")

vix = vix.reset_index()
vix['Date'] = pd.to_datetime(df['Date']).dt.date
vix = vix[['Date', 'Close']]
vix = vix.rename(columns={'Close': 'VIX'})
print(vix)

x = pd.read_csv('NG_daily5.csv')
x['Date'] = pd.to_datetime(x['Date'])
vix['Date'] = pd.to_datetime(vix['Date'])
x = pd.merge(x, vix, on='Date', how='left')
x.to_csv('NG_daily6.csv', index=False)


#%% Load ENTSOE data via Energy-Charts

# 1. SETUP PARAMETERS (Energy-Charts API)
country_code = "de"  # 'de' for Germany (covers DE-LU bidding zone)
bzn = "DE-LU"
start_date = "2020-01-01"
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

print(f"Fetching power mix and prices for {country_code.upper()} from Energy-Charts ({start_date} to {end_date})...\n")

# 2. FETCH POWER GENERATION MIX
gen_url = "https://api.energy-charts.info/public_power"
gen_params = {
    "country": country_code,
    "start": start_date,
    "end": end_date
}

gen_response = requests.get(gen_url, params=gen_params)

if gen_response.status_code == 200:
    gen_data = gen_response.json()
    
    # Energy-Charts returns timestamps under 'unix_seconds'
    timestamps = pd.to_datetime(gen_data["unix_seconds"], unit="s")
    
    # Production types contain power values (usually in MW) for each source
    prod_dict = {}
    for item in gen_data["production_types"]:
        prod_dict[item["name"]] = item["data"]
        
    df_generation = pd.DataFrame(prod_dict, index=timestamps)
    df_generation.index.name = "timestamp"
    print("Successfully downloaded power generation mix.")
else:
    print(f"Failed to fetch generation data: {gen_response.status_code} - {gen_response.text}")

# 3. FETCH DAY-AHEAD POWER PRICES
price_url = "https://api.energy-charts.info/price"
price_params = {
    "bzn": bzn,
    "start": start_date,
    "end": end_date
}

price_response = requests.get(price_url, params=price_params)

if price_response.status_code == 200:
    price_data = price_response.json()
    
    # Prices endpoint also uses 'unix_seconds'
    price_timestamps = pd.to_datetime(price_data["unix_seconds"], unit="s")
    df_prices = pd.DataFrame({
        "price_eur_per_mwh": price_data["price"]
    }, index=price_timestamps)
    df_prices.index.name = "timestamp"
    
    print("Successfully downloaded power prices.")
else:
    print(f"Failed to fetch price data: {price_response.status_code} - {price_response.text}")

# 4. MERGE, RESAMPLE TO DAILY, AND EXPORT
if 'df_generation' in locals() and 'df_prices' in locals():
    # Combine generation and prices into a master hourly dataframe
    master_df = df_generation.join(df_prices, how="outer")
    
    # Resample from hourly to daily averages
    daily_df = master_df.resample("D").mean().reset_index()
    
    output_filename = f"energy_charts_{country_code}_daily_2020_onwards.csv"
    daily_df.to_csv(output_filename, index=False)
    
    print(f"\nSUCCESS! Saved daily power mix and price data to {output_filename}")
    print(daily_df.head(10))

#%% Load EU+UK power generation mix data

# 1. SETUP PARAMETERS
start_date = "2020-01-01"
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
gen_url = "https://api.energy-charts.info/public_power"

def fetch_generation(country_code):
    print(f"Fetching data for '{country_code}' from Energy-Charts...")
    response = requests.get(gen_url, params={"country": country_code, "start": start_date, "end": end_date})
    if response.status_code == 200:
        data = response.json()
        timestamps = pd.to_datetime(data["unix_seconds"], unit="s")
        prod_dict = {item["name"]: item["data"] for item in data["production_types"]}
        df = pd.DataFrame(prod_dict, index=timestamps)
        return df
    else:
        raise Exception(f"Failed to fetch {country_code}: {response.status_code} - {response.text}")

# 2. DOWNLOAD EU AND UK SEPARATELY & ADD TOGETHER
# Download EU aggregate and UK individually with only 2 API calls total
df_eu = fetch_generation("eu")
df_uk = fetch_generation("uk")

# Align indexes and sum the generation values across matching columns and timestamps
combined_hourly = df_eu.add(df_uk, fill_value=0)

# 3. RESAMPLE TO DAILY AND EXPORT
daily_df = combined_hourly.resample("D").mean().reset_index()
daily_df.rename(columns={"index": "timestamp"}, inplace=True)

output_filename = "energy_charts_eu_plus_uk_summed_daily.csv"
daily_df.to_csv(output_filename, index=False)

print(f"\nSUCCESS! Combined EU + UK data saved to {output_filename}")
print(daily_df.head(10))

#%%

x = pd.read_csv('NG_daily6.csv')
x['Date'] = pd.to_datetime(x['Date'])
gen = pd.read_csv('energy_charts_eu_plus_uk_summed_daily.csv')
gen = gen.rename(columns={'timestamp':'Date'})
gen['Date'] = pd.to_datetime(gen['Date'])
x = pd.merge(x, gen, on='Date', how='left')
x.to_csv('NG_daily7.csv', index=False)

#%% Load DE and IT power prices

# Correct bidding zones based on Energy-Charts specifications
# IT-Calabria uses 'IT-CALA' for the API query, mapped back to 'IT-Calabria' for user output
zones = {"DE-LU": "DE-LU", "IT-Calabria": "IT-Calabria"}

all_data = []

for zone_name, bzn_code in zones.items():
  url = "https://api.energy-charts.info/price"
  # Providing a start date; omitting 'end' defaults to current/latest available data
  params = {"bzn": bzn_code, "start": "2021-01-01", "end": "2026-07-20"}

  response = requests.get(url, params=params)

  if response.status_code == 200:
    data = response.json()

    if "unix_seconds" in data and "price" in data:
      timestamps = pd.to_datetime(data["unix_seconds"], unit="s")
      prices = data["price"]

      df_temp = pd.DataFrame({
          "Timestamp": timestamps,
          "Zone": zone_name,  # Keeps user-friendly name 'IT-Calabria'
          "Price_EUR_MWh": prices,
      })
      all_data.append(df_temp)
      print(
          f"Successfully downloaded {len(df_temp)} hourly records for"
          f" {zone_name}"
      )
    else:
      print(f"Unexpected data structure received for {zone_name}")
  else:
    print(
        f"Failed to fetch data for {zone_name}. Status code:"
        f" {response.status_code}"
    )

if all_data:
  final_df = pd.concat(all_data, ignore_index=True)
  final_df.to_csv("energy_charts_prices_2021_onwards.csv", index=False)
  print(
      "Price data successfully saved to energy_charts_prices_2021_onwards.csv"
  )
else:
  print("No data collected.")


x = pd.read_csv('NG_daily7.csv')
x['Date'] = pd.to_datetime(x['Date']).dt.date
prices = pd.read_csv('energy_charts_prices_2021_onwards.csv')
prices['Timestamp'] = pd.to_datetime(prices['Timestamp']).dt.date
prices = prices.groupby(['Timestamp', 'Zone'])['Price_EUR_MWh'].mean().reset_index()
prices = prices.pivot(index='Timestamp', columns='Zone', values='Price_EUR_MWh').reset_index()
prices = prices.rename(columns={'Timestamp':'Date', 'DE-LU':'DE_price(EUR/MWh)', 'IT-Calabria':'IT_price(EUR/MWh)'})
prices['Date'] = pd.to_datetime(prices['Date']).dt.date

x = pd.merge(x, prices, on='Date', how='left')
x.to_csv('NG_daily8.csv', index=False)

#%% Process date into final version

x = pd.read_csv('NG_daily8.csv')
x['Date'] = pd.to_datetime(x['Date'])
x = x[x['Date'] >= '2021-01-01']
x['no_gas'] = x['JKM(USD/mmbtu)'] + x['HH(USD/mmbtu)'] + x['TTF(USD/mmbtu)']
x = x[~x['no_gas'].isna()]
x = x.drop(columns=['no_gas'])
print(x)
x.to_csv('NG_daily9.csv', index=False)


#%% Graphing to check data and EDA

# Load data
x = pd.read_csv('NG_daily9.csv')
x = x[x['Date'] >= '2024-01-01']

data_melt = x.melt(
    id_vars=['Date'], 
    var_name='Series', 
    value_name='Value'
)

first_var = 'TTF(USD/mmbtu)'
second_var = 'Rome_CDD'

color_mapping = {
    first_var : '#1f77b4',   # Custom Blue
    second_var : '#ff7f0e'      # Custom Orange
}

# Select both series for the dual-axis chart
filtered_df = data_melt

# Create separate Plotly Express figures for each series
fig1 = px.line(
    filtered_df[filtered_df['Series'] == first_var], 
    x='Date', 
    y='Value', 
    color='Series',
    color_discrete_map=color_mapping
)
fig2 = px.line(
    filtered_df[filtered_df['Series'] == second_var], 
    x='Date', 
    y='Value', 
    color='Series',
    color_discrete_map=color_mapping
)

# Create a subplot figure with a secondary y-axis enabled
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add both traces to the figure
fig.add_traces(fig1.data + fig2.data)

# Route the Storage series to the secondary y-axis
fig.update_traces(yaxis="y2", selector=dict(name=second_var))

# Apply line styling across traces
fig.update_traces(line=dict(width=1.2))

# Configure custom ranges for both y-axes
fig.update_yaxes(title_text=first_var, secondary_y=False)
fig.update_yaxes(title_text=second_var, secondary_y=True)

# Layout configurations
fig.update_layout(
    title=f'Gas: {first_var} and {second_var}',
    hovermode='x unified',
    xaxis_tickangle=-45,
    template='plotly_white'
)

fig.show()

# %% Further charts

x = pd.read_csv('NG_daily9.csv')
x = x[x['Date'] >= '2024-01-01']

first_var = 'corridor_NO'
second_var = 'TTF(USD/mmbtu)'

# Assuming 'filtered_df' is your subset of the melted dataframe
fig = px.scatter(
    x, 
    x=first_var, 
    y=second_var,
    title='Gas Metrics Scatter Plot',
    labels={'Date': 'Date', 'Value': 'Value'},
)

# Optional styling
fig.update_traces(marker=dict(size=6, color='firebrick')) # Adjust marker size if needed
fig.update_layout(
    xaxis_tickangle=-45,
    template='plotly_white'
)

fig.show()