#%% Import libraries

import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px

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
data = data[data['Date'] >= '2015-01-01']
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data = data.set_index('Date')
data = data.resample('MS').mean()
data = data.reset_index()
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

data = data.rename(columns={'TTF':'TTF(USD/mmbtu)', 'JKM':'JKM(USD/mmbtu)', 'HH':'HH(USD/mmbtu)'})
output_filename = "NG_monthly0.csv"
data.to_csv(output_filename, index=False)

#%% Graphing natural gas prices to check data

print(data_melt.info())

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

#%% GIE storage and sendout data using API


API_KEY = GIE_API_KEY
headers = {
    "x-key": API_KEY,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# 1. Establish your target timeframe constraints
start_date = "2015-01-01"
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

storage_df['Date'] = pd.to_datetime(data['Date'])
storage_df = storage_df.sort_values('Date')
storage_df = storage_df.set_index('Date')
storage_df = storage_df.resample('MS').mean()
storage_df = storage_df.reset_index()
print(storage_df)

# %%

x = pd.read_csv('NG_monthly0.csv')
x = pd.merge(data, storage_df, on='Date', how='outer')
x.to_csv('NG_monthly1.csv', index=False)
print(x)

#%% Load ALSI LNG sendout data

API_KEY = ALSI_API_KEY
headers = {
    "x-key": API_KEY,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# 1. Define yearly intervals to bypass the server's loop block
years = list(range(2015, 2027))
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

#%% Process ALSI LNG sendout data

# 2. Build and structure the full historical DataFrame
lng_df = pd.DataFrame(all_lng_data)
lng_df['gasDayStart'] = pd.to_datetime(lng_df['gasDayStart'])
lng_df = lng_df.sort_values('gasDayStart', ascending=True)
lng_df = lng_df[lng_df['gasDayStart'] >= '2015-01-01']
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

#%% Add LNG sendout data to data file

lng = lng_df[['gasDayStart', 'sendOut']]
lng = lng.rename(columns={'gasDayStart':'Date', 'sendOut': 'LNG_sendout(GWh/d)'})
lng.to_csv('x.csv')
lng['Date'] = pd.to_datetime(lng['Date'])
lng = lng.sort_values('Date')
lng = lng.set_index('Date')
lng['LNG_sendout(GWh/d)'] = pd.to_numeric(lng['LNG_sendout(GWh/d)'], errors='coerce')
lng = lng.resample('MS').mean() 
lng = lng.reset_index()
print(lng.head(20))
print(lng.tail(20))
x = pd.read_csv('NG_monthly0.csv')
x['Date'] = pd.to_datetime(x['Date'])
x = pd.merge(x, lng, on='Date', how='outer')
print(x)
x.to_csv('NG_monthly2.csv', index=False)
print(x)

# %% Load temperature data

# 1. SETUP LOCATIONS & PARAMETERS
locations = {
    "Berlin": {"lat": 52.5200, "lon": 13.4050},
    "London": {"lat": 51.5074, "lon": -0.1278},
    "Rome": {"lat": 41.9028, "lon": 12.4964},
}

start_date = "2015-01-01"
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

base_url = "https://archive-api.open-meteo.com/v1/archive"
all_city_dfs = []

print(
    f"Fetching historical daily weather data from Open-Meteo ({start_date} to {end_date})..."
)

# 2. FETCH DATA PER LOCATION FROM OPEN-METEO ARCHIVE API# ==========================================================
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

# 3. COMBINE, CALCULATE CDD/HDD, AND RESAMPLE TO MONTHLY
if all_city_dfs:
  combined_df = pd.concat(all_city_dfs, ignore_index=True)

  # Calculate daily CDD and HDD
  combined_df["HDD"] = (18.0 - combined_df["temp_mean"]).clip(lower=0)
  combined_df["CDD"] = (combined_df["temp_mean"] - 22.0).clip(lower=0)

  monthly_dfs = []

  # Group by city so resampling happens independently per location
  for city, group in combined_df.groupby("city"):
    group = group.sort_values("date").set_index("date")

    # Resample to Month Start ('MS'):
    # - Temperatures get averaged (.mean())
    # - Degree Days get accumulated (.sum())
    resampled = group.resample("MS").agg({
        "temp_mean": "mean",
        "temp_max": "mean",
        "temp_min": "mean",
        "HDD": "sum",
        "CDD": "sum"
    }).reset_index()

    resampled["city"] = city
    monthly_dfs.append(resampled)

  # Combine back into a single dataframe
  final_output = pd.concat(monthly_dfs, ignore_index=True)

  # Reorder columns nicely
  final_output = final_output[
      ["date", "city", "temp_mean", "temp_max", "temp_min", "HDD", "CDD"]
  ].sort_values(["date", "city"])

  output_filename = "cdd_hdd_open_meteo_monthly.csv"
  final_output.to_csv(output_filename, index=False)

  print(f"\nSUCCESS! Saved monthly data to {output_filename}")
  print(final_output.head(10))
else:
  print("No data retrieved.")

#%% CDD and HDD processing

monthly_meteo = pd.read_csv('cdd_hdd_open_meteo_monthly.csv')
x = monthly_meteo.pivot(index='date',columns='city', values='HDD')
x = x.rename(columns={'Berlin':'Berlin_HDD', 'London': 'London_HDD', 'Rome': 'Rome_HDD'})
x = x.reset_index()
y = monthly_meteo.pivot(index='date',columns='city', values='CDD')
y = y.rename(columns={'Berlin':'Berlin_CDD', 'London': 'London_CDD', 'Rome': 'Rome_CDD'})
y = y.reset_index()
z = pd.merge(x, y, on='date', how='outer')
z = z.rename(columns={'date':'Date'})
print(z)

x = pd.read_csv('NG_monthly2.csv')
x = pd.merge(x, z, on='Date', how='outer')
x.to_csv('NG_monthly3.csv', index=False)


#%% Load FX data

# Download daily EUR/USD data
eurusd = yf.Ticker("EURUSD=X")
df = eurusd.history(start='2015-01-01', interval="1d")
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
df = df[['Date', 'Close']]
df = df.rename(columns={'Close': 'USD-EUR'})
df = df.set_index('Date')
df = df.resample('MS').mean()
df = df.reset_index()
print(df)

x = pd.read_csv('NG_monthly3.csv')
x['Date'] = pd.to_datetime(x['Date'])
df['Date'] = pd.to_datetime(df['Date'])
x = pd.merge(x, df, on='Date', how='left')
x.to_csv('NG_monthly4.csv', index=False)


#%% Load VIX data

# Download daily VIX history
vix = yf.Ticker("^VIX")
vix = vix.history(start='2015-01-01', interval="1d")

vix = vix.reset_index()
vix['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
vix = vix[['Date', 'Close']]
vix = vix.rename(columns={'Close': 'VIX'})
vix = vix.set_index('Date')
vix = vix.resample('MS').mean()
vix = vix.reset_index()
print(vix)

x = pd.read_csv('NG_monthly4.csv')
x['Date'] = pd.to_datetime(x['Date'])
vix['Date'] = pd.to_datetime(vix['Date'])
x = pd.merge(x, vix, on='Date', how='left')
x.to_csv('NG_monthly5.csv', index=False)

#%% Load EU+UK power generation mix data

# 1. SETUP PARAMETERS
start_date = "2015-01-01"
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

output_filename = "energy_charts_eu_plus_uk_summed_daily_monthly.csv"
daily_df.to_csv(output_filename, index=False)

print(f"\nSUCCESS! Combined EU + UK data saved to {output_filename}")
print(daily_df.head(10))

#%%

x = pd.read_csv('NG_monthly5.csv')
x['Date'] = pd.to_datetime(x['Date'])
gen = pd.read_csv('energy_charts_eu_plus_uk_summed_daily_monthly.csv')
gen = gen.rename(columns={'timestamp':'Date'})
gen['Date'] = pd.to_datetime(gen['Date'])
gen = gen.set_index('Date')
gen = gen.resample('MS').mean()
gen = gen.reset_index()
print(gen)

x = pd.merge(x, gen, on='Date', how='left')
x.to_csv('NG_monthly6.csv', index=False)


#%% Load DE and IT power prices

# Eurostat API endpoint for dataset 'nrg_cb_gasm'
dataset_code = 'nrg_cb_gasm'
url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset_code}"

# Define query parameters for EU-27, natural gas, million cubic metres, and your 4 flows
params = {
    'format': 'JSON',
    'lang': 'en',
    'geo': 'EU27_2020',
    'siec': 'G3000',
    'unit': 'MIO_M3',
    'nrg_bal': ['PRO', 'TI_EGC', 'FC_IND', 'FC_O_S'],
    'sinceTimePeriod': '2015-01'
}

print(f"Fetching dataset '{dataset_code}' from Eurostat API...")
response = requests.get(url, params=params)

if response.status_code == 200:
    result = response.json()
    
    values = result.get('value', {})
    dimensions = result.get('dimension', {})
    
    nrg_bal_index = dimensions.get('nrg_bal', {}).get('category', {}).get('index', {})
    time_index = dimensions.get('time', {}).get('category', {}).get('index', {})
    
    rev_nrg_bal = {v: k for k, v in nrg_bal_index.items()}
    rev_time = {v: k for k, v in time_index.items()}
    
    n_times = len(time_index)
    
    data_rows = []
    for pos_str, val in values.items():
        pos = int(pos_str)
        t_idx = pos % n_times
        nrg_idx = pos // n_times
        
        time_code = rev_time.get(t_idx)
        nrg_code = rev_nrg_bal.get(nrg_idx)
        
        if time_code and nrg_code:
            data_rows.append({
                'Date': time_code,
                'nrg_bal': nrg_code,
                'Value': val
            })
            
    df = pd.DataFrame(data_rows)
    
    # Format Date to Month Start ('MS') datetime format
    df['Date'] = pd.to_datetime(df['Date'] + '-01')
    
    # Filter strictly for Jan 2015 onwards
    df = df[df['Date'] >= '2015-01-01']
    
    # Pivot flows into distinct user-friendly columns
    final_pivot = df.pivot(index='Date', columns='nrg_bal', values='Value').reset_index()
    
    rename_mapping = {
        'PRO': 'Indigenous_Production(MIO_M3)',
        'TI_EGC': 'Transformation_Electricity_Heat(MIO_M3)',
        'FC_IND': 'Final_Consumption_Industry(MIO_M3)',
        'FC_O_S': 'Final_Consumption_Other(MIO_M3)'
    }
    final_pivot = final_pivot.rename(columns=rename_mapping)
    
    print("\nSuccessfully compiled all 4 Eurostat datasets from Jan 2015 onwards:")
    print(final_pivot.head(15))
    
    final_pivot.to_csv('eurostat_gas_balance_monthly.csv', index=False)
else:
    print(f"Failed to fetch data. Status code: {response.status_code} - {response.text}")


# %% Compile and finalize all data

y = pd.read_excel('22 July 2026 Eurostat.xlsx', sheet_name='Transform')
y['Date'] = pd.to_datetime(y['Date'] + '-01')
y['Date'] = pd.to_datetime(y['Date']).dt.tz_localize(None)
print(y)
x = pd.read_csv('NG_monthly7.csv')
x['Date'] = pd.to_datetime(x['Date'])
x = pd.merge(x, y, on='Date', how='left')
print(x)
x.to_csv('NG_monthly8.csv')
