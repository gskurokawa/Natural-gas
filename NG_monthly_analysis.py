#%%

import pandas as pd

x = pd.read_csv('NG_monthly9.csv')
x['Date'] = pd.to_datetime(x['Date'])
x['Production(bcm)'] = pd.to_numeric(x['Production(bcm)'], errors='coerce')
x['LNG_sendout(bcm)'] = pd.to_numeric(x['LNG_sendout(bcm)'], errors='coerce')
pipelist = ['BY', 'DZ', 'LY', 'MA', 'NO', 'RU-Baltic', 'TR', 'UA', 'DE', 'ES']
x['Piped'] = 0
for i in pipelist:
    x['Piped'] = x['Piped'] + x['bcm_' + i]
print(x)
x['Net_supply'] = (
    x['LNG_sendout(bcm)'] 
    + x['Production(bcm)'] 
    + x['Piped'] 
    - x['CH_RS_net_exports'] 
    - x['bcm_ES']
)
x['Total(bcm)'] = pd.to_numeric(x['Total(bcm)'], errors='coerce')
x['Calc_storage_change'] = x['Net_supply'] - x['Total(bcm)']
print(x[['Net_supply', 'Total(bcm)']])
print(x[['Calc_storage_change', 'Av_storage(bcm)']])
x['Act_storage_change'] = -x['Av_storage(bcm)'].diff(-1)
print(x['Act_storage_change'])
x['Diff_storage_changes'] = x['Act_storage_change'] - x['Calc_storage_change']
print(x['Diff_storage_changes'])
x.to_csv('diff.csv')


import plotly.express as px

# 1. Ensure 'Date' is in datetime format so Plotly sorts and plots it correctly
x['Date'] = pd.to_datetime(x['Date'])

# 2. Reshape the dataframe from wide to long format for Plotly Express
df_melted = x.melt(
    id_vars=['Date'], 
    value_vars=['Calc_storage_change', 'Act_storage_change'],
    var_name='Metric', 
    value_name='Value'
)

# 3. Create the line plot
fig = px.line(
    df_melted, 
    x='Date', 
    y='Value', 
    color='Metric',
    title='Storage Change vs Average Storage',
    labels={'Value': 'bcm', 'Date': 'Date'}
)

# 4. Display the plot
fig.show()

#%%

import plotly.express as px

# 1. Ensure 'Date' is in datetime format so Plotly sorts and plots it correctly
x['Date'] = pd.to_datetime(x['Date'])

# 2. Reshape the dataframe from wide to long format for Plotly Express
df_melted = x.melt(
    id_vars=['Date'], 
    value_vars=['Net_supply', 'Total(bcm)'],
    var_name='Metric', 
    value_name='Value'
)

# 3. Create the line plot
fig = px.line(
    df_melted, 
    x='Date', 
    y='Value', 
    color='Metric',
    title='Storage Change vs Average Storage',
    labels={'Value': 'bcm', 'Date': 'Date'}
)

# 4. Display the plot
fig.show()


# %%
import plotly.express as px

# Create a scatter plot with one variable on the x-axis and the other on the y-axis
fig = px.scatter(
    x, 
    x='Act_storage_change', 
    y='Calc_storage_change',
    title='Storage Change vs Average Storage',
    labels={
        'Av_storage(bcm)': 'Average Storage (bcm)', 
        'Calc_storage_change': 'Calculated Storage Change (bcm)'
    },
    trendline='ols'  # Optional: adds a trendline if you want to see the correlation
)

fig.show()
# %%
import pandas as pd
import plotly.express as px

# 1. Ensure 'Date' is in datetime format
x['Date'] = pd.to_datetime(x['Date'])

# 2. Filter the dataframe to only include dates after 2018-01-01
x_filtered = x[x['Date'] >= '2022-01-01']

# 3. Create the scatter plot using the filtered data
fig = px.scatter(
    x_filtered, 
    x='Act_storage_change', 
    y='Calc_storage_change',
    title='Storage Change vs Average Storage (Post-2018)',
    labels={
        'Av_storage(bcm)': 'Average Storage (bcm)', 
        'Calc_storage_change': 'Calculated Storage Change (bcm)'
    },
    trendline='ols'  # Optional: keeps the trendline for the filtered range
)

fig.show()
# %%
import plotly.express as px

# Create a line plot for diff_storage_changes over Date
fig = px.line(
    x, 
    x='Date', 
    y='Diff_storage_changes',
    title='Difference in Storage Changes over Time (Post-2018)',
    labels={
        'Date': 'Date', 
        'diff_storage_changes': 'Difference in Storage Changes (bcm)'
    }
)

fig.show()
# %%
