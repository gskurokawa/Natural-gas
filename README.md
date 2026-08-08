# TTF Natural Gas: Forecasting Studies

This repository collects a set of out-of-sample econometric studies of Dutch TTF
natural gas prices — asking how much of the price, and of its volatility, can be
forecast — together with an application to the options market.

The recurring finding is consistent across frequencies: the TTF price *level* is
close to a random walk and is difficult to forecast even under perfect foresight
of fundamentals, whereas its *volatility* is forecastable out of sample.

## Repository structure

- **Monthly/** — Month-ahead (H = 1) study on monthly data (2015–2026). Covers
  price forecasting, cross-hub cointegration and transmission among TTF, JKM, and
  Henry Hub via a VECM, and GARCH volatility. Contains the paper (markdown), 
  code, and dataset. The dataset can be generated from the build .py file. The analysis
  is found in the .ipynb file.

- **Daily/** — Week-ahead (7-day) study on daily data (2022–2026). Tests whether
  European and global fundamentals, granted perfect foresight, can beat a random
  walk for the TTF price level (they cannot), and shows that volatility is
  forecastable with a HAR-X(+VIX) model. Contains the paper (markdown),
  code, and dataset. The dataset can be generated from the build .py file. The analysis
  is found in the .ipynb file.

- **Application/** — A short note applying the daily HAR-X(+VIX) volatility
  forecast to TTF options: implied volatilities are backed out from an option
  chain and compared with the forecast to read off the variance risk premium.

- **References/** — Academic references cited in the daily and monthly studies.

The PDF version of the monthly and daily TTF papers are in the main directory here.
