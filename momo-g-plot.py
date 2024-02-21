import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def get_ccxt_data(exchange, symbol, timeframe, limit):
    exchange = getattr(ccxt, exchange)()
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df[['close']]

def calculate_ema(data, window):
    return data['close'].ewm(span=window, adjust=False).mean()

def calculate_ema_gradient(data, ema_window):
    ema = calculate_ema(data, ema_window)
    ema_gradient = np.gradient(ema.astype(float).values, data.index.astype(np.int64))
    return ema_gradient

def min_max_scale(series, window):
    # Use a fixed lookback window for min-max scaling
    min_value = series.rolling(window=window, min_periods=1).min()
    max_value = series.rolling(window=window, min_periods=1).max()
    scaled_series = 2 * (series - min_value) / (max_value - min_value) - 1  # Scale to [-1, 1]
    return scaled_series

def simulate_strategy(data, ema_gradient_scaled, leverage):
    # Use the min-max scaled EMA gradient as the position size
    positions = ema_gradient_scaled.shift(1) * leverage
    return pd.Series(positions, index=data.index)

def calculate_equity(data, positions):
    # Assuming we start with 100% of equity and trade with positions
    returns = data['close'].pct_change() * positions
    equity_curve = (1 + returns.fillna(0)).cumprod() * 100  # Starting with 100% equity
    return equity_curve

def calculate_drawdown(equity_curve):
    # Calculate drawdown from equity curve
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return drawdown * 100  # Convert to percentage

# Fetch daily data for the last 365 days from a cryptocurrency exchange using ccxt
exchange = 'binance'  # Use any exchange available in ccxt
symbol = 'BTC/USDT'
timeframe = '1d'
limit = 1000  # Number of data points to fetch
leverage = 1.5  # Change leverage here
min_max_lookback = 252  # Fixed lookback window for min-max scaling

# Fetch data and calculate indicators
ticker_data = get_ccxt_data(exchange, symbol, timeframe, limit)
ema_gradient = calculate_ema_gradient(ticker_data, 50)
ema_gradient_scaled = min_max_scale(pd.Series(ema_gradient, index=ticker_data.index), window=min_max_lookback)
positions = simulate_strategy(ticker_data, ema_gradient_scaled, leverage)
equity_curve = calculate_equity(ticker_data, positions)
drawdown = calculate_drawdown(equity_curve)

# Create a colormap for position size
norm = Normalize(vmin=positions.min(), vmax=positions.max())
cmap = plt.get_cmap('RdYlGn')  # Red (negative) to Green (positive)

# Plot the price chart with colored background based on the position values
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(ticker_data.index, ticker_data['close'], label='Price')

for i in range(1, len(ticker_data)):
    color = cmap(norm(positions.iloc[i]))
    ax.axvspan(ticker_data.index[i-1], ticker_data.index[i], color=color, alpha=0.3)

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title(f'{symbol}')

# Create a ScalarMappable to show the colormap scale
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array, as we don't need to map a specific value
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Position Size')

plt.show()
