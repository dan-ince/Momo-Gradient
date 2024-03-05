# Imports
import streamlit as st
import ccxt
import pandas as pd
import numpy as np

# Retrieve data from CCXT
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

#Min-max scale the feature so it's usable for sizing
def min_max_scale(series, window):
    # Use a fixed lookback window
    min_value = series.rolling(window=window, min_periods=1).min()
    max_value = series.rolling(window=window, min_periods=1).max()
    scaled_series = 2 * (series - min_value) / (max_value - min_value) - 1  # Scale to [-1, 1]
    return scaled_series

def simulate_strategy(data, ema_gradient_scaled, leverage):
    # Use the min-max scaled EMA gradient as the position size
    positions = ema_gradient_scaled * leverage
    return pd.Series(positions, index=data.index)

def calculate_equity(data, positions):
    returns = data['close'].pct_change() * positions.shift(1)
    equity_curve = (1 + returns.fillna(0)).cumprod() * 100  # Starting with 100% equity
    return equity_curve

def calculate_drawdown(equity_curve):
    # Calculate drawdown from equity curve
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return drawdown * 100

def calculate_normal_returns(data):
    returns = data['close'].pct_change().fillna(0)
    normal_returns = (1 + returns).cumprod() * 100
    return normal_returns

def calculate_sharpe_ratio(equity_curve, daily_risk_free_rate):
    daily_returns = equity_curve.pct_change()
    sharpe_ratio = (daily_returns.mean() - daily_risk_free_rate) / daily_returns.std() * np.sqrt(252)
    return sharpe_ratio

def main():
    #Set title, ticker text input and tabs
    st.title('Feature Momentum Dashboard')
    ticker = st.text_input('Ticker', 'BTC')
    tab1, tab2, tab3, tab4 = st.tabs(["Price", "Feature", "Performance", "Drawdown"])

    # Fetch data from Okex due to Binance location restrictions
    exchange = 'okx'  
    symbol = f'{ticker}/USDT'
    timeframe = '1d'
    limit = 1000  # Number of data points to fetch
    leverage = 1  # Change leverage here
    min_max_lookback = 252  # Fixed lookback window for min-max scaling
    daily_risk_free_rate = 0.0001  # Example daily risk-free rate

    try:
        ticker_data = get_ccxt_data(exchange, symbol, timeframe, limit)
        ema_length = st.slider('EMA Length', 1, 200, 50)
        ema_gradient = calculate_ema_gradient(ticker_data, ema_length)

        # Divider to improve readability
        st.divider()

        # Min-max scale the entire EMA gradient time series with a fixed lookback window
        ema_gradient_scaled = min_max_scale(pd.Series(ema_gradient, index=ticker_data.index), window=min_max_lookback)

        # Simulate the trading strategy based on the min-maxed EMA gradient
        positions = simulate_strategy(ticker_data, ema_gradient_scaled, leverage)

        # Print the most recent position size for following-along
        pos = positions.iloc[-1]
        st.write(f"Position: {pos:.3}")

        # Calculate equity curve
        equity_curve = calculate_equity(ticker_data, positions)

        # Calculate drawdown
        drawdown = calculate_drawdown(equity_curve)

        sharpe_ratio = calculate_sharpe_ratio(equity_curve, daily_risk_free_rate)

        # Organize charts via tabs to improve readability
        with tab1:
            # Price
            st.subheader(f'{symbol} Price Chart')
            ema = calculate_ema(ticker_data, ema_length)
            chart_data = pd.DataFrame({'Price': ticker_data['close'], 'EMA': ema})
            st.line_chart(chart_data, use_container_width=True)

        with tab2:
            # Feature strength
            st.subheader(f'Feature Strength')
            st.line_chart(ema_gradient_scaled, use_container_width=True)
            
        with tab3:
            # Performance including the asset's buy and hold performance
            normal_returns = calculate_normal_returns(ticker_data)
            performance = pd.DataFrame({'Strategy': equity_curve, 'Asset': normal_returns})
            st.subheader('Performance')
            st.line_chart(performance, use_container_width=True)
            st.write(f"Sharpe Ratio: {sharpe_ratio:.3f}")
            
        with tab4:   
            #Drawdown
            st.subheader('Drawdown')
            st.line_chart(drawdown, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing {symbol} on {exchange}: {e}")

if __name__ == "__main__":
    main()
