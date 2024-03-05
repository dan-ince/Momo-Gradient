import streamlit as st
import ccxt
import pandas as pd
import numpy as np

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
    positions = ema_gradient_scaled * leverage
    pos_change = np.diff(positions)
    pos_change_sum = np.cumsum(pos_change)
    return pd.Series(positions, index=data.index)

def calculate_equity(data, positions):
    # Assuming we start with 100% of equity and trade with positions
    returns = data['close'].pct_change() * positions.shift(1)
    equity_curve = (1 + returns.fillna(0)).cumprod() * 100  # Starting with 100% equity
    return equity_curve

def calculate_drawdown(equity_curve):
    # Calculate drawdown from equity curve
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return drawdown * 100  # Convert to percentage

def main():
    st.title('Gradient Momentum Dashboard')
    ticker = st.text_input('Ticker', 'BTC')
    tab1, tab2, tab3, tab4 = st.tabs(["Price", "Feature", "Performance", "Drawdown"])

    # Fetch daily data for the last 365 days from a cryptocurrency exchange using ccxt
    exchange = 'okx'  # Use any exchange available in ccxt
    symbol = f'{ticker}/USDT'
    timeframe = '1d'
    limit = 1000  # Number of data points to fetch
    leverage = 1  # Change leverage here
    min_max_lookback = 252  # Fixed lookback window for min-max scaling

    try:
        ticker_data = get_ccxt_data(exchange, symbol, timeframe, limit)

        ema_length = st.slider('EMA Length', 1, 200, 50)
        ema_gradient = calculate_ema_gradient(ticker_data, ema_length)

        st.divider()

        # Min-max scale the entire EMA gradient time series with a fixed lookback window
        ema_gradient_scaled = min_max_scale(pd.Series(ema_gradient, index=ticker_data.index), window=min_max_lookback)

        # Simulate the trading strategy based on the min-maxed EMA gradient
        positions = simulate_strategy(ticker_data, ema_gradient_scaled, leverage)

        # Print the most recent close price, its corresponding min-max scale score, and the date
        pos = positions.iloc[-1]
        st.write(f"Position: {pos:.3}")

        # Calculate equity curve
        equity_curve = calculate_equity(ticker_data, positions)

        # Calculate drawdown
        drawdown = calculate_drawdown(equity_curve)

        # Organize charts side by side with wider columns and padding
        with tab1:
            # Plot the price chart and EMA on the left side
            st.subheader(f'{symbol} Price Chart')
            ema = calculate_ema(ticker_data, ema_length)
            chart_data = pd.DataFrame({'Price': ticker_data['close'], 'EMA': ema})
            st.line_chart(chart_data, use_container_width=True)

        with tab2:
            # Plot the Min-Max scaled EMA gradient, equity curve, and drawdown on the right side
            st.subheader(f'Min-Max Scaled EMA Gradient for {symbol}')
            st.line_chart(ema_gradient_scaled, use_container_width=True)
            
        with tab3:
            st.subheader('Equity Curve')
            st.line_chart(equity_curve, use_container_width=True)
            
        with tab4:    
            st.subheader('Drawdown')
            st.line_chart(drawdown, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing {symbol} on {exchange}: {e}")

if __name__ == "__main__":
    main()
