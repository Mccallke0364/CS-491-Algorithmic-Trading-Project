import datetime
import pandas as pd
import numpy as np
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def get_data(symbol, data_client):
    """
    Retrieve stock bar data for a given symbol
    
    :param symbol: Stock symbol to retrieve data for
    :param data_client: Alpaca Stock Data Client
    :return: DataFrame with stock bar data
    """
    # Calculate start date (41 days back)
    start_date = datetime.datetime.now() - datetime.timedelta(365)
    start_time = pd.to_datetime(start_date.date()).tz_localize('America/New_York')
    
    # Create bar data request
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_time
    )
    
    # Retrieve and convert bars
    bars_df = data_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
    
    return bars_df

def generate_moving_average_signals(data, short_window=20, long_window=50):
    """
    Generate trading signals based on moving average crossover
    
    :param data: Historical price DataFrame
    :param short_window: Short-term moving average window
    :param long_window: Long-term moving average window
    :return: DataFrame with trading signals
    """
    # Calculate moving averages
    data['short_ma'] = data['close'].rolling(window=short_window).mean()
    data['long_ma'] = data['close'].rolling(window=long_window).mean()
    
    # Generate signals
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1  # Buy signal
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1  # Sell signal
    
    return data

def backtest_strategy(
    symbol, 
    data_client,
    initial_capital=10000, 
    commission=0,  # Reduced commission 
    short_window=50,  # Longer windows
    long_window=200
):
    """
    Perform an improved moving average crossover backtest
    
    :param symbol: Stock symbol to backtest
    :param data_client: Alpaca Stock Data Client
    :param initial_capital: Starting capital for backtest
    :param commission: Per trade commission cost
    :param short_window: Short-term moving average window
    :param long_window: Long-term moving average window
    :return: Backtest results and performance metrics
    """
    # Fetch historical data
    bars = get_data(symbol, data_client)
    
    # Generate trading signals with additional confirmation
    bars_with_signals = generate_moving_average_signals(
        bars.copy(),  # Use a copy to avoid SettingWithCopyWarning 
        short_window=short_window, 
        long_window=long_window
    )
    
    # Add trend strength filter
    bars_with_signals['trend_strength'] = abs(bars_with_signals['short_ma'] - bars_with_signals['long_ma']) / bars_with_signals['long_ma']
    
    # Modify signals to reduce unnecessary trading
    bars_with_signals['filtered_signal'] = bars_with_signals['signal'].copy()
    bars_with_signals.loc[
        (bars_with_signals['trend_strength'] < 0.02) |  # Weak trend
        (bars_with_signals['signal'].diff() == 0),     # No change from previous state
        'filtered_signal'
    ] = 0
    
    # Remove NaN rows
    bars_with_signals = bars_with_signals.dropna()
    
    # Initialize portfolio tracking with explicit float dtype
    portfolio = pd.DataFrame(
        index=bars_with_signals.index, 
        columns=['signal', 'position', 'trades', 'commission', 'portfolio_value', 'buy_and_hold_value'],
        dtype=float
    )
    
    portfolio['signal'] = bars_with_signals['filtered_signal']
    portfolio['position'] = portfolio['signal'].diff()
    
    # Calculate trades and commission
    portfolio['trades'] = portfolio['position'] * bars_with_signals['close']
    portfolio['commission'] = np.abs(portfolio['position']) * commission
    
    # Calculate cumulative performance
    # Use .loc for setting values to avoid chained assignment warnings
    portfolio.loc[portfolio.index[0], 'portfolio_value'] = initial_capital
    portfolio.loc[portfolio.index[0], 'buy_and_hold_value'] = initial_capital
    
    # Track buy and hold performance
    shares_bought = initial_capital / bars_with_signals['close'].iloc[0]
    
    for i in range(1, len(portfolio)):
        # Trading strategy performance
        prev_value = portfolio.loc[portfolio.index[i-1], 'portfolio_value']
        trade_value = portfolio.loc[portfolio.index[i], 'trades']
        trade_commission = portfolio.loc[portfolio.index[i], 'commission']
        
        # Use .loc for setting values
        portfolio.loc[portfolio.index[i], 'portfolio_value'] = (
            prev_value + trade_value - trade_commission
        )
        
        # Buy and hold performance
        portfolio.loc[portfolio.index[i], 'buy_and_hold_value'] = (
            shares_bought * bars_with_signals['close'].iloc[i]
        )
    
    # Calculate performance metrics
    strategy_return = (portfolio['portfolio_value'].iloc[-1] - initial_capital) / initial_capital * 100
    buy_and_hold_return = (portfolio['buy_and_hold_value'].iloc[-1] - initial_capital) / initial_capital * 100
    max_drawdown = (portfolio['portfolio_value'].max() - portfolio['portfolio_value'].min()) / portfolio['portfolio_value'].max() * 100
    
    # Print and visualize results
    print(f"Backtest Results for {symbol}")
    print(f"Strategy Total Return: {strategy_return:.2f}%")
    print(f"Buy and Hold Total Return: {buy_and_hold_return:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Final Strategy Portfolio Value: ${portfolio['portfolio_value'].iloc[-1]:.2f}")
    print(f"Final Buy and Hold Portfolio Value: ${portfolio['buy_and_hold_value'].iloc[-1]:.2f}")
    
    # Optional visualization
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.plot(portfolio.index, portfolio['portfolio_value'], label='Strategy')
        plt.plot(portfolio.index, portfolio['buy_and_hold_value'], label='Buy and Hold')
        plt.title(f'{symbol} Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")
    
    return portfolio

def main():
    # Alpaca API Credentials (REPLACE WITH YOUR ACTUAL CREDENTIALS)
    API_KEY = 'PKKPO8AP76EAC7AXVYS4'
    API_SECRET = 'XBjvLBP31v85nvRVf4ScTcJzrEweIdOv9V5BSuzI'
    
    # Initialize Alpaca Stock Data Client
    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    
    # Run backtest
    backtest_strategy(
        symbol='VOO',  # Example with Apple stock
        data_client=data_client,
        initial_capital=10000,
        short_window=5,
        long_window=30
    )

if __name__ == "__main__":
    main()

