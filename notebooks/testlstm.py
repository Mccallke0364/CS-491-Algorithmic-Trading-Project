import requests
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

API_KEY_POLYGON='q6YjvzTWAp_OkhFvfxwfgrtIVOpddl_V'

"""##IMPORT HISTORICAL DATA"""

#fetches historical data for one ticker
def get_historical_stock_data(ticker, start_date, end_date):
    url = f" {'POLYGON_API_URL'}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=5000&apiKey=q6YjvzTWAp_OkhFvfxwfgrtIVOpddl_V"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for non-200 status codes

        # Check if response is valid JSON
        data = response.json()

        # Check if 'results' is in the response (indicates data is present)
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            df['t'] = pd.to_datetime(df['t'], unit='ms')  # Convert timestamp to date
            df.set_index('t', inplace=True)
            return df[['o', 'h', 'l', 'c', 'v']]  # Keep OHLC and volume columns
        else:
            print(f"No data available for {ticker} in the specified date range.")
            return pd.DataFrame()  # Return an empty DataFrame if no data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"JSON decode error for {ticker}: {e}")
        return pd.DataFrame()

#Gets data for multiple tickers
def get_data_for_multiple_tickers(tickers, start_date, end_date):
  stock_data = {}
  for ticker in tickers:
    data = get_historical_stock_data(ticker, start_date, end_date)
    print(f"fetched data for {ticker}")
    time.sleep(1)
    if data is not None:
      stock_data[ticker] = data
  return stock_data


API_KEY_BENZINGA='2e033fcd1cc24a8c94ed4e17251f7da1'

def get_government_trades_data(ticker, start_date, end_date):
    #get government trades data from benzinga
    url = "https://api.benzinga.com/api/v1/gov/usa/congress/trades"
    querystring = {"token":"2e033fcd1cc24a8c94ed4e17251f7da1","date_from":"2020-01-01","date_to":"2024-01-01"}

    response = requests.request("GET", url, params=querystring)

    # Check if the response is a list and handle accordingly
    data = response.json()
    if isinstance(data, list):
        # Assuming the data is in the first element of the list if it's a list
        trades_data = data[0].get("data", []) if data else []
    else:
        # If it's a dictionary, proceed as before
        trades_data = data.get("data", [])

    trade_entries = []
    for trade in trades_data:
      # Extract relevant fields from each trade entry
          trade_entry = {
              "ticker": trade["security"]["ticker"],
              "transaction_type": trade["transaction_type"],
              "transaction_date": trade["transaction_date"],
              "amount": trade["amount"],
              "member_name": trade["filer_info"]["member_name"],
              "chamber": trade["chamber"],
              "state": trade["filer_info"]["state"]
            }
          trade_entries.append(trade_entry)

    #process into data frame
    trades_df = pd.DataFrame(trade_entries)
    print(trades_df.head())
    trades_df["transaction_date"] = pd.to_datetime(trades_df["transaction_date"])
    trades_df.set_index('transaction_date', inplace=True)
    return trades_df

tickers = np.array(["CAT", "NSRGY", "TSLA", "CMG", "AAPL"])



def combine_data(stock_data, government_trades):
    """
    Combine stock price data with government trades data in a formatted structure.

    Parameters:
    stock_data (dict): Dictionary of stock data DataFrames for multiple tickers from get_data_for_multiple_tickers.
    government_trades (pd.DataFrame): DataFrame of government trades data from get_government_trades.

    Returns:
    pd.DataFrame: Combined and formatted DataFrame containing stock prices and government trades information.
    """
    combined_data = []

    # Iterate through each ticker's data in stock_data
    for ticker, stock_df in stock_data.items():
        # Format the index to match date formatting
        stock_df.index = pd.to_datetime(stock_df.index).strftime('%Y-%m-%d')

        # Filter government trades data for the current ticker and format date
        gov_trades_ticker = government_trades[government_trades['ticker'] == ticker].copy()
        # Use the index directly instead of 'transaction_date' column
        # Changed this line to directly format the DatetimeIndex
        gov_trades_ticker.index = gov_trades_ticker.index.strftime('%Y-%m-%d')

        # Merge stock data and government trades data on date
        merged_df = stock_df.merge(
            gov_trades_ticker,
            left_index=True,
            right_index=True,
            how="left",
            suffixes=("", "_gov")
        )

        # Ensure ticker column is added for clarity
        merged_df["ticker"] = ticker
        combined_data.append(merged_df)

    # Concatenate all ticker DataFrames into a single DataFrame
    combined_df = pd.concat(combined_data)

    # Clean up by filling NaN values in government trades columns with blanks
    combined_df.fillna({
        'transaction_type': '',
        'amount': '',
        'member_name': '',
        'chamber': '',
        'state': ''
    }, inplace=True)

    # Reorder columns for better readability
    cols = ['ticker', 'o', 'h', 'l', 'c', 'v', 'transaction_type', 'amount', 'member_name', 'chamber', 'state']
    combined_df = combined_df[cols]

    return combined_df

# Get stock data
start_date = '2020-01-01'
end_date = '2024-01-01'
stock_data = get_data_for_multiple_tickers(tickers, start_date, end_date)

# Get government trades data
government_trades = get_government_trades_data(tickers, start_date, end_date)

# Combine the data
combined_data = combine_data(stock_data, government_trades)

print(combined_data.head())

"""##PREPROCESS DATA
###normalize stock prices and generate a simple moving average
"""

from sklearn.preprocessing import MinMaxScaler

#function to create sequences
def create_sequences(df, window_size=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)  # Scale DataFrame and keep it as np.array
    X, y = [], []

    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size, :-1])  # Include all columns except the last
        y.append(scaled_data[i + window_size, -1])  # Target is the last column (returns)

    return np.array(X), np.array(y)

# Initialize lists to store sequences for multiple stocks
X_list, y_list = [], []

# Iterate through each ticker to calculate indicators and sequences
for ticker, df in combined_data.items():
    # Calculate moving averages and returns
    df['SMA_10'] = df['c'].rolling(window=10).mean()
    df['SMA_50'] = df['c'].rolling(window=50).mean()
    df['Returns'] = df['c'].pct_change()
    df.dropna(inplace=True)  # Drop NaN values from rolling calculations

    # Create sequences from the scaled DataFrame
    X, y = create_sequences(df, window_size=30)
    X_list.append(X)
    y_list.append(y)

# Concatenate all sequences for the multi-stock model
X_multi = np.concatenate(X_list, axis=0)
y_multi = np.concatenate(y_list, axis=0)

# Adjust y_multi to have the correct shape
num_stocks = len(tickers)  # Number of assets
y_multi = np.repeat(y_multi, num_stocks).reshape(-1, num_stocks)

print(f"X_multi shape: {X_multi.shape}, y_multi shape: {y_multi.shape}")
# format --> X(total_samples, window_size, num_features); Y(total_samples, num_stocks)

"""###make sequences of input to predict the next day"""



"""##BUILD LSTM MODEL"""

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(num_stocks, activation='sigmoid')  # Sigmoid for probability output
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

"""####TRAINING THE MODEL"""

history = model.fit(X_multi, y_multi, epochs=5, batch_size=32, validation_split=0.2)

"""####MAKE PREDICTIONS AND SIGNAL BUY OR SELL"""

def generate_multi_stock_signals(data, model, threshold=0.5):
    predictions = model.predict(data)
    signals = [
        'Buy' if pred > threshold else 'Sell'
        for pred in predictions[0]
    ]
    return predictions[0], signals

# Predict for the latest data
latest_data = X_multi[-1].reshape(1, X_multi.shape[1], X_multi.shape[2])
predicted_returns, signals = generate_multi_stock_signals(latest_data, model)

for i, ticker in enumerate(tickers):
    print(f"{ticker}: Predicted Return: {predicted_returns[i]:.2%}, Signal: {signals[i]}")

import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()