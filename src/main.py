import os
from dotenv import load_dotenv
from data_collection.polygon_data import get_data_for_multiple_tickers
from data_collection.bezinga_data import get_government_trades_data
from data_collection.usaspending_data import get_usaspending_data_from_zip
from preprocessing.preprocess_data import combine_data, create_sequences
from model.lstm_model import build_model, train_model
from model.utils import generate_multi_stock_signals, plot_training_history

# Load API keys and URLs from the .env file
load_dotenv()
API_KEY_POLYGON = os.getenv('POLYGON_API_KEY')
API_KEY_BEZINGA = os.getenv('BENZINGA_API_KEY')
API_KEY_USASPENDING = os.getenv('USASPENDING_API_KEY')

# Define tickers and date range
# Arbitrarily picked 5 defense and construction stocks to demonstrate for now
# TODO use Principle Component analysis
tickers = ['NGL', 'TSLA', 'AAPL', 'V', 'NSRGY']
start_date = '2024-01-01'
end_date = '2024-12-31'
zip_file_path = '/Users/collinkozlowski/CS 485/CS-491-Algorithmic-Trading-Project/src/data_collection/raw_data/FY2024_All_Contracts_Full_20241106.zip'


# Fetch data
stock_data = get_data_for_multiple_tickers(tickers, start_date, end_date)
#government_trades = get_government_trades_data(tickers, start_date, end_date)
usaspending_data = get_usaspending_data_from_zip(zip_file_path)

# Process data
combined_data = combine_data(stock_data, usaspending_data)
print(combined_data.head())

# Initialize lists to store sequences for multiple stocks
X_list, y_list = [], []

for ticker, df in combined_data.items():
    df['SMA_10'] = df['c'].rolling(window=10).mean()
    df['SMA_50'] = df['c'].rolling(window=50).mean()
    df['Returns'] = df['c'].pct_change()
    df.dropna(inplace=True)
    X, y = create_sequences(df, window_size=30)
    X_list.append(X)
    y_list.append(y)

X_multi = np.concatenate(X_list, axis=0)
y_multi = np.concatenate(y_list, axis=0)

num_stocks = len(tickers)
y_multi = np.repeat(y_multi, num_stocks).reshape(-1, num_stocks)

print(f"X_multi shape: {X_multi.shape}, y_multi shape: {y_multi.shape}")

# Build and train the model
input_shape = (X_multi.shape[1], X_multi.shape[2])
model = build_model(input_shape, num_stocks)
history = train_model(model, X_multi, y_multi, epochs=20, batch_size=64, validation_split=0.2)

# Plot training history
plot_training_history(history)

# Generate buy/sell signals
latest_data = X_multi[-1].reshape(1, X_multi.shape[1], X_multi.shape[2])
predicted_returns, signals = generate_multi_stock_signals(latest_data, model)

for i, ticker in enumerate(tickers):
    print(f"{ticker}: Predicted Return: {predicted_returns[i]:.2%}, Signal: {signals[i]}")

# TODO: Implement PCA to reduce dimensions and select the most relevant stocks 
# Define the number of components
# TODO: experiment with the number of stocks we can use. currently, polygon and bezinga only let us request 5 at a time


# TODO: Fit PCA on the combined dataset and transform it
# TODO: Evaluate and select the top n most relevant stocks based on PCA results 

# TODO: Re-train the LSTM model using the reduced dataset from PCA 
#  Update X_multi and y_multi with the selected stocks and re-train the model


# TODO: make process continuous where each day before market close:
#   - run PCA on the past 30 days of market data to select the most relevant stocks
#   - re-train the LSTM model with the selected stocks
#   - hold or buy stocks that remained in the list
#   - sell stocks that are no longer in the list
#   - buy stocks that were added to the list