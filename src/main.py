import os
import numpy as np
import pandas as pd
from data_collection.polygon_data import get_data_for_multiple_tickers
#from data_collection.bezinga_data import get_government_trades_data
from data_collection.usaspending_data import get_usaspending_data
from preprocessing.preprocess_data import merge_data, create_sequences
from model.lstm_model import build_model, train_model
from model.utils import generate_multi_stock_signals, plot_training_history

# Define tickers and date range
# Arbitrarily picked 5 defense and construction stocks to demonstrate for now
# TODO use Principle Component analysis
tickers = ['NGL', 'TSLA', 'AAPL', 'V', 'NSRGY']
start_date = '2023-10-01'
end_date = '2024-12-30'
filepath = 'data_collection/usaspending_data.csv'


# Fetch data
stock_data = get_data_for_multiple_tickers(tickers, start_date, end_date)
print('')
#government_trades = get_government_trades_data(tickers, start_date, end_date)
usaspending_data = get_usaspending_data(filepath)

# # Process data
# combined_data = combine_data(stock_data, usaspending_data)
# print(combined_data.head())
# print(stock_data.head())
merged_data = usaspending_data
for data_frame in stock_data:
    print(type(stock_data))
    print(type(data_frame))
    print(usaspending_data.head())
    print(type(stock_data[data_frame].index[0]))
    print(type(usaspending_data.index[0]))
    print(stock_data[data_frame].head())
    df = stock_data[data_frame]  

    merged_data = pd.merge(merged_data, df, right_index=True, left_index=True)
    print(merged_data)

print(merged_data)
with open("merged_data.txt", "a") as f:
    f.write(merged_data.head(50).to_string())
merged_data.rename_axis("Date", inplace=True)
# merged_data = merge_data(stock_data, usaspending_data)
# print(merged_data)
# Exception(stop)
# Initialize lists to store sequences for multiple stocks

# X_list, y_list = [], []

# X, y = create_sequences(merged_data)
# X_list.append(X)
# y_list.append(y)
# print("complete_1")
# if not X_list:
#     X_multi = np.concatenate(X_list, axis=0)
#     y_multi = np.concatenate(y_list, axis=0)
# print("complete_2")
# num_stocks = len(tickers)
# y_multi = np.repeat(y_multi, num_stocks).reshape(-1, num_stocks)

# print(f"X_multi shape: {X_multi.shape}, y_multi shape: {y_multi.shape}")
train_data, test_data = split_train_test(merged_data)
train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)
build_model((train_seq.shape[1],train_seq.shape[2]))
# Build and train the model
input_shape = (X_multi.shape[1], X_multi.shape[2])
model = build_model(input_shape, num_stocks)
history = train_model(model, X_multi, y_multi, epochs=20, batch_size=64, validation_split=0.2)
print("complete_3")
# Plot training history
plot_training_history(history)

# Generate buy/sell signals
latest_data = X_multi[-1].reshape(1, X_multi.shape[1], X_multi.shape[2])
predicted_returns, signals = generate_multi_stock_signals(latest_data, model)
print("complete_4")
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

