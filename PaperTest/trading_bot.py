import time
import papertest
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()
tf.get_logger().setLevel('ERROR')

# Load the pre-trained model
model = tf.keras.models.load_model('LSTM.keras')
model.summary()




def generate_multi_stock_signals(data, model, threshold):
    """
    Generates trading signals for multiple stocks.
   """
    signals = [0] * 5
    test_predictions = model.predict(data)
    prediction = test_predictions[len(test_predictions)-1]
    for i, pred in enumerate(prediction):
        if pred > threshold:
            signals[i] = True
        else:
            signals[i] = False
        

    return signals

def buy_stock(symbol, cash):
    """
    Buys a stock based on the available cash.
    """
    try:
        price = papertest.get_latest_stock_price(symbol)
        qty = int(cash / price)  # Calculate quantity to buy
        sell_price = price * 0.998
        buy_price = price * 1.002
        
        print(f"Price: {price}, Sell Price: {sell_price}, Buy Price: {buy_price}")
        papertest.buyorder(symbol, qty)
        time.sleep(10)
        papertest.sellStop(symbol, sell_price, qty)
    except Exception as e:
        print(f"Failed to buy {symbol}: {e}")

def close_position(symbol):
    """
    Closes a position if it exists.
    """
    try:
        # Retrieve the list of positions
        positions = papertest.getPostion()

        # Check if any position's symbol matches the given symbol
        if any(pos.symbol == symbol for pos in positions):  # Accessing 'symbol' via dot notation
            papertest.closePostion(symbol)
            print(f"Closed position for {symbol}")
        else:
            print(f"No position found for {symbol}")
    except Exception as e:
        print(f"Failed to close position for {symbol}: {e}")
def main():
    merged_df = pd.read_csv('lstm_input.csv', parse_dates=['Date'], header=0, index_col=0)
    merged_df.index = pd.to_datetime(merged_df.index, unit='ms')

    df = merged_df.drop(columns=['total_outlayed_amount'])
    df = df.fillna(0)

    #df = df.sample(frac = 0.3)
    #NORMALIZE NUMERICAL COLUMNS
    numeric_cols = df.select_dtypes(include=np.number)
    numeric_df = df[numeric_cols.columns.tolist()]

    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(numeric_df)

    #CREATE SEQUENCES
    X_list, Y_list = [], []
    #define sequence length
    window_size = 30

    target_cols = ['c_NGL', 'c_TSLA', 'c_AAPL', 'c_V', 'c_NSRGY']
    target_indices = []

    # print
    for col in target_cols:
        target_indices.append(numeric_df.columns.get_loc(col))




    for i in range(len(numeric_data) - window_size):
        X_list.append(numeric_data[i: i+window_size])
        Y_list.append(numeric_data[i+window_size, target_indices])

    X = np.array(X_list)
    y = np.array(Y_list) 
    # X=X_list
    # y=Y_list
    #implement 80|20 train test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[: train_size], X[train_size:]
    y_train, y_test = y[: train_size], y[train_size:]

    # Reshape for LSTM input (add batch dimension)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    # List of symbols from PCA analysis
    tickers = ['NGL', 'TSLA', 'AAPL', 'V', 'NSRGY']
    cash = float(papertest.getcash())
    # Load data for each symbol
    stock_data = []
    signals= generate_multi_stock_signals(X_test,model,threshold=-0.0)
    amountOfStocks = (100/sum(signals)-1)*.01
    print(signals)
    for i, signal in enumerate(signals):
        if signal:
            buy_stock(tickers[i],cash*amountOfStocks)
        else:
            close_position(tickers[i])
            

if __name__ == "__main__":  
    main()
