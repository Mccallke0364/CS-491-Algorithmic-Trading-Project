import tensorflow as tf
import time
import papertest
import numpy as np
tf.get_logger().setLevel('ERROR')

# Load the pre-trained model
model = tf.keras.models.load_model('my_checkpoint.keras')
model.summary()

# Threshold for buy/sell signals
THRESHOLD = 0.5

def generate_multi_stock_signals(data, model, threshold=THRESHOLD):
    """
    Generates trading signals for multiple stocks.
    """
    predictions = model.predict(data)
    signals = [
        True if pred > threshold else False
        for pred in predictions[0]
    ]
    return predictions[0], signals 

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
        positions = papertest.getPostion()
        if any(pos['symbol'] == symbol for pos in positions):
            papertest.closePostion(symbol)
            print(f"Closed position for {symbol}")
    except Exception as e:
        print(f"Failed to close position for {symbol}: {e}")

# List of symbols from PCA analysis
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Load data for each symbol
stock_data = []
for symbol in symbols:
    data = papertest.getData(symbol)
    stock_data.append(np.expand_dims(data, axis=0))

# Predict and act for each stock
cash = float(papertest.getcash())
for idx, symbol in enumerate(symbols):
    data = stock_data[idx]
    predicted_returns, signals = generate_multi_stock_signals(data, model)

    if signals[0]:
        print(f"{symbol}: Predicted Return: {predicted_returns[0]:.2%}, Signal: {signals[0]}")
        buy_stock(symbol, cash)
        #time.sleep(60)
        close_position(symbol)
    else:
        print(f"{symbol}: No buy signal. Predicted Return: {predicted_returns[0]:.2%}")

# Close positions for stocks not in the PCA list
current_positions = papertest.getPostion()
for position in current_positions:
    symbol = position['symbol']
    if symbol not in symbols:
        close_position(symbol)

