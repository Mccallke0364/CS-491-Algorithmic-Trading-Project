import tensorflow as tf
import requests
import pandas as pd
import numpy as np
import datetime
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import alpaca_trade_api as api
import time
import papertest 
import numpy as np 
tf.get_logger().setLevel('ERROR')
model = tf.keras.models.load_model('my_checkpoint.keras')
model.summary()
symbol = 'AAPL'
data = papertest.getData(symbol)
data = np.expand_dims(data, axis=0)  

def generate_multi_stock_signals(data, model, threshold=0.5):
    predictions = model.predict(data)
    signals = [
        True if pred > threshold else False
        for pred in predictions[0]
    ]
    return predictions[0], signals
predicted_returns, signals = generate_multi_stock_signals(data, model)

if(signals[0]):
    price = papertest.get_latest_stock_price(symbol)
    qty = int(float(papertest.getcash())) / price 
    sellPrice = price*0.998
    buyPrice = 1.002*price
    print("price: ",price,"sell price: ",sellPrice,"buy price :", buyPrice)
    papertest.buyorder(symbol,int(qty))
    time.sleep(20)
    papertest.sellStop(symbol,sellPrice,qty)
    




print(f"AAPL: Predicted Return: {predicted_returns[0]:.2%}, Signal: {signals[0]}")

