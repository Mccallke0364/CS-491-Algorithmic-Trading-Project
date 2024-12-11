from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
from datetime import datetime, timedelta
import datetime
import alpaca_trade_api as api
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest,StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import pandas as pd
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockQuotesRequest, StockBarsRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus


app = Flask(__name__)
CORS(app)

# %%
# Alpaca API credentials
ALPACA_API_KEY = 'PKKPO8AP76EAC7AXVYS4'
ALPACA_SECRET_KEY = 'XBjvLBP31v85nvRVf4ScTcJzrEweIdOv9V5BSuzI'
ALPACA_PAPER = True  # Set to True for paper trading
BASE_URL = "https://paper-api.alpaca.markets/"
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

alpaca = api.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL)

def fetch_stock_data(symbol="AAPL", period="1d", interval="1m"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period, interval=interval)

    # 格式化数据
    formatted_data = []
    for date, row in data.iterrows():
        formatted_data.append([
            date.strftime("%Y-%m-%d %H:%M:%S"),  # 格式化日期和时间
            row['Open'],  # 开盘价
            row['High'],  # 最高价
            row['Low'],  # 最低价
            row['Close'],  # 收盘价
            row['Volume']  # 成交量
        ])
    return formatted_data


@app.route('/get_stock_data')
def get_stock_data():
    try:
        # Fetch positions using the getPosition function
        positions = alpaca.list_positions()
        return jsonify({"status": "success", "data": positions}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
