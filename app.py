from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)



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
    stock_data = fetch_stock_data(symbol="AAPL", period="1d", interval="1m")
    return jsonify(stock_data)


if __name__ == '__main__':
    app.run(debug=True)
