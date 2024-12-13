# %%
import datetime
import alpaca_trade_api as api
import time
# import alpaca
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest,StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import pandas as pd
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockQuotesRequest, StockBarsRequest
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderStatus


# %%
# Alpaca API credentials
ALPACA_API_KEY = 'PKKPO8AP76EAC7AXVYS4'
ALPACA_SECRET_KEY = 'XBjvLBP31v85nvRVf4ScTcJzrEweIdOv9V5BSuzI'
ALPACA_PAPER = True  # Set to True for paper trading
BASE_URL = "https://paper-api.alpaca.markets/"
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

alpaca = api.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL)

# what to do if you can't get stock 
def accountCash():
    account = alpaca.get_account()
    account_cash = float(account.cash)
    # Print your account balance
    print(account_cash)
accountCash()

# Sympobl not avaible Unit test 
def get_latest_stock_price(symbol):
    latest_trade = alpaca.get_latest_trade(symbol)
   
    return latest_trade.p
lastest_apple = get_latest_stock_price(symbol='AAPL')
def closePostion(symbol):
    alpaca.close_position(symbol)  



# buy order exeptions 
def buyorder(symbol,qty):
    # preparing market order
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                        )

    # Market order
    market_order = trading_client.submit_order(
                    order_data=market_order_data
                )
    return market_order
        # Time in force (good till canceled)
# how to handle empty or if alpaca is down 
#gets current postion and than return it to you in an array with json inside
def getPostion():
    postions = alpaca.list_positions()
    return postions


# %%
def getcash():
    return alpaca.get_account().cash


# %%

#postion diddn't close 
def sellAll():
    print(alpaca.close_all_positions())


#couldn't sell 
def sellOrder(symbol,qty):
    alpaca.submit_order(
        symbol=symbol,           # The ticker symbol for the asset (e.g., 'BTCUSD')
        qty=int(qty),                 # Quantity of the asset to sell
        side='sell',             # The side of the order (sell)
        type='market',           # Order type (market order in this case)
        time_in_force='gtc'      # Time in force (good till canceled)
    )
#sell stop wash
def sellStop(symbol,price,qty):
    market_order_data = StopOrderRequest(
                        symbol=symbol,
                        stop_price= round(price, 2),
                        side=OrderSide.SELL,
                        qty = int(qty),
                        time_in_force='gtc'
                        )
    # Market order
    market_order = trading_client.submit_order(
                    order_data=market_order_data
                )
    return market_order
#date and time not got 
def getData(symbol):
    start_date = datetime.datetime.now() - datetime.timedelta(41)
    start_time = pd.to_datetime(start_date.date()).tz_localize('America/New_York')

# It's generally best to explicitly provide an end time but will default to 'now' if not
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_time
        )

    bars_df = data_client.get_stock_bars(request_params).df.tz_convert('America/New_York', level=1)
    return bars_df
'''
symbol = "SPY"
price = get_latest_stock_price(symbol)
qty = int(float(getcash())) / price 
qty = qty*0.8
sellPrice = price*0.998
buyPrice = 1.002*price
print("price: ",price,"sell price: ",sellPrice,"buy price :", buyPrice)
buyorder(symbol,int(qty))
while( sellPrice < price ):
    time.sleep(1)
    if(price > buyPrice):
        sellPrice = price*0.999
        buyPrice = 1.002*price
        print("new sell price", sellPrice, "new buy",buyPrice)
        
    price = get_latest_stock_price(symbol)
sellOrder(symbol,int(qty))

# %%
sellAll()

'''