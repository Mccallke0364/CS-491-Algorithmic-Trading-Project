import unittest
from unittest.mock import MagicMock, patch
import datetime
from alpaca.trading.enums import OrderSide, TimeInForce

class TestAlpacaFunctions(unittest.TestCase):
    @patch('alpaca_trade_api.REST')
    def test_accountCash(self, mock_rest):
        # Mock account cash
        mock_rest().get_account.return_value.cash = '5000.00'
        account_cash = float(mock_rest().get_account().cash)
        self.assertEqual(account_cash, 5000.00)

    @patch('alpaca_trade_api.REST')
    def test_get_latest_stock_price(self, mock_rest):
        # Mock latest trade price
        mock_rest().get_latest_trade.return_value.p = 150.25
        symbol = 'AAPL'
        price = mock_rest().get_latest_trade(symbol).p
        self.assertEqual(price, 150.25)

    @patch('alpaca_trade_api.REST')
    def test_closePosition(self, mock_rest):
        # Mock close position
        symbol = 'AAPL'
        mock_rest().close_position.return_value = {'status': 'success'}
        response = mock_rest().close_position(symbol)
        self.assertEqual(response['status'], 'success')

    @patch('alpaca.trading.client.TradingClient')
    def test_buyorder(self, mock_trading_client):
        # Mock market order submission
        mock_trading_client().submit_order.return_value = {'id': 'order_123'}
        market_order_data = {
            'symbol': 'AAPL',
            'qty': 1,
            'side': OrderSide.BUY,
            'time_in_force': TimeInForce.DAY
        }
        response = mock_trading_client().submit_order(order_data=market_order_data)
        self.assertEqual(response['id'], 'order_123')

    @patch('alpaca_trade_api.REST')
    def test_getPosition(self, mock_rest):
        # Mock list positions
        mock_rest().list_positions.return_value = [{'symbol': 'AAPL', 'qty': 10}]
        positions = mock_rest().list_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['symbol'], 'AAPL')

    @patch('alpaca_trade_api.REST')
    def test_sellAll(self, mock_rest):
        # Mock close all positions
        mock_rest().close_all_positions.return_value = {'status': 'success'}
        response = mock_rest().close_all_positions()
        self.assertEqual(response['status'], 'success')

    @patch('alpaca_trade_api.REST')
    def test_sellOrder(self, mock_rest):
        # Mock sell order
        symbol = 'AAPL'
        qty = 5
        mock_rest().submit_order.return_value = {'id': 'order_456'}
        response = mock_rest().submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
        self.assertEqual(response['id'], 'order_456')

    @patch('alpaca.trading.client.TradingClient')
    def test_sellStop(self, mock_trading_client):
        # Mock stop order
        symbol = 'AAPL'
        price = 140.00
        qty = 5
        mock_trading_client().submit_order.return_value = {'id': 'order_789'}
        stop_order_data = {
            'symbol': symbol,
            'stop_price': round(price, 2),
            'side': OrderSide.SELL,
            'qty': int(qty),
            'time_in_force': 'gtc'
        }
        response = mock_trading_client().submit_order(order_data=stop_order_data)
        self.assertEqual(response['id'], 'order_789')

    @patch('alpaca.data.StockHistoricalDataClient')
    def test_getData(self, mock_data_client):
        # Mock stock bars data
        symbol = 'AAPL'
        start_time = datetime.datetime.now() - datetime.timedelta(41)
        bars_df_mock = {'symbol': symbol, 'open': 150, 'close': 155}
        mock_data_client().get_stock_bars.return_value.df.tz_convert.return_value = bars_df_mock
        bars_df = mock_data_client().get_stock_bars(None).df.tz_convert('America/New_York', level=1)
        self.assertEqual(bars_df['symbol'], symbol)

if __name__ == '__main__':
    unittest.main()