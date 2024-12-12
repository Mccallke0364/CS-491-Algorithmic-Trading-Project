import unittest
from unittest.mock import patch, MagicMock
import numpy as np

class TestTradingFunctions(unittest.TestCase):
    @patch('papertest.get_latest_stock_price')
    @patch('papertest.getcash')
    @patch('papertest.buyorder')
    @patch('papertest.sellStop')
    def test_buy_stock(self, mock_sellStop, mock_buyorder, mock_getcash, mock_get_latest_stock_price):
        mock_get_latest_stock_price.return_value = 150.0
        mock_getcash.return_value = 1000.0
        mock_buyorder.return_value = None
        mock_sellStop.return_value = None

        from trading_bot import buy_stock 
        from unittest.mock import call # Import the refined buy_stock function
        expected_calls = [
            call('AAPL', 6),
            call('MSFT', 6),
            call('GOOGL', 6),
            call('AMZN', 6),
            call('TSLA', 6),
        ]
        expected_sell_calls = [
        call('AAPL', 149.7, 6),
        call('MSFT', 149.7, 6),
        call('GOOGL', 149.7, 6),
        call('AMZN', 149.7, 6),
        call('TSLA', 149.7, 6)
        ]
        mock_buyorder.assert_has_calls(expected_calls, any_order=True)
        mock_sellStop.assert_has_calls(expected_sell_calls)

    @patch('papertest.getPostion')
    @patch('papertest.closePostion')
    def test_close_position(self, mock_closePostion, mock_getPostion):
        mock_getPostion.return_value = [{'symbol': 'AAPL'}, {'symbol': 'MSFT'}]

        from trading_bot import close_position
        close_position('AAPL')

        mock_closePostion.assert_called_once_with('AAPL')

    @patch('trading_bot.generate_multi_stock_signals')
    def test_generate_multi_stock_signals(self, mock_generate_signals):
        mock_generate_signals.return_value = (np.array([0.6, 0.4]), [True, False])

        from trading_bot import generate_multi_stock_signals
        data = np.zeros((1, 5, 10))  # Example input
        model = MagicMock()
        predictions, signals = generate_multi_stock_signals(data, model)

        self.assertEqual(predictions[0], 0.6)
        self.assertTrue(signals[0])

if __name__ == '__main__':
    unittest.main()
