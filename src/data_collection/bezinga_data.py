# import requests
# import pandas as pd
# import os
# from dotenv import load_dotenv

# load_dotenv()
# BENZINGA_API_URL = os.getenv('BENZINGA_API_URL')
# BENZINGA_API_KEY = os.getenv('BENZINGA_API_KEY')

# def get_government_trades_data(tickers, start_date, end_date):
#     """
#     Fetches government trades data from Benzinga and processes it into a DataFrame.

#     Parameters:
#     tickers (list): List of stock ticker symbols to fetch data for.
#     start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
#     end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.

#     Returns:
#     pd.DataFrame: A DataFrame containing processed government trades data.
#     """
#     url = "https://api.benzinga.com/api/v1/gov/usa/congress/trades"

#     querystring = {"token":"2e033fcd1cc24a8c94ed4e17251f7da1","date_from":"2024-10-01","date_to":"2024-11-01","tickers": ",".join(tickers)}

#     response = requests.request("GET", url, params=querystring)

#     print(response.text)

#     data = response.json()

#     if isinstance(data, list):
#         trades_data = data[0].get("data", []) if data else []
#     else:
#         trades_data = data.get("data", [])

#     trade_entries = []

#     for trade in trades_data:
#         trade_entry = {
#             "ticker": trade["security"]["ticker"],
#             "transaction_type": trade["transaction_type"],
#             "transaction_date": trade["transaction_date"],
#             "amount": trade["amount"],
#             "member_name": trade["filer_info"]["member_name"],
#             "chamber": trade["chamber"],
#             "state": trade["filer_info"]["state"]
#         }
#         trade_entries.append(trade_entry)

#     trades_df = pd.DataFrame(trade_entries)
#     trades_df["transaction_date"] = pd.to_datetime(trades_df["transaction_date"])
#     trades_df.set_index('transaction_date', inplace=True)

#     # Calculate amount bought, amount sold, and number of trades per ticker
#     summary_data = []

#     for ticker in tickers:
#         ticker_data = trades_df[trades_df['ticker'] == ticker]
#         amount_bought = ticker_data[ticker_data['transaction_type'] == 'Purchase']['amount'].sum()
#         amount_sold = ticker_data[ticker_data['transaction_type'] == 'Sale']['amount'].sum()
#         num_trades = ticker_data.shape[0]
        
#         summary_data.append({
#             'ticker': ticker,
#             'amount_bought': amount_bought,
#             'amount_sold': amount_sold,
#             'num_trades': num_trades
#         })

#     summary_df = pd.DataFrame(summary_data)
#     return summary_df


import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
BENZINGA_API_KEY = os.getenv('BENZINGA_API_KEY')

def get_all_government_trades_data(start_date, end_date):
    """
    Fetches all government trades data from Benzinga within a specified date range.

    Parameters:
    start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.

    Returns:
    list: A list of trade entries.
    """
    url = "https://api.benzinga.com/api/v1/gov/usa/congress/trades"
    all_trades = []
    page = 1

    while True:
        querystring = {
            "token": BENZINGA_API_KEY,
            "date_from": start_date,
            "date_to": end_date,
            "page": page,
        }
        response = requests.get(url, params=querystring)
        print(f"Request URL: {response.url}")  # Debugging: Print the full request URL
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        trades_data = data.get("data", [])
        all_trades.extend(trades_data)

        if len(trades_data) < 1000:  # Assuming each page returns 1000 items
            break
        page += 1

    return all_trades

def filter_trades_by_tickers(all_trades, tickers):
    """
    Filters trades data by specified tickers.

    Parameters:
    all_trades (list): A list of all trade entries.
    tickers (list): List of stock ticker symbols to filter data for.

    Returns:
    pd.DataFrame: A DataFrame containing filtered government trades data.
    """
    trade_entries = []

    for trade in all_trades:
        if trade["security"]["ticker"] in tickers:
            trade_entry = {
                "ticker": trade["security"]["ticker"],
                "transaction_type": trade["transaction_type"],
                "transaction_date": trade.get("transaction_date"),
                "amount": trade["amount"],
                "member_name": trade["filer_info"]["member_name"],
                "chamber": trade["chamber"],
                "state": trade["filer_info"]["state"]
            }
            trade_entries.append(trade_entry)

    trades_df = pd.DataFrame(trade_entries)

    if 'transaction_date' not in trades_df.columns:
        print("transaction_date column is missing from the DataFrame.")
        print(trades_df.head())
        return pd.DataFrame()

    trades_df["transaction_date"] = pd.to_datetime(trades_df["transaction_date"])
    trades_df.set_index('transaction_date', inplace=True)

    return trades_df

def get_government_trades_data(tickers, start_date, end_date):
    """
    Fetches and filters government trades data from Benzinga by tickers within a specified date range.

    Parameters:
    tickers (list): List of stock ticker symbols to fetch data for.
    start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing filtered government trades data.
    """
    all_trades = get_all_government_trades_data(start_date, end_date)
    if not all_trades:
        print("No trades data received from the API.")
        return pd.DataFrame()

    trades_df = filter_trades_by_tickers(all_trades, tickers)

    if trades_df.empty:
        print("No data available for the specified tickers.")
        return pd.DataFrame()

    # Calculate amount bought, amount sold, and number of trades per ticker
    summary_data = []

    for ticker in tickers:
        ticker_data = trades_df[trades_df['ticker'] == ticker]
        amount_bought = ticker_data[ticker_data['transaction_type'] == 'Purchase']['amount'].sum()
        amount_sold = ticker_data[ticker_data['transaction_type'] == 'Sale']['amount'].sum()
        num_trades = ticker_data.shape[0]
        
        summary_data.append({
            'ticker': ticker,
            'amount_bought': amount_bought,
            'amount_sold': amount_sold,
            'num_trades': num_trades
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df
