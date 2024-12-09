# Functions for retrieving and processing Polygon.io data
import requests
import time
import pandas as pd
import numpy as np
import os

POLYGON_API_KEY='q6YjvzTWAp_OkhFvfxwfgrtIVOpddl_V'
POLYGON_API_URL='https://api.polygon.io'

def get_historical_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a given ticker from Polygon.io.

    Parameters:
    ticker (str): The stock ticker symbol to fetch data for.
    start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing OHLCV data for the ticker.
    """
    
    url = f"{POLYGON_API_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        data = response.json()
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            # print(df.t[3])
            df['t_a'] = pd.to_datetime(df['t'], unit ="ms", yearfirst=True)
            df["t"] =  df['t_a'].dt.date
            
            
            df.set_index('t', inplace=True)
            print(type(df.index))
            df.rename(columns={'o': f'o_{ticker}', 'h': f'h_{ticker}', 'l': f'l_{ticker}', 'c': f'c_{ticker}','v':f'v_{ticker}'}, inplace=True)
            # print(df.head())
            
          
            df[f'{ticker}_SMA_10'] = df[f'c_{ticker}'].rolling(window=10).mean()
            df[f'{ticker}_SMA_50'] = df[f'c_{ticker}'].rolling(window=50).mean()
            df[f'{ticker}_Returns'] = df[f'c_{ticker}'].pct_change()
            df.dropna(inplace=True)
            
            return df[[f'o_{ticker}', f'h_{ticker}', f'l_{ticker}', f'c_{ticker}',f'v_{ticker}', f'{ticker}_SMA_10',f'{ticker}_SMA_50',f'{ticker}_Returns']]#df[['o', 'h', 'l', 'c', 'v']]
        else:
            print(f"No data available for {ticker} in the specified date range.")
            return pd.DataFrame()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"JSON decode error for {ticker}: {e}")
        return pd.DataFrame()

def get_data_for_multiple_tickers(tickers, start_date, end_date):
    """
    Fetches historical stock data for multiple tickers from Polygon.io.

    Parameters:
    tickers (list): A list of stock ticker symbols to fetch data for.
    start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching data in 'YYYY-MM-DD' format.

    Returns:
    dict: A dictionary with ticker symbols as keys and their corresponding DataFrames as values.
    """
    
    stock_data = {}
    for ticker in tickers:
        data = get_historical_stock_data(ticker, start_date, end_date)
        print(f"Fetched data for {ticker}")
        time.sleep(1)
        if not data.empty:
            stock_data[ticker] = data
    return stock_data
