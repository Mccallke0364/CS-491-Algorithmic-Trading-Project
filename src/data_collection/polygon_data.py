# Functions for retrieving and processing Polygon.io data
import requests
import time
import pandas as pd
import numpy as np
import os

POLYGON_API_KEY='q6YjvzTWAp_OkhFvfxwfgrtIVOpddl_V'
POLYGON_API_URL='https://api.polygon.io'

def get_historical_stock_data(ticker, start_date, end_date, POLYGON_API_KEY='q6YjvzTWAp_OkhFvfxwfgrtIVOpddl_V', POLYGON_API_URL='https://api.polygon.io'):
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
            # print(type(df.index))
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

def get_data_for_multiple_tickers(tickers=['NGL', 'TSLA', 'AAPL', 'V', 'NSRGY'], start_date= '2023-10-01', end_date = '2024-12-30'):
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

def merge_dataframes(starting_df, dict_stock_dfs):
    """ merges the dataframes"""
    merged_data=starting_df
        #identifies the starting dataframe for subsequent merges
    for data_frame in dict_stock_dfs:
        #will iterate through the keys of the dictionaries of stock dataframes, these keys will be the tickers of the stocks
        df_to_add = dict_stock_dfs[data_frame]
            #accesses the current dataframe in the dictionary of stock dataframes  
        merged_data = pd.merge(merged_data, df_to_add, right_index=True, left_index=True)
            #merges the usa spending for each date with the corresponding stock data for that date
            #since the dates are the indicies, the merge occurs on the indicies
            #each stock dataframe has to have column titles that are unique to it's stock so that the stocks can all be in the same dataframe without overwriting eachothers data
                # ie every stock dataframe has data for o h l c and v so we add the stock ticker to the column name as an extra identifier
    merged_data.rename_axis("Date", inplace=True)
        #retains the original index identifier so that the index can be accessed using the keyword "Date" in future code
    return merged_data