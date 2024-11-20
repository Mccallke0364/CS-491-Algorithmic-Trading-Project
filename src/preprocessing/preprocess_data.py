# Data cleaning, normalizing, feature engineering, scaling
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def combine_data(stock_data, government_spending):
    """
    Combines stock price data with government spending data into a formatted structure.

    Parameters:
    stock_data (dict): Dictionary of stock data DataFrames for multiple tickers.
    government_spending (pd.DataFrame): DataFrame of USASpending data.

    Returns:
    pd.DataFrame: Combined and formatted DataFrame containing stock prices and government spending information.
    """

    # Combine all stock data into a single DataFrame
    stock_dataframes = []
    for ticker, data in stock_data.items():
        data['ticker'] = ticker
        stock_dataframes.append(data)
    combined_stock_df = pd.concat(stock_dataframes, ignore_index=True)
    combined_stock_df['date'] = combined_stock_df.index.date
    print(type(combined_stock_df.index))

    # Merge the dataframes on the date column
    combined_df = pd.merge(combined_stock_df, government_spending, left_on='date', right_on='action_date', how='outer')

    # Fill or handle missing data
    combined_df.fillna({
        'ticker': 'Unknown',
        'funding_agency_name': 'Unknown',
        'federal_action_obligation': 0,
        'recipient_name': 'Unknown'
    }, inplace=True)

    # Drop unnecessary columns
    combined_df.drop(columns=['action_date'], inplace=True)

    return combined_df

def create_sequences(df, window_size=30):
    """
    Creates sequences of data for training the LSTM model.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be sequenced.
    window_size (int): The size of the window for creating sequences. Default is 30.

    Returns:
    tuple: A tuple containing two arrays, X and y. X is the array of input sequences, and y is the array of target values.
    """
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = [], []

    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size, :-1])
        y.append(scaled_data[i + window_size, -1])

    return np.array(X), np.array(y)









import pandas as pd
import numpy as np

def combine_data(stock_data, government_spending):
    """
    Combines stock price data with government spending data into a formatted structure.

    Parameters:
    stock_data (dict): Dictionary of stock data DataFrames for multiple tickers.
    government_spending (pd.DataFrame): DataFrame of USASpending data.

    Returns:
    pd.DataFrame: Combined and formatted DataFrame containing stock prices and government spending information.
    """

    # Combine all stock data into a single DataFrame
    stock_dataframes = []
    for ticker, data in stock_data.items():
        data['ticker'] = ticker
        stock_dataframes.append(data)
    combined_stock_df = pd.concat(stock_dataframes, ignore_index=True)

    combined_stock_df.index = pd.to_datetime(combined_stock_df.index)
    combined_stock_df['date'] = combined_stock_df.index.date

    # Merge the dataframes on the date column
    combined_df = pd.merge(combined_stock_df, government_spending, left_on='date', right_on='action_date', how='outer')

     # Fill or handle missing data
    combined_df.fillna({
        'ticker': 'Unknown',
        'agency': 'Unknown',
        'federal_action_obligation': 0,
        'recipient_name': 'Unknown',
        'o': 0,
        'h': 0,
        'l': 0,
        'c': 0,
        'v': 0
    }, inplace=True)

    # Drop unnecessary columns
    combined_df.drop(columns=['action_date'], inplace=True)

    return combined_df

def create_sequences(df, window_size=30):
    """
    Creates sequences of data for training the LSTM model.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be sequenced.
    window_size (int): The size of the window for creating sequences. Default is 30.

    Returns:
    tuple: A tuple containing two arrays, X and y. X is the array of input sequences, and y is the array of target values.
    """
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X, y = [], []

    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size, :-1])
        y.append(scaled_data[i + window_size, -1])

    return np.array(X), np.array(y)
