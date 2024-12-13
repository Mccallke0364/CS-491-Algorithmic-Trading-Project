# Data cleaning, normalizing, feature engineering, scaling
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def merge_data(stock_data, gov_data):
    """
    Merges stock data and government spending data on the date for each ticker.
    
    Parameters:
    stock_data (dict): A dictionary with ticker symbols as keys and their corresponding DataFrames as values.
    gov_data (pd.DataFrame): A DataFrame containing the government spending data.
    
    Returns:
    pd.DataFrame: A merged DataFrame with stock data and government spending data.
    """
    print("Stock Data Columns:", stock_data.keys())
    print("USASpending Data Columns:", gov_data.columns.tolist())
    
    # Ensure the Date column in government spending data is datetime
    gov_data['Date'] = pd.to_datetime(gov_data['Date'])


    merged_data = pd.DataFrame()
    for ticker, df in stock_data.items():
        df.reset_index(inplace=True)  # Ensure the date is a column
        df.rename(columns={'t': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        df['Ticker'] = ticker
        df['Date'] = pd.to_datetime(df['Date'])
        merged = pd.merge(df, gov_data, on='Date', how='inner')  # Merge on the Date column
        merged_data = pd.concat([merged_data, merged], ignore_index=True)
    return merged_data

def split_train_test(df):
    Ms = MinMaxScaler()
    scaled_df = pd.DataFrame(Ms.fit_transform(df), columns=df.columns)

    training_size= round(len(df)*0.80)
    train_data= df[:training_size]
    test_data=df[training_size:]
    return train_data, test_data, Ms

 



def create_sequences(df, window_size=30, column_A="Date"):
    """
    Prepares data for LSTM by reshaping it into a 3D array.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing
    window_size (int): The number of lagged time steps.
    
    Returns:
    np.array, np.array: The reshaped features and targets.
    """
    sequences = []
    labels = []
    strt_idx = 0
    for stp_idx in range(window_size, len(df)):
        sequences.append(df.iloc[strt_idx:stp_idx].values)
        labels.append(df.iloc[stp_idx].values)
        strt_idx+=1
    return(np.array(sequences), np.array(labels))
    


    #previous thought which was close but not complete
    print(df)
    values = df.values#drop(columns=[column_A]).values
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i-window_size:i, :-1])
        y.append(values[i, -1])
    return np.array(X), np.array(y)
