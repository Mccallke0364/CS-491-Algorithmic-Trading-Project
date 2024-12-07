import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


#POLYGON DATA COLLECTION
# Functions for retrieving and processing Polygon.io data

POLYGON_API_URL = os.getenv('POLYGON_API_URL')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

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
            return df[[f'o_{ticker}', f'h_{ticker}', f'l_{ticker}', f'c_{ticker}',f'v_{ticker}']]#df[['o', 'h', 'l', 'c', 'v']]
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

#FETCH GOVERNMENT SPENDING DATA

def get_usaspending_data(filepath):
    """
    Loads government spending data from a CSV file.
    
    Parameters:
    filepath (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: A DataFrame containing the government spending data.
    """
    df = pd.read_csv(filepath, parse_dates=['Date'], header=0, index_col=0)
    print(df.head())
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

#PREPROCESS THE DATA
# Data cleaning, normalizing, feature engineering, scaling
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

def create_sequences(df, tickers, window_size=30):
    """
    Create sequence data for multiple stocks
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged dataset
    tickers : list
        List of stock tickers
    window_size : int
        Number of time steps in each sequence
    
    Returns:
    --------
    tuple
        Input sequences (X) and target values (y)
    """
    # Identify columns for each stock
    stock_columns = {
        ticker: [f'{col}_{ticker}' for col in ['o', 'h', 'l', 'c', 'v']]
        for ticker in tickers
    }
    
    # Prepare sequences
    X, y = [], []
    for i in range(len(df) - window_size):
        # Create sequence of features
        sequence = df[[col for cols in stock_columns.values() for col in cols]].iloc[i:i+window_size].values
        X.append(sequence)
        
        # Target could be next day's close prices
        next_closes = [df[f'c_{ticker}'].iloc[i+window_size] for ticker in tickers]
        y.append(next_closes)
    
    return np.array(X), np.array(y)


#TRAIN THE MODEL
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
from keras.losses import Huber

def build_model(input_shape, num_stocks):
    """
    Builds and compiles an LSTM model.

    Parameters:
    input_shape (tuple): The shape of the input data (time_steps, num_features).
    num_stocks (int): The number of stocks (output dimensions).

    Returns:
    keras.models.Sequential: Compiled LSTM model.
    """
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(num_stocks, activation='tanh')  # Using tanh activation
    ])

    model.compile(optimizer=RMSprop(), loss=Huber())  # Using RMSprop and Huber Loss
    model.summary()
    return model

def train_model(model, X, y, epochs=20, batch_size=64, validation_split=0.2):
    """
    Trains the LSTM model.

    Parameters:
    model (keras.models.Sequential): The compiled LSTM model.
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Target values.
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.
    validation_split (float): Fraction of data to use for validation.

    Returns:
    keras.callbacks.History: History object containing training history.
    """
    
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history

def generate_multi_stock_signals(data, model, threshold=0.5):
    """
    Generates buy/sell signals based on model predictions.

    Parameters:
    data (numpy.ndarray): Input data for prediction.
    model (keras.models.Sequential): The trained model.
    threshold (float): Threshold for deciding buy/sell signals.

    Returns:
    tuple: Predicted returns and buy/sell signals.
    """
    
    predictions = model.predict(data)
    signals = [
        'Buy' if pred > threshold else 'Sell'
        for pred in predictions[0]
    ]
    return predictions[0], signals

def plot_training_history(history):
    """
    Plots training and validation loss over epochs.

    Parameters:
    history (keras.callbacks.History): History object containing training history.
    """
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()




#MAIN DRIVER FUNCTION

def main():
    # Load API keys and URLs from the .env file
    load_dotenv()
    API_KEY_POLYGON = os.getenv('POLYGON_API_KEY')
    API_KEY_USASPENDING = os.getenv('USASPENDING_API_KEY')

    # Define tickers and date range
    # Arbitrarily picked 5 defense and construction stocks to demonstrate for now
    # TODO use Principle Component analysis
    tickers = ['NGL', 'TSLA', 'AAPL', 'V', 'NSRGY']
    start_date = '2023-10-01'
    end_date = '2024-12-30'
    filepath = 'src/data_collection/usaspending_data.csv'
    


    print('FETCHING STOCK AND USASPENDING DATA')
    stock_data = get_data_for_multiple_tickers(tickers, start_date, end_date)
    print('')
    #government_trades = get_government_trades_data(tickers, start_date, end_date)
    usaspending_data = get_usaspending_data(filepath)
    print('DATA FETCH COMPLETE\n')

    merged_data = usaspending_data
    for data_frame in stock_data:
        print(type(stock_data))
        print(type(data_frame))
        print(usaspending_data.head())
        print(type(stock_data[data_frame].index[0]))
        print(type(usaspending_data.index[0]))
        print(stock_data[data_frame].head())
        merged_data = pd.merge(merged_data, stock_data[data_frame], right_index=True, left_index=True)
        

    merged_data = merged_data.sample(frac=0.1, random_state=42)
    print(merged_data)
    print('MERGED DATA COMPLETE \n')
    merged_data.rename_axis("Date", inplace=True)
    merged_data.to_csv('lstm_input.csv', index='False')
    df = pd.read_csv('lstm_input.csv', parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])

    #drop non_numeric columns
    numeric_cols = df.columns.difference(['Date', 'awarding_agency_name', 'recipient_name', 'action_type'])

    #standardize the data
    print('NORMALIZING THE DATA\n')
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(df[numeric_cols])
    print('NORMALIZATION COMPLETE\n')

    # Initialize lists to store sequences for multiple stocks
    X_list, y_list = [], []


    print('CREATING SEQUENCES')
    X, y = create_sequences(df, tickers, window_size=30)

    X_multi = np.concatenate(X_list, axis=0)
    y_multi = np.concatenate(y_list, axis=0)

    num_stocks = len(tickers)
    y_multi = np.repeat(y_multi, num_stocks).reshape(-1, num_stocks)

    print(f"X_multi shape: {X_multi.shape}, y_multi shape: {y_multi.shape}")

    # Build and train the model
    input_shape = (X_multi.shape[1], X_multi.shape[2])
    model = build_model(input_shape, num_stocks)
    history = train_model(model, X_multi, y_multi, epochs=20, batch_size=64, validation_split=0.2)

    # Plot training history
    plot_training_history(history)

    # Generate buy/sell signals
    latest_data = X_multi[-1].reshape(1, X_multi.shape[1], X_multi.shape[2])
    predicted_returns, signals = generate_multi_stock_signals(latest_data, model)

    for i, ticker in enumerate(tickers):
        print(f"{ticker}: Predicted Return: {predicted_returns[i]:.2%}, Signal: {signals[i]}")

    # TODO: Implement PCA to reduce dimensions and select the most relevant stocks 
    # Define the number of components
    # TODO: experiment with the number of stocks we can use. currently, polygon and bezinga only let us request 5 at a time


    # TODO: Fit PCA on the combined dataset and transform it
    # TODO: Evaluate and select the top n most relevant stocks based on PCA results 

    # TODO: Re-train the LSTM model using the reduced dataset from PCA 
    #  Update X_multi and y_multi with the selected stocks and re-train the model


    # TODO: make process continuous where each day before market close:
    #   - run PCA on the past 30 days of market data to select the most relevant stocks
    #   - re-train the LSTM model with the selected stocks
    #   - hold or buy stocks that remained in the list
    #   - sell stocks that are no longer in the list
    #   - buy stocks that were added to the list

