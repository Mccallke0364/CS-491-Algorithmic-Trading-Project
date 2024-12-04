import pandas as pd

def get_usaspending_data(filepath):
    """
    Loads government spending data from a CSV file.
    
    Parameters:
    filepath (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: A DataFrame containing the government spending data.
    """
    return pd.read_csv(filepath, parse_dates=['Date'])