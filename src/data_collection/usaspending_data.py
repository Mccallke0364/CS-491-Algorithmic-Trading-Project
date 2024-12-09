import pandas as pd

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
    # df.set_index(["Date"])
    # print(type(df.index))
    return df