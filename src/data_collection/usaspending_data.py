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
    df.index = pd.to_datetime(df.index, unit='ms')
    print_df(df.sort_index(), "usa_spend")
 
    # df.set_index(["Date"])
    # print(type(df.index))
    return df

def print_df(df, filename):
   with open(f"{filename}.txt", "w") as f:
        f.write(df.head(50).to_string())
    