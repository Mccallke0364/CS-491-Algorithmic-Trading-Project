
import pandas as pd
import requests
import time
import numpy as np
import os

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
    
def print_df(df, filename, location="dataframes/"):
    with open(f"{location}{filename}.txt", "w") as f:
        f.write(df.to_string())
    with open(f"{location}{filename}.csv", "w") as f_csv:
        f_csv.write(df.to_csv())

def get_data_for_all_stocks():
    dict={}
    for a in os.listdir("dataframes"):
        if a.endswith(".csv"):
            if a!="usa_spend.csv":
                date = "t"
                ticker = a[:-4]
                df = pd.read_csv(f"dataframes/{a}", parse_dates=[date], header=0, index_col=0)
                df.rename(columns={'vw': f'vw_{ticker}', "n": f"n_{ticker}"}, inplace=True)
                dict[a[:-4]] = df[[f'o_{ticker}', f'h_{ticker}', f'l_{ticker}', f'c_{ticker}',f'v_{ticker}', f"vw_{ticker}", f"n_{ticker}", f'{ticker}_SMA_10',f'{ticker}_SMA_50',f'{ticker}_Returns']]
            else:
                date = "Date"
                dataframe = pd.read_csv(f"dataframes/{a}", parse_dates=[date], header=0, index_col=0)
                continue
        
    return dict, dataframe

dict_stock_dfs, usa_spending_data = get_data_for_all_stocks()

full_dataframe = merge_dataframes(usa_spending_data, dict_stock_dfs)
    #merges all the stock dataframes and usa spending based on the date the data was collected

# print_df(full_dataframe, "full_df")
print(full_dataframe.head())