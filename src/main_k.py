import os
import numpy as np
import pandas as pd
from data_collection.polygon_data import *
#from data_collection.bezinga_data import get_government_trades_data
from data_collection.usaspending_data import *
from preprocessing.preprocess_data import *
from model.lstm_model import *
from model.utils import *


################---------Collect data from polygon.io and usa spending---------################
dict_stock_dfs = get_data_for_multiple_tickers() 
    # a dictionary of each stocks dataframe 
    # set up with the default stocks of ['NGL', 'TSLA', 'AAPL', 'V', 'NSRGY'] and dates start_date = '2023-10-01' end_date = '2024-12-30' 
usa_spending_data = get_usaspending_data() 
    # dataframe of the usa spending data for each date 
    # set up with the default path to be 'data_collection/usaspending_data.csv'
usa_spending_data = usa_spending_data.filter(["total_obligations",  "total_outlayed_amount"])
full_dataframe = merge_dataframes(usa_spending_data, dict_stock_dfs)
    #merges all the stock dataframes and usa spending based on the date the data was collected

print_df(full_dataframe, "full_df")

################---------Process data for model---------################
train_data, test_data = split_train_test(full_dataframe)
    #split data 80:20 for training and testing
train_seq, train_label = create_sequences(train_data)
    #creating the sequence for training
test_seq, test_label = create_sequences(test_data)
    #creating the sequence for testing


print(train_seq)
################---------Build model---------################
# model = model_atmpt_2((train_seq.shape[1], train_seq.shape[2]))
    #building model, automatically takes the default number of stocks, (5), since it is not specified in the call
model = build_model((train_seq.shape[1], train_seq.shape[2]))

model_frame = train_model(full_dataframe, model, train_seq, train_label, test_seq, test_label)
print_df(model_frame, "implementdf")
# print_df(implement_model(full_dataframe, model, train_seq, train_label, test_seq, test_label), "full_model_1")
    #used train_model_pre_split since the data has already been split into training and testing data
    #anything not specified has a default in the function call
