import os
import numpy as np
import pandas as pd
import matplotlib as plt
from data_collection.polygon_data import *
#from data_collection.bezinga_data import get_government_trades_data
from data_collection.usaspending_data import *
from preprocessing.preprocess_data import *
from model.lstm_collin import *
from model.utils import *
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import RMSprop
from keras.losses import Huber



################---------Collect data from polygon.io and usa spending---------################
"""time.sleep(60)
dict_stock_dfs = get_data_for_multiple_tickers() 
    # a dictionary of each stocks dataframe 
    # set up with the default stocks of ['NGL', 'TSLA', 'AAPL', 'V', 'NSRGY'] and dates start_date = '2023-10-01' end_date = '2024-12-30' 
usa_spending_data = get_usaspending_data(filepath='data_collection/usaspending_data.csv') 
    # dataframe of the usa spending data for each date 
    # set up with the default path to be 'data_collection/usaspending_data.csv'
usa_spending_data = usa_spending_data.filter(["total_obligations",  "total_outlayed_amount"])
full_dataframe = merge_dataframes(usa_spending_data, dict_stock_dfs)
    #merges all the stock dataframes and usa spending based on the date the data was collected

print_df(full_dataframe, "full_df")"""
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


tickers = []
def get_data_for_all_stocks():
    dict={}
    for a in os.listdir("../notebooks/dataframes/"):
       
        if a.endswith(".csv"):
            if a!="usa_spend.csv":
                # print(a)
                date = "t"
                ticker = a[:-4]
                tickers.append(ticker)
                df = pd.read_csv(f"../notebooks/dataframes/{a}", parse_dates=[date], header=0, index_col=0)
                df.rename(columns={'vw': f'vw_{ticker}', "n": f"n_{ticker}"}, inplace=True)
                # df.rename(columns={f'{ticker}_Returns': f'{ticker}'}, inplace=True)
                dict[ticker]= df[[f"{ticker}_Returns"]]
                # dict[a[:-4]] = df[[f'o_{ticker}', f'h_{ticker}', f'l_{ticker}', f'c_{ticker}',f'v_{ticker}', f"vw_{ticker}", f"n_{ticker}", f'{ticker}_SMA_10',f'{ticker}_SMA_50',f'{ticker}_Returns']]
            else:
                date = "Date"
                dataframe = pd.read_csv(f"../notebooks/dataframes/{a}", parse_dates=[date], header=0, index_col=0)
                
                continue
        
    return dict, dataframe[["total_outlayed_amount"]], tickers

dict_stock_dfs, usa_spending_data, tickers = get_data_for_all_stocks()

full_dataframe = merge_dataframes(usa_spending_data, dict_stock_dfs)
# print([a for a in full_dataframe])

full_dataframe.dropna(inplace=True)
full_dataframe = full_dataframe.drop_duplicates()
print(full_dataframe)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(full_dataframe)
full_dataframe = pd.DataFrame(scaled_data)
# print_df(full_dataframe, "full_df")


################---------Process data for model---------################
train_data, test_data, Ms = split_train_test(full_dataframe)
    #split data 80:20 for training and testing
train_seq, train_label = create_sequences(train_data)
    #creating the sequence for training
test_seq, test_label = create_sequences(test_data)
    #creating the sequence for testing


# print(train_seq)
################---------Build model---------################
# model = model_atmpt_2((train_seq.shape[1], train_seq.shape[2]))
    #building model, automatically takes the default number of stocks, (5), since it is not specified in the call

print('_________________________________building_model_________________________________')
model = build_model((train_seq.shape[1], train_seq.shape[2]))

print('_________________________________training_model_________________________________')
history = model.fit(train_seq, train_label, validation_data=(test_seq, test_label), epochs=3, batch_size=64)
model.save('trained_model.keras')

print('_________________________________generating_predictions_________________________________')
ranked_signals= generate_multi_stock_signals(test_seq, model, tickers, threshold=.5)
hist_df = pd.DataFrame(history.history) 
print(hist_df)
test_p_df= pd.DataFrame(ranked_signals['Expected Return Rate'])
print_df(test_p_df, "test_predict_a")
# test_inverse_predicted = Ms.inverse_transform(test_predictions)
# test_i_p_df= pd.DataFrame(test_inverse_predicted)
# print_df(test_i_p_df, "inverse_predict_a")
new_df = pd.concat([full_dataframe, test_p_df])
print('Ticker signals: ', signals)

print('_________________________________plotting_results_________________________________')
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Loss', fontsize=14)
ax.set_title('Model Loss Over Epochs', fontsize=16)
ax.legend(fontsize=12)  
plt.savefig("test_collin.png", dpi=500, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(
    ranked_signals["Ticker"], 
    ranked_signals["Expected Return Rate"], 
    color=['green' if signal == 'Buy' else 'red' for signal in ranked_signals["Signal"]]
)
plt.xlabel("Ticker")
plt.ylabel("Expected Return Rate")
plt.title("Ranked Buy and Sell Signals by Expected Return Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ranked_signals_plot.png")
plt.show()





# plt.savefig("test.png", dpi=500, bbox_inches="tight")

# model_2 = model_atmpt_2((train_seq.shape[1], train_seq.shape[2]))

# modela_2= implement_model(full_dataframe, model_2, train_seq, train_label, test_seq, test_label)
# test_predicted = modela_2.predict(test_seq)
# test_p_df= pd.DataFrame(test_predicted)
# print_df(test_p_df, "test_predict_a")
# test_inverse_predicted = Ms.inverse_transform(test_predicted)
# test_i_p_df= pd.DataFrame(test_inverse_predicted)
# print_df(test_i_p_df, "inverse_predict_a")
# # new_df = pd.concat(full_df, test_p_df)
# model_implementation=model_2
# # print(model_implementation.history['accuracy'])
# # plt.plot(model_implementation.history.history['accuracy'])
# # plt.plot(model_implementation.history.history['val_accuracy'],ls="--")
# # plt.plot(model_implementation.history.history['loss'])
# # plt.plot(model_implementation.history.history['val_loss'])
# # plt.title('Model Accuracy')
# # plt.xlabel("Epoch")
# # plt.ylabel("Accuracy")
# plt.legend(['train','val','loss','val loss'], loc="best")
# plt.savefig("test.png", dpi=500, bbox_inches="tight")
# # print_df(model_frame, "implementdf")
# # print_df(implement_model(full_dataframe, model, train_seq, train_label, test_seq, test_label), "full_model_1")
#     #used train_model_pre_split since the data has already been split into training and testing data
#     #anything not specified has a default in the function call
