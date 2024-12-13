# from main_k.py import full_dataframe
import os
import numpy as np
import pandas as pd
import json
from data_collection.polygon_data import *
#from data_collection.bezinga_data import get_government_trades_data
from data_collection.usaspending_data import *
from preprocessing.preprocess_data import *
from model.lstm_model import *
from model.utils import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Bidirectional
from keras.optimizers import RMSprop
from keras.losses import Huber
import tensorflow as tf
import keras
import sklearn
import json
import seaborn as sns
# import matplotlib.pyplot as plt
# run_in_parallel(main_k.py)
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
    dict_a={}
    for a in os.listdir("../notebooks/dataframes/"):
       
        if a.endswith(".csv"):
            if a!="usa_spend.csv":
                # print(a)
                date = "t"
                ticker = a[:-4]
                df = pd.read_csv(f"../notebooks/dataframes/{a}", parse_dates=[date], header=0, index_col=0)
                df.rename(columns={'vw': f'vw_{ticker}', "n": f"n_{ticker}"}, inplace=True)
                # df.rename(columns={f'{ticker}_Returns': f'{ticker}'}, inplace=True)
                dict_a[ticker]= df[[f"{ticker}_Returns"]]
                # dict[a[:-4]] = df[[f'o_{ticker}', f'h_{ticker}', f'l_{ticker}', f'c_{ticker}',f'v_{ticker}', f"vw_{ticker}", f"n_{ticker}", f'{ticker}_SMA_10',f'{ticker}_SMA_50',f'{ticker}_Returns']]
            else:
                date = "Date"
                dataframe = pd.read_csv(f"../notebooks/dataframes/{a}", parse_dates=[date], header=0, index_col=0)
                
                continue
        
    return dict_a, dataframe[["total_outlayed_amount"]]

dict_stock_dfs, usa_spending_data = get_data_for_all_stocks()

full_dataframe = merge_dataframes(usa_spending_data, dict_stock_dfs)

full_dataframe.dropna(inplace=True)
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

# fig, ax = plt.subplots()
# sns.lineplot(data=full_dataframe, ax=ax)
# fig.savefig("returns.png")
# for column in full_dataframe.T.itertuples():
    # sns.lineplot(


# sns.lineplot(x=data=full_dataframe)


new_model = tf.keras.models.load_model('saved_models/one_d_20')

with open('history_one_d_20.json', 'r') as f:
    saved_history = json.load(f)

fig_1, ax_1 = plt.subplots()
ax_1.plot(saved_history['loss'], label='Training Loss', color='b', linewidth=2)
ax_1.plot(saved_history['val_loss'], label='Validation Loss', color='r', linestyle='--', linewidth=2)
ax_1.set_xlabel('Epoch', fontsize=14)
ax_1.set_ylabel('Loss', fontsize=14)
ax_1.set_title('Model Loss Over Epochs', fontsize=16)
ax_1.legend(fontsize=12)
fig_1.savefig("loss_chart_model_1_20.png", bbox_inches="tight", dpi=500)



# training_results = history.history  # This contains the history dictionary

# Plot loss and validation loss
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(training_results['loss'], label='Training Loss', color='b', linewidth=2)
# ax.plot(training_results['val_loss'], label='Validation Loss', color='r', linestyle='--', linewidth=2)

# # Labels, title, and legend
# ax.set_xlabel('Epoch', fontsize=14)
# ax.set_ylabel('Loss', fontsize=14)
# ax.set_title('Model Loss Over Epochs', fontsize=16)
# ax.legend(fontsize=12)
# fig.savefig("test.png", dpi=500, bbox_inches="tight")

model_2 = tf.keras.models.load_model("trained_model_20.keras")

# model_implementation= history_2.history
test_predicted = model_2.predict(test_seq)
test_p_df= pd.DataFrame(test_predicted)
print_df(test_p_df, "test_predict_a")
test_inverse_predicted = Ms.inverse_transform(test_predicted)
test_i_p_df= pd.DataFrame(test_inverse_predicted)
print_df(test_i_p_df, "inverse_predict_a")
new_df = pd.concat([full_dataframe, test_p_df])

print_df(new_df, "full_predict")


with open('history_trained_20.json', 'r') as f:
    saved_history = json.load(f)
fig_2, ax_2 = plt.subplots()
ax_2.plot(saved_history['loss'], label='Training Loss', color='b', linewidth=2)
ax_2.plot(saved_history['val_loss'], label='Validation Loss', color='r', linestyle='--', linewidth=2)
ax_2.set_xlabel('Epoch', fontsize=14)
ax_2.set_ylabel('Loss', fontsize=14)
ax_2.set_title('Model Loss Over Epochs', fontsize=16)
ax_2.legend(fontsize=12)


# ax.plot(model_implementation['accuracy'])
# ax.plot(model_implementation['val_accuracy'],ls="--")

# plt.title('Model Accuracy')
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend(['train','val','loss','val loss'], loc="best")
fig_2.savefig("test_2_20.png", dpi=500, bbox_inches="tight")

