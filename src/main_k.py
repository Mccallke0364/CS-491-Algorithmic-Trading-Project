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
import keras
import sklearn

from concurrent.futures import ThreadPoolExecutor
import os

# Function to run on each thread (example task)
def process_task(task_id):
    print(f"Running task {task_id} on thread {os.getpid()}")
    # Add your task code here
    return task_id

def run_in_parallel():
    num_threads = os.cpu_count()  # Number of threads, usually same as number of cores
    print(f"Using {num_threads} threads.")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_task, range(num_threads)))
    
    print(f"Results: {results}")

if __name__ == "__main__":
    run_in_parallel()
import tensorflow as tf

# Limit TensorFlow to a specific number of CPU threads (optional)
# Set this to the number of threads you want to use
num_threads = os.cpu_count()
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_threads, inter_op_parallelism_threads=num_threads)
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Now you can train your model as usual with multiple threads


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
# print([a for a in full_dataframe])
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
    #creating the sequence for testing


# print(train_seq)
################---------Build model---------################
# model = model_atmpt_2((train_seq.shape[1], train_seq.shape[2]))
    #building model, automatically takes the default number of stocks, (5), since it is not specified in the call
input_shape = (train_seq.shape[1], train_seq.shape[2])
num_stocks=44

model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(num_stocks, activation='tanh')  # Using tanh activation
    ])

model.compile(optimizer=RMSprop(), loss=Huber())
# model.fit(train_seq, train_label, validation_data=(test_seq, test_label), epochs=3, batch_size=64)

# # trained_model = train_model(full_dataframe, model, train_seq, train_label, test_seq, test_label) #returns a History object that contains history dictionary
# # trained_model.save('trained_model.keras')


# # Plot loss and validation loss
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(training_results['loss'], label='Training Loss', color='b', linewidth=2)
# ax.plot(training_results['val_loss'], label='Validation Loss', color='r', linestyle='--', linewidth=2)

# # Labels, title, and legend
# ax.set_xlabel('Epoch', fontsize=14)
# ax.set_ylabel('Loss', fontsize=14)
# ax.set_title('Model Loss Over Epochs', fontsize=16)
# ax.legend(fontsize=12)

# Input shape and number of stocks
# input_shape = (train_seq.shape[1], train_seq.shape[2])
# num_stocks = 44

# # Build the model
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=input_shape),
#     Dropout(0.2),
#     LSTM(50),
#     Dropout(0.2),
#     Dense(num_stocks, activation='tanh')  # Using tanh activation
# ])

# # Compile the model
# model.compile(optimizer=RMSprop(), loss=Huber())

# Fit the model and get the training history
history = model.fit(train_seq, train_label, validation_data=(test_seq, test_label), epochs=20, batch_size=64)
with open('history_one_d_20.json', 'w') as f:
    json.dump(history.history, f)
model.save("saved_models/one_d_20")
# Extract the training history

# Display the plot
# plt.grid(True)
# fig.savefig("")

model_2 = Sequential()
model_2.add(LSTM(units=50, return_sequences=True, input_shape = input_shape))

model_2.add(Dropout(0.1)) 
model_2.add(LSTM(units=50))

model_2.add(Dense(num_stocks, activation="tanh"))

model_2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

print(model_2.summary())

epochs=20
batch_size=64
verbose=1
history_2 = model_2.fit(train_seq, train_label, epochs=epochs, batch_size=batch_size, validation_data=(test_seq, test_label), verbose=verbose)

# Optionally, save the model
with open('history_trained_20.json', 'w') as f:
    json.dump(history.history, f)
model.save('trained_model_20.keras')


# Display plot
# plt.grid(True)
# plt.show()
# training_results = trained_model.history
# print(type(training_results))
# print(trained_model)


# # pre_trained_model = keras.models.load_model("src/trained_model.trained_model.keras")
# # print(training_results.keys())

# # model_implementation=model
# # Plot loss
# fig, ax = plt.subplots()
# for a in trained_model.history:
#     ax.plot(a)
# # plt.plot(training_results.history['val_loss'], label='Validation Loss', linestyle='--', marker='x')

# # Title and labels
# ax.set_title('Model Performance Over Epochs', fontsize=16)
# ax.set_xlabel('Epoch', fontsize=14)
# ax.set_ylabel('Value', fontsize=14)

# # Legends and grid
# ax.legend(fontsize=12)
# ax.grid(alpha=0.3)

# Show plot
# plt.tight_layout()
# plt.show()
# plt.legend(['train','val','loss','val loss'], loc="best")


    # return model
# input_shape=train_seq.shape[1], train_seq.shape[2]))
# modela_2= implement_model(full_dataframe, model_2, train_seq, train_label, test_seq, test_label)

# model_implementation=modela_2
  # This contains the history dictionary

# Plot loss and validation loss
# fig, ax = plt.subplots()
# ax.plot(training_results['loss'], label='Training Loss', color='b', linewidth=2)
# ax.plot(training_results['val_loss'], label='Validation Loss', color='r', linestyle='--', linewidth=2)

# Labels, title, and legend

# print_df(model_frame, "implementdf")
# print_df(implement_model(full_dataframe, model, train_seq, train_label, test_seq, test_label), "full_model_1")
    #used train_model_pre_split since the data has already been split into training and testing data
    #anything not specified has a default in the function call
