#IMPORT NECESSARY LIBRARIES
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

#DEFINE TICKERS AND DATE RANGE
tickers = ['NGL', 'TSLA', 'AAPL', 'V', 'NSRGY']
start_date = '2023-10-01'
end_date = '2024-12-30'

#LOAD MERGED DATASET FROM CSV 'lstm_input.csv'

# Date	        total_obligations	total_outlayed_amount	awarding_agency_name	recipient_name	action_type	o_NGL	h_NGL	l_NGL	c_NGL	v_NGL	o_TSLA	h_TSLA	l_TSLA	c_TSLA	v_TSLA	o_AAPL	h_AAPL	l_AAPL	c_AAPL	v_AAPL	o_V	h_V	l_V	c_V	v_V	o_NSRGY	h_NSRGY	l_NSRGY	c_NSRGY	v_NSRGY
# 10/28/2024	0		            Department of Housing and Urban Development	UNITED WHOLESALE MORTGAGE, LLC	NEW	4.18	4.24	4.06	4.08	96477	270	273.536	262.24	262.51	104072158	233.32	234.73	232.55	233.4	32964281	282.04	284.64	281.53	284.19	4118591	97.48	97.8399	97.425	97.55	337937
# 10/7/2024	    680		            Department of Agriculture	REDACTED DUE TO PII	NEW	4.37	4.65	4.37	4.63	226475	249	249.83	240.7	240.83	65363518	224.5	225.69	221.33	221.69	37595470	277.6	277.615	273.24	273.79	4160646	98.23	98.34	97.44	97.48	839147
# 11/12/2024	15000		        Department of Homeland Security	WRIGHT NATIONAL FLOOD INSURANCE COMPANY	NEW	4.42	4.47	4.13	4.13	270025	342.74	345.84	323.31	328.49	155039630	224.55	225.59	223.355	224.23	37632672	309.04	310.55	308.11	309.85	4459893	88.58	88.638	87.89	88.5	2274438


print('LOADING DATA FROM CSV..._______________________________________\n')

merged_df = pd.read_csv('lstm_input.csv', parse_dates=['Date'], header=0, index_col=0)
merged_df.index = pd.to_datetime(merged_df.index, unit='ms')

df = merged_df.drop(columns=['total_outlayed_amount'])
df = df.fillna(0)
#df = df.sample(frac = 0.3)

print(df.head())

print("PREPROCESSING THE DATA..._______________________________________\n")

#NORMALIZE NUMERICAL COLUMNS
numeric_cols = df.select_dtypes(include=np.number)
numeric_df = df[numeric_cols.columns.tolist()]
print(numeric_df.head())

scaler = StandardScaler()
numeric_data = scaler.fit_transform(numeric_df)
print(numeric_data)

#CREATE SEQUENCES
print('CREATING SEQUENCES FOR LSTM INPUT_______________________________________\n')
X_list, Y_list = [], []
#define sequence length
window_size = 30

target_cols = ['c_NGL', 'c_TSLA', 'c_AAPL', 'c_V', 'c_NSRGY']
target_indices = []

# print
for col in target_cols:
    target_indices.append(numeric_df.columns.get_loc(col))


print('target columns: ', target_cols, '\n')
print('target indices: ', target_indices, '\n')

for i in range(len(numeric_data) - window_size):
    X_list.append(numeric_data[i: i+window_size])
    Y_list.append(numeric_data[i+window_size, target_indices])

X = np.array(X_list)
y = np.array(Y_list) 
# X=X_list
# y=Y_list
#implement 80|20 train test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[: train_size], X[train_size:]
y_train, y_test = y[: train_size], y[train_size:]

# Reshape for LSTM input (add batch dimension)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

print(type(X_train))
print(type(y_train))

print(X_train.shape)  
print(y_train.shape)

print('BUILDING THE LSTM MODEL..._______________________________________\n')
#BUILD MODEL
num_stocks = 5
model = Sequential()
#model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        #Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(num_stocks, activation='tanh')  # Using tanh activation
    ])

model.compile(optimizer=RMSprop(), loss=Huber())  # Using RMSprop and Huber Loss
model.summary()

#TRAIN MODEL
print('TRAINING THE LSTM MODEL..._______________________________________\n')
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)
print('model has finished training\n')


# Predict on the training set
train_predictions = model.predict(X_train)

# Predict on the test set
test_predictions = model.predict(X_test)

# # If you want to inverse scale the predictions, you can use the scaler's inverse_transform method:
# train_predictions_scaled = scaler.inverse_transform(train_predictions)
# test_predictions_scaled = scaler.inverse_transform(test_predictions)

# Optionally, print the predictions
print("Training predictions:")
print(train_predictions)

print("Testing predictions:")
print(test_predictions)



plt.figure(figsize=(10, 5))
plt.plot(y_test[:, 0], label='Actual')
plt.plot(test_predictions[:, 0], label='Predicted')
plt.legend()
plt.title('Predictions vs Actual for c_NGL')
plt.show()

corr_matrix = df.corr()
print(corr_matrix['c_NGL'].sort_values(ascending=False))



# Calculate the evaluation metrics for the test set
mse = mean_squared_error(y_test, test_predictions)
mae = mean_absolute_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f"Test Set MSE: {mse}")
print(f"Test Set MAE: {mae}")
print(f"Test Set R2 Score: {r2}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

