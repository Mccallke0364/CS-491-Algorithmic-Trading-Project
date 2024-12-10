from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import RMSprop
from keras.losses import Huber
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler


def build_model(input_shape, num_stocks=42):
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
    print(model.summary())
    return model

def model_atmpt_2(input_shape, num_stocks=5):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape = input_shape))

    model.add(Dropout(0.1)) 
    model.add(LSTM(units=50))

    model.add(Dense(42))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    print(model.summary())
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


def implement_model(df, model, train_seq, train_label, test_seq, test_label, epochs=3, batch_size=64, verbose=1):
    """
    Trains the LSTM model.

    Parameters:
    model (keras.models.Sequential): The compiled LSTM model.
    train_seq (numpy.ndarray): Input features.
    train_label (numpy.ndarray): Target values.
    validataion_data : tuple of test seq and label
    epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.

    Returns:
    keras.callbacks.History: History object containing training history.
    """
    model.fit(train_seq, train_label, epochs=epochs, batch_size=batch_size, validation_data=(test_seq, test_label))
    test_predicted = model.predict(test_seq)
    test_p_df= pd.DataFrame(test_predicted)
    print_df(test_p_df, "test_predict_a")
    test_inverse_predicted = MMS.inverse_transform(test_predicted)
    test_i_p_df= pd.DataFrame(test_inverse_predicted)
    print_df(test_i_p_df, "inverse_predict_a")
    new_df = pd.concat(df, test_p_df)
    return new_df

def print_df(df, filename):
   with open(f"{filename}.txt", "w") as f:
        f.write(df.head(50).to_string())