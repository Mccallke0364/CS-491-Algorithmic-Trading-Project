from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
from keras.losses import Huber

def build_model(input_shape, num_stocks):
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
    model.summary()
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

