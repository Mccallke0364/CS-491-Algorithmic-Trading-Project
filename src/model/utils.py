import numpy as np
import matplotlib.pyplot as plt

def generate_multi_stock_signals(data, model, threshold=0.5):
    """
    Generates buy/sell signals based on model predictions.

    Parameters:
    data (numpy.ndarray): Input data for prediction.
    model (keras.models.Sequential): The trained model.
    threshold (float): Threshold for deciding buy/sell signals.

    Returns:
    tuple: Predicted returns and buy/sell signals.
    """
    
    predictions = model.predict(data)
    signals = [
        'Buy' if pred > threshold else 'Sell'
        for pred in predictions[0]
    ]
    return predictions[0], signals

def plot_training_history(history):
    """
    Plots training and validation loss over epochs.

    Parameters:
    history (keras.callbacks.History): History object containing training history.
    """
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()
