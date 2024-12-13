import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_multi_stock_signals(data, model, tickers, threshold=0.5):
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
        # Create a DataFrame
    print(f"Length of predictions: {len(predictions[0])}")
    print(f"Length of signals: {len(signals)}")
    print(f"Length of tickers: {len(tickers)}")
    results_df = pd.DataFrame({
        "Ticker": tickers,
        "Expected Return Rate": predictions[0],
        "Signal": signals
    })
    results_df = results_df.sort_values(by="Expected Return Rate", ascending=False)

    return results_df

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
