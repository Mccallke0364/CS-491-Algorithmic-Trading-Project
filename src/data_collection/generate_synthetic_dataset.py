import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Parameters for the synthetic dataset
num_days = 365  # Number of days of data
num_stocks = 5  # Number of different stock tickers
start_date = datetime(2023, 1, 1)

# Generate date range
dates = [start_date + timedelta(days=i) for i in range(num_days)]

# Generate synthetic stock data
tickers = ['AAPL', 'MSFT', 'GOOGL','CAT', 'NSRGY']
stock_data = {
    ticker: {
        'Date': dates,
        'Open': np.random.uniform(100, 150, num_days),
        'High': np.random.uniform(150, 200, num_days),
        'Low': np.random.uniform(90, 140, num_days),
        'Close': np.random.uniform(100, 150, num_days),
        'Volume': np.random.randint(1_000_000, 5_000_000, num_days),
    } for ticker in tickers
}

# Convert to DataFrame and combine
stock_dfs = []
for ticker, data in stock_data.items():
    df = pd.DataFrame(data)
    df['Ticker'] = ticker
    stock_dfs.append(df)

combined_stock_df = pd.concat(stock_dfs)

# Generate synthetic government spending data
gov_data = {
    'Date': dates,
    'Total Obligations': np.random.uniform(1_000_000, 10_000_000, num_days),
    'Total Outlays': np.random.uniform(900_000, 9_000_000, num_days),
    'Budget Function': ['Defense', 'Healthcare', 'Education'] * (num_days // 3 + 1),
    'Award Type': ['Contract', 'Grant', 'Loan'] * (num_days // 3 + 1),
    'Recipient': ['ABC Corp', 'XYZ Inc', '123 Ltd'] * (num_days // 3 + 1),
    'Agency': ['DoD', 'HHS', 'ED'] * (num_days // 3 + 1)
}

# Adjust lengths of non-date lists to match the number of days
for key in ['Budget Function', 'Award Type', 'Recipient', 'Agency']:
    gov_data[key] = gov_data[key][:num_days]

gov_df = pd.DataFrame(gov_data)

# Merge the data
merged_data = pd.merge(combined_stock_df, gov_df, on='Date', how='left')

# Example of how the final DataFrame might look
print(merged_data)

# Save to CSV for LSTM input
merged_data.to_csv('synthetic_lstm_input.csv', index=False)
