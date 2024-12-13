import pandas as pd


data=  pd.read_csv('DTS_OpCashBal_20191211_20241210(2).csv')

df = pd.DataFrame(data)
print(df.columns)

# Rename columns to remove spaces, if necessary
df.rename(columns=lambda x: x.strip(), inplace=True)

# Convert Date column to datetime format
df["Record Date"] = pd.to_datetime(df["Record Date"])

# Sort by date
df = df.sort_values(by="Record Date")
df = df.drop("")
# Consolidate the data into one row per date
consolidated_df = df.pivot_table(
    index="Record Date", 
    columns="Type of Account", 
    values="Opening Balance Today",  # Replace with actual numeric column
    aggfunc="sum"
).reset_index()

# Rename columns for readability
consolidated_df.columns = [
    "Date", 
    "Opening Balance", 
    "Deposits", 
    "Withdrawals", 
    "Closing Balance"
]

# Calculate missing balances
consolidated_df["Opening Balance"] = consolidated_df["Closing Balance"].shift(1)
consolidated_df["Closing Balance"] = (
    consolidated_df["Opening Balance"] 
    + consolidated_df["Deposits"] 
    - consolidated_df["Withdrawals"]
)

print(consolidated_df)