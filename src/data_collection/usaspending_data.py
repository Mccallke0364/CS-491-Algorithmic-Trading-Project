import pandas as pd
import os
from zipfile import ZipFile

def get_usaspending_data_from_zip(zip_file_path):
    """
    Loads and combines USASpending data from multiple CSV files within a zip file.

    Parameters:
    zip_file_path (str): Path to the zip file containing USASpending CSV files.

    Returns:
    pd.DataFrame: Combined DataFrame of USASpending data from all CSVs in the zip file.
    """
    usaspending_data = []
    
    with ZipFile(zip_file_path, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        i=0
        for csv_file in csv_files:
            with zip_ref.open(csv_file) as f:
                df = pd.read_csv(f, usecols=[
                    'action_date', 'federal_action_obligation', 'recipient_name', 'funding_agency_name'
                ])
                df['action_date'] = pd.to_datetime(df['action_date']).dt.date
                usaspending_data.append(df)
                i+=1
                print(f'pulled data from the csv #{i}:')
    
    combined_usaspending_df = pd.concat(usaspending_data, ignore_index=True)
    return combined_usaspending_df
