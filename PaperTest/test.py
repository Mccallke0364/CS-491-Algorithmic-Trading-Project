import requests
import json
import os
from datetime import datetime

# Example: Base URL for the DTS API (replace with the correct URL)
API_BASE_URL = "https://api.fiscal.treasury.gov/services/api/daily-treasury-statements" 

# Set headers for the API request (update if authentication is required)
HEADERS = {
    "Accept": "application/json",  # Update to the accepted format of the API
    "User-Agent": "DTS-Downloader/1.0"
}

def fetch_dts_data(start_date: str, end_date: str, output_dir: str):
    """
    Fetches Daily Treasury Statement data from the API and saves it as JSON files.

    :param start_date: Start date for the data in YYYY-MM-DD format.
    :param end_date: End date for the data in YYYY-MM-DD format.
    :param output_dir: Directory to save the downloaded files.
    """
    params = {
        "start_date": start_date,  # Replace with actual parameter name
        "end_date": end_date       # Replace with actual parameter name
    }

    try:
        response = requests.get(API_BASE_URL, headers=HEADERS, params=params)
        response.raise_for_status()  # Raise an error for HTTP errors

        data = response.json()  # Parse the JSON response

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save data to a file
        output_file = os.path.join(output_dir, f"DTS_{start_date}_to_{end_date}.json")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Data saved successfully to {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    start_date = "2024-12-01"
    end_date = "2024-12-12"
    output_dir = "./dts_data"

    # Download the data
    fetch_dts_data(start_date, end_date, output_dir)