
## Quick Setup
* TLDR quick setup/guide with project opened in VS Code
    1. `python -m venv .venv`
    2. Unix: `source bin/activate`. Windows: `Scripts\Activate.ps1`. For Windows, you may have to set a security exception in a Powershell terminal. `set-executionpolicy remotesigned`
    3. `pip install -r requirements.txt`
    4. If using Juypter Notebook
        * `python -m ipykernel install --user --name=project_kernel`
    5. You may have to restart VS Code to do the next step
    6. For every Jupyter Notebook, select the ipykernel

# Algorithmic Trading Project with Government and Market Data
This project uses a combination of government trade insights, market data, and economic indicators to build an LSTM-based algorithmic trading model. The model leverages data from the Benzinga Government Trades API, Polygon.io for market data, and the USASpending API to predict market movements and make informed buy/sell decisions.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Data Sources](#data-sources)
6. [Model Training and Evaluation](#model-training-and-evaluation)


## Project Overview
This project integrates three main datasets:

Market Data: Retrieved from Polygon.io, providing historical and real-time stock information.
Government Trade Data: Retrieved from the Benzinga Government Trades API, providing insights into insider trading by government officials.
Government Investment Data: Pulled from the USASpending API, providing data on government spending patterns and economic indicators.
The project prepares these datasets, processes them, and feeds them into an LSTM model, which makes trading predictions based on a combination of historical trends, government trade insights, and investment data.

## Installation
### Requirements
Python 3.8+
Required Libraries: Listed in requirements.txt
- run pip install -r requirements.txt
### API Keys
You need API keys for Polygon.io, Benzinga Government Trades API, and USASpending.
Store these in a .env file or in src/config.py.

## Configuration
Place API keys and other configuration details in config.py or .env for secure and centralized management.

## Usage
### 1. Data Collection
Run main.py to collect data from all sources. This will retrieve and store raw data in the data/raw directory.

### 2. Data Processing
The preprocess_data.py script in the src/preprocessing/ directory handles data cleaning, merging, and feature engineering.

### 3. Model Training
Once the data is processed, lstm_model.py in src/model/ builds and trains the LSTM model.

### 4. Predicting and Generating Signals
After training, the model will use live market data to make buy/sell predictions based on government trading activity and spending insights.

## Running the Project
To run the full pipeline:
- run "python main.py"

## Data Sources
Polygon.io: Provides historical and live stock market data.
Benzinga Government Trades API: Supplies information on government insider trades.
USASpending API: Gives details on government spending.

# Model Training and Evaluation
The model training process is managed in lstm_model.py:

 - Input: Merged data from all sources.
 - Output: Buy/Sell signals, confidence scores, and expected return rates based on trained LSTM predictions.
 - Evaluation: Mean Squared Error and accuracy metrics are used for model validation.
