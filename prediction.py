# @ Author: Bertan Berker
# @ Language: Python
# @ File: prediction.py


import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import yfinance as yf


# Function to get historical Bitcoin data from Yahoo Finance
def get_bitcoin_data():
    bitcoin_data = yf.download('BTC-USD', start='2020-01-01', end='2023-01-01')
    return bitcoin_data

# Fetch Bitcoin data
bitcoin_data = get_bitcoin_data()

# Drop rows with missing 'open' values
bitcoin = bitcoin_data.dropna(subset=['Open'])

# Splitting data into training and testing data sets (80-20%)
train_bitcoin = bitcoin.iloc[:int(.80*len(bitcoin)), :]
test_bitcoin = bitcoin.iloc[int(.20*len(bitcoin)):, :] 

# Defining the features and target variable
features = ['Open', 'Volume']
target = 'Close'

# Create and train model
model = xgb.XGBRegressor()
model.fit(train_bitcoin[features], train_bitcoin[target])

# Make and show the predictions on the test data
value_predictions = model.predict(test_bitcoin[features])

accuracy_value = model.score(test_bitcoin[features], test_bitcoin[target])
print(accuracy_value)

plt.plot(bitcoin['Close'], label = 'Bitcoin Price')
plt.plot(test_bitcoin['Close'].index, value_predictions, label  = "Predictions")
plt.legend()
plt.show()