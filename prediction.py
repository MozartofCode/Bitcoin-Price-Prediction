# @ Author: Bertan Berker
# @ Language: Python
# @ File: prediction.py

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the inflation dataset
bitcoin_data = pd.read_csv('bitcoin.csv')
bitcoin = bitcoin_data.dropna(subset=['open', 'Volume_USD', 'Volume_BTC'])

# Assuming df is your DataFrame
bitcoin['Date'] = pd.to_datetime(bitcoin['Date'])
bitcoin = bitcoin.sort_values(by='Date')

# Splitting data into training and testing data sets (80-20%)
train_bitcoin = bitcoin.iloc[:int(.80*len(bitcoin)), :]
test_bitcoin = bitcoin.iloc[int(.80*len(bitcoin)):, :] 

# Defining the features and target variable
features = ['open', 'Volume_USD', 'Volume_BTC']
target = 'close'

# Create and train model
model = xgb.XGBRegressor()
model.fit(train_bitcoin[features], train_bitcoin[target])

# Make and show the predictions on the test data
value_predictions = model.predict(test_bitcoin[features])

accuracy_value = model.score(test_bitcoin[features], test_bitcoin[target])
print(accuracy_value)

plt.plot(bitcoin['close'], label = 'Bitcoin Price')
plt.plot(test_bitcoin['close'].index, value_predictions, label  = "Predictions")
plt.legend()
plt.show()