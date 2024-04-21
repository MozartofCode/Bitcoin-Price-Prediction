# @ Author: Bertan Berker
# @ Language: Python
# @ File: prediction.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Getting bitcoin dataset
bitcoinData = pd.read_csv('bitcoin.csv')
X = bitcoinData.drop(columns= ['Close', 'SNo', 'Name', 'Symbol', 'Date'])
y = bitcoinData['Close']


X = X[-120:]
y = y[-120:]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features using Min-Max Scaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert y_train and y_test to NumPy arrays
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network model
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = X_train.shape[1]
hidden_size = 6400
model = StockPredictor(input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.5)

# Train the model
epochs = 100
train_losses = []
test_losses = []
test_accuracy = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()

    with torch.no_grad():
        y_test_pred = model(X_test_tensor)
        test_loss = criterion(y_test_pred.squeeze(), y_test_tensor)
        test_losses.append(test_loss.item())


    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
 

# Plot training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# Plot predictions vs. actual values
# Plot predictions vs. actual values
plt.plot(range(1, len(y_test) + 1), y_test, label='Actual')  # Use the length of y_test to determine the x-axis ticks
plt.plot(range(1, len(predictions) + 1), predictions, label='Predicted')  # Use the length of predictions to determine the x-axis ticks
plt.xlabel('Time (Days)')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()
