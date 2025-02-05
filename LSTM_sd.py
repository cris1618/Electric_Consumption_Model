import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import seaborn as sns
import torch 
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from functions import calculate_accuracy, calculate_nrmse, create_sequences
import pickle

# Import the synthetic data
df = pd.read_csv("SyntethicData/synthetic_energy_data.csv")

# Preprocess the data for the model
df = pd.get_dummies(df, columns=["Season", "Heating Type", "Cooling Type"], drop_first=True)

# Create features to account for the sequentiality of the data 
#df["Consumption_lag1"] = df["KWh Consumption"].shift(1)
#df["Consumption_rolling_mean"] = df["KWh Consumption"].rolling(window=3).mean()
df.dropna(inplace=True)

# Split features and targets
X = df.drop("KWh Consumption", axis=1)
feature_cols = X.columns.tolist()

# Save to use them in the API
with open("feature_column.pkl", "wb") as  f:
    pickle.dump(feature_cols, f)

y = df["KWh Consumption"]

# Normalize features 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for LSTM training
sequence_length = 10
X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        lstm_out = self.dropout(lstm_out)
        predictions = self.fc(lstm_out[:,-1,:])
        return predictions
    
# Parameters
input_dim = X_seq.shape[2]
hidden_dim = 50
output_dim = 1

model = LSTMModel(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).view(-1,1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=1618)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Training loop 
epochs = 1000
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

# Save the model
torch.save(model.state_dict(), "Trained_Models/synthetic_data_energy_first_test.pth")
print("Model saved")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test).numpy()
    y_pred_train = model(X_train).numpy()


mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
nrmse_train = calculate_nrmse(y_train, y_pred_train)
accuracy_train = calculate_accuracy(nrmse_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
nrmse_test = calculate_nrmse(y_test, y_pred_test)
accuracy_test = calculate_accuracy(nrmse_test)

print(f"Training Set - MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, R^2: {r2_train:.2f}, Accuracy: {accuracy_train:.2f}%")
print(f"Testing Set - MAE: {mae_test:.2f}, MSE: {mse_test:.2f}, R^2: {r2_test:.2f}, Accuracy: {accuracy_test:.2f}%")