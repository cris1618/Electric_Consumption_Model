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
from functions import calculate_accuracy, calculate_nrmse

# Import the dataframe
df = pd.read_csv("appliances+energy+prediction/energydata_complete.csv")

# Convert to datetime
df["date"] = pd.to_datetime(df["date"])

# Extract time-based features 
df["hour"] = df["date"].dt.hour
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# Get specific time of the day
def time_of_day(hour):
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"

df["time_of_day"] = df["hour"].apply(time_of_day)
df = pd.get_dummies(df, columns=["time_of_day"], drop_first=True)

# Create features to account for the sequentiality of the data (avoid time-series)
df["Appliances_lag1"] = df["Appliances"].shift(1)
df["Appliances_rolling_mean"] = df["Appliances"].rolling(window=3).mean()

# Since Appliencies is highly skewed, apply log trasformation
#df["Appliances_log"] = np.log1p(df["Appliances"])

# Drop missing values introduced by lag/rolling features
df.dropna(inplace=True)

# Split the data
X = df.drop(["Appliances", "date", "T9", "T6", "rv1", "rv2", "Windspeed"], axis=1) # drop "date" or not
y = df["Appliances"]

# Apply log transformation to target
#y_log = np.log1p(y)  # Stabilizes variance

# Normalize features for the LSTM
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Normalize targets
target_scaler = MinMaxScaler()
#y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Reeshape data for LSTM
def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

for i in range(len(X_seq)):
    assert len(X_seq[i]) == sequence_length, f"Sequence {i} does not have the correct length!"
assert len(X_seq) == len(y_seq)

# Create the model
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

assert X_tensor.min() >= 0 and X_tensor.max() <= 1, "Features are not normalized between 0 and 1!"
#assert y_tensor.min() >= 0 and y_tensor.max() <= 1, "Targets are not normalized between 0 and 1!"

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=1618)
assert not set(X_train.flatten()).intersection(set(X_test.flatten())), "Error, X_train and X_test Intersect"
assert not set(y_train.flatten()).intersection(set(y_test.flatten()))

assert len(X_train) + len(X_test) == len(X_tensor), "Train-test split is inconsistent with the original data!"
assert len(y_train) + len(y_test) == len(y_tensor)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Training Loop
epochs = 500
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

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test).numpy()
    y_pred_train = model(X_train).numpy()

# Reverse normalization
#y_test_pred_normalized = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
#y_test_original_normalized = target_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

# Reverse log transformation
#y_test_pred_original = np.expm1(y_test_pred_normalized)
#y_test_original = np.expm1(y_test_original_normalized)


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

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
residuals = y_test - y_pred_test
ax[0].scatter(y_test, residuals, alpha=0.5)
ax[0].axhline(0, color='red', linestyle='--')
ax[0].set_xlabel("Actual Values")
ax[0].set_ylabel("Residuals")
ax[0].set_title("Residual Plot")

time_index = range(len(y_test[:50]))
ax[1].plot(time_index, y_test[:50], label="Actual Value", color="red", linestyle="-", marker="o")
ax[1].plot(time_index, y_pred_test[:50], label="Predicted Value", color="blue", linestyle="--", marker="x")
ax[1].set_xlabel("Time (h)")
ax[1].set_ylabel("Appliances Energy Consumption (Wh)")
ax[1].set_title("Prediction Results of Household Appliance Energy Consumption")
plt.legend()
plt.grid(0.5)
plt.show()