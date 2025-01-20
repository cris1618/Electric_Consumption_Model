import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
X = df.drop(["Appliances", "date"], axis=1)
y = df["Appliances"]

# Standardize and divide data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(1,1,figsize=(10,8))
ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker="o")
ax.set_title("Cumulative Explained Variance")
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.grid(0.5)
#plt.show()

# Retain 95% of variance
n_components = sum(pca.explained_variance_ratio_.cumsum() <= 0.95) + 1
print(f"Number of Components explaining 95% variance: {n_components}")

# Refit with right number of components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Make data range from 0 to 1 for NN training
min_max_scaler = MinMaxScaler(feature_range=(0,1))
X_tr = min_max_scaler.fit_transform(X_pca)

X_tensor = torch.tensor(X_tr, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=1618)

# Create Dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model
class EnergyNN(nn.Module):
    def __init__(self, input_size):
        super(EnergyNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1)
        )
    
    def forward(self, x):
        return self.model(x)

input_size = X_train.shape[1]
model = EnergyNN(input_size)

# Loss and optimizer
criterion = nn.MSELoss()
#criterion = nn.SmoothL1Loss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # Halve the LR every 50 epochs

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    scheduler.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluate
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()

            predictions.append(pred.numpy())
            actuals.append(y_batch.numpy())
    
    predictions = torch.cat([torch.tensor(p) for p in predictions]).numpy()
    actuals = torch.cat([torch.tensor(a) for a in actuals]).numpy()

    return total_loss / len(test_loader), predictions, actuals

# compute test loss
test_loss, y_pred, y_actual = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: (MSE): {test_loss:.4f}")