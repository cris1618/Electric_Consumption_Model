import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
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
from sklearn.pipeline import Pipeline

# Import the dataframe
df = pd.read_csv("appliances+energy+prediction/energydata_complete.csv")

# Convert to datetime
df["date"] = pd.to_datetime(df["date"])

# Extract time-based features 
#df["hour"] = df["date"].dt.hour
#df["day_of_week"] = df["date"].dt.dayofweek
#df["month"] = df["date"].dt.month

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

#df["time_of_day"] = df["hour"].apply(time_of_day)
#df = pd.get_dummies(df, columns=["time_of_day"], drop_first=True)

# Create features to account for the sequentiality of the data (avoid time-series)
#df["Appliances_lag1"] = df["Appliances"].shift(1)
#df["Appliances_rolling_mean"] = df["Appliances"].rolling(window=3).mean()

# Since Appliencies is highly skewed, apply log trasformation
#df["Appliances_log"] = np.log1p(df["Appliances"])

# Drop missing values introduced by lag/rolling features
df.dropna(inplace=True)

# Split the data
X = df.drop(["Appliances", "date", "T9", "T6", "rv1", "rv2", "Windspeed"], axis=1)
y = df["Appliances"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Pipelines
"""pipelines = {
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # Retain 95% variance
        ('model', RandomForestRegressor(random_state=1618))
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('model', SVR(kernel='rbf'))
    ]),
    "Linear Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('model', LinearRegression())
    ]),
    "XGBoost": Pipeline([
        ('scaler', StandardScaler()),  # Only scaling
        ('model', XGBRegressor(random_state=1618))
    ])
}

# Define parameter grids
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1]
    },
    "Linear Regression": {},  # No hyperparameters
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}"""

"""correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

sns.histplot(df["Appliances"], kde=True)
plt.title("Distribution of Appliances Energy Consumption")
plt.show()"""

# Perform Nested cross-validation
# Outer loop for testing different models
"""outer_cv = KFold(n_splits=5, shuffle=True, random_state=1618)
cv_results = {}

for model_name, pipeline in pipelines.items():
    print(f"Evaluating {model_name}...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        cv=3,
        scoring="neg_mean_absolute_error"
        )
    
    scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring="neg_mean_absolute_error")
    cv_results[model_name] = {"MAE": -np.mean(scores), "STD": np.std(scores)}

# Results
print("Nested CV Results:")
for model_name, results in cv_results.items():
    print(f"{model_name}: MAE = {results['MAE']:.2f} ± {results['STD']:.4f}")"""

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1618)

"""param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=1618),
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",  # Use MAE as the scoring metric
    cv=5,  # 5-fold cross-validation
    verbose=1,  # To show progress
    n_jobs=-1  # Use all available cores
)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)"""

# Parameters taken from the paper
model = XGBRegressor(colsample_bytree=1.0, learning_rate=0.2, max_depth=7, n_estimators=200, subsample=1.0)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training Set - MAE: {mae_train:.2f}, MSE: {mse_train:.2f}, R^2: {r2_train:.2f}")
print(f"Testing Set - MAE: {mae_test:.2f}, MSE: {mse_test:.2f}, R^2: {r2_test:.2f}")

"""fig, ax = plt.subplots(1, 2, figsize=(10,8))
# Transformed scale
ax[0].scatter(y_test, y_pred, alpha=0.5)
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax[0].set_xlabel("Actual (Log Scale)")
ax[0].set_ylabel("Predicted (Log Scale)")
ax[0].set_title("Predicted vs Actual (Log Scale)")

# Original scale
ax[1].scatter(y_test_original, y_pred_original, alpha=0.5)
ax[1].plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
ax[1].set_xlabel("Actual (Original Scale)")
ax[1].set_ylabel("Predicted (Original Scale)")
ax[1].set_title("Predicted vs Actual (Original Scale)")
plt.show()"""


"""Nested CV Results:
Random Forest: MAE = 20.82 ± 0.9219
SVM: MAE = 22.67 ± 0.8022
XGBoost: MAE = 20.56 ± 0.9219
Linear Regression: MAE = 22.75 ± 0.9125"""
