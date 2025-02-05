import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Define the api
app = FastAPI()

# Import the LSTM from training file
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

# Import trained model (weights)
input_dim = 8 
hidden_dim = 50
output_dim = 1

model = LSTMModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("Trained_Models/synthetic_data_energy_first_test.pth"))
model.eval()  # Set to evaluation mode
print("Model loaded")

# Import the data
df = pd.read_csv("SyntethicData/synthetic_energy_data.csv")
scaler = MinMaxScaler()
# Preprocess the data for the model
df = pd.get_dummies(df, columns=["Season", "Heating Type", "Cooling Type"], drop_first=True)
df.dropna(inplace=True)

# Split features and targets
X = df.drop("KWh Consumption", axis=1)
X_scaled = scaler.fit_transform(X)

# Define API request structure
class EnergyInput(BaseModel):
    season: str
    size: int
    occupants: int
    heating_type: str
    cooling_type: str

with open("feature_column.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Prediction endpoint
@app.post("/predict")
def predict_energy(data: EnergyInput):
    print("Received Request:", data)
    input_df = pd.DataFrame([{
        "Season": data.season,
        "Size": data.size,
        "Occupants": data.occupants,
        "Heating Type": data.heating_type,
        "Cooling Type": data.cooling_type
    }])

    # Apply One-Hot Encoding (get_dummies)
    input_df = pd.get_dummies(input_df, columns=["Season", "Heating Type", "Cooling Type"], drop_first=False)

    # Ensure all expected columns exist 
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Fill missing columns with 0

    # Reorder columns to match training order
    input_df = input_df[feature_columns]

    # Normalize using the same scaler
    input_data = scaler.transform(input_df)

    # Convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Make Prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return {"KWh Consumption": prediction}


