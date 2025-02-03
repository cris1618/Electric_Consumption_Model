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

# Import the synthetic data
df = pd.read_csv("SyntethicData/synthetic_energy_data.csv")
print(df.head())