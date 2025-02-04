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