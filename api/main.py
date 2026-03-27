
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import numpy as np
import os

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="StellarNet API")

# -------------------------------
# Model Definition
# -------------------------------

class TunedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, 5)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, 5)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, 3)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = torch.mean(x, dim=2)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# -------------------------------
# Load Model (ROBUST PATH FIX)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # api/
ROOT_DIR = os.path.dirname(BASE_DIR)                        # project root
MODEL_PATH = os.path.join(ROOT_DIR, "notebooks", "model.pth")

print("Loading model from:", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = TunedCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -------------------------------
# Input Schema
# -------------------------------
class InputData(BaseModel):
    data: List[float]

# -------------------------------
# Normalization (same as training)
# -------------------------------
def normalize_curve(curve):
    curve = np.array(curve, dtype=np.float32)
    mean = curve.mean()
    std = curve.std() + 1e-8
    return (curve - mean) / std

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def home():
    return {"message": "StellarNet API is running"}

@app.post("/predict")
def predict(input: InputData):
    arr = np.array(input.data)

    # Validate input length
    if len(arr) != 3197:
        return {"error": "Input must contain exactly 3197 values"}

    # Check for invalid values
    if not np.isfinite(arr).all():
        return {"error": "Input contains NaN or Inf values"}

    # Normalize
    arr = normalize_curve(arr)

    # Convert to tensor
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    # Apply threshold
    threshold = 0.7
    prediction = "planet" if prob > threshold else "no planet"

    return {
        "probability": round(prob, 4),
        "prediction": prediction
    }