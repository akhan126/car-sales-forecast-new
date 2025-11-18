import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import os

# Load dataset
data_path = os.path.join("data", "monthly-car-sales.csv")
df = pd.read_csv(data_path)

# Sort by Month (string YYYY-MM) correctly
df = df.sort_values("Month")

# Rename column for convenience
df.rename(columns={"Sales": "sales"}, inplace=True)

# Create lag features
for lag in range(1, 7):
    df[f"lag_{lag}"] = df["sales"].shift(lag)

# Remove rows with NaNs
df = df.dropna().reset_index(drop=True)

# Features and target
X = df[[f"lag_{i}" for i in range(1, 7)]]
y = df["sales"]

# Train XGBRegressor
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X, y)

# Save to JSON as required
os.makedirs("model", exist_ok=True)
model.save_model("model/model.json")

print("Model trained successfully and saved to model/model.json")
