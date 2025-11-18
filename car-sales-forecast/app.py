from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path


app = FastAPI()

# Load pretrained XGBoost model (Booster)
model = xgb.Booster()
model.load_model("model/model.json")

# HTML file path
BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = BASE_DIR / "frontend" / "index.html"

@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        content = HTML_FILE.read_text()
        return HTMLResponse(content=content)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading HTML</h1><p>{e}</p>")

@app.post("/predict")
def predict(
    lag1: float = Form(...),
    lag2: float = Form(...),
    lag3: float = Form(...),
    lag4: float = Form(...),
    lag5: float = Form(...),
    lag6: float = Form(...)
):
    # Build DataFrame just like training
    df = pd.DataFrame([{
        "lag_1": lag1,
        "lag_2": lag2,
        "lag_3": lag3,
        "lag_4": lag4,
        "lag_5": lag5,
        "lag_6": lag6
    }])

    dmatrix = xgb.DMatrix(df)
    prediction = model.predict(dmatrix)[0]

    return {"prediction": float(prediction)}