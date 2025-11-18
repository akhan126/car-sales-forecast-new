from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from xgboost import XGBRegressor

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model", "model.json")
model = XGBRegressor()
model.load_model(model_path)

app = FastAPI()

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SalesInput(BaseModel):
    sales: list[float]   # exactly 6 values

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join("frontend", "index.html"))

@app.post("/predict")
async def predict_sales(input: SalesInput):
    if len(input.sales) != 6:
        return {"error": "Please provide exactly 6 values"}

    arr = np.array(input.sales).reshape(1, -1)
    pred = model.predict(arr)[0]

    return {"prediction": float(pred)}
