from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Load pre-trained objects ---
scaler = joblib.load("data/lstm_scaler.pkl")
label_encoder = joblib.load("data/lstm_label_encoder.pkl")
model = tf.keras.models.load_model("data/lstm_model.h5")

# The features in the same order as your training!
FEATURES = [
    "close", "volume", "RSI_14", "DMA_20", "DMA_50", "DMA_100", 
    "SUPPORT_20", "RESIST_20", "PE", "PB"
]
SEQUENCE_LENGTH = 20

class StockRequest(BaseModel):
    stock_name: str
    features: list  # list of dicts, length=20

@app.post("/predict")
def predict_signal(req: StockRequest):
    if len(req.features) != SEQUENCE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Must provide {SEQUENCE_LENGTH} days of features.")
    try:
        X = np.array([[float(day[feat]) for feat in FEATURES] for day in req.features])
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape((1, SEQUENCE_LENGTH, len(FEATURES)))
        pred = model.predict(X_scaled)
        pred_label_idx = np.argmax(pred, axis=1)[0]
        pred_label = label_encoder.inverse_transform([pred_label_idx])[0]

        # Optionally log
        print(f"Predicted for {req.stock_name}: {pred_label}")

        return {
            "stock_name": req.stock_name,
            "prediction": pred_label,
            "confidence": float(np.max(pred))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
