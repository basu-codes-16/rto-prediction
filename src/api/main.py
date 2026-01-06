from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Initialize FastAPI
app = FastAPI(
    title="RTO Prediction API",
    description="Predict Return-to-Origin probability for e-commerce orders",
    version="1.0.0"
)

# Load model artifacts
model = joblib.load('models/production/model.pkl')
scaler = joblib.load('models/production/scaler.pkl')
label_encoders = joblib.load('models/production/label_encoders.pkl')

# Request schema
class OrderFeatures(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    day_of_month: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)
    is_month_start: int = Field(..., ge=0, le=1)
    is_month_end: int = Field(..., ge=0, le=1)
    is_metro: int = Field(..., ge=0, le=1)
    pin_rto_rate: float = Field(..., ge=0, le=1)
    state_rto_rate: float = Field(..., ge=0, le=1)
    pin_order_count: int = Field(..., ge=0)
    state_order_count: int = Field(..., ge=0)
    quantity: int = Field(..., ge=1)
    meesho_price: float = Field(..., ge=0)
    final_price: float = Field(..., ge=0)
    shipping_charges_total: float = Field(..., ge=0)
    price_per_unit: float = Field(..., ge=0)
    discount_amount: float = Field(..., ge=0)
    discount_pct: float = Field(..., ge=0, le=100)
    shipping_to_price_ratio: float = Field(..., ge=0)
    has_valid_pincode: int = Field(..., ge=0, le=1)
    has_state: int = Field(..., ge=0, le=1)
    address_quality_score: float = Field(..., ge=0, le=1)
    product_rto_rate: float = Field(..., ge=0, le=1)
    product_order_count: int = Field(..., ge=0)
    product_length: int = Field(..., ge=0)
    price_category: str = Field(..., description="low, medium, high, or premium")
    state_clean: str = Field(..., description="State name")

    class Config:
        json_schema_extra = {
            "example": {
                "day_of_week": 2,
                "day_of_month": 15,
                "month": 8,
                "is_weekend": 0,
                "is_month_start": 0,
                "is_month_end": 0,
                "is_metro": 1,
                "pin_rto_rate": 0.25,
                "state_rto_rate": 0.35,
                "pin_order_count": 50,
                "state_order_count": 200,
                "quantity": 1,
                "meesho_price": 1200,
                "final_price": 1200,
                "shipping_charges_total": 50,
                "price_per_unit": 1200,
                "discount_amount": 0,
                "discount_pct": 0,
                "shipping_to_price_ratio": 0.042,
                "has_valid_pincode": 1,
                "has_state": 1,
                "address_quality_score": 1.0,
                "product_rto_rate": 0.3,
                "product_order_count": 10,
                "product_length": 50,
                "price_category": "medium",
                "state_clean": "Maharashtra"
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    rto_probability: float
    rto_prediction: int
    risk_level: str
    confidence: float

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "online",
        "model": "RTO Prediction v1.0",
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_rto(features: OrderFeatures):
    """
    Predict RTO probability for an order.
    
    Returns:
    - rto_probability: Probability of RTO (0-1)
    - rto_prediction: Binary prediction (0=Delivered, 1=RTO)
    - risk_level: Low, Medium, or High
    - confidence: Model confidence (0-1)
    """
    try:
        # Encode categorical features
        price_cat_encoded = label_encoders['price_category'].transform([features.price_category])[0]
        state_encoded = label_encoders['state_clean'].transform([features.state_clean])[0]
        
        # Create feature array
        feature_array = np.array([[
            features.day_of_week,
            features.day_of_month,
            features.month,
            features.is_weekend,
            features.is_month_start,
            features.is_month_end,
            features.is_metro,
            features.pin_rto_rate,
            features.state_rto_rate,
            features.pin_order_count,
            features.state_order_count,
            features.quantity,
            features.meesho_price,
            features.final_price,
            features.shipping_charges_total,
            features.price_per_unit,
            features.discount_amount,
            features.discount_pct,
            features.shipping_to_price_ratio,
            features.has_valid_pincode,
            features.has_state,
            features.address_quality_score,
            features.product_rto_rate,
            features.product_order_count,
            features.product_length,
            price_cat_encoded,
            state_encoded
        ]])
        
        # Scale features
        feature_array_scaled = scaler.transform(feature_array)
        
        # Predict
        prediction = model.predict(feature_array_scaled)[0]
        probability = model.predict_proba(feature_array_scaled)[0][1]
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(probability - 0.5) * 2
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            rto_probability=float(probability),
            rto_prediction=int(prediction),
            risk_level=risk_level,
            confidence=float(confidence)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(orders: list[OrderFeatures]):
    """Batch prediction endpoint."""
    results = []
    for order in orders:
        try:
            result = predict_rto(order)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results, "total": len(orders)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)