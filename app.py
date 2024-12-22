from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict, List

app = FastAPI(
    title="Shipment Delay Prediction API",
    description="Predict if a shipment will be delayed based on various factors"
)

# Load the trained model and preprocessors
try:
    model = joblib.load('random_forest_model.joblib')
    encoders = joblib.load('encoders.joblib')
    scaler = joblib.load('scaler.joblib')
except:
    raise Exception("Model files not found. Please run train_model.py first.")

# Get valid values for each categorical feature
valid_values = {
    'Origin': list(encoders['Origin'].classes_),
    'Destination': list(encoders['Destination'].classes_),
    'Vehicle Type': list(encoders['Vehicle Type'].classes_),
    'Weather Conditions': list(encoders['Weather Conditions'].classes_),
    'Traffic Conditions': list(encoders['Traffic Conditions'].classes_)
}

class ShipmentInput(BaseModel):
    origin: str
    destination: str
    vehicle_type: str
    distance: float
    weather_conditions: str
    traffic_conditions: str

@app.post("/predict")
async def predict_delay(shipment: ShipmentInput):
    """
    Predict shipment delay with the following features:
    - origin: Must be one of {valid_values['Origin']}
    - destination: Must be one of {valid_values['Destination']}
    - vehicle_type: Must be one of {valid_values['Vehicle Type']}
    - distance: Distance in kilometers (float)
    - weather_conditions: Must be one of {valid_values['Weather Conditions']}
    - traffic_conditions: Must be one of {valid_values['Traffic Conditions']}
    """
    try:
        # Encode categorical variables
        encoded_features = []
        for feature, value in {
            'Origin': shipment.origin,
            'Destination': shipment.destination,
            'Vehicle Type': shipment.vehicle_type,
            'Weather Conditions': shipment.weather_conditions,
            'Traffic Conditions': shipment.traffic_conditions
        }.items():
            encoder = encoders[feature]
            try:
                encoded_value = encoder.transform([value])[0]
                encoded_features.append(encoded_value)
            except:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid value for {feature}. Valid values are: {', '.join(encoder.classes_)}"
                )
        
        # Add numerical features
        features = encoded_features + [shipment.distance]
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return {
            "delayed": bool(prediction),
            "probability": float(probability),
            "prediction_text": "Delayed" if prediction else "On Time"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/valid-values")
async def get_valid_values():
    """Get valid values for all categorical features"""
    return valid_values

@app.get("/")
async def root():
    return {
        "message": "Shipment Delay Prediction API",
        "usage": "Send POST request to /predict with shipment details",
        "docs": "Visit /docs for interactive API documentation",
        "valid_values": "Visit /valid-values to see valid values for categorical features"
    }
