from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Dict, Optional, List
import pandas as pd
import joblib
import logging
from datetime import datetime
import json
from pathlib import Path

from config import API_CONFIG, MODELS_DIR, LOGS_DIR
from features import CustomFeatureTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ShipmentInput(BaseModel):
    origin: str
    destination: str
    vehicle_type: str
    distance: float
    weather_conditions: str
    traffic_conditions: str
    shipment_date: str
    planned_delivery_date: str

    @validator('distance')
    def validate_distance(cls, v):
        if v <= 0:
            raise ValueError("Distance must be positive")
        return v

    @validator('shipment_date', 'planned_delivery_date')
    def validate_dates(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")

class ModelResponse(BaseModel):
    prediction: Dict[str, bool]
    probability: Dict[str, float]
    feature_importance: Dict[str, List[Dict[str, float]]]
    model_confidence: float
    explanation: str

app = FastAPI(
    title=API_CONFIG['title'],
    description=API_CONFIG['description'],
    version=API_CONFIG['version'],
    docs_url=API_CONFIG['docs_url'],
    redoc_url=API_CONFIG['redoc_url']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and preprocessor
try:
    feature_transformer = joblib.load(MODELS_DIR / 'feature_transformer.joblib')
    models = {
        name: joblib.load(MODELS_DIR / f'{name}_model.joblib')
        for name in ['random_forest', 'xgboost', 'lightgbm']
    }
    feature_importances = {
        name: pd.read_csv(MODELS_DIR / f'{name}_feature_importance.csv')
        for name in ['random_forest', 'xgboost', 'lightgbm']
    }
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise RuntimeError("Model loading failed")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests"""
    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"Path: {request.url.path} "
        f"Duration: {duration:.3f}s "
        f"Status: {response.status_code}"
    )
    
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

def get_model_confidence(probabilities: Dict[str, float]) -> float:
    """Calculate model confidence based on prediction agreement"""
    values = list(probabilities.values())
    return 1 - (max(values) - min(values))

def generate_explanation(
    predictions: Dict[str, bool],
    probabilities: Dict[str, float],
    feature_importance: Dict[str, List[Dict[str, float]]]
) -> str:
    """Generate human-readable explanation of the prediction"""
    # Get majority prediction
    delayed = sum(predictions.values()) > len(predictions) / 2
    
    # Get average probability
    avg_prob = sum(probabilities.values()) / len(probabilities)
    
    # Get top features
    important_features = []
    for model, importance in feature_importance.items():
        if importance:
            important_features.extend(importance[:2])
    
    explanation = (
        f"The shipment is predicted to be {'delayed' if delayed else 'on time'} "
        f"with {avg_prob:.1%} probability. "
        f"This prediction is based primarily on "
        f"the following factors: {', '.join(f['feature'] for f in important_features[:3])}"
    )
    
    return explanation

@app.post("/predict", response_model=ModelResponse)
async def predict_delay(shipment: ShipmentInput):
    """
    Predict shipment delay using multiple ML models
    
    This endpoint:
    1. Processes input features
    2. Makes predictions using multiple models
    3. Provides feature importance and explanation
    4. Calculates model confidence
    """
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([shipment.dict()])
        
        # Transform features
        X = feature_transformer.transform(input_df)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        importance = {}
        
        for name, model in models.items():
            predictions[name] = bool(model.predict(X)[0])
            probabilities[name] = float(model.predict_proba(X)[0][1])
            
            # Get feature importance for this prediction
            if name in feature_importances:
                importance[name] = feature_importances[name].to_dict('records')
        
        # Calculate model confidence
        confidence = get_model_confidence(probabilities)
        
        # Generate explanation
        explanation = generate_explanation(predictions, probabilities, importance)
        
        # Log prediction
        logger.info(
            f"Prediction made: "
            f"delayed={predictions} "
            f"confidence={confidence:.3f}"
        )
        
        return ModelResponse(
            prediction=predictions,
            probability=probabilities,
            feature_importance=importance,
            model_confidence=confidence,
            explanation=explanation
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/metadata")
async def model_metadata():
    """Get metadata about trained models"""
    return {
        name: {
            "feature_importance": importance.to_dict('records')[:5]
        }
        for name, importance in feature_importances.items()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
