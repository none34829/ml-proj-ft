from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'categorical_features': [
        'Origin', 'Destination', 'Vehicle Type',
        'Weather Conditions', 'Traffic Conditions'
    ],
    'numerical_features': ['Distance (km)'],
    'date_features': [
        'Shipment Date', 'Planned Delivery Date'
    ]
}

# API configuration
API_CONFIG = {
    'title': 'Shipment Delay Prediction API',
    'description': '''
    An advanced ML-powered API for predicting shipment delays in logistics.
    
    Features:
    - Multiple ML models (Random Forest, XGBoost, LightGBM)
    - Sophisticated feature engineering
    - Comprehensive error handling
    - Model performance monitoring
    ''',
    'version': '2.0.0',
    'docs_url': '/docs',
    'redoc_url': '/redoc'
}
