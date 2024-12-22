import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
import mlflow
import shap
import joblib
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path

from config import MODEL_CONFIG, MODELS_DIR, LOGS_DIR, FEATURE_CONFIG
from features import CustomFeatureTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create required directories
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

class ShipmentDelayPredictor:
    """Advanced ML model for shipment delay prediction"""
    
    def __init__(self):
        self.feature_transformer = CustomFeatureTransformer()
        self.models = {}
        self.feature_importance = {}
        self.shap_values = {}
    
    def train(self, data: pd.DataFrame, target_col: str = 'Delayed'):
        """Train multiple models with sophisticated techniques"""
        logger.info("Starting model training pipeline")
        
        # Prepare target
        y = (data[target_col] == 'Yes').astype(int)
        
        # Transform features
        logger.info("Performing feature engineering")
        X = self.feature_transformer.fit_transform(data)
        
        # Initialize MLflow
        mlflow.set_experiment("Shipment Delay Prediction")
        
        # Store best model and its metrics
        best_model_name = None
        best_f1_score = 0
        
        # Train multiple models
        for model_name in MODEL_CONFIG.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'='*50}")
            
            with mlflow.start_run(run_name=model_name):
                # Optimize hyperparameters
                best_params = self._optimize_hyperparameters(model_name, X, y)
                logger.info(f"Best parameters for {model_name}:")
                for param, value in best_params.items():
                    logger.info(f"{param}: {value}")
                
                # Train model with best parameters
                model = self._get_model(model_name, best_params)
                
                # Perform cross-validation
                cv_results = self._cross_validate(model, X, y)
                
                # Log detailed metrics
                logger.info(f"\nPerformance Metrics for {model_name}:")
                logger.info("-" * 30)
                for metric, value in cv_results.items():
                    logger.info(f"{metric}: {value:.4f}")
                logger.info("-" * 30)
                
                # Track best model
                if cv_results['f1'] > best_f1_score:
                    best_f1_score = cv_results['f1']
                    best_model_name = model_name
                
                # Log metrics to MLflow
                for metric, value in cv_results.items():
                    mlflow.log_metric(f"cv_{metric}", value)
                
                # Train final model on full dataset
                model.fit(X, y)
                self.models[model_name] = model
                
                # Calculate and log feature importance
                self.feature_importance[model_name] = self._get_feature_importance(model, model_name)
                if self.feature_importance[model_name] is not None:
                    logger.info(f"\nTop 5 Important Features for {model_name}:")
                    logger.info(self.feature_importance[model_name].head().to_string())
                
                # Calculate SHAP values
                self.shap_values[model_name] = self._calculate_shap_values(model, X, model_name)
                
                # Save model
                self._save_model(model, model_name)
                
                # Log model to MLflow
                mlflow.sklearn.log_model(model, model_name)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Model training completed")
        logger.info(f"Best performing model: {best_model_name} (F1 Score: {best_f1_score:.4f})")
        logger.info(f"{'='*50}")
    
    def _optimize_hyperparameters(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna"""
        logger.info(f"Optimizing hyperparameters for {model_name}")
        
        def objective(trial):
            params = self._get_trial_params(trial, model_name)
            model = self._get_model(model_name, params)
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, X, y,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1
            )
            
            return cv_results['test_score'].mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def _get_trial_params(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter search space for each model"""
        if model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
        elif model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
        else:  # lightgbm
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
    
    def _get_model(self, model_name: str, params: Dict[str, Any]):
        """Get model instance with specified parameters"""
        if model_name == 'random_forest':
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
        else:  # lightgbm
            return lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)
    
    def _cross_validate(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation with multiple metrics"""
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }
        
        cv_results = cross_validate(
            model, X, y,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring=scoring,
            n_jobs=-1
        )
        
        # Convert numpy values to Python floats for MLflow compatibility
        return {
            metric: float(cv_results[f'test_{metric}'].mean())
            for metric in scoring.keys()
        }
    
    def _get_feature_importance(self, model, model_name: str) -> pd.DataFrame:
        """Get feature importance for the model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            return None
        
        feature_names = (
            [f'{col}_encoded' for col in FEATURE_CONFIG['categorical_features']] +
            self.feature_transformer._get_numerical_features(pd.DataFrame())
        )
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def _calculate_shap_values(self, model, X: np.ndarray, model_name: str):
        """Calculate SHAP values for model interpretability"""
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X)
    
    def _save_model(self, model, model_name: str):
        """Save model and associated artifacts"""
        # Save model
        model_path = MODELS_DIR / f'{model_name}_model.joblib'
        joblib.dump(model, model_path)
        
        # Save feature importance
        if self.feature_importance[model_name] is not None:
            importance_path = MODELS_DIR / f'{model_name}_feature_importance.csv'
            self.feature_importance[model_name].to_csv(importance_path, index=False)
        
        # Save SHAP values
        shap_path = MODELS_DIR / f'{model_name}_shap_values.joblib'
        joblib.dump(self.shap_values[model_name], shap_path)
    
    def predict(self, data: pd.DataFrame) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Make predictions using all trained models"""
        X = self.feature_transformer.transform(data)
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)[0]
            probabilities[model_name] = model.predict_proba(X)[0][1]
        
        return predictions, probabilities

if __name__ == "__main__":
    # Load data
    data = pd.read_excel('AI ML Internship Training Data.xlsx')
    
    # Train models
    predictor = ShipmentDelayPredictor()
    predictor.train(data)
    
    # Save feature transformer
    predictor.feature_transformer.save('feature_transformer.joblib')
