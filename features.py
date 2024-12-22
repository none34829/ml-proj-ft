import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from typing import List, Dict, Union
from config import FEATURE_CONFIG, MODELS_DIR

class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    """Advanced feature transformer with sophisticated engineering techniques"""
    
    def __init__(self):
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_stats: Dict[str, Dict] = {}
    
    def _engineer_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated distance-based features"""
        # Log transform distance for better distribution
        df['distance_log'] = np.log1p(df['Distance (km)'])
        
        # Distance buckets based on quantiles
        df['distance_bucket'] = pd.qcut(df['Distance (km)'], q=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        
        # Calculate z-score of distance
        df['distance_zscore'] = (df['Distance (km)'] - df['Distance (km)'].mean()) / df['Distance (km)'].std()
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract sophisticated temporal features"""
        for date_col in FEATURE_CONFIG['date_features']:
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Extract basic temporal features
            df[f'{date_col}_month'] = df[date_col].dt.month
            df[f'{date_col}_day'] = df[date_col].dt.day
            df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
            df[f'{date_col}_quarter'] = df[date_col].dt.quarter
            
            # Is month end/start
            df[f'{date_col}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)
            df[f'{date_col}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
            
            # Is weekend
            df[f'{date_col}_is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Calculate planned transit time
        df['planned_transit_days'] = (
            pd.to_datetime(df['Planned Delivery Date']) - 
            pd.to_datetime(df['Shipment Date'])
        ).dt.total_seconds() / (24 * 60 * 60)
        
        return df
    
    def _engineer_categorical_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between categorical variables"""
        # Origin-Destination pair frequency encoding
        od_freq = df.groupby(['Origin', 'Destination']).size() / len(df)
        df['route_frequency'] = df.apply(lambda x: od_freq.get((x['Origin'], x['Destination']), 0), axis=1)
        
        # Weather-Traffic interaction
        df['weather_traffic_interaction'] = df['Weather Conditions'] + '_' + df['Traffic Conditions']
        
        # Vehicle type per route
        vehicle_route_freq = df.groupby(['Origin', 'Destination', 'Vehicle Type']).size() / len(df)
        df['vehicle_route_frequency'] = df.apply(
            lambda x: vehicle_route_freq.get((x['Origin'], x['Destination'], x['Vehicle Type']), 0), 
            axis=1
        )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sophisticated missing value handling"""
        # Store missing value statistics
        self.feature_stats['missing'] = df.isnull().sum().to_dict()
        
        # For categorical features, use mode with frequency encoding
        for cat_feature in FEATURE_CONFIG['categorical_features']:
            if df[cat_feature].isnull().any():
                # Calculate value frequencies
                freq = df[cat_feature].value_counts(normalize=True)
                # Add small random noise to prevent identical imputed values
                df[cat_feature] = df[cat_feature].fillna(
                    df[cat_feature].mode()[0]
                )
                df[f'{cat_feature}_was_missing'] = df[cat_feature].isnull().astype(int)
        
        return df
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature transformer"""
        df = X.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Store feature statistics
        for feature in FEATURE_CONFIG['numerical_features']:
            self.feature_stats[feature] = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max()
            }
        
        # Fit label encoders for categorical features
        for cat_feature in FEATURE_CONFIG['categorical_features']:
            self.encoders[cat_feature] = LabelEncoder()
            self.encoders[cat_feature].fit(df[cat_feature].astype(str))
        
        # Engineer features
        df = self._engineer_distance_features(df)
        df = self._engineer_temporal_features(df)
        df = self._engineer_categorical_interactions(df)
        
        # Prepare final feature matrix
        numerical_features = self._get_numerical_features(df)
        self.scaler.fit(df[numerical_features])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the data"""
        df = X.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Engineer features
        df = self._engineer_distance_features(df)
        df = self._engineer_temporal_features(df)
        df = self._engineer_categorical_interactions(df)
        
        # Encode categorical features
        for cat_feature in FEATURE_CONFIG['categorical_features']:
            df[f'{cat_feature}_encoded'] = self.encoders[cat_feature].transform(df[cat_feature].astype(str))
        
        # Scale numerical features
        numerical_features = self._get_numerical_features(df)
        df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        # Select final features
        final_features = (
            [f'{col}_encoded' for col in FEATURE_CONFIG['categorical_features']] +
            numerical_features
        )
        
        return df[final_features].values
    
    def _get_numerical_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of all numerical features including engineered ones"""
        return (
            FEATURE_CONFIG['numerical_features'] +
            ['distance_log', 'distance_zscore', 'route_frequency',
             'vehicle_route_frequency', 'planned_transit_days']
        )
    
    def save(self, filename: str):
        """Save the transformer"""
        joblib.dump(self, MODELS_DIR / filename)
    
    @classmethod
    def load(cls, filename: str) -> 'CustomFeatureTransformer':
        """Load the transformer"""
        return joblib.load(MODELS_DIR / filename)
