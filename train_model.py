import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime

def load_and_preprocess_data():
    # Load the data
    df = pd.read_excel('AI ML Internship Training Data.xlsx')
    
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Handle missing values
    df = df.fillna({
        'Weather Conditions': df['Weather Conditions'].mode()[0],
        'Traffic Conditions': df['Traffic Conditions'].mode()[0],
        'Vehicle Type': df['Vehicle Type'].mode()[0]
    })
    
    # Clean Distance column - remove any commas and convert to float
    df['Distance (km)'] = df['Distance (km)'].astype(str).str.replace(',', '').astype(float)
    
    print("\nValue Counts for Categorical Variables:")
    for col in ['Origin', 'Destination', 'Vehicle Type', 'Weather Conditions', 'Traffic Conditions']:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    # Create label encoders for categorical variables
    categorical_columns = ['Origin', 'Destination', 'Vehicle Type', 'Weather Conditions', 'Traffic Conditions']
    encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Save encoders
    joblib.dump(encoders, 'encoders.joblib')
    
    # Create feature matrix - only using features available at prediction time
    feature_columns = [f'{col}_encoded' for col in categorical_columns] + ['Distance (km)']
    X = df[feature_columns]
    y = df['Delayed'].map({'Yes': 1, 'No': 0})
    
    # Print class distribution
    print("\nClass Distribution:")
    print(y.value_counts(normalize=True))
    print(f"Total samples: {len(y)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.joblib')
    
    return X_scaled, y

def train_and_evaluate_models(X, y):
    # Split the data with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Add cross-validation
    from sklearn.model_selection import cross_val_score
    
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # Limit tree depth to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Train on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Print feature importances for Random Forest
        if name == 'random_forest':
            feature_cols = [f'{col}_encoded' for col in ['Origin', 'Destination', 'Vehicle Type', 
                                                        'Weather Conditions', 'Traffic Conditions']] + ['Distance (km)']
            importances = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nFeature Importances for {name}:")
            print(importances)
        
        # Save the model
        joblib.dump(model, f'{name}_model.joblib')
    
    return results

def main():
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data()
    
    print("Training and evaluating models...")
    results = train_and_evaluate_models(X, y)
    
    # Print results with cross-validation scores
    for model_name, metrics in results.items():
        print(f"\nResults for {model_name}:")
        print(f"Test Set Metrics:")
        print(f"accuracy: {metrics['accuracy']:.4f}")
        print(f"precision: {metrics['precision']:.4f}")
        print(f"recall: {metrics['recall']:.4f}")
        print(f"f1: {metrics['f1']:.4f}")
        print(f"\nCross-validation scores:")
        print(f"Mean accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

if __name__ == "__main__":
    main()
