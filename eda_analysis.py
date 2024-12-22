import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_theme()
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

def create_output_dirs():
    """Create directories for saving plots and analysis results"""
    os.makedirs('analysis_results/plots', exist_ok=True)
    os.makedirs('analysis_results/stats', exist_ok=True)

def load_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    try:
        df = pd.read_excel('AI ML Internship Training Data.xlsx')
        print(f"Dataset Shape: {df.shape}")
        print("\nDataset Info:")
        df.info()
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def analyze_missing_values(df):
    """Analyze and visualize missing values"""
    try:
        print("\nAnalyzing missing values...")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        # Print missing value summary
        print("\nMissing Value Summary:")
        for col in df.columns:
            if missing[col] > 0:
                print(f"{col}: {missing[col]} values ({missing_percent[col]:.2f}%)")
        
        return True
    except Exception as e:
        print(f"Error in missing value analysis: {str(e)}")
        return False

def analyze_numerical_features(df):
    """Analyze numerical features"""
    try:
        print("\nAnalyzing numerical features...")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Basic statistics
        stats_df = df[numerical_cols].describe()
        print("\nNumerical Features Statistics:")
        print(stats_df)
        
        return True
    except Exception as e:
        print(f"Error in numerical feature analysis: {str(e)}")
        return False

def analyze_categorical_features(df):
    """Analyze categorical features"""
    try:
        print("\nAnalyzing categorical features...")
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts().head())
            print(f"Unique values: {df[col].nunique()}")
        
        return True
    except Exception as e:
        print(f"Error in categorical feature analysis: {str(e)}")
        return False

def analyze_target_variable(df):
    """Analyze target variable (if present)"""
    try:
        print("\nAnalyzing target variable...")
        if 'Delayed' in df.columns:
            delayed_counts = df['Delayed'].value_counts()
            print("\nDelay Distribution:")
            print(delayed_counts)
            print(f"\nDelay Rate: {(delayed_counts.get(1, 0) / len(df)) * 100:.2f}%")
        else:
            print("No 'Delayed' column found in the dataset")
        
        return True
    except Exception as e:
        print(f"Error in target variable analysis: {str(e)}")
        return False

def detect_outliers(df):
    """Detect and analyze outliers in numerical features"""
    try:
        print("\nDetecting outliers...")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            print(f"\nOutliers in {col}:")
            print(f"Number of outliers: {len(outliers)}")
            print(f"Percentage of outliers: {(len(outliers)/len(df))*100:.2f}%")
        
        return True
    except Exception as e:
        print(f"Error in outlier detection: {str(e)}")
        return False

def generate_summary():
    """Generate a text summary of key findings"""
    print("\n=== EDA Summary ===")
    print("1. Data Quality:")
    print("   - Check the missing value analysis above")
    print("   - Review the outlier detection results")
    print("\n2. Feature Characteristics:")
    print("   - Review the numerical and categorical feature statistics")
    print("\n3. Target Variable:")
    print("   - Check the delay distribution and rate")
    print("\nRecommendations:")
    print("1. Handle missing values appropriately")
    print("2. Consider feature engineering based on the distributions")
    print("3. Address any class imbalance in the target variable")

def main():
    """Main function to run the EDA"""
    create_output_dirs()
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Run analyses
    analyses = [
        analyze_missing_values,
        analyze_numerical_features,
        analyze_categorical_features,
        analyze_target_variable,
        detect_outliers
    ]
    
    for analysis in analyses:
        try:
            analysis(df)
        except Exception as e:
            print(f"Error in {analysis.__name__}: {str(e)}")
    
    generate_summary()
    print("\nEDA completed!")

if __name__ == "__main__":
    main()
