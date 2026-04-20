"""
Copyright © 2026 StellarMind - EarthGuard Asteroid Defense AI
Author: Gouragopal Mohapatra & Arijit Kumar Mohanty
File: scaler.py
Purpose: Data scaling functions for asteroid dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import joblib

class DataScaler:
    """
    Data scaling class for asteroid dataset
    Supports multiple scaling methods
    """
    
    def __init__(self, method='robust'):
        """
        Initialize scaler
        
        Parameters:
        method: 'robust', 'standard', or 'minmax'
        """
        self.method = method
        self.scaler = None
        self.fitted = False
        
        if method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'robust', 'standard', or 'minmax'")
    
    def fit(self, X):
        """Fit the scaler to data"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.numeric_cols = numeric_cols
        self.scaler.fit(X[numeric_cols])
        self.fitted = True
        print(f"✓ Scaler fitted on {len(numeric_cols)} numeric columns")
        return self
    
    def transform(self, X):
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X_scaled = X.copy()
        X_scaled[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X_scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled):
        """Inverse transform to original scale"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        X_original = X_scaled.copy()
        X_original[self.numeric_cols] = self.scaler.inverse_transform(X_scaled[self.numeric_cols])
        return X_original
    
    def save(self, filepath):
        """Save scaler to disk"""
        joblib.dump({
            'scaler': self.scaler,
            'method': self.method,
            'numeric_cols': self.numeric_cols,
            'fitted': self.fitted
        }, filepath)
        print(f"✓ Scaler saved to {filepath}")
    
    def load(self, filepath):
        """Load scaler from disk"""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.method = data['method']
        self.numeric_cols = data['numeric_cols']
        self.fitted = data['fitted']
        print(f"✓ Scaler loaded from {filepath}")
        return self


class FeatureSelector:
    """Feature selection utilities"""
    
    @staticmethod
    def remove_low_variance(df, threshold=0.01):
        """Remove features with very low variance"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var()
        low_var_cols = variances[variances < threshold].index.tolist()
        
        df = df.drop(columns=low_var_cols)
        print(f"✓ Removed {len(low_var_cols)} low-variance features")
        return df
    
    @staticmethod
    def remove_high_correlation(df, correlation_threshold=0.95):
        """Remove highly correlated features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().abs()
        
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
        
        df = df.drop(columns=high_corr_cols)
        print(f"✓ Removed {len(high_corr_cols)} highly correlated features")
        return df


if __name__ == "__main__":
    # Test the scaler
    df = pd.read_csv('../../data/processed/asteroid_cleaned.csv')
    
    # Separate features and target
    target = 'is_potentially_hazardous_asteroid'
    X = df.drop(columns=[target])
    y = df[target]
    
    # Scale features
    scaler = DataScaler(method='robust')
    X_scaled = scaler.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Scaled shape: {X_scaled.shape}")
    print(f"Scaled mean (first feature): {X_scaled[X_scaled.columns[0]].mean():.4f}")
    print(f"Scaled std (first feature): {X_scaled[X_scaled.columns[0]].std():.4f}")