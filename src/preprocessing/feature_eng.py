"""
Copyright © 2026 StellarMind - EarthGuard Asteroid Defense AI
Author: Gouragopal Mohapatra & Arijit Kumar Mohanty
File: feature_eng.py
Purpose: Feature engineering functions for asteroid dataset
"""

import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Feature engineering class for asteroid dataset
    Creates new features and handles categorical encoding
    """
    
    def __init__(self):
        self.created_features = []
        self.encoding_maps = {}
    
    def create_orbital_ratios(self, df):
        """Create ratio-based features from orbital parameters"""
        
        if 'aphelion_distance' in df.columns and 'perihelion_distance' in df.columns:
            df['orbit_ratio'] = df['aphelion_distance'] / (df['perihelion_distance'] + 1e-6)
            self.created_features.append('orbit_ratio')
        
        if 'semi_major_axis' in df.columns and 'perihelion_distance' in df.columns:
            df['axis_perihelion_ratio'] = df['semi_major_axis'] / (df['perihelion_distance'] + 1e-6)
            self.created_features.append('axis_perihelion_ratio')
        
        if 'orbital_period' in df.columns and 'semi_major_axis' in df.columns:
            # Kepler's third law verification
            df['kepler_check'] = df['orbital_period'] / (df['semi_major_axis'] ** 1.5 + 1e-6)
            self.created_features.append('kepler_check')
        
        return df
    
    def create_risk_indicators(self, df):
        """Create risk-based indicator features"""
        
        # Earth proximity indicator
        if 'perihelion_distance' in df.columns:
            df['earth_proximity'] = (df['perihelion_distance'] < 1.3).astype(int)
            self.created_features.append('earth_proximity')
        
        # High eccentricity indicator
        if 'eccentricity' in df.columns:
            df['high_eccentricity'] = (df['eccentricity'] > 0.5).astype(int)
            self.created_features.append('high_eccentricity')
        
        # Size risk indicator
        if 'estimated_diameter_kilometers_max' in df.columns:
            df['large_asteroid'] = (df['estimated_diameter_kilometers_max'] > 1.0).astype(int)
            self.created_features.append('large_asteroid')
        
        return df
    
    def encode_categorical_smart(self, df, categorical_cols, max_categories=10):
        """
        Smart encoding for categorical columns
        - Low cardinality: One-hot encoding
        - Medium cardinality: Frequency encoding
        - High cardinality: Top-N encoding
        """
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            unique_count = df[col].nunique()
            
            if unique_count <= max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
                self.created_features.extend(dummies.columns.tolist())
                print(f"  ✓ {col}: One-hot encoded -> {dummies.shape[1]} columns")
                
            elif unique_count <= 50:
                # Frequency encoding
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df_encoded[col + '_freq'] = df[col].map(freq_map)
                df_encoded = df_encoded.drop(columns=[col])
                self.created_features.append(col + '_freq')
                self.encoding_maps[col] = freq_map
                print(f"  ✓ {col}: Frequency encoded -> 1 column")
                
            else:
                # Top-N encoding (keep top categories as binary)
                top_cats = df[col].value_counts().head(max_categories).index
                for cat in top_cats:
                    new_col = f"{col}_is_{cat}"
                    df_encoded[new_col] = (df[col] == cat).astype(int)
                    self.created_features.append(new_col)
                df_encoded = df_encoded.drop(columns=[col])
                print(f"  ✓ {col}: Top-{max_categories} encoded -> {max_categories} columns")
        
        return df_encoded
    
    def engineer_features(self, df, categorical_cols=None):
        """Run all feature engineering steps"""
        print("=== Starting Feature Engineering ===")
        
        df = self.create_orbital_ratios(df)
        df = self.create_risk_indicators(df)
        
        if categorical_cols:
            df = self.encode_categorical_smart(df, categorical_cols)
        
        print(f"✓ Feature engineering completed")
        print(f"  Created {len(self.created_features)} new features")
        print(f"  Final shape: {df.shape}")
        
        return df
    
    def get_created_features(self):
        """Return list of created features"""
        return self.created_features


if __name__ == "__main__":
    # Test the feature engineer
    df = pd.read_csv('../../data/processed/asteroid_cleaned.csv')
    engineer = FeatureEngineer()
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    engineered_df = engineer.engineer_features(df, categorical_cols=cat_cols)
    print(f"\nCreated features: {engineer.get_created_features()[:5]}")