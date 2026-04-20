"""
Copyright © 2026 StellarMind - EarthGuard Asteroid Defense AI
Author: Gouragopal Mohapatra & Arijit Kumar Mohanty
File: cleaner.py
Purpose: Data cleaning functions for asteroid dataset
"""

import pandas as pd
import numpy as np

class AsteroidDataCleaner:
    """
    Cleaner class for asteroid dataset
    Handles missing values, duplicates, and basic data cleaning
    """
    
    def __init__(self):
        self.removed_columns = []
        self.cleaning_log = []
    
    def remove_duplicates(self, df):
        """Remove duplicate rows from dataframe"""
        initial_shape = df.shape
        df = df.drop_duplicates()
        removed = initial_shape[0] - df.shape[0]
        self.cleaning_log.append(f"Removed {removed} duplicate rows")
        return df
    
    def remove_identifier_columns(self, df):
        """Remove unnecessary identifier columns"""
        id_cols = ['Unnamed: 0', 'id', 'neo_reference_id', 'orbit_id', 'name']
        cols_to_remove = [col for col in id_cols if col in df.columns]
        
        if cols_to_remove:
            self.removed_columns.extend(cols_to_remove)
            df = df.drop(columns=cols_to_remove)
            self.cleaning_log.append(f"Removed identifier columns: {cols_to_remove}")
        
        return df
    
    def remove_text_heavy_columns(self, df):
        """Remove columns with too many unique text values"""
        text_cols = ['orbit_class_description', 'orbit_determination_date', 
                     'first_observation_date', 'last_observation_date', 'equinox']
        cols_to_remove = [col for col in text_cols if col in df.columns]
        
        if cols_to_remove:
            self.removed_columns.extend(cols_to_remove)
            df = df.drop(columns=cols_to_remove)
            self.cleaning_log.append(f"Removed text-heavy columns: {cols_to_remove}")
        
        return df
    
    def handle_missing_values(self, df, strategy='median'):
        """Handle missing values in numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        missing_before = df[numeric_cols].isnull().sum().sum()
        
        if strategy == 'median':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mean':
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        missing_after = df[numeric_cols].isnull().sum().sum()
        self.cleaning_log.append(f"Handled {missing_before - missing_after} missing values using {strategy}")
        
        return df
    
    def clean(self, df):
        """Run all cleaning steps"""
        print("=== Starting Data Cleaning ===")
        
        df = self.remove_duplicates(df)
        df = self.remove_identifier_columns(df)
        df = self.remove_text_heavy_columns(df)
        df = self.handle_missing_values(df)
        
        print(f"✓ Cleaning completed")
        print(f"  Final shape: {df.shape}")
        print(f"  Removed columns: {self.removed_columns}")
        
        return df
    
    def get_cleaning_log(self):
        """Return cleaning log"""
        return self.cleaning_log


if __name__ == "__main__":
    # Test the cleaner
    df = pd.read_csv('../../data/raw/nasa_asteroid.csv')
    cleaner = AsteroidDataCleaner()
    cleaned_df = cleaner.clean(df)
    print("\nCleaning Log:")
    for log in cleaner.get_cleaning_log():
        print(f"  • {log}")