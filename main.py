"""
Copyright © 2026 StellarMind - EarthGuard Asteroid Defense AI
Author: Gouragopal Mohapatra & Arijit Kumar Mohanty
File: main.py
Purpose: Main entry point for Asteroid Risk Prediction System
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import src modules
from src.preprocessing.cleaner import AsteroidDataCleaner
from src.preprocessing.feature_eng import FeatureEngineer
from src.preprocessing.scaler import DataScaler, FeatureSelector
from src.training.train_rf import RandomForestTrainer
from src.training.train_dt import DecisionTreeTrainer
from src.training.train_regression import LogisticRegressionTrainer
from src.evaluation.metrics import ModelEvaluator, ModelComparator
from src.utils.logger import get_logger
from src.utils.config import get_config

# Initialize logger and config
logger = get_logger()
config = get_config()

class AsteroidRiskPredictor:
    """
    Main class for Asteroid Risk Prediction System
    """
    
    def __init__(self):
        self.logger = logger
        self.config = config
        self.models = {}
        self.best_model = None
    
    def load_data(self):
        """Load raw asteroid data"""
        self.logger.info("Loading data...")
        raw_path = self.config.get('data.raw_path')
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Data file not found: {raw_path}")
        
        df = pd.read_csv(raw_path)
        self.logger.info(f"Data loaded: {df.shape}")
        return df
    
    def preprocess(self, df):
        """Preprocess the data"""
        self.logger.info("Starting preprocessing...")
        
        # Clean data
        cleaner = AsteroidDataCleaner()
        df_clean = cleaner.clean(df)
        
        # Feature engineering
        engineer = FeatureEngineer()
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        df_feat = engineer.engineer_features(df_clean, categorical_cols=categorical_cols)
        
        # Feature selection
        selector = FeatureSelector()
        df_feat = selector.remove_low_variance(df_feat, threshold=0.01)
        df_feat = selector.remove_high_correlation(df_feat, correlation_threshold=0.95)
        
        self.logger.info(f"Preprocessing completed: {df_feat.shape}")
        return df_feat
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        self.logger.info("Starting model training...")
        
        # Random Forest
        if self.config.get('models.random_forest.enabled'):
            rf_trainer = RandomForestTrainer()
            rf_trainer.train_baseline(X_train, y_train)
            rf_trainer.cross_validate(X_train, y_train)
            rf_trainer.save_model()
            self.models['Random Forest'] = rf_trainer.model
            self.logger.info("Random Forest trained successfully")
        
        # Decision Tree
        if self.config.get('models.decision_tree.enabled'):
            dt_trainer = DecisionTreeTrainer()
            dt_trainer.train_baseline(X_train, y_train)
            dt_trainer.cross_validate(X_train, y_train)
            dt_trainer.save_model()
            self.models['Decision Tree'] = dt_trainer.model
            self.logger.info("Decision Tree trained successfully")
        
        # Logistic Regression
        if self.config.get('models.logistic_regression.enabled'):
            lr_trainer = LogisticRegressionTrainer()
            lr_trainer.train_baseline(X_train, y_train)
            lr_trainer.cross_validate(X_train, y_train)
            lr_trainer.save_model()
            self.models['Logistic Regression'] = lr_trainer.model
            self.logger.info("Logistic Regression trained successfully")
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        self.logger.info("Evaluating models...")
        
        comparator = ModelComparator()
        
        for name, model in self.models.items():
            comparator.add_model(model, name, X_test, y_test)
        
        results_df = comparator.compare()
        
        # Find best model
        best_idx = results_df['F1-Score'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[best_model_name]
        
        # Save results
        results_df.to_csv(self.config.get('paths.reports_dir') + 'model_results.csv', index=False)
        self.logger.info(f"Best model: {best_model_name}")
        
        return results_df
    
    def run_pipeline(self):
        """Run complete pipeline"""
        print("="*60)
        print("🌟 STELLARMIND - EARTHGUARD ASTEROID DEFENSE AI 🌟")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Preprocess
        df_processed = self.preprocess(df)
        
        # Split data
        target_col = self.config.get('data.target_column')
        feature_cols = [col for col in df_processed.columns if col != target_col]
        
        X = df_processed[feature_cols]
        y = df_processed[target_col]
        
        # Scale data
        scaler = DataScaler(method=self.config.get('preprocessing.scaling_method'))
        X_scaled = scaler.fit_transform(X)
        scaler.save(self.config.get('paths.models_dir') + 'scaler.pkl')
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=self.config.get('data.test_size'),
            random_state=self.config.get('data.random_state'),
            stratify=y
        )
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY ✅")
        print("="*60)
        
        return self.best_model, results
    
    def predict(self, features_df, model_name='best'):
        """Make predictions on new data"""
        if model_name == 'best':
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)[:, 1]
        
        return predictions, probabilities


if __name__ == "__main__":
    # Run the pipeline
    predictor = AsteroidRiskPredictor()
    best_model, results = predictor.run_pipeline()
    
    print("\n🎯 Asteroid Risk Prediction System is ready!")
    print("   Use predictor.predict() for new asteroid data")