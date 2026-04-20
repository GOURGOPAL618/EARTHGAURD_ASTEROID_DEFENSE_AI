"""
Copyright © 2026 StellarMind - EarthGuard Asteroid Defense AI
Author: Gouragopal Mohapatra & Arijit Kumar Mohanty
File: train_rf.py
Purpose: Random Forest model training for asteroid risk prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class RandomForestTrainer:
    """
    Random Forest trainer for asteroid risk prediction
    Includes hyperparameter tuning and cross-validation
    """
    
    def __init__(self, random_state=42):
        self.model = None
        self.best_params = None
        self.cv_scores = None
        self.random_state = random_state
    
    def train_baseline(self, X_train, y_train):
        """Train baseline Random Forest model"""
        print("=== Training Baseline Random Forest ===")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        train_score = self.model.score(X_train, y_train)
        
        print(f"✓ Baseline model trained")
        print(f"  Training accuracy: {train_score:.4f}")
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV"""
        print("=== Hyperparameter Tuning for Random Forest ===")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, 
                                     class_weight='balanced',
                                     n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"✓ Best parameters: {self.best_params}")
        print(f"✓ Best cross-validation F1: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def cross_validate(self, X_train, y_train, cv=5):
        """Perform cross-validation"""
        print(f"=== {cv}-Fold Cross Validation ===")
        
        if self.model is None:
            self.train_baseline(X_train, y_train)
        
        scores = cross_val_score(self.model, X_train, y_train, 
                                  cv=cv, scoring='f1')
        
        self.cv_scores = scores
        print(f"Cross-validation F1 scores: {scores}")
        print(f"Mean F1: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def feature_importance(self, feature_names, top_n=15):
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n=== Top {top_n} Important Features ===")
        print(importance_df.head(top_n))
        
        return importance_df
    
    def save_model(self, filepath='../../models/random_forest.pkl'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
        
    def load_model(self, filepath='../../models/random_forest.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.cv_scores = model_data['cv_scores']
        print(f"✓ Model loaded from {filepath}")
        return self.model


if __name__ == "__main__":
    # Test the trainer
    train_df = pd.read_csv('../../data/processed/train.csv')
    target = 'is_potentially_hazardous_asteroid'
    
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    
    trainer = RandomForestTrainer()
    
    # Train baseline
    model = trainer.train_baseline(X_train, y_train)
    
    # Cross-validate
    trainer.cross_validate(X_train, y_train)
    
    # Feature importance
    importance = trainer.feature_importance(X_train.columns)
    
    # Save model
    trainer.save_model()