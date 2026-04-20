"""
Copyright © 2026 StellarMind - EarthGuard Asteroid Defense AI
Author: [Your Name] & Arijit Kumar Mohanty
File: metrics.py
Purpose: Evaluation metrics and utilities for asteroid risk prediction
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, matthews_corrcoef,
                            log_loss, average_precision_score)

class ModelEvaluator:
    """
    Comprehensive model evaluation for asteroid risk prediction
    """
    
    def __init__(self, model, model_name="Model"):
        self.model = model
        self.model_name = model_name
        self.metrics = {}
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        print(f"=== Evaluating {self.model_name} ===")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate all metrics
        self.metrics = {
            'Model': self.model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'Log Loss': log_loss(y_test, y_proba),
            'ROC-AUC': roc_auc_score(y_test, y_proba),
            'Avg Precision': average_precision_score(y_test, y_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.metrics['True Negatives'] = cm[0, 0]
        self.metrics['False Positives'] = cm[0, 1]
        self.metrics['False Negatives'] = cm[1, 0]
        self.metrics['True Positives'] = cm[1, 1]
        
        # Print results
        print(f"Accuracy:  {self.metrics['Accuracy']:.4f}")
        print(f"Precision: {self.metrics['Precision']:.4f}")
        print(f"Recall:    {self.metrics['Recall']:.4f}")
        print(f"F1-Score:  {self.metrics['F1-Score']:.4f}")
        print(f"ROC-AUC:   {self.metrics['ROC-AUC']:.4f}")
        
        return self.metrics, y_pred, y_proba
    
    def get_confusion_matrix(self, X_test, y_test):
        """Return confusion matrix as DataFrame"""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        return pd.DataFrame(cm, 
                           index=['Actual Non-Hazardous', 'Actual Hazardous'],
                           columns=['Predicted Non-Hazardous', 'Predicted Hazardous'])
    
    def get_classification_report(self, X_test, y_test):
        """Return classification report as DataFrame"""
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, 
                                       target_names=['Non-Hazardous', 'Hazardous'],
                                       output_dict=True)
        return pd.DataFrame(report).transpose()
    
    def get_metrics_df(self):
        """Return metrics as DataFrame"""
        return pd.DataFrame([self.metrics])


class ModelComparator:
    """
    Compare multiple models for asteroid risk prediction
    """
    
    def __init__(self):
        self.models = {}
        self.results = []
    
    def add_model(self, model, name, X_test, y_test):
        """Add a model for comparison"""
        evaluator = ModelEvaluator(model, name)
        metrics, _, _ = evaluator.evaluate(X_test, y_test)
        self.models[name] = evaluator
        self.results.append(metrics)
    
    def compare(self):
        """Compare all models"""
        results_df = pd.DataFrame(self.results)
        
        print("="*70)
        print("MODEL COMPARISON RESULTS")
        print("="*70)
        print(results_df.round(4))
        
        # Find best model for each metric
        print("\n" + "="*70)
        print("BEST MODEL FOR EACH METRIC")
        print("="*70)
        
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            best_idx = results_df[metric].idxmax()
            best_model = results_df.loc[best_idx, 'Model']
            best_value = results_df.loc[best_idx, metric]
            print(f"{metric:15s}: {best_model:20s} ({best_value:.4f})")
        
        # Overall best model (by F1-Score)
        best_idx = results_df['F1-Score'].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_f1 = results_df.loc[best_idx, 'F1-Score']
        
        print("\n" + "="*70)
        print(f"🏆 OVERALL BEST MODEL: {best_model} (F1-Score: {best_f1:.4f})")
        print("="*70)
        
        return results_df
    
    def get_results_df(self):
        """Return results as DataFrame"""
        return pd.DataFrame(self.results)


if __name__ == "__main__":
    # Test the evaluator
    import joblib
    
    # Load test data
    test_df = pd.read_csv('../../data/processed/test.csv')
    target = 'is_potentially_hazardous_asteroid'
    
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    
    # Load models
    rf = joblib.load('../../models/random_forest.pkl')['model']
    dt = joblib.load('../../models/decision_tree.pkl')['model']
    lr = joblib.load('../../models/logistic_regression.pkl')['model']
    
    # Compare models
    comparator = ModelComparator()
    comparator.add_model(rf, 'Random Forest', X_test, y_test)
    comparator.add_model(dt, 'Decision Tree', X_test, y_test)
    comparator.add_model(lr, 'Logistic Regression', X_test, y_test)
    
    results = comparator.compare()