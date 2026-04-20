"""
Copyright © 2026 StellarMind - EarthGuard Asteroid Defense AI
Author: Gouragopal Mohapatra & Arijit Kumar Mohanty
File: compare_models.py
Purpose: Compare all trained models and select best one
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    """
    Visual comparison of all trained models
    """
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.probabilities = {}
    
    def load_models(self, model_paths):
        """
        Load trained models
        
        model_paths: dict {'model_name': 'path/to/model.pkl'}
        """
        for name, path in model_paths.items():
            model_data = joblib.load(path)
            self.models[name] = model_data['model']
            print(f"✓ Loaded {name}")
    
    def predict_all(self, X_test):
        """Get predictions from all models"""
        for name, model in self.models.items():
            self.predictions[name] = model.predict(X_test)
            self.probabilities[name] = model.predict_proba(X_test)[:, 1]
        
        print("✓ Predictions completed for all models")
        return self.predictions, self.probabilities
    
    def plot_roc_curves(self, y_test, save_path=None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 7))
        
        for name, proba in self.probabilities.items():
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, y_test, save_path=None):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 7))
        
        for name, proba in self.probabilities.items():
            precision, recall, _ = precision_recall_curve(y_test, proba)
            plt.plot(recall, precision, lw=2, label=f'{name}')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, y_test, save_path=None):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, pred) in enumerate(self.predictions.items()):
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, pred)
            
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], 
                       cmap='RdYlGn', cbar=False)
            axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            
            # Add text annotations
            axes[idx].text(0.5, -0.15, f'TN: {cm[0,0]} | FP: {cm[0,1]}', 
                          transform=axes[idx].transAxes, ha='center', fontsize=9)
            axes[idx].text(0.5, -0.25, f'FN: {cm[1,0]} | TP: {cm[1,1]}', 
                          transform=axes[idx].transAxes, ha='center', fontsize=9)
        
        plt.suptitle('Confusion Matrices - Model Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_comparison_bar_chart(self, metrics_df, save_path=None):
        """Plot bar chart comparing model metrics"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        
        for idx, (name, color) in enumerate(zip(self.models.keys(), colors)):
            values = [metrics_df.loc[idx, m] for m in metrics_to_plot]
            bars = ax.bar(x + (idx - 1) * width, values, width, label=name, color=color)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', 
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", 
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def get_best_model(self, metrics_df, metric='F1-Score'):
        """Get the best model based on specified metric"""
        best_idx = metrics_df[metric].idxmax()
        best_model = metrics_df.loc[best_idx, 'Model']
        best_score = metrics_df.loc[best_idx, metric]
        
        return best_model, best_score


if __name__ == "__main__":
    # Test the comparison
    from metrics import ModelComparator
    
    # Load test data
    test_df = pd.read_csv('../../data/processed/test.csv')
    target = 'is_potentially_hazardous_asteroid'
    
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    
    # Create comparator
    comparator = ModelComparator()
    
    # Load and evaluate models
    model_paths = {
        'Random Forest': '../../models/random_forest.pkl',
        'Decision Tree': '../../models/decision_tree.pkl',
        'Logistic Regression': '../../models/logistic_regression.pkl'
    }
    
    comparison = ModelComparison()
    comparison.load_models(model_paths)
    comparison.predict_all(X_test)
    
    # Plot comparisons
    comparison.plot_roc_curves(y_test)
    comparison.plot_precision_recall_curves(y_test)
    comparison.plot_confusion_matrices(y_test)