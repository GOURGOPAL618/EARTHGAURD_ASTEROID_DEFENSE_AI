"""
Copyright © 2026 StellarMind - EarthGuard Asteroid Defense AI
Author: Gouragopal Mohapatra & Arijit Kumar Mohanty
File: config.py
Purpose: Configuration management for asteroid risk prediction system
"""

import os
import json
import yaml

class Config:
    """
    Configuration manager for StellarMind project
    """
    
    def __init__(self, config_path='../../config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuration loaded from {self.config_path}")
            return config
        else:
            print(f"⚠️ Config file not found at {self.config_path}")
            print("Using default configuration")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            'data': {
                'raw_path': '../data/raw/nasa_asteroid.csv',
                'processed_path': '../data/processed/',
                'test_size': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'scaling_method': 'robust',
                'imputation_strategy': 'median',
                'remove_outliers': True
            },
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'class_weight': 'balanced'
                },
                'decision_tree': {
                    'max_depth': 10,
                    'class_weight': 'balanced'
                },
                'logistic_regression': {
                    'max_iter': 1000,
                    'class_weight': 'balanced'
                }
            },
            'training': {
                'cv_folds': 5,
                'scoring_metric': 'f1',
                'hyperparameter_tuning': True
            },
            'paths': {
                'models_dir': '../models/',
                'reports_dir': '../reports/',
                'logs_dir': '../logs/'
            }
        }
    
    def get(self, key, default=None):
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """Set configuration value by dot notation key"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        print(f"✓ Set {key} = {value}")
    
    def save(self, path=None):
        """Save configuration to file"""
        save_path = path or self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        print(f"✓ Configuration saved to {save_path}")
    
    def display(self):
        """Display current configuration"""
        print("="*50)
        print("CURRENT CONFIGURATION")
        print("="*50)
        print(yaml.dump(self.config, default_flow_style=False))


# Global config instance
stellar_config = Config()

def get_config():
    """Get global config instance"""
    return stellar_config


if __name__ == "__main__":
    # Test the config
    config = get_config()
    config.display()
    
    # Get specific values
    print(f"\nData path: {config.get('data.raw_path')}")
    print(f"Test size: {config.get('data.test_size')}")
    
    # Set new value
    config.set('training.cv_folds', 10)
    
    # Save config
    # config.save()