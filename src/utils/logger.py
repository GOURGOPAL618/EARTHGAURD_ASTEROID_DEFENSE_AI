"""
Copyright © 2026 StellarMind - EarthGuard Asteroid Defense AI
Author: Gouragopal Mohapatra & Arijit Kumar Mohanty
File: logger.py
Purpose: Logging utilities for asteroid risk prediction system
"""

import logging
import datetime
import os

class ProjectLogger:
    """
    Custom logger for StellarMind project
    """
    
    def __init__(self, log_dir='../../logs', log_level=logging.INFO):
        self.log_dir = log_dir
        self.logger = None
        self.setup_logger(log_level)
    
    def setup_logger(self, log_level):
        """Setup logger configuration"""
        
        # Create logs directory if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create logger
        self.logger = logging.getLogger('StellarMind')
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        log_filename = f"{self.log_dir}/stellar_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("="*60)
        self.logger.info("StellarMind Logger Initialized")
        self.logger.info(f"Log file: {log_filename}")
        self.logger.info("="*60)
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_model_training(self, model_name, params, score):
        """Log model training details"""
        self.logger.info(f"Model Training: {model_name}")
        self.logger.info(f"  Parameters: {params}")
        self.logger.info(f"  Score: {score:.4f}")
    
    def log_prediction(self, model_name, n_samples, accuracy):
        """Log prediction details"""
        self.logger.info(f"Prediction: {model_name}")
        self.logger.info(f"  Samples: {n_samples}")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")


# Global logger instance
stellar_logger = ProjectLogger()

def get_logger():
    """Get global logger instance"""
    return stellar_logger


if __name__ == "__main__":
    # Test the logger
    logger = get_logger()
    
    logger.info("Testing logger functionality")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    logger.log_model_training("Random Forest", {"n_estimators": 100}, 0.95)