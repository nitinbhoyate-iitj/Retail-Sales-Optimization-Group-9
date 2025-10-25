"""
Logging utility for the project
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_dir: str = None,
    log_level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_dir is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(log_dir) / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class DataQualityLogger:
    """Logger specifically for data quality and cleaning operations"""
    
    def __init__(self, log_dir: str = None):
        """
        Initialize data quality logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.logger = setup_logger('data_quality', log_dir)
        self.cleaning_log = []
    
    def log_cleaning_step(self, step: str, details: dict):
        """
        Log a data cleaning step
        
        Args:
            step: Name of the cleaning step
            details: Dictionary containing details of the operation
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'details': details
        }
        self.cleaning_log.append(log_entry)
        self.logger.info(f"{step}: {details}")
    
    def log_missing_values(self, df, before: bool = True):
        """Log missing value statistics"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        status = "before" if before else "after"
        self.logger.info(f"Missing values {status} cleaning:")
        for col, count in missing[missing > 0].items():
            self.logger.info(f"  {col}: {count} ({missing_pct[col]:.2f}%)")
    
    def log_outliers(self, column: str, n_outliers: int, method: str):
        """Log outlier detection results"""
        self.logger.info(
            f"Outliers detected in {column}: {n_outliers} using {method} method"
        )
    
    def log_transformation(self, column: str, transformation: str):
        """Log data transformation"""
        self.logger.info(f"Applied {transformation} transformation to {column}")
    
    def save_cleaning_report(self, output_path: str = None):
        """
        Save cleaning log to file
        
        Args:
            output_path: Path to save the cleaning report
        """
        if output_path is None:
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = log_dir / f"cleaning_report_{timestamp}.log"
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA CLEANING REPORT\n")
            f.write("="*80 + "\n\n")
            
            for entry in self.cleaning_log:
                f.write(f"[{entry['timestamp']}] {entry['step']}\n")
                f.write(f"Details: {entry['details']}\n")
                f.write("-"*80 + "\n")
        
        self.logger.info(f"Cleaning report saved to {output_path}")

