"""
Data cleaning and preprocessing pipeline for retail sales data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.db_operations import MongoDBHandler
from utils.logger import setup_logger, DataQualityLogger


class DataCleaner:
    """Data cleaning utilities for retail sales data"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize Data Cleaner
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.cleaning_config = self.config['cleaning']
        self.logger = setup_logger('data_cleaner')
        self.quality_logger = DataQualityLogger()
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'auto',
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Handle missing values in dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                     ('auto', 'drop', 'mean', 'median', 'mode', 'forward_fill')
            threshold: Threshold for dropping columns (fraction of missing values)
            
        Returns:
            Cleaned DataFrame
        """
        self.quality_logger.log_missing_values(df, before=True)
        
        if threshold is None:
            threshold = self.cleaning_config['missing_value_threshold']
        
        df_clean = df.copy()
        
        # Drop columns with too many missing values
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            self.quality_logger.log_cleaning_step(
                'drop_high_missing_columns',
                {
                    'columns_dropped': cols_to_drop,
                    'threshold': threshold
                }
            )
        
        # Handle remaining missing values
        if strategy == 'drop':
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna()
            rows_dropped = initial_rows - len(df_clean)
            self.quality_logger.log_cleaning_step(
                'drop_missing_rows',
                {'rows_dropped': rows_dropped}
            )
        
        elif strategy in ['mean', 'median', 'mode', 'auto']:
            # Separate numeric and categorical columns
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
            
            # Impute numeric columns
            if len(numeric_cols) > 0:
                if strategy == 'auto':
                    impute_strategy = 'median'
                else:
                    impute_strategy = strategy if strategy != 'mode' else 'median'
                
                imputer = SimpleImputer(strategy=impute_strategy)
                df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                
                self.quality_logger.log_cleaning_step(
                    'impute_numeric',
                    {
                        'columns': list(numeric_cols),
                        'strategy': impute_strategy
                    }
                )
            
            # Impute categorical columns
            if len(categorical_cols) > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                df_clean[categorical_cols] = imputer.fit_transform(df_clean[categorical_cols])
                
                self.quality_logger.log_cleaning_step(
                    'impute_categorical',
                    {
                        'columns': list(categorical_cols),
                        'strategy': 'most_frequent'
                    }
                )
        
        elif strategy == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill')
            self.quality_logger.log_cleaning_step(
                'forward_fill',
                {'method': 'ffill'}
            )
        
        self.quality_logger.log_missing_values(df_clean, before=False)
        
        return df_clean
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: str = None,
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Detect and handle outliers
        
        Args:
            df: Input DataFrame
            columns: List of columns to check. If None, checks all numeric columns.
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        if method is None:
            method = self.cleaning_config['outlier_method']
        
        if threshold is None:
            threshold = self.cleaning_config['outlier_threshold']
        
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
            
            initial_count = len(df_clean)
            
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outlier_mask = z_scores > threshold
            
            else:
                self.logger.warning(f"Unknown outlier method: {method}")
                continue
            
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                # Cap outliers instead of removing them
                if method == 'iqr':
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                
                self.quality_logger.log_outliers(col, n_outliers, method)
        
        return df_clean
    
    def standardize_dates(
        self,
        df: pd.DataFrame,
        date_columns: List[str],
        output_format: str = '%Y-%m-%d'
    ) -> pd.DataFrame:
        """
        Standardize date formats
        
        Args:
            df: Input DataFrame
            date_columns: List of date column names
            output_format: Desired output format
            
        Returns:
            DataFrame with standardized dates
        """
        df_clean = df.copy()
        date_formats = self.cleaning_config['date_formats']
        
        for col in date_columns:
            if col not in df_clean.columns:
                continue
            
            # Try to parse dates with multiple formats
            parsed_dates = None
            for fmt in date_formats:
                try:
                    parsed_dates = pd.to_datetime(df_clean[col], format=fmt, errors='coerce')
                    if parsed_dates.notna().sum() > 0:
                        break
                except:
                    continue
            
            # If no format worked, try pandas' automatic parsing
            if parsed_dates is None or parsed_dates.notna().sum() == 0:
                parsed_dates = pd.to_datetime(df_clean[col], errors='coerce')
            
            df_clean[col] = parsed_dates
            
            self.quality_logger.log_transformation(col, 'date_standardization')
        
        return df_clean
    
    def standardize_text(
        self,
        df: pd.DataFrame,
        text_columns: List[str] = None,
        lowercase: bool = True,
        remove_whitespace: bool = True
    ) -> pd.DataFrame:
        """
        Standardize text columns
        
        Args:
            df: Input DataFrame
            text_columns: List of text columns. If None, uses all object columns.
            lowercase: Convert to lowercase
            remove_whitespace: Remove leading/trailing whitespace
            
        Returns:
            DataFrame with standardized text
        """
        df_clean = df.copy()
        
        if text_columns is None:
            text_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        for col in text_columns:
            if col not in df_clean.columns:
                continue
            
            if remove_whitespace:
                df_clean[col] = df_clean[col].astype(str).str.strip()
            
            if lowercase:
                df_clean[col] = df_clean[col].astype(str).str.lower()
            
            self.quality_logger.log_transformation(col, 'text_standardization')
        
        return df_clean
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: List[str] = None,
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for identifying duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        duplicates_removed = initial_rows - len(df_clean)
        
        self.quality_logger.log_cleaning_step(
            'remove_duplicates',
            {
                'duplicates_removed': duplicates_removed,
                'subset': subset,
                'keep': keep
            }
        )
        
        return df_clean
    
    def validate_data_types(
        self,
        df: pd.DataFrame,
        schema: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Validate and convert data types
        
        Args:
            df: Input DataFrame
            schema: Dictionary mapping column names to expected types
            
        Returns:
            DataFrame with corrected types
        """
        df_clean = df.copy()
        
        for col, dtype in schema.items():
            if col not in df_clean.columns:
                continue
            
            try:
                if dtype in ['int', 'int64']:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')
                elif dtype in ['float', 'float64']:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                elif dtype == 'str':
                    df_clean[col] = df_clean[col].astype(str)
                elif dtype == 'datetime':
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                elif dtype == 'category':
                    df_clean[col] = df_clean[col].astype('category')
                
                self.quality_logger.log_transformation(col, f'type_conversion_to_{dtype}')
            
            except Exception as e:
                self.logger.warning(f"Could not convert {col} to {dtype}: {e}")
        
        return df_clean


class CleaningPipeline:
    """End-to-end data cleaning pipeline"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize Cleaning Pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.db_handler = MongoDBHandler()
        self.cleaner = DataCleaner(config_path)
        self.logger = setup_logger('cleaning_pipeline')
    
    def clean_shopping_dataset(
        self,
        input_collection: str = 'raw_sales',
        output_collection: str = 'cleaned_sales'
    ) -> bool:
        """
        Clean the Kaggle Customer Shopping Dataset
        
        Args:
            input_collection: Source MongoDB collection
            output_collection: Destination MongoDB collection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting cleaning pipeline for {input_collection}")
            
            # Load data from MongoDB
            df = self.db_handler.read_to_dataframe(input_collection)
            self.logger.info(f"Loaded {len(df)} records from {input_collection}")
            
            initial_shape = df.shape
            
            # Step 1: Remove duplicates
            df = self.cleaner.remove_duplicates(df)
            
            # Step 2: Standardize date columns
            if 'invoice_date' in df.columns:
                df = self.cleaner.standardize_dates(df, ['invoice_date'])
            
            # Step 3: Standardize text columns
            text_cols = ['gender', 'category', 'payment_method', 'shopping_mall']
            df = self.cleaner.standardize_text(df, text_cols)
            
            # Step 4: Validate and convert data types
            schema = {
                'invoice_no': 'str',
                'customer_id': 'str',
                'gender': 'category',
                'age': 'int64',
                'category': 'category',
                'quantity': 'int64',
                'price': 'float64',
                'payment_method': 'category',
                'invoice_date': 'datetime',
                'shopping_mall': 'category'
            }
            df = self.cleaner.validate_data_types(df, schema)
            
            # Step 5: Handle missing values
            df = self.cleaner.handle_missing_values(df, strategy='auto')
            
            # Step 6: Handle outliers in numeric columns
            numeric_cols = ['age', 'quantity', 'price']
            df = self.cleaner.handle_outliers(df, numeric_cols)
            
            # Step 7: Add derived columns
            if 'price' in df.columns and 'quantity' in df.columns:
                df['total_amount'] = df['price'] * df['quantity']
            
            if 'invoice_date' in df.columns:
                df['year'] = df['invoice_date'].dt.year
                df['month'] = df['invoice_date'].dt.month
                df['day'] = df['invoice_date'].dt.day
                df['day_of_week'] = df['invoice_date'].dt.dayofweek
                df['quarter'] = df['invoice_date'].dt.quarter
            
            final_shape = df.shape
            
            self.logger.info(
                f"Cleaning completed: {initial_shape} -> {final_shape}"
            )
            
            # Save to MongoDB
            count = self.db_handler.insert_dataframe(df, output_collection)
            self.logger.info(
                f"Saved {count} cleaned records to {output_collection}"
            )
            
            # Save cleaning report
            self.cleaner.quality_logger.save_cleaning_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in cleaning pipeline: {e}")
            return False
    
    def clean_generic_dataset(
        self,
        input_collection: str,
        output_collection: str,
        date_columns: List[str] = None,
        text_columns: List[str] = None,
        numeric_columns: List[str] = None
    ) -> bool:
        """
        Generic cleaning pipeline for any dataset
        
        Args:
            input_collection: Source MongoDB collection
            output_collection: Destination MongoDB collection
            date_columns: List of date column names
            text_columns: List of text column names
            numeric_columns: List of numeric column names
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting generic cleaning pipeline for {input_collection}")
            
            # Load data
            df = self.db_handler.read_to_dataframe(input_collection)
            self.logger.info(f"Loaded {len(df)} records")
            
            # Remove duplicates
            df = self.cleaner.remove_duplicates(df)
            
            # Standardize dates
            if date_columns:
                df = self.cleaner.standardize_dates(df, date_columns)
            
            # Standardize text
            if text_columns:
                df = self.cleaner.standardize_text(df, text_columns)
            
            # Handle missing values
            df = self.cleaner.handle_missing_values(df, strategy='auto')
            
            # Handle outliers
            if numeric_columns:
                df = self.cleaner.handle_outliers(df, numeric_columns)
            
            # Save to MongoDB
            count = self.db_handler.insert_dataframe(df, output_collection)
            self.logger.info(f"Saved {count} cleaned records to {output_collection}")
            
            # Save cleaning report
            self.cleaner.quality_logger.save_cleaning_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in cleaning pipeline: {e}")
            return False


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Data cleaning pipeline'
    )
    parser.add_argument(
        '--input-collection',
        type=str,
        required=True,
        help='Input MongoDB collection name'
    )
    parser.add_argument(
        '--output-collection',
        type=str,
        required=True,
        help='Output MongoDB collection name'
    )
    parser.add_argument(
        '--shopping-dataset',
        action='store_true',
        help='Use specialized cleaning for Kaggle Shopping Dataset'
    )
    
    args = parser.parse_args()
    
    pipeline = CleaningPipeline()
    
    if args.shopping_dataset:
        success = pipeline.clean_shopping_dataset(
            args.input_collection,
            args.output_collection
        )
    else:
        success = pipeline.clean_generic_dataset(
            args.input_collection,
            args.output_collection
        )
    
    if success:
        print(f"✓ Successfully cleaned data from {args.input_collection} to {args.output_collection}")
    else:
        print("✗ Cleaning pipeline failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

