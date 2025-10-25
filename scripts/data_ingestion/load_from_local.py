"""
Script to load data from local filesystem into MongoDB
"""

import os
import sys
from pathlib import Path
import pandas as pd
from typing import List, Optional
import glob

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.db_operations import MongoDBHandler
from utils.logger import setup_logger


class LocalDataLoader:
    """Load data from local filesystem into MongoDB"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize Local Data Loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.db_handler = MongoDBHandler()
        self.logger = setup_logger('local_data_loader')
        
        # Get project root and dataset path
        self.project_root = Path(__file__).parent.parent.parent
        self.dataset_path = self.project_root / self.config['local']['dataset_path']
        
        # Create dataset directories if they don't exist
        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(
            self.project_root / self.config['local']['raw_data_path'],
            exist_ok=True
        )
        
        self.logger.info(f"Initialized LocalDataLoader with path: {self.dataset_path}")
    
    def list_files(
        self,
        directory: str = None,
        file_pattern: str = '*.csv'
    ) -> List[str]:
        """
        List all files in directory matching pattern
        
        Args:
            directory: Directory path. If None, uses default dataset path.
            file_pattern: Glob pattern for files
            
        Returns:
            List of file paths
        """
        if directory is None:
            directory = self.dataset_path
        else:
            directory = Path(directory)
        
        files = list(directory.glob(file_pattern))
        self.logger.info(
            f"Found {len(files)} files matching {file_pattern} in {directory}"
        )
        return [str(f) for f in files]
    
    def read_file(
        self,
        file_path: str,
        file_type: str = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Read file into DataFrame
        
        Args:
            file_path: Path to file
            file_type: Type of file (csv, excel, parquet). If None, infers from extension.
            **kwargs: Additional arguments for pandas read function
            
        Returns:
            DataFrame or None if failed
        """
        try:
            file_path = Path(file_path)
            
            # Infer file type from extension if not provided
            if file_type is None:
                ext = file_path.suffix.lower()
                if ext == '.csv':
                    file_type = 'csv'
                elif ext in ['.xlsx', '.xls']:
                    file_type = 'excel'
                elif ext == '.parquet':
                    file_type = 'parquet'
                else:
                    self.logger.error(f"Unsupported file extension: {ext}")
                    return None
            
            # Read file based on type
            if file_type == 'csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_type == 'excel':
                df = pd.read_excel(file_path, **kwargs)
            elif file_type == 'parquet':
                df = pd.read_parquet(file_path, **kwargs)
            else:
                self.logger.error(f"Unsupported file type: {file_type}")
                return None
            
            self.logger.info(
                f"Read file: {file_path.name} - Shape: {df.shape}"
            )
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def load_to_mongodb(
        self,
        file_path: str,
        collection_name: str,
        file_type: str = None,
        **read_kwargs
    ) -> bool:
        """
        Load data from local file into MongoDB
        
        Args:
            file_path: Path to file
            collection_name: MongoDB collection name
            file_type: Type of file (csv, excel, parquet)
            **read_kwargs: Additional arguments for pandas read function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read file
            df = self.read_file(file_path, file_type, **read_kwargs)
            
            if df is None:
                return False
            
            # Insert into MongoDB
            count = self.db_handler.insert_dataframe(df, collection_name)
            self.logger.info(
                f"Successfully loaded {count} records from {file_path} "
                f"into {collection_name}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data to MongoDB: {e}")
            return False
    
    def load_folder_to_mongodb(
        self,
        directory: str = None,
        collection_name: str = None,
        file_pattern: str = '*.csv',
        file_type: str = None,
        **read_kwargs
    ) -> int:
        """
        Load all files from folder into MongoDB collection
        
        Args:
            directory: Directory path. If None, uses default dataset path.
            collection_name: MongoDB collection name
            file_pattern: Glob pattern for files
            file_type: Type of files (csv, excel, parquet)
            **read_kwargs: Additional arguments for pandas read function
            
        Returns:
            Number of files successfully loaded
        """
        files = self.list_files(directory, file_pattern)
        
        if not files:
            self.logger.warning("No files found to load")
            return 0
        
        success_count = 0
        for file_path in files:
            if self.load_to_mongodb(file_path, collection_name, file_type, **read_kwargs):
                success_count += 1
        
        self.logger.info(
            f"Successfully loaded {success_count}/{len(files)} files"
        )
        return success_count
    
    def load_kaggle_dataset(
        self,
        file_path: str,
        collection_name: str = 'raw_sales'
    ) -> bool:
        """
        Load the Kaggle Customer Shopping Dataset
        Dataset: https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset
        
        Args:
            file_path: Path to the shopping.csv file
            collection_name: MongoDB collection name
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Loading Kaggle Customer Shopping Dataset")
        
        # Expected columns in the dataset
        expected_columns = [
            'invoice_no', 'customer_id', 'gender', 'age', 'category',
            'quantity', 'price', 'payment_method', 'invoice_date', 'shopping_mall'
        ]
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            self.logger.info(f"Dataset shape: {df.shape}")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Basic validation
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                self.logger.warning(
                    f"Expected columns not found: {missing_cols}"
                )
            
            # Load to MongoDB
            count = self.db_handler.insert_dataframe(df, collection_name)
            self.logger.info(
                f"Successfully loaded {count} shopping records into {collection_name}"
            )
            
            # Log basic statistics
            self.logger.info(f"Date range: {df['invoice_date'].min()} to {df['invoice_date'].max()}")
            self.logger.info(f"Number of customers: {df['customer_id'].nunique()}")
            self.logger.info(f"Number of categories: {df['category'].nunique()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading Kaggle dataset: {e}")
            return False
    
    def save_to_local(
        self,
        df: pd.DataFrame,
        filename: str,
        subdirectory: str = 'raw',
        file_format: str = 'csv'
    ) -> Optional[str]:
        """
        Save DataFrame to local storage
        
        Args:
            df: DataFrame to save
            filename: Name of file to save
            subdirectory: Subdirectory within dataset path (raw, cleaned, transformed)
            file_format: Format to save (csv, parquet, excel)
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Create subdirectory if needed
            save_dir = self.dataset_path / subdirectory
            os.makedirs(save_dir, exist_ok=True)
            
            # Add appropriate extension
            if not any(filename.endswith(ext) for ext in ['.csv', '.parquet', '.xlsx']):
                if file_format == 'csv':
                    filename += '.csv'
                elif file_format == 'parquet':
                    filename += '.parquet'
                elif file_format == 'excel':
                    filename += '.xlsx'
            
            file_path = save_dir / filename
            
            # Save based on format
            if file_format == 'csv':
                df.to_csv(file_path, index=False)
            elif file_format == 'parquet':
                df.to_parquet(file_path, index=False)
            elif file_format == 'excel':
                df.to_excel(file_path, index=False)
            else:
                self.logger.error(f"Unsupported format: {file_format}")
                return None
            
            self.logger.info(f"Saved file to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving file: {e}")
            return None


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Load data from local filesystem into MongoDB'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='File path to load'
    )
    parser.add_argument(
        '--directory',
        type=str,
        help='Directory to load all files from'
    )
    parser.add_argument(
        '--collection',
        type=str,
        required=True,
        help='MongoDB collection name'
    )
    parser.add_argument(
        '--file-type',
        type=str,
        choices=['csv', 'excel', 'parquet'],
        help='File type to load'
    )
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='*.csv',
        help='File pattern to match (when using --directory)'
    )
    parser.add_argument(
        '--kaggle-dataset',
        action='store_true',
        help='Load the Kaggle Customer Shopping Dataset'
    )
    
    args = parser.parse_args()
    
    loader = LocalDataLoader()
    
    if args.kaggle_dataset and args.file:
        success = loader.load_kaggle_dataset(args.file, args.collection)
        if success:
            print(f"✓ Successfully loaded Kaggle dataset into {args.collection}")
        else:
            print("✗ Failed to load Kaggle dataset")
            sys.exit(1)
    
    elif args.file:
        success = loader.load_to_mongodb(
            args.file,
            args.collection,
            args.file_type
        )
        if success:
            print(f"✓ Successfully loaded {args.file} into {args.collection}")
        else:
            print(f"✗ Failed to load {args.file}")
            sys.exit(1)
    
    elif args.directory:
        count = loader.load_folder_to_mongodb(
            args.directory,
            args.collection,
            args.file_pattern,
            args.file_type
        )
        print(f"✓ Successfully loaded {count} files into {args.collection}")
    
    else:
        parser.error("Either --file or --directory must be provided")


if __name__ == '__main__':
    main()

