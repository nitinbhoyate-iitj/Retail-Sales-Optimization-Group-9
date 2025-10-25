"""
Unified data loader that supports both S3 and local sources
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.logger import setup_logger
from .load_from_s3 import S3DataLoader
from .load_from_local import LocalDataLoader


class DataLoader:
    """
    Unified data loader supporting both S3 and local filesystem
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize unified data loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logger('data_loader')
        
        # Initialize both loaders
        self.s3_loader = S3DataLoader(config_path)
        self.local_loader = LocalDataLoader(config_path)
        
        self.logger.info("Initialized unified DataLoader")
    
    def load_data(
        self,
        source: str,
        source_path: str,
        collection_name: str,
        file_type: str = 'csv',
        **kwargs
    ) -> bool:
        """
        Load data from either S3 or local filesystem
        
        Args:
            source: Data source ('s3' or 'local')
            source_path: Path to data (S3 key or local file path)
            collection_name: MongoDB collection name
            file_type: Type of file (csv, excel, parquet)
            **kwargs: Additional arguments for read functions
            
        Returns:
            True if successful, False otherwise
        """
        source = source.lower()
        
        if source == 's3':
            self.logger.info(f"Loading from S3: {source_path}")
            return self.s3_loader.load_to_mongodb(
                source_path,
                collection_name,
                file_type,
                **kwargs
            )
        
        elif source == 'local':
            self.logger.info(f"Loading from local: {source_path}")
            return self.local_loader.load_to_mongodb(
                source_path,
                collection_name,
                file_type,
                **kwargs
            )
        
        else:
            self.logger.error(f"Unsupported source: {source}")
            return False
    
    def load_kaggle_shopping_dataset(
        self,
        source: str,
        source_path: str,
        collection_name: str = 'raw_sales'
    ) -> bool:
        """
        Load the Kaggle Customer Shopping Dataset from either S3 or local
        Dataset: https://www.kaggle.com/datasets/mehmettahiraslan/customer-shopping-dataset
        
        Args:
            source: Data source ('s3' or 'local')
            source_path: Path to shopping.csv (S3 key or local file path)
            collection_name: MongoDB collection name
            
        Returns:
            True if successful, False otherwise
        """
        source = source.lower()
        
        self.logger.info(
            f"Loading Kaggle Customer Shopping Dataset from {source}: {source_path}"
        )
        
        if source == 's3':
            return self.s3_loader.load_to_mongodb(
                source_path,
                collection_name,
                file_type='csv'
            )
        
        elif source == 'local':
            return self.local_loader.load_kaggle_dataset(
                source_path,
                collection_name
            )
        
        else:
            self.logger.error(f"Unsupported source: {source}")
            return False
    
    def load_batch(
        self,
        batch_config: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Load multiple datasets based on configuration
        
        Args:
            batch_config: Dictionary with dataset configurations
                Example:
                {
                    'sales': {
                        'source': 'local',
                        'path': 'dataset/sales.csv',
                        'collection': 'raw_sales',
                        'file_type': 'csv'
                    },
                    'products': {
                        'source': 's3',
                        'path': 'raw_data/products.csv',
                        'collection': 'raw_products',
                        'file_type': 'csv'
                    }
                }
        
        Returns:
            Dictionary with dataset names and success status
        """
        results = {}
        
        for dataset_name, config in batch_config.items():
            self.logger.info(f"Loading dataset: {dataset_name}")
            
            try:
                success = self.load_data(
                    source=config['source'],
                    source_path=config['path'],
                    collection_name=config['collection'],
                    file_type=config.get('file_type', 'csv'),
                    **config.get('read_kwargs', {})
                )
                results[dataset_name] = success
                
                if success:
                    self.logger.info(f"✓ Successfully loaded {dataset_name}")
                else:
                    self.logger.error(f"✗ Failed to load {dataset_name}")
                    
            except Exception as e:
                self.logger.error(f"Error loading {dataset_name}: {e}")
                results[dataset_name] = False
        
        # Summary
        success_count = sum(results.values())
        total_count = len(results)
        self.logger.info(
            f"Batch loading completed: {success_count}/{total_count} successful"
        )
        
        return results


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Unified data loader for S3 and local filesystem'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['s3', 'local'],
        help='Data source (s3 or local)'
    )
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to data (S3 key or local file path)'
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
        default='csv',
        choices=['csv', 'excel', 'parquet'],
        help='File type'
    )
    parser.add_argument(
        '--kaggle-dataset',
        action='store_true',
        help='Load as Kaggle Customer Shopping Dataset'
    )
    
    args = parser.parse_args()
    
    loader = DataLoader()
    
    if args.kaggle_dataset:
        success = loader.load_kaggle_shopping_dataset(
            args.source,
            args.path,
            args.collection
        )
    else:
        success = loader.load_data(
            args.source,
            args.path,
            args.collection,
            args.file_type
        )
    
    if success:
        print(f"✓ Successfully loaded data into {args.collection}")
    else:
        print("✗ Failed to load data")
        sys.exit(1)


if __name__ == '__main__':
    main()

