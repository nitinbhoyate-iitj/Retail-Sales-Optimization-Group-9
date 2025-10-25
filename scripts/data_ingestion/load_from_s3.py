"""
Script to load data from AWS S3 bucket into MongoDB
"""

import os
import sys
from pathlib import Path
import pandas as pd
import io
from typing import List, Optional, Dict
import boto3
from botocore.exceptions import ClientError

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config, get_s3_client
from utils.db_operations import MongoDBHandler
from utils.logger import setup_logger


class S3DataLoader:
    """Load data from AWS S3 into MongoDB"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize S3 Data Loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.s3_client = get_s3_client()
        self.db_handler = MongoDBHandler()
        self.logger = setup_logger('s3_data_loader')
        
        self.bucket_name = os.getenv(
            'S3_BUCKET_NAME',
            self.config['aws']['s3_bucket']
        )
        self.logger.info(f"Initialized S3DataLoader for bucket: {self.bucket_name}")
    
    def list_files(self, prefix: str = '') -> List[str]:
        """
        List all files in S3 bucket with given prefix
        
        Args:
            prefix: S3 prefix/folder path
            
        Returns:
            List of file keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                self.logger.warning(f"No files found with prefix: {prefix}")
                return []
            
            files = [obj['Key'] for obj in response['Contents']]
            self.logger.info(f"Found {len(files)} files with prefix: {prefix}")
            return files
            
        except ClientError as e:
            self.logger.error(f"Error listing S3 files: {e}")
            return []
    
    def download_file(self, s3_key: str, local_path: str = None) -> Optional[str]:
        """
        Download file from S3 to local storage
        
        Args:
            s3_key: S3 object key
            local_path: Local path to save file. If None, saves to temp location.
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            if local_path is None:
                # Save to temp location in dataset folder
                project_root = Path(__file__).parent.parent.parent
                local_path = project_root / "dataset" / "temp" / Path(s3_key).name
                os.makedirs(local_path.parent, exist_ok=True)
            
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            self.logger.info(f"Downloaded {s3_key} to {local_path}")
            return str(local_path)
            
        except ClientError as e:
            self.logger.error(f"Error downloading file {s3_key}: {e}")
            return None
    
    def read_csv_from_s3(self, s3_key: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Read CSV file directly from S3 into DataFrame
        
        Args:
            s3_key: S3 object key
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame or None if failed
        """
        try:
            obj = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), **kwargs)
            self.logger.info(
                f"Read CSV from S3: {s3_key} - Shape: {df.shape}"
            )
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading CSV from S3 {s3_key}: {e}")
            return None
    
    def read_excel_from_s3(self, s3_key: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Read Excel file directly from S3 into DataFrame
        
        Args:
            s3_key: S3 object key
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            DataFrame or None if failed
        """
        try:
            obj = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            df = pd.read_excel(io.BytesIO(obj['Body'].read()), **kwargs)
            self.logger.info(
                f"Read Excel from S3: {s3_key} - Shape: {df.shape}"
            )
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading Excel from S3 {s3_key}: {e}")
            return None
    
    def read_parquet_from_s3(self, s3_key: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Read Parquet file directly from S3 into DataFrame
        
        Args:
            s3_key: S3 object key
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            DataFrame or None if failed
        """
        try:
            obj = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            df = pd.read_parquet(io.BytesIO(obj['Body'].read()), **kwargs)
            self.logger.info(
                f"Read Parquet from S3: {s3_key} - Shape: {df.shape}"
            )
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading Parquet from S3 {s3_key}: {e}")
            return None
    
    def load_to_mongodb(
        self,
        s3_key: str,
        collection_name: str,
        file_type: str = 'csv',
        **read_kwargs
    ) -> bool:
        """
        Load data from S3 directly into MongoDB
        
        Args:
            s3_key: S3 object key
            collection_name: MongoDB collection name
            file_type: Type of file (csv, excel, parquet)
            **read_kwargs: Additional arguments for pandas read function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read data based on file type
            if file_type.lower() == 'csv':
                df = self.read_csv_from_s3(s3_key, **read_kwargs)
            elif file_type.lower() in ['excel', 'xlsx', 'xls']:
                df = self.read_excel_from_s3(s3_key, **read_kwargs)
            elif file_type.lower() == 'parquet':
                df = self.read_parquet_from_s3(s3_key, **read_kwargs)
            else:
                self.logger.error(f"Unsupported file type: {file_type}")
                return False
            
            if df is None:
                return False
            
            # Insert into MongoDB
            count = self.db_handler.insert_dataframe(df, collection_name)
            self.logger.info(
                f"Successfully loaded {count} records from {s3_key} "
                f"into {collection_name}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data to MongoDB: {e}")
            return False
    
    def load_folder_to_mongodb(
        self,
        s3_prefix: str,
        collection_name: str,
        file_pattern: str = '*.csv',
        file_type: str = 'csv',
        **read_kwargs
    ) -> int:
        """
        Load all files from S3 folder into MongoDB collection
        
        Args:
            s3_prefix: S3 folder prefix
            collection_name: MongoDB collection name
            file_pattern: Pattern to match files
            file_type: Type of files (csv, excel, parquet)
            **read_kwargs: Additional arguments for pandas read function
            
        Returns:
            Number of files successfully loaded
        """
        files = self.list_files(s3_prefix)
        
        # Filter files by pattern
        import fnmatch
        pattern_files = [
            f for f in files 
            if fnmatch.fnmatch(Path(f).name, file_pattern)
        ]
        
        self.logger.info(
            f"Found {len(pattern_files)} files matching pattern {file_pattern}"
        )
        
        success_count = 0
        for file_key in pattern_files:
            if self.load_to_mongodb(file_key, collection_name, file_type, **read_kwargs):
                success_count += 1
        
        self.logger.info(
            f"Successfully loaded {success_count}/{len(pattern_files)} files"
        )
        return success_count
    
    def upload_to_s3(
        self,
        local_file_path: str,
        s3_key: str = None
    ) -> bool:
        """
        Upload local file to S3
        
        Args:
            local_file_path: Path to local file
            s3_key: S3 key. If None, uses filename with raw_data prefix
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if s3_key is None:
                prefix = self.config['aws']['s3_raw_data_prefix']
                s3_key = prefix + Path(local_file_path).name
            
            self.s3_client.upload_file(
                local_file_path,
                self.bucket_name,
                s3_key
            )
            self.logger.info(f"Uploaded {local_file_path} to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            self.logger.error(f"Error uploading to S3: {e}")
            return False


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Load data from S3 into MongoDB'
    )
    parser.add_argument(
        '--s3-key',
        type=str,
        help='S3 object key to load'
    )
    parser.add_argument(
        '--s3-prefix',
        type=str,
        help='S3 prefix/folder to load all files from'
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
        help='File type to load'
    )
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='*.csv',
        help='File pattern to match (when using --s3-prefix)'
    )
    
    args = parser.parse_args()
    
    loader = S3DataLoader()
    
    if args.s3_key:
        success = loader.load_to_mongodb(
            args.s3_key,
            args.collection,
            args.file_type
        )
        if success:
            print(f"✓ Successfully loaded {args.s3_key} into {args.collection}")
        else:
            print(f"✗ Failed to load {args.s3_key}")
            sys.exit(1)
    
    elif args.s3_prefix:
        count = loader.load_folder_to_mongodb(
            args.s3_prefix,
            args.collection,
            args.file_pattern,
            args.file_type
        )
        print(f"✓ Successfully loaded {count} files into {args.collection}")
    
    else:
        parser.error("Either --s3-key or --s3-prefix must be provided")


if __name__ == '__main__':
    main()

