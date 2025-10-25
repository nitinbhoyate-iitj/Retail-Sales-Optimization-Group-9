"""
Configuration loader and client initialization utilities
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import boto3
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default path.
        
    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_s3_client():
    """
    Initialize and return AWS S3 client
    
    Returns:
        boto3 S3 client
    """
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
    else:
        # Use default credentials (IAM role, AWS CLI config, etc.)
        s3_client = boto3.client('s3', region_name=aws_region)
    
    return s3_client


def get_mongo_client(config: Dict[str, Any] = None):
    """
    Initialize and return MongoDB client
    
    Args:
        config: Configuration dictionary. If None, loads from default config.
        
    Returns:
        pymongo MongoClient
    """
    if config is None:
        config = load_config()
    
    # Try to get MongoDB URI from environment variable first
    mongodb_uri = os.getenv('MONGODB_URI')
    
    if mongodb_uri:
        # Use URI from environment
        client = MongoClient(mongodb_uri)
    else:
        # Build connection string from config
        mongo_config = config['mongodb']
        host = mongo_config.get('host', 'localhost')
        port = mongo_config.get('port', 27017)
        username = os.getenv('MONGODB_USERNAME')
        password = os.getenv('MONGODB_PASSWORD')
        
        if username and password:
            connection_string = f"mongodb://{username}:{password}@{host}:{port}/"
        else:
            connection_string = f"mongodb://{host}:{port}/"
        
        # Additional connection parameters
        connection_params = mongo_config.get('connection', {})
        
        client = MongoClient(
            connection_string,
            maxPoolSize=connection_params.get('max_pool_size', 50),
            serverSelectionTimeoutMS=connection_params.get('timeout_ms', 5000)
        )
    
    return client


def get_database(db_name: str = None):
    """
    Get MongoDB database instance
    
    Args:
        db_name: Database name. If None, uses name from config.
        
    Returns:
        MongoDB database instance
    """
    config = load_config()
    
    if db_name is None:
        db_name = os.getenv('MONGODB_DATABASE', config['mongodb']['database'])
    
    client = get_mongo_client(config)
    return client[db_name]

