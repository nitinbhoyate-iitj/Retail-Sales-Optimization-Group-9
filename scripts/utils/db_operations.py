"""
MongoDB database operations handler
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import BulkWriteError, ConnectionFailure
from .config_loader import get_database, load_config
from .logger import setup_logger


class MongoDBHandler:
    """Handler for MongoDB operations"""
    
    def __init__(self, db_name: str = None):
        """
        Initialize MongoDB handler
        
        Args:
            db_name: Database name. If None, uses name from config.
        """
        self.config = load_config()
        self.db = get_database(db_name)
        self.logger = setup_logger('mongodb_handler')
        self.logger.info(f"Connected to MongoDB database: {self.db.name}")
    
    def insert_dataframe(
        self,
        df: pd.DataFrame,
        collection_name: str,
        batch_size: int = 1000
    ) -> int:
        """
        Insert pandas DataFrame into MongoDB collection
        
        Args:
            df: DataFrame to insert
            collection_name: Name of the collection
            batch_size: Number of documents to insert at once
            
        Returns:
            Number of documents inserted
        """
        collection = self.db[collection_name]
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                result = collection.insert_many(batch, ordered=False)
                total_inserted += len(result.inserted_ids)
            except BulkWriteError as e:
                # Continue even if some documents fail
                total_inserted += e.details['nInserted']
                self.logger.warning(
                    f"Batch insert warning: {e.details['nInserted']} inserted, "
                    f"{len(e.details['writeErrors'])} errors"
                )
        
        self.logger.info(
            f"Inserted {total_inserted} documents into {collection_name}"
        )
        return total_inserted
    
    def read_to_dataframe(
        self,
        collection_name: str,
        query: Dict = None,
        projection: Dict = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Read MongoDB collection into pandas DataFrame
        
        Args:
            collection_name: Name of the collection
            query: MongoDB query filter
            projection: Fields to include/exclude
            limit: Maximum number of documents to return
            
        Returns:
            DataFrame containing the data
        """
        collection = self.db[collection_name]
        
        if query is None:
            query = {}
        
        cursor = collection.find(query, projection)
        
        if limit:
            cursor = cursor.limit(limit)
        
        df = pd.DataFrame(list(cursor))
        
        # Remove MongoDB _id field if present
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        
        self.logger.info(
            f"Read {len(df)} documents from {collection_name}"
        )
        return df
    
    def update_documents(
        self,
        collection_name: str,
        query: Dict,
        update: Dict,
        upsert: bool = False
    ) -> int:
        """
        Update documents in a collection
        
        Args:
            collection_name: Name of the collection
            query: Query to match documents
            update: Update operations
            upsert: Whether to insert if no match found
            
        Returns:
            Number of documents modified
        """
        collection = self.db[collection_name]
        result = collection.update_many(query, update, upsert=upsert)
        
        self.logger.info(
            f"Updated {result.modified_count} documents in {collection_name}"
        )
        return result.modified_count
    
    def delete_documents(
        self,
        collection_name: str,
        query: Dict
    ) -> int:
        """
        Delete documents from a collection
        
        Args:
            collection_name: Name of the collection
            query: Query to match documents to delete
            
        Returns:
            Number of documents deleted
        """
        collection = self.db[collection_name]
        result = collection.delete_many(query)
        
        self.logger.info(
            f"Deleted {result.deleted_count} documents from {collection_name}"
        )
        return result.deleted_count
    
    def create_index(
        self,
        collection_name: str,
        index_fields: List[tuple],
        index_name: str = None
    ):
        """
        Create index on a collection
        
        Args:
            collection_name: Name of the collection
            index_fields: List of (field, direction) tuples
            index_name: Optional name for the index
        """
        collection = self.db[collection_name]
        result = collection.create_index(index_fields, name=index_name)
        
        self.logger.info(
            f"Created index '{result}' on {collection_name}"
        )
        return result
    
    def aggregate(
        self,
        collection_name: str,
        pipeline: List[Dict]
    ) -> List[Dict]:
        """
        Run aggregation pipeline on a collection
        
        Args:
            collection_name: Name of the collection
            pipeline: Aggregation pipeline
            
        Returns:
            List of aggregation results
        """
        collection = self.db[collection_name]
        results = list(collection.aggregate(pipeline))
        
        self.logger.info(
            f"Aggregation on {collection_name} returned {len(results)} results"
        )
        return results
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        Get statistics for a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary containing collection statistics
        """
        collection = self.db[collection_name]
        
        stats = {
            'document_count': collection.count_documents({}),
            'indexes': collection.index_information(),
            'collection_name': collection_name
        }
        
        return stats
    
    def drop_collection(self, collection_name: str):
        """
        Drop a collection
        
        Args:
            collection_name: Name of the collection to drop
        """
        self.db[collection_name].drop()
        self.logger.warning(f"Dropped collection: {collection_name}")
    
    def test_connection(self) -> bool:
        """
        Test MongoDB connection
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.db.client.server_info()
            self.logger.info("MongoDB connection successful")
            return True
        except ConnectionFailure:
            self.logger.error("MongoDB connection failed")
            return False

