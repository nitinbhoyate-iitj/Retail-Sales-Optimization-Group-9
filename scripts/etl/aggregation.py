"""
Data aggregation utilities using MongoDB aggregation pipeline
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.db_operations import MongoDBHandler
from utils.logger import setup_logger


class DataAggregator:
    """MongoDB aggregation pipeline utilities"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize Data Aggregator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.db_handler = MongoDBHandler()
        self.logger = setup_logger('data_aggregator')
    
    def aggregate_sales_by_period(
        self,
        collection_name: str,
        period: str = 'day',
        date_field: str = 'invoice_date'
    ) -> List[Dict]:
        """
        Aggregate sales by time period
        
        Args:
            collection_name: MongoDB collection name
            period: Time period ('day', 'week', 'month', 'year')
            date_field: Name of date field
            
        Returns:
            List of aggregation results
        """
        # Define date grouping based on period
        if period == 'day':
            date_group = {
                'year': {'$year': f'${date_field}'},
                'month': {'$month': f'${date_field}'},
                'day': {'$dayOfMonth': f'${date_field}'}
            }
        elif period == 'week':
            date_group = {
                'year': {'$isoWeekYear': f'${date_field}'},
                'week': {'$isoWeek': f'${date_field}'}
            }
        elif period == 'month':
            date_group = {
                'year': {'$year': f'${date_field}'},
                'month': {'$month': f'${date_field}'}
            }
        elif period == 'year':
            date_group = {
                'year': {'$year': f'${date_field}'}
            }
        else:
            raise ValueError(f"Invalid period: {period}")
        
        pipeline = [
            {
                '$group': {
                    '_id': date_group,
                    'total_sales': {'$sum': '$total_amount'},
                    'total_quantity': {'$sum': '$quantity'},
                    'transaction_count': {'$sum': 1},
                    'unique_customers': {'$addToSet': '$customer_id'},
                    'avg_transaction': {'$avg': '$total_amount'}
                }
            },
            {
                '$project': {
                    '_id': 1,
                    'total_sales': 1,
                    'total_quantity': 1,
                    'transaction_count': 1,
                    'unique_customers': {'$size': '$unique_customers'},
                    'avg_transaction': 1
                }
            },
            {
                '$sort': {'_id': 1}
            }
        ]
        
        results = self.db_handler.aggregate(collection_name, pipeline)
        self.logger.info(f"Aggregated sales by {period}: {len(results)} results")
        
        return results
    
    def aggregate_by_category(
        self,
        collection_name: str,
        category_field: str = 'category'
    ) -> List[Dict]:
        """
        Aggregate sales by category
        
        Args:
            collection_name: MongoDB collection name
            category_field: Name of category field
            
        Returns:
            List of aggregation results
        """
        pipeline = [
            {
                '$group': {
                    '_id': f'${category_field}',
                    'total_sales': {'$sum': '$total_amount'},
                    'total_quantity': {'$sum': '$quantity'},
                    'transaction_count': {'$sum': 1},
                    'unique_customers': {'$addToSet': '$customer_id'},
                    'avg_transaction': {'$avg': '$total_amount'},
                    'avg_price': {'$avg': '$price'}
                }
            },
            {
                '$project': {
                    'category': '$_id',
                    'total_sales': 1,
                    'total_quantity': 1,
                    'transaction_count': 1,
                    'unique_customers': {'$size': '$unique_customers'},
                    'avg_transaction': 1,
                    'avg_price': 1,
                    '_id': 0
                }
            },
            {
                '$sort': {'total_sales': -1}
            }
        ]
        
        results = self.db_handler.aggregate(collection_name, pipeline)
        self.logger.info(f"Aggregated by category: {len(results)} results")
        
        return results
    
    def aggregate_customer_rfm(
        self,
        collection_name: str,
        customer_field: str = 'customer_id',
        date_field: str = 'invoice_date',
        amount_field: str = 'total_amount'
    ) -> List[Dict]:
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics
        
        Args:
            collection_name: MongoDB collection name
            customer_field: Name of customer ID field
            date_field: Name of date field
            amount_field: Name of amount field
            
        Returns:
            List of RFM results
        """
        pipeline = [
            {
                '$group': {
                    '_id': f'${customer_field}',
                    'last_purchase': {'$max': f'${date_field}'},
                    'first_purchase': {'$min': f'${date_field}'},
                    'frequency': {'$sum': 1},
                    'monetary': {'$sum': f'${amount_field}'},
                    'avg_transaction': {'$avg': f'${amount_field}'}
                }
            },
            {
                '$project': {
                    'customer_id': '$_id',
                    'last_purchase': 1,
                    'first_purchase': 1,
                    'frequency': 1,
                    'monetary': 1,
                    'avg_transaction': 1,
                    '_id': 0
                }
            },
            {
                '$sort': {'monetary': -1}
            }
        ]
        
        results = self.db_handler.aggregate(collection_name, pipeline)
        self.logger.info(f"Calculated RFM for {len(results)} customers")
        
        return results
    
    def aggregate_top_products(
        self,
        collection_name: str,
        category_field: str = 'category',
        limit: int = 10
    ) -> List[Dict]:
        """
        Get top performing products/categories
        
        Args:
            collection_name: MongoDB collection name
            category_field: Name of category field
            limit: Number of top items to return
            
        Returns:
            List of top products
        """
        pipeline = [
            {
                '$group': {
                    '_id': f'${category_field}',
                    'total_sales': {'$sum': '$total_amount'},
                    'total_quantity': {'$sum': '$quantity'},
                    'transaction_count': {'$sum': 1}
                }
            },
            {
                '$sort': {'total_sales': -1}
            },
            {
                '$limit': limit
            },
            {
                '$project': {
                    'category': '$_id',
                    'total_sales': 1,
                    'total_quantity': 1,
                    'transaction_count': 1,
                    '_id': 0
                }
            }
        ]
        
        results = self.db_handler.aggregate(collection_name, pipeline)
        self.logger.info(f"Retrieved top {limit} products")
        
        return results
    
    def aggregate_by_demographics(
        self,
        collection_name: str,
        demographic_fields: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Aggregate sales by demographic fields
        
        Args:
            collection_name: MongoDB collection name
            demographic_fields: List of demographic fields
            
        Returns:
            Dictionary of aggregation results by demographic
        """
        if demographic_fields is None:
            demographic_fields = ['gender', 'age']
        
        results = {}
        
        for field in demographic_fields:
            pipeline = [
                {
                    '$group': {
                        '_id': f'${field}',
                        'total_sales': {'$sum': '$total_amount'},
                        'transaction_count': {'$sum': 1},
                        'avg_transaction': {'$avg': '$total_amount'},
                        'unique_customers': {'$addToSet': '$customer_id'}
                    }
                },
                {
                    '$project': {
                        field: '$_id',
                        'total_sales': 1,
                        'transaction_count': 1,
                        'avg_transaction': 1,
                        'unique_customers': {'$size': '$unique_customers'},
                        '_id': 0
                    }
                },
                {
                    '$sort': {'total_sales': -1}
                }
            ]
            
            results[field] = self.db_handler.aggregate(collection_name, pipeline)
            self.logger.info(f"Aggregated by {field}: {len(results[field])} results")
        
        return results
    
    def aggregate_payment_methods(
        self,
        collection_name: str,
        payment_field: str = 'payment_method'
    ) -> List[Dict]:
        """
        Aggregate sales by payment method
        
        Args:
            collection_name: MongoDB collection name
            payment_field: Name of payment method field
            
        Returns:
            List of aggregation results
        """
        pipeline = [
            {
                '$group': {
                    '_id': f'${payment_field}',
                    'total_sales': {'$sum': '$total_amount'},
                    'transaction_count': {'$sum': 1},
                    'avg_transaction': {'$avg': '$total_amount'}
                }
            },
            {
                '$project': {
                    'payment_method': '$_id',
                    'total_sales': 1,
                    'transaction_count': 1,
                    'avg_transaction': 1,
                    'percentage': {
                        '$multiply': [
                            {'$divide': ['$transaction_count', {'$sum': '$transaction_count'}]},
                            100
                        ]
                    },
                    '_id': 0
                }
            },
            {
                '$sort': {'total_sales': -1}
            }
        ]
        
        results = self.db_handler.aggregate(collection_name, pipeline)
        self.logger.info(f"Aggregated by payment method: {len(results)} results")
        
        return results
    
    def save_aggregation_to_collection(
        self,
        results: List[Dict],
        output_collection: str
    ) -> int:
        """
        Save aggregation results to a new collection
        
        Args:
            results: List of aggregation results
            output_collection: Name of output collection
            
        Returns:
            Number of documents inserted
        """
        if not results:
            self.logger.warning("No results to save")
            return 0
        
        df = pd.DataFrame(results)
        count = self.db_handler.insert_dataframe(df, output_collection)
        self.logger.info(f"Saved {count} aggregation results to {output_collection}")
        
        return count


def main():
    """Main function for testing"""
    aggregator = DataAggregator()
    print("✓ Data Aggregator initialized successfully")


if __name__ == '__main__':
    main()

