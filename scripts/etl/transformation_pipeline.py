"""
End-to-end transformation pipeline for retail sales data
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.db_operations import MongoDBHandler
from utils.logger import setup_logger
from feature_engineering import FeatureEngineer


class TransformationPipeline:
    """Complete transformation pipeline from cleaned data to analysis-ready features"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize Transformation Pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.db_handler = MongoDBHandler()
        self.feature_engineer = FeatureEngineer(config_path)
        self.logger = setup_logger('transformation_pipeline')
    
    def transform_shopping_dataset(
        self,
        input_collection: str = 'cleaned_sales',
        output_collection: str = 'transformed_sales'
    ) -> bool:
        """
        Transform the Kaggle Customer Shopping Dataset
        
        Args:
            input_collection: Source MongoDB collection
            output_collection: Destination MongoDB collection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting transformation pipeline for {input_collection}")
            
            # Load cleaned data
            df = self.db_handler.read_to_dataframe(input_collection)
            self.logger.info(f"Loaded {len(df)} records from {input_collection}")
            
            # Sort by date for time series operations
            if 'invoice_date' in df.columns:
                df = df.sort_values('invoice_date').reset_index(drop=True)
            
            # Step 1: Date features
            if 'invoice_date' in df.columns:
                df = self.feature_engineer.create_date_features(df, 'invoice_date')
                self.logger.info("Created date features")
            
            # Step 2: Customer features (RFM analysis)
            if 'customer_id' in df.columns and 'total_amount' in df.columns:
                df = self.feature_engineer.create_customer_features(
                    df,
                    customer_col='customer_id',
                    date_col='invoice_date',
                    amount_col='total_amount'
                )
                self.logger.info("Created customer features")
            
            # Step 3: Product/Category features
            if 'category' in df.columns and 'total_amount' in df.columns:
                df = self.feature_engineer.create_product_features(
                    df,
                    category_col='category',
                    amount_col='total_amount'
                )
                self.logger.info("Created product/category features")
            
            # Step 4: Aggregate features by customer
            if 'customer_id' in df.columns and 'total_amount' in df.columns:
                df = self.feature_engineer.create_aggregate_features(
                    df,
                    group_cols=['customer_id'],
                    agg_col='total_amount',
                    agg_funcs=['mean', 'sum', 'count']
                )
                self.logger.info("Created customer aggregate features")
            
            # Step 5: Aggregate features by category
            if 'category' in df.columns and 'total_amount' in df.columns:
                df = self.feature_engineer.create_aggregate_features(
                    df,
                    group_cols=['category'],
                    agg_col='total_amount',
                    agg_funcs=['mean', 'sum']
                )
                self.logger.info("Created category aggregate features")
            
            # Step 6: Time-based lag and rolling features
            # Group by customer and create lag features
            if 'customer_id' in df.columns and 'total_amount' in df.columns:
                df = df.sort_values(['customer_id', 'invoice_date'])
                df = self.feature_engineer.create_lag_features(
                    df,
                    value_col='total_amount',
                    lag_periods=[1, 7, 14],
                    group_cols=['customer_id']
                )
                self.logger.info("Created lag features")
            
            # Step 7: Rolling window features
            if 'customer_id' in df.columns and 'total_amount' in df.columns:
                df = self.feature_engineer.create_rolling_features(
                    df,
                    value_col='total_amount',
                    windows=[7, 14, 30],
                    group_cols=['customer_id'],
                    agg_funcs=['mean', 'std']
                )
                self.logger.info("Created rolling window features")
            
            # Step 8: Categorical encoding
            cat_cols = ['gender', 'category', 'payment_method', 'shopping_mall']
            df = self.feature_engineer.create_categorical_features(
                df,
                cat_cols=[col for col in cat_cols if col in df.columns],
                encoding_type='label'
            )
            self.logger.info("Encoded categorical features")
            
            # Step 9: Additional business logic features
            if 'price' in df.columns and 'quantity' in df.columns:
                # Price per unit category statistics
                avg_price_by_category = df.groupby('category')['price'].mean().to_dict()
                df['price_vs_category_avg'] = df.apply(
                    lambda x: x['price'] / avg_price_by_category.get(x['category'], 1),
                    axis=1
                )
            
            # Customer lifetime value proxy
            if 'customer_id' in df.columns and 'total_amount' in df.columns:
                customer_ltv = df.groupby('customer_id')['total_amount'].sum().to_dict()
                df['customer_ltv'] = df['customer_id'].map(customer_ltv)
            
            self.logger.info(f"Transformation completed. Final shape: {df.shape}")
            
            # Save to MongoDB
            count = self.db_handler.insert_dataframe(df, output_collection)
            self.logger.info(
                f"Saved {count} transformed records to {output_collection}"
            )
            
            # Also save to S3 as Parquet if configured
            try:
                local_path = Path(__file__).parent.parent.parent / "dataset" / "transformed"
                local_path.mkdir(parents=True, exist_ok=True)
                
                parquet_file = local_path / f"{output_collection}.parquet"
                df.to_parquet(parquet_file, index=False)
                self.logger.info(f"Saved transformed data to {parquet_file}")
            except Exception as e:
                self.logger.warning(f"Could not save parquet file: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in transformation pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_aggregated_views(
        self,
        input_collection: str = 'transformed_sales',
        output_prefix: str = 'agg'
    ) -> bool:
        """
        Create pre-aggregated views for reporting
        
        Args:
            input_collection: Source MongoDB collection
            output_prefix: Prefix for output collections
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Creating aggregated views")
            
            df = self.db_handler.read_to_dataframe(input_collection)
            
            # Daily aggregation
            if 'invoice_date' in df.columns:
                daily_agg = df.groupby(pd.Grouper(key='invoice_date', freq='D')).agg({
                    'total_amount': ['sum', 'mean', 'count'],
                    'quantity': ['sum', 'mean'] if 'quantity' in df.columns else [],
                    'customer_id': 'nunique' if 'customer_id' in df.columns else []
                }).reset_index()
                
                daily_agg.columns = [
                    'date', 'total_sales', 'avg_transaction',
                    'transaction_count', 'total_quantity',
                    'avg_quantity', 'unique_customers'
                ]
                
                self.db_handler.insert_dataframe(
                    daily_agg,
                    f'{output_prefix}_daily_sales'
                )
                self.logger.info("Created daily aggregation")
            
            # Category aggregation
            if 'category' in df.columns:
                category_agg = df.groupby('category').agg({
                    'total_amount': ['sum', 'mean', 'count'],
                    'quantity': ['sum', 'mean'] if 'quantity' in df.columns else [],
                    'customer_id': 'nunique' if 'customer_id' in df.columns else []
                }).reset_index()
                
                category_agg.columns = [
                    'category', 'total_sales', 'avg_transaction',
                    'transaction_count', 'total_quantity',
                    'avg_quantity', 'unique_customers'
                ]
                
                self.db_handler.insert_dataframe(
                    category_agg,
                    f'{output_prefix}_category_sales'
                )
                self.logger.info("Created category aggregation")
            
            # Customer segmentation
            if 'customer_id' in df.columns:
                customer_agg = df.groupby('customer_id').agg({
                    'total_amount': ['sum', 'mean', 'count'],
                    'invoice_date': ['min', 'max'] if 'invoice_date' in df.columns else []
                }).reset_index()
                
                customer_agg.columns = [
                    'customer_id', 'total_spent', 'avg_transaction',
                    'transaction_count', 'first_purchase', 'last_purchase'
                ]
                
                self.db_handler.insert_dataframe(
                    customer_agg,
                    f'{output_prefix}_customer_summary'
                )
                self.logger.info("Created customer aggregation")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating aggregated views: {e}")
            return False


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Data transformation pipeline'
    )
    parser.add_argument(
        '--input-collection',
        type=str,
        default='cleaned_sales',
        help='Input MongoDB collection name'
    )
    parser.add_argument(
        '--output-collection',
        type=str,
        default='transformed_sales',
        help='Output MongoDB collection name'
    )
    parser.add_argument(
        '--create-aggregations',
        action='store_true',
        help='Also create aggregated views'
    )
    
    args = parser.parse_args()
    
    pipeline = TransformationPipeline()
    
    success = pipeline.transform_shopping_dataset(
        args.input_collection,
        args.output_collection
    )
    
    if success:
        print(f"✓ Successfully transformed data from {args.input_collection} to {args.output_collection}")
        
        if args.create_aggregations:
            agg_success = pipeline.create_aggregated_views(args.output_collection)
            if agg_success:
                print("✓ Successfully created aggregated views")
    else:
        print("✗ Transformation pipeline failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

