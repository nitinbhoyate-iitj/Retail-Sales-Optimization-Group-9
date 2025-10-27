#!/usr/bin/env python3
"""
End-to-end pipeline runner for Retail Sales Optimization
"""

import sys
from pathlib import Path
import argparse

# Add scripts directory to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'scripts'))

from utils.logger import setup_logger
from data_ingestion.data_loader import DataLoader
from data_cleaning.cleaning_pipeline import CleaningPipeline
from etl.transformation_pipeline import TransformationPipeline

def main():
    """Run the complete data pipeline"""
    
    parser = argparse.ArgumentParser(
        description='Run complete data pipeline for Retail Sales Optimization'
    )
    parser.add_argument(
        '--data-source',
        type=str,
        default='local',
        choices=['s3', 'local'],
        help='Data source (s3 or local)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='dataset/customer_shopping_data.csv',
        help='Path to data file (S3 key or local path)'
    )
    parser.add_argument(
        '--skip-ingestion',
        action='store_true',
        help='Skip data ingestion step'
    )
    parser.add_argument(
        '--skip-cleaning',
        action='store_true',
        help='Skip data cleaning step'
    )
    parser.add_argument(
        '--skip-transformation',
        action='store_true',
        help='Skip data transformation step'
    )
    
    args = parser.parse_args()
    
    logger = setup_logger('pipeline_runner')
    
    print("="*80)
    print("RETAIL SALES OPTIMIZATION - DATA PIPELINE")
    print("="*80)
    print()
    
    # Step 1: Data Ingestion
    if not args.skip_ingestion:
        print("Step 1: Data Ingestion")
        print("-" * 80)
        
        loader = DataLoader()
        success = loader.load_kaggle_shopping_dataset(
            source=args.data_source,
            source_path=args.data_path,
            collection_name='raw_sales'
        )
        
        if success:
            print("✓ Data ingestion completed successfully")
        else:
            print("✗ Data ingestion failed")
            return 1
        
        print()
    else:
        print("⊘ Skipping data ingestion")
        print()
    
    # Step 2: Data Cleaning
    if not args.skip_cleaning:
        print("Step 2: Data Cleaning")
        print("-" * 80)
        
        cleaner = CleaningPipeline()
        success = cleaner.clean_shopping_dataset(
            input_collection='raw_sales',
            output_collection='cleaned_sales'
        )
        
        if success:
            print("✓ Data cleaning completed successfully")
        else:
            print("✗ Data cleaning failed")
            return 1
        
        print()
    else:
        print("⊘ Skipping data cleaning")
        print()
    
    # Step 3: Feature Engineering & Transformation
    if not args.skip_transformation:
        print("Step 3: Feature Engineering & Transformation")
        print("-" * 80)
        
        transformer = TransformationPipeline()
        success = transformer.transform_shopping_dataset(
            input_collection='cleaned_sales',
            output_collection='transformed_sales'
        )
        
        if success:
            print("✓ Data transformation completed successfully")
        else:
            print("✗ Data transformation failed")
            return 1
        
        # Create aggregations
        print("\nCreating aggregated views...")
        agg_success = transformer.create_aggregated_views('transformed_sales')
        
        if agg_success:
            print("✓ Aggregations created successfully")
        else:
            print("⚠ Aggregation creation had issues (non-fatal)")
        
        print()
    else:
        print("⊘ Skipping data transformation")
        print()
    
    # Pipeline Complete
    print("="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print()
    print("Next Steps:")
    print("1. Launch Jupyter notebooks for analysis:")
    print("   jupyter notebook notebooks/eda/EDA_sales.ipynb")
    print()
    print("2. Run the dashboard:")
    print("   cd dashboards && streamlit run app.py")
    print()
    print("3. Build ML models:")
    print("   jupyter notebook notebooks/modeling/modeling_forecasting.ipynb")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

