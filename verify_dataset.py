#!/usr/bin/env python3
"""
Dataset verification script
Verifies the Kaggle Customer Shopping Dataset structure and compatibility
"""

import pandas as pd
import sys
from pathlib import Path

def verify_dataset(file_path):
    """Verify dataset structure and contents"""
    
    print("="*80)
    print("DATASET VERIFICATION")
    print("="*80)
    print()
    
    # Expected columns
    expected_columns = [
        'invoice_no',
        'customer_id', 
        'gender',
        'age',
        'category',
        'quantity',
        'price',
        'payment_method',
        'invoice_date',
        'shopping_mall'
    ]
    
    try:
        # Load dataset
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded successfully")
        print()
        
        # Check shape
        print(f"Dataset Shape: {df.shape}")
        print(f"  - Rows (transactions): {df.shape[0]:,}")
        print(f"  - Columns (attributes): {df.shape[1]}")
        print()
        
        # Check columns
        print("Column Verification:")
        actual_columns = df.columns.tolist()
        print(f"  Expected: {expected_columns}")
        print(f"  Actual:   {actual_columns}")
        
        if actual_columns == expected_columns:
            print("  ✓ All columns match!")
        else:
            missing = set(expected_columns) - set(actual_columns)
            extra = set(actual_columns) - set(expected_columns)
            if missing:
                print(f"  ✗ Missing columns: {missing}")
            if extra:
                print(f"  ⚠ Extra columns: {extra}")
        print()
        
        # Data types
        print("Data Types:")
        for col in df.columns:
            print(f"  {col:20} -> {df[col].dtype}")
        print()
        
        # Missing values
        print("Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("  ✓ No missing values")
        else:
            for col in missing[missing > 0].index:
                pct = (missing[col] / len(df)) * 100
                print(f"  {col:20} -> {missing[col]:,} ({pct:.2f}%)")
        print()
        
        # Sample data
        print("Sample Records:")
        print(df.head(3).to_string())
        print()
        
        # Basic statistics
        print("Numeric Columns Summary:")
        print(df[['age', 'quantity', 'price']].describe())
        print()
        
        # Categorical columns
        print("Categorical Columns:")
        categorical_cols = ['gender', 'category', 'payment_method', 'shopping_mall']
        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                print(f"  {col:20} -> {unique_count} unique values")
                print(f"    Top 3: {df[col].value_counts().head(3).to_dict()}")
        print()
        
        # Date range
        print("Date Range:")
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
        print(f"  From: {df['invoice_date'].min()}")
        print(f"  To:   {df['invoice_date'].max()}")
        print(f"  Span: {(df['invoice_date'].max() - df['invoice_date'].min()).days} days")
        print()
        
        # Calculated field
        df['total_amount'] = df['price'] * df['quantity']
        total_sales = df['total_amount'].sum()
        print("Sales Metrics:")
        print(f"  Total Sales: ${total_sales:,.2f}")
        print(f"  Average Transaction: ${df['total_amount'].mean():,.2f}")
        print(f"  Unique Customers: {df['customer_id'].nunique():,}")
        print()
        
        print("="*80)
        print("✓ DATASET VERIFICATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print()
        print("Next Steps:")
        print("1. Run the pipeline:")
        print("   python run_pipeline.py --data-source local --data-path dataset/customer_shopping_data.csv")
        print()
        print("2. Or run individual steps:")
        print("   python scripts/data_ingestion/load_from_local.py \\")
        print("     --file dataset/customer_shopping_data.csv \\")
        print("     --collection raw_sales \\")
        print("     --kaggle-dataset")
        print()
        
        return True
        
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        print()
        print("Please ensure the dataset is placed in the correct location:")
        print("  dataset/customer_shopping_data.csv")
        return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Get project root
    project_root = Path(__file__).parent
    dataset_path = project_root / "dataset" / "customer_shopping_data.csv"
    
    success = verify_dataset(dataset_path)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

