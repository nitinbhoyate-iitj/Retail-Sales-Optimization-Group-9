"""
Feature engineering for retail sales data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_loader import load_config
from utils.logger import setup_logger


class FeatureEngineer:
    """Feature engineering for sales forecasting and analysis"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize Feature Engineer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.feature_config = self.config['features']
        self.logger = setup_logger('feature_engineer')
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        lag_periods: List[int] = None,
        group_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            df: Input DataFrame
            value_col: Column to create lags for
            lag_periods: List of lag periods
            group_cols: Columns to group by (e.g., product_id, store_id)
            
        Returns:
            DataFrame with lag features
        """
        if lag_periods is None:
            lag_periods = self.feature_config['lag_periods']
        
        df_features = df.copy()
        
        for lag in lag_periods:
            col_name = f'{value_col}_lag_{lag}'
            
            if group_cols:
                df_features[col_name] = df_features.groupby(group_cols)[value_col].shift(lag)
            else:
                df_features[col_name] = df_features[value_col].shift(lag)
            
            self.logger.info(f"Created lag feature: {col_name}")
        
        return df_features
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        windows: List[int] = None,
        group_cols: List[str] = None,
        agg_funcs: List[str] = None
    ) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Input DataFrame
            value_col: Column to calculate rolling statistics for
            windows: List of window sizes
            group_cols: Columns to group by
            agg_funcs: Aggregation functions ('mean', 'std', 'min', 'max', 'sum')
            
        Returns:
            DataFrame with rolling features
        """
        if windows is None:
            windows = self.feature_config['rolling_windows']
        
        if agg_funcs is None:
            agg_funcs = ['mean', 'std']
        
        df_features = df.copy()
        
        for window in windows:
            for func in agg_funcs:
                col_name = f'{value_col}_rolling_{func}_{window}'
                
                if group_cols:
                    df_features[col_name] = df_features.groupby(group_cols)[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).agg(func)
                    )
                else:
                    df_features[col_name] = df_features[value_col].rolling(
                        window=window, min_periods=1
                    ).agg(func)
                
                self.logger.info(f"Created rolling feature: {col_name}")
        
        return df_features
    
    def create_date_features(
        self,
        df: pd.DataFrame,
        date_col: str
    ) -> pd.DataFrame:
        """
        Extract date-based features
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with date features
        """
        df_features = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features[date_col]):
            df_features[date_col] = pd.to_datetime(df_features[date_col])
        
        # Extract date components
        df_features['year'] = df_features[date_col].dt.year
        df_features['month'] = df_features[date_col].dt.month
        df_features['day'] = df_features[date_col].dt.day
        df_features['day_of_week'] = df_features[date_col].dt.dayofweek
        df_features['day_of_year'] = df_features[date_col].dt.dayofyear
        df_features['week_of_year'] = df_features[date_col].dt.isocalendar().week
        df_features['quarter'] = df_features[date_col].dt.quarter
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        df_features['is_month_start'] = df_features[date_col].dt.is_month_start.astype(int)
        df_features['is_month_end'] = df_features[date_col].dt.is_month_end.astype(int)
        
        # Season
        df_features['season'] = df_features['month'].apply(
            lambda x: 1 if x in [12, 1, 2] else (  # Winter
                      2 if x in [3, 4, 5] else (    # Spring
                      3 if x in [6, 7, 8] else 4))  # Summer, Fall
        )
        
        self.logger.info(f"Created date features from {date_col}")
        
        return df_features
    
    def create_categorical_features(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        encoding_type: str = 'label'
    ) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            cat_cols: List of categorical columns
            encoding_type: Type of encoding ('label', 'onehot', 'frequency')
            
        Returns:
            DataFrame with encoded features
        """
        df_features = df.copy()
        
        for col in cat_cols:
            if col not in df_features.columns:
                continue
            
            if encoding_type == 'label':
                # Label encoding
                df_features[f'{col}_encoded'] = pd.Categorical(df_features[col]).codes
            
            elif encoding_type == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df_features[col], prefix=col, drop_first=True)
                df_features = pd.concat([df_features, dummies], axis=1)
            
            elif encoding_type == 'frequency':
                # Frequency encoding
                freq_map = df_features[col].value_counts(normalize=True).to_dict()
                df_features[f'{col}_freq'] = df_features[col].map(freq_map)
            
            self.logger.info(f"Encoded categorical feature: {col} using {encoding_type}")
        
        return df_features
    
    def create_aggregate_features(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        agg_col: str,
        agg_funcs: List[str] = None
    ) -> pd.DataFrame:
        """
        Create aggregate features by grouping
        
        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_col: Column to aggregate
            agg_funcs: Aggregation functions
            
        Returns:
            DataFrame with aggregate features
        """
        if agg_funcs is None:
            agg_funcs = ['mean', 'sum', 'std', 'min', 'max', 'count']
        
        df_features = df.copy()
        
        for func in agg_funcs:
            col_name = f'{agg_col}_{func}_by_{"_".join(group_cols)}'
            
            agg_result = df_features.groupby(group_cols)[agg_col].transform(func)
            df_features[col_name] = agg_result
            
            self.logger.info(f"Created aggregate feature: {col_name}")
        
        return df_features
    
    def create_ratio_features(
        self,
        df: pd.DataFrame,
        numerator_col: str,
        denominator_col: str,
        feature_name: str = None
    ) -> pd.DataFrame:
        """
        Create ratio features
        
        Args:
            df: Input DataFrame
            numerator_col: Numerator column
            denominator_col: Denominator column
            feature_name: Name for the ratio feature
            
        Returns:
            DataFrame with ratio feature
        """
        df_features = df.copy()
        
        if feature_name is None:
            feature_name = f'{numerator_col}_to_{denominator_col}_ratio'
        
        # Avoid division by zero
        df_features[feature_name] = np.where(
            df_features[denominator_col] != 0,
            df_features[numerator_col] / df_features[denominator_col],
            0
        )
        
        self.logger.info(f"Created ratio feature: {feature_name}")
        
        return df_features
    
    def create_customer_features(
        self,
        df: pd.DataFrame,
        customer_col: str = 'customer_id',
        date_col: str = 'invoice_date',
        amount_col: str = 'total_amount'
    ) -> pd.DataFrame:
        """
        Create customer-level features (RFM and more)
        
        Args:
            df: Input DataFrame
            customer_col: Customer ID column
            date_col: Date column
            amount_col: Amount column
            
        Returns:
            DataFrame with customer features
        """
        df_features = df.copy()
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_features[date_col]):
            df_features[date_col] = pd.to_datetime(df_features[date_col])
        
        reference_date = df_features[date_col].max() + timedelta(days=1)
        
        # RFM features
        rfm = df_features.groupby(customer_col).agg({
            date_col: lambda x: (reference_date - x.max()).days,  # Recency
            'invoice_no': 'count',  # Frequency
            amount_col: 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = [customer_col, 'recency', 'frequency', 'monetary']
        
        # Merge back
        df_features = df_features.merge(rfm, on=customer_col, how='left')
        
        # Additional customer features
        if 'quantity' in df_features.columns:
            customer_stats = df_features.groupby(customer_col).agg({
                amount_col: ['mean', 'std', 'min', 'max'],
                'quantity': ['mean', 'sum']
            }).reset_index()
            
            customer_stats.columns = [
                customer_col,
                'customer_avg_amount',
                'customer_std_amount',
                'customer_min_amount',
                'customer_max_amount',
                'customer_avg_quantity',
                'customer_total_quantity'
            ]
        else:
            customer_stats = df_features.groupby(customer_col).agg({
                amount_col: ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            customer_stats.columns = [
                customer_col,
                'customer_avg_amount',
                'customer_std_amount',
                'customer_min_amount',
                'customer_max_amount'
            ]
        
        df_features = df_features.merge(customer_stats, on=customer_col, how='left')
        
        self.logger.info("Created customer features (RFM and statistics)")
        
        return df_features
    
    def create_product_features(
        self,
        df: pd.DataFrame,
        category_col: str = 'category',
        amount_col: str = 'total_amount'
    ) -> pd.DataFrame:
        """
        Create product/category-level features
        
        Args:
            df: Input DataFrame
            category_col: Category column
            amount_col: Amount column
            
        Returns:
            DataFrame with product features
        """
        df_features = df.copy()
        
        # Category statistics
        if 'quantity' in df_features.columns:
            category_stats = df_features.groupby(category_col).agg({
                amount_col: ['mean', 'sum', 'count'],
                'quantity': ['mean', 'sum']
            }).reset_index()
            
            category_stats.columns = [
                category_col,
                'category_avg_amount',
                'category_total_amount',
                'category_transaction_count',
                'category_avg_quantity',
                'category_total_quantity'
            ]
        else:
            category_stats = df_features.groupby(category_col).agg({
                amount_col: ['mean', 'sum', 'count']
            }).reset_index()
            
            category_stats.columns = [
                category_col,
                'category_avg_amount',
                'category_total_amount',
                'category_transaction_count'
            ]
        
        df_features = df_features.merge(category_stats, on=category_col, how='left')
        
        # Category rank by sales
        category_rank = df_features.groupby(category_col)[amount_col].sum().rank(ascending=False).to_dict()
        df_features['category_rank'] = df_features[category_col].map(category_rank)
        
        self.logger.info("Created product/category features")
        
        return df_features


def main():
    """Main function for testing"""
    engineer = FeatureEngineer()
    print("✓ Feature Engineer initialized successfully")


if __name__ == '__main__':
    main()

