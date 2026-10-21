"""
Feature Engineering Module for DemandWise
=========================================
Computes leakage-safe retrospective rolling features and exact 7-day forward demand-surge targets.
All features for prediction date t are calculated strictly using information available on or before date t.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

SURGE_THRESHOLD_MULTIPLIER: float = 1.30  # Configurable 30% surge threshold
MIN_HISTORY_DAYS: int = 31

NUMERIC_FEATURES: List[str] = [
    'sales_today',
    'past_7d_avg',
    'past_14d_avg',
    'past_30d_avg',
    'past_7d_std',
    'sales_7d_ago',
    'sales_14d_ago',
    'sales_30d_ago',
    'sales_vs_7d_avg',
    'ratio_7d_14d',
    'ratio_7d_30d',
    'coef_variation_7d',
    'onpromotion',
    'promo_7d',
    'promo_delta',
    'is_weekend',
    'month'
]

CATEGORICAL_FEATURES: List[str] = [
    'category',
    'store_name',
    'day_of_week'
]

ALL_MODEL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def compute_features_and_target(
    df: pd.DataFrame,
    surge_multiplier: float = SURGE_THRESHOLD_MULTIPLIER
) -> pd.DataFrame:
    """
    Computes retrospective features (<= t) and future surge target ([t+1..t+7])
    independently for each (store_name, category) time-series.
    
    Leakage Protection:
    - Lagged and rolling features use strictly past and current values.
    - Forward sales are strictly used for target calculation and never included in features.
    - Rows with incomplete future 7-day target windows have target=NaN.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.sort_values(['store_name', 'category', 'date']).reset_index(drop=True)
    
    grouped = df.groupby(['store_name', 'category'], observed=True)
    
    # 1. Base Rolling Statistics strictly <= t
    df['sales_today'] = df['sales'].astype(float)
    df['past_7d_avg'] = grouped['sales'].transform(lambda s: s.rolling(7, min_periods=7).mean())
    df['past_14d_avg'] = grouped['sales'].transform(lambda s: s.rolling(14, min_periods=14).mean())
    df['past_30d_avg'] = grouped['sales'].transform(lambda s: s.rolling(30, min_periods=30).mean())
    df['past_7d_std'] = grouped['sales'].transform(lambda s: s.rolling(7, min_periods=7).std()).fillna(0.0)
    
    # 2. Lagged Sales (Historical points)
    df['sales_7d_ago'] = grouped['sales'].shift(7)
    df['sales_14d_ago'] = grouped['sales'].shift(14)
    df['sales_30d_ago'] = grouped['sales'].shift(30)
    
    # 3. Rolling Promotions
    df['onpromotion'] = df['promotion'].astype(int)
    df['promo_7d'] = grouped['promotion'].transform(lambda s: s.rolling(7, min_periods=7).mean())
    df['promo_delta'] = df['onpromotion'].astype(float) - df['promo_7d'].fillna(0.0)
    
    # 4. Engineered Ratios & Dynamics (strictly <= t)
    df['sales_vs_7d_avg'] = (df['sales_today'] - df['past_7d_avg']) / (df['past_7d_avg'] + 1e-5)
    df['ratio_7d_14d'] = df['past_7d_avg'] / (df['past_14d_avg'] + 1e-5)
    df['ratio_7d_30d'] = df['past_7d_avg'] / (df['past_30d_avg'] + 1e-5)
    df['coef_variation_7d'] = df['past_7d_std'] / (df['past_7d_avg'] + 1e-5)
    
    # 5. Calendar Context
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['month'] = df['date'].dt.month
    
    # Ensure categorical columns are strings
    df['category'] = df['category'].astype(str)
    df['store_name'] = df['store_name'].astype(str)
    
    # 6. Forward Target Formulation: Average sales in [t+1 .. t+7]
    df['future_7d_avg'] = grouped['sales'].transform(
        lambda s: s.iloc[::-1].rolling(7, min_periods=7).mean().iloc[::-1].shift(-1)
    )
    
    # Exclude rows without complete windows or with a non-positive baseline.
    valid_target = (
        df['future_7d_avg'].notna() &
        df['past_7d_avg'].notna() &
        (df['past_7d_avg'] > 0.0)
    )
    is_surge = (
        (df['future_7d_avg'] >= surge_multiplier * df['past_7d_avg']) &
        (df['future_7d_avg'] > df['past_7d_avg'])
    )
    df['demand_surge'] = np.nan
    df.loc[valid_target, 'demand_surge'] = is_surge.loc[valid_target].astype(float)
    
    return df


def get_feature_matrix_for_inference(
    df: pd.DataFrame,
    store_name: str,
    category: Optional[str] = None
) -> pd.DataFrame:
    """
    Extracts the latest feature-complete row(s) for a given store and category
    to perform live forward prediction for the upcoming 7 days.
    """
    processed = compute_features_and_target(df)
    
    # Filter for store
    store_df = processed[processed['store_name'] == store_name].copy()
    if store_df.empty:
        return pd.DataFrame()
        
    if category:
        store_df = store_df[store_df['category'] == category].copy()
        if store_df.empty:
            return pd.DataFrame()
            
    # Match the feature-complete rows used during training.
    valid_features = store_df.dropna(subset=['past_30d_avg', 'sales_30d_ago']).copy()
    if valid_features.empty:
        return pd.DataFrame()
        
    # Get latest row per category
    latest = valid_features.sort_values('date').groupby('category', as_index=False).tail(1)
    return latest.reset_index(drop=True)
