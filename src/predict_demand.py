"""
Prediction Module for DemandWise
================================
Runs inference using the saved Scikit-learn pipeline on the latest available data.
Outputs estimated probabilities, 7-day forecast horizons, and actionable recommendations.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.feature_engineering import (
    compute_features_and_target,
    ALL_MODEL_FEATURES,
)
from src.train_model import load_trained_model
from src.recommendations import generate_recommendation_and_explanation


def _get_recent_observed_surge(sub_df: pd.DataFrame) -> Dict[str, Any]:
    """Compare the latest complete 7-day sales window with the preceding 7 days."""
    daily_sales = (
        sub_df.assign(date=pd.to_datetime(sub_df['date']).dt.normalize())
        .groupby('date')['sales']
        .sum()
        .sort_index()
    )
    recent = daily_sales.tail(14)
    has_consecutive_days = (
        recent.index.to_series().diff().dropna() == pd.Timedelta(days=1)
    ).all()
    if len(recent) < 14 or not has_consecutive_days:
        return {
            'observed_surge_available': False,
            'observed_surge': False,
            'observed_recent_avg': None,
            'observed_previous_avg': None,
            'observed_change_pct': None,
            'observed_window_start': None,
            'observed_window_end': None,
        }

    previous_avg = float(recent.iloc[:7].mean())
    recent_avg = float(recent.iloc[7:].mean())
    change_pct = (
        ((recent_avg / previous_avg) - 1.0) * 100.0
        if previous_avg > 0
        else None
    )
    return {
        'observed_surge_available': True,
        'observed_surge': bool(
            previous_avg > 0
            and recent_avg >= 1.30 * previous_avg
            and recent_avg > previous_avg
        ),
        'observed_recent_avg': recent_avg,
        'observed_previous_avg': previous_avg,
        'observed_change_pct': change_pct,
        'observed_window_start': recent.index[7].strftime('%Y-%m-%d'),
        'observed_window_end': recent.index[-1].strftime('%Y-%m-%d'),
    }


def predict_category_surge(
    store_df: pd.DataFrame,
    store_name: str,
    category: str,
    pipeline: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates a 7-day demand surge prediction for a specific store and category.
    """
    # 1. Load pipeline if not provided
    if pipeline is None or metadata is None:
        pipe, meta = load_trained_model()
        pipeline = pipeline or pipe
        metadata = metadata or meta
        
    if pipeline is None or metadata is None:
        return {
            'success': False,
            'error': "The model is not trained yet. Train it from Model & Report before requesting predictions."
        }
        
    # 2. Filter and compute features
    sub_df = store_df[(store_df['store_name'] == store_name) & (store_df['category'] == category)].copy()
    if sub_df.empty:
        return {
            'success': False,
            'error': f"No data records found for store '{store_name}' and category '{category}'."
        }
        
    featured = compute_features_and_target(sub_df)
    
    # Match training's requirement for every feature, including the 30-day lag.
    valid_history = featured.dropna(
        subset=['past_30d_avg', 'sales_30d_ago']
    ).copy()
    if valid_history.empty:
        return {
            'success': False,
            'error': (
                f"Insufficient history for {category} in {store_name}. "
                "This model requires at least 31 consecutive calendar days so every "
                "training feature, including the 30-day lag, is available."
            )
        }
        
    latest_row = valid_history.sort_values('date').tail(1).iloc[0]
    eval_date = latest_row['date']
    forecast_start = eval_date + pd.Timedelta(days=1)
    forecast_end = eval_date + pd.Timedelta(days=7)
    
    # 3. Model Inference
    input_df = pd.DataFrame([latest_row[ALL_MODEL_FEATURES].to_dict()])
    expected_features = getattr(pipeline, 'feature_names_in_', None)
    if expected_features is not None and list(expected_features) != list(input_df.columns):
        return {
            'success': False,
            'error': (
                "The loaded model's feature names or order do not match the inference "
                "data. Retrain the model with the current feature pipeline."
            )
        }
    
    try:
        prob = float(pipeline.predict_proba(input_df)[0][1])
    except Exception as e:
        return {
            'success': False,
            'error': f"Inference execution failed: {str(e)}"
        }
        
    if 'optimal_decision_threshold' not in metadata:
        return {
            'success': False,
            'error': "Model metadata does not include a validated decision threshold. Retrain the model."
        }
    threshold = float(metadata['optimal_decision_threshold'])
    is_surge = prob >= threshold
    
    # Badge determination
    dist = abs(prob - threshold)
    if is_surge:
        badge = "SURGE"
        status_label = "Surge Likely"
        badge_color = "#DC2626"
    elif dist <= 0.08:
        badge = "WATCH"
        status_label = "Elevated Demand (Watch)"
        badge_color = "#D97706"
    else:
        badge = "NORMAL"
        status_label = "No Surge Detected"
        badge_color = "#15803D"
        
    feature_values = {
        'sales_today': float(latest_row['sales_today']),
        'past_7d_avg': float(latest_row['past_7d_avg']),
        'past_14d_avg': float(latest_row['past_14d_avg']),
        'past_30d_avg': float(latest_row['past_30d_avg']) if pd.notna(latest_row['past_30d_avg']) else float(latest_row['past_7d_avg']),
        'sales_vs_7d_avg': float(latest_row['sales_vs_7d_avg']),
        'onpromotion': int(latest_row['onpromotion']),
        'day_of_week': str(latest_row['day_of_week'])
    }
    
    base_result = {
        'success': True,
        'store_name': store_name,
        'category': category,
        'is_surge': is_surge,
        'probability': prob,
        'probability_pct': round(prob * 100, 1),
        'threshold_used': threshold,
        'badge_status': badge,
        'badge_color': badge_color,
        'status_label': status_label,
        'evaluation_date': eval_date.strftime('%Y-%m-%d'),
        'forecast_start': forecast_start.strftime('%Y-%m-%d'),
        'forecast_end': forecast_end.strftime('%Y-%m-%d'),
        'forecast_horizon_label': f"{forecast_start.strftime('%b %d, %Y')} – {forecast_end.strftime('%b %d, %Y')} (Next 7 Days)",
        'feature_values': feature_values
    }
    base_result.update(_get_recent_observed_surge(sub_df))
    
    # Add textual explanation & advisory
    recs = generate_recommendation_and_explanation(base_result)
    base_result['explanation'] = recs['explanation']
    base_result['advisory'] = recs['advisory']
    
    return base_result


def predict_all_categories_for_store(
    store_df: pd.DataFrame,
    store_name: str,
    pipeline: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generates predictions for all categories present in the selected store.
    Returns list of prediction dicts sorted by probability descending.
    """
    categories = sorted(store_df[store_df['store_name'] == store_name]['category'].unique().tolist())
    results = []
    
    for cat in categories:
        res = predict_category_surge(store_df, store_name, cat, pipeline=pipeline, metadata=metadata)
        if res.get('success'):
            results.append(res)
            
    return sorted(results, key=lambda x: x['probability'], reverse=True)
