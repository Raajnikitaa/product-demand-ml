"""
Model Training Module for DemandWise
====================================
Executes leakage-safe chronological train/validation/test splitting with 7-day purge gaps,
fits the scikit-learn Logistic Regression pipeline, optimizes decision threshold on validation,
evaluates on held-out test data, and persists the pipeline and metadata.

Temporal leakage prevention:
- Features for date t use only data <= t.
- The 7-day forward surge target uses sales from [t+1..t+7].
- A 7-day purge gap is placed between each split boundary so that forward-label windows
  from training rows cannot overlap into the validation or test periods.
- Preprocessing is fitted strictly on the training partition.
- The decision threshold is tuned on the validation slice only.
- The held-out test set is never touched until final evaluation.

Training data selection:
- Callers must explicitly pass store_paths: a list of validated dataset file paths.
- The function will NOT silently discover or combine datasets without the caller's explicit choice.
- Schema compatibility and duplicate/overlap checks are run before training.
"""

import os
import time
import json
import joblib
import datetime
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from src.data_loader import load_single_store_excel, aggregate_daily_category_sales, build_complete_store_calendar
from src.feature_engineering import (
    compute_features_and_target,
    ALL_MODEL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MIN_HISTORY_DAYS,
    SURGE_THRESHOLD_MULTIPLIER
)
from src.preprocessing import build_model_pipeline
from src.evaluate_model import (
    find_optimal_validation_threshold,
    evaluate_predictions,
    compute_evaluation_curves,
    evaluate_naive_baseline
)

def get_default_models_dir() -> str:
    candidates = [
        os.path.abspath('models'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'),
        os.path.abspath(os.path.join('..', 'models'))
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, 'model_pipeline.joblib')):
            return c
    return candidates[0]

DEFAULT_MODELS_DIR = get_default_models_dir()


def _save_model_artifacts(
    pipeline: Any,
    metadata: Dict[str, Any],
    models_dir: str
) -> Dict[str, Any]:
    """Write a versioned pipeline, then atomically publish metadata and its legacy alias."""
    model_id = uuid.uuid4().hex
    artifact_name = f'model_pipeline_{model_id}.joblib'
    artifact_path = os.path.join(models_dir, artifact_name)
    compatibility_path = os.path.join(models_dir, 'model_pipeline.joblib')
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    temporary_artifact = artifact_path + '.tmp'
    temporary_compatibility = compatibility_path + f'.{model_id}.tmp'
    temporary_metadata = os.path.join(
        models_dir, f'model_metadata_{model_id}.json.tmp'
    )

    metadata = dict(metadata)
    metadata['model_id'] = model_id
    metadata['pipeline_artifact'] = artifact_name
    try:
        joblib.dump(pipeline, temporary_artifact)
        os.replace(temporary_artifact, artifact_path)
        with open(temporary_metadata, 'w', encoding='utf-8') as file:
            json.dump(metadata, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_metadata, metadata_path)
        joblib.dump(pipeline, temporary_compatibility)
        os.replace(temporary_compatibility, compatibility_path)
    finally:
        for temporary_path in (
            temporary_artifact,
            temporary_metadata,
            temporary_compatibility,
        ):
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    return metadata


# ---------------------------------------------------------------------------
# Training data pre-flight checks
# ---------------------------------------------------------------------------

def validate_training_sources(store_paths: List[str]) -> Dict[str, Any]:
    """
    Validates a list of store dataset file paths for use as training data.

    Checks:
    1. All files exist and can be parsed.
    2. All files share the same required schema (after alias mapping).
    3. No critical duplicate records after combination (same date + store + category in multiple files).
    4. Minimum total record count for meaningful training.

    Returns a dict:
      - is_valid: bool
      - errors: List[str]
      - warnings: List[str]
      - files_accepted: List[str]
      - files_rejected: List[Dict]
      - total_rows: int
      - date_min: str
      - date_max: str
    """
    from src.data_validation import validate_excel_file, REQUIRED_COLUMNS

    errors: List[str] = []
    warnings: List[str] = []
    files_accepted: List[str] = []
    files_rejected: List[Dict[str, Any]] = []
    loaded_frames: List[pd.DataFrame] = []
    file_metas: List[Dict[str, Any]] = []

    if not store_paths:
        return {
            'is_valid': False,
            'errors': ['No dataset files were provided. Select at least one store dataset for training.'],
            'warnings': [],
            'files_accepted': [],
            'files_rejected': [],
            'total_rows': 0,
            'date_min': None,
            'date_max': None,
        }

    for path in store_paths:
        fname = os.path.basename(path)
        if not os.path.exists(path):
            files_rejected.append({'file': fname, 'reason': 'File not found at configured path.'})
            continue

        val = validate_excel_file(path)
        if not val['is_valid']:
            files_rejected.append({
                'file': fname,
                'reason': 'Validation failed: ' + '; '.join(val['errors'])
            })
            continue

        df = val['cleaned_df']
        files_accepted.append(fname)
        loaded_frames.append(df)
        file_metas.append({
            'file': fname,
            'path': path,
            'store_names': val.get('store_names', []),
            'date_min': val.get('date_min'),
            'date_max': val.get('date_max'),
            'row_count': val.get('row_count', 0),
            'revenue_calculated': val.get('revenue_calculated', False),
        })

    if not loaded_frames:
        errors.append(
            'No valid datasets could be loaded from the provided file paths. '
            'Resolve the file issues listed in "files_rejected" before training.'
        )
        return {
            'is_valid': False,
            'errors': errors,
            'warnings': warnings,
            'files_accepted': files_accepted,
            'files_rejected': files_rejected,
            'total_rows': 0,
            'date_min': None,
            'date_max': None,
        }

    # Combine and check for cross-file duplicates
    combined = pd.concat(loaded_frames, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])

    dup_key_mask = combined.duplicated(subset=['date', 'store_name', 'category', 'product'], keep=False)
    dup_count = int(dup_key_mask.sum())
    if dup_count > 0:
        warnings.append(
            f'Found {dup_count} potentially overlapping records across the selected files '
            '(same date, store, category, and product). These will be aggregated before training, '
            'which may double-count sales if the same data appears in multiple files.'
        )

    total_rows = len(combined)
    date_min = combined['date'].min().strftime('%Y-%m-%d') if not combined.empty else None
    date_max = combined['date'].max().strftime('%Y-%m-%d') if not combined.empty else None
    unique_days = int(combined['date'].dt.normalize().nunique())

    if total_rows < 500:
        warnings.append(
            f'Only {total_rows:,} total rows available for training. '
            'Logistic Regression results may be unreliable with very small datasets.'
        )
    if unique_days < 60:
        warnings.append(
            f'Only {unique_days} unique calendar days across all selected datasets. '
            'At least 90 days are recommended for reliable chronological splits.'
        )

    return {
        'is_valid': True,
        'errors': errors,
        'warnings': warnings,
        'files_accepted': files_accepted,
        'files_rejected': files_rejected,
        'file_metas': file_metas,
        'combined_df': combined,
        'total_rows': total_rows,
        'date_min': date_min,
        'date_max': date_max,
    }


# ---------------------------------------------------------------------------
# Chronological splitting
# ---------------------------------------------------------------------------

def prepare_chronological_splits(
    df: pd.DataFrame,
    test_days: int = 60,
    val_days: int = 30,
    purge_gap_days: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partitions time-series data chronologically with 7-day purge gaps to prevent
    forward-label leakage across split boundaries.

    Layout:
    [Train Core] --- (7d Purge Gap) ---> [Validation Slice] --- (7d Purge Gap) ---> [Test Set]

    The 7-day purge gap is mandatory because each row's target window extends 7 days
    into the future. Without the gap, the last 7 training days' targets would use
    observations that fall into the validation period, causing leakage.

    Preprocessing (scaler, encoder) is fitted only on train_df.
    Threshold tuning uses only val_df.
    Final evaluation uses only test_df (never touched during tuning).
    """
    valid = df.dropna(subset=['demand_surge', 'past_30d_avg', 'sales_30d_ago']).copy()
    valid['demand_surge'] = valid['demand_surge'].astype(int)

    max_date = valid['date'].max()

    # Test boundary
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    test_df = valid[valid['date'] >= test_start].copy()

    # Validation boundary — at least purge_gap_days before test_start
    train_horizon_end = test_start - pd.Timedelta(days=purge_gap_days + 1)
    val_start = train_horizon_end - pd.Timedelta(days=val_days - 1)
    val_df = valid[(valid['date'] >= val_start) & (valid['date'] <= train_horizon_end)].copy()

    # Train core boundary — at least purge_gap_days before val_start
    train_core_end = val_start - pd.Timedelta(days=purge_gap_days + 1)
    train_df = valid[valid['date'] <= train_core_end].copy()

    meta = {
        'total_valid_rows': len(valid),
        'train_start': train_df['date'].min().strftime('%Y-%m-%d') if not train_df.empty else 'N/A',
        'train_end': train_df['date'].max().strftime('%Y-%m-%d') if not train_df.empty else 'N/A',
        'train_rows': len(train_df),
        'train_surge_count': int(train_df['demand_surge'].sum()) if not train_df.empty else 0,
        'train_no_surge_count': int((train_df['demand_surge'] == 0).sum()) if not train_df.empty else 0,
        'train_surge_pct': round(float(train_df['demand_surge'].mean() * 100), 2) if not train_df.empty else 0.0,
        'val_purge_start': (train_core_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        'val_purge_end': (val_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        'val_start': val_df['date'].min().strftime('%Y-%m-%d') if not val_df.empty else 'N/A',
        'val_end': val_df['date'].max().strftime('%Y-%m-%d') if not val_df.empty else 'N/A',
        'val_rows': len(val_df),
        'val_surge_count': int(val_df['demand_surge'].sum()) if not val_df.empty else 0,
        'val_no_surge_count': int((val_df['demand_surge'] == 0).sum()) if not val_df.empty else 0,
        'val_surge_pct': round(float(val_df['demand_surge'].mean() * 100), 2) if not val_df.empty else 0.0,
        'test_purge_start': (train_horizon_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        'test_purge_end': (test_start - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        'test_start': test_df['date'].min().strftime('%Y-%m-%d') if not test_df.empty else 'N/A',
        'test_end': test_df['date'].max().strftime('%Y-%m-%d') if not test_df.empty else 'N/A',
        'test_rows': len(test_df),
        'test_surge_count': int(test_df['demand_surge'].sum()) if not test_df.empty else 0,
        'test_no_surge_count': int((test_df['demand_surge'] == 0).sum()) if not test_df.empty else 0,
        'test_surge_pct': round(float(test_df['demand_surge'].mean() * 100), 2) if not test_df.empty else 0.0,
    }
    return train_df, val_df, test_df, meta


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
    store_paths: Optional[List[str]] = None,
    models_dir: Optional[str] = None,
    surge_multiplier: float = SURGE_THRESHOLD_MULTIPLIER
) -> Dict[str, Any]:
    """
    Executes the full end-to-end model training, validation tuning, and test evaluation.
    Saves model artifacts and returns the execution report.

    Args:
        store_paths: Explicit list of validated .xlsx file paths to use as training data.
                     If None, falls back to discovering files in the default data/ directory.
                     Callers should explicitly pass the selected store dataset paths to ensure
                     transparency about which data was used.
        models_dir:  Directory to save model artifacts. Defaults to DEFAULT_MODELS_DIR.
        surge_multiplier: The multiplier defining a demand surge (default 1.30 = +30%).

    Leakage prevention summary:
        - Features at time t use only data <= t (rolling windows, lags, ratios).
        - Target for t uses sales from [t+1..t+7]; rows without a full 7-day forward window are excluded.
        - 7-day purge gaps between train/val and val/test boundaries prevent forward-label overlap.
        - Preprocessing is fitted only on train_df.
        - Decision threshold is tuned only on val_df (not the test set).
        - Test set metrics are computed once, on the untouched holdout.
    """
    t_start = time.time()
    target_models_dir = models_dir or DEFAULT_MODELS_DIR
    os.makedirs(target_models_dir, exist_ok=True)

    # 1. Resolve and validate training data sources
    if store_paths is None:
        from src.data_loader import discover_excel_files, get_default_data_dir
        data_dir = os.path.abspath(get_default_data_dir())
        store_paths = [
            path for path in discover_excel_files(data_dir)
            if os.path.dirname(os.path.abspath(path)) == data_dir
        ]

    if not store_paths:
        raise ValueError(
            "No dataset files available. Place Excel workbooks in data/ or select "
            "store datasets explicitly in the training-data selection step."
        )

    preflight = validate_training_sources(store_paths)
    if not preflight['is_valid']:
        raise ValueError(
            "Training data pre-flight validation failed:\n" +
            "\n".join(preflight['errors'])
        )

    # Use the already-combined and validated DataFrame from the preflight
    raw_df = preflight['combined_df']
    file_metas = preflight.get('file_metas', [])

    # 2. Aggregate to daily store-category records and build complete calendar
    daily_agg = aggregate_daily_category_sales(raw_df)
    calendar_df, _ = build_complete_store_calendar(daily_agg)

    if calendar_df is None or calendar_df.empty:
        raise ValueError("No usable data after aggregation and calendar construction.")

    # 3. Compute features and forward target (leakage-safe)
    featured_df = compute_features_and_target(calendar_df, surge_multiplier=surge_multiplier)

    # 4. Chronological splits with 7-day purge gaps
    train_df, val_df, test_df, split_meta = prepare_chronological_splits(featured_df)

    if len(train_df) < 50:
        raise ValueError(
            f"Training partition has only {len(train_df)} usable rows after feature engineering "
            "and purge gaps. Add more historical data before training."
        )
    if len(val_df) < 10:
        raise ValueError(
            f"Validation partition has only {len(val_df)} rows. Increase the dataset date range."
        )
    if len(test_df) < 10:
        raise ValueError(
            f"Test partition has only {len(test_df)} rows. Increase the dataset date range."
        )
    if train_df['demand_surge'].nunique() < 2:
        raise ValueError(
            "The training partition contains only one target class. Select training data "
            "with both surge and no-surge examples; a classifier cannot be fitted otherwise."
        )
    if int(val_df['demand_surge'].sum()) == 0:
        raise ValueError(
            "The validation partition contains no surge examples, so model and threshold "
            "selection cannot be evaluated. Add representative training history or choose "
            "a different dataset."
        )

    # 5. Compare class-weighted and unweighted Logistic Regression on validation only.
    validation_candidates: List[Dict[str, Any]] = []
    candidate_pipelines: Dict[Optional[str], Any] = {}
    for class_weight in ('balanced', None):
        candidate = build_model_pipeline(class_weight=class_weight)
        candidate.fit(train_df[ALL_MODEL_FEATURES], train_df['demand_surge'])
        val_probs = candidate.predict_proba(val_df[ALL_MODEL_FEATURES])[:, 1]
        threshold, validation_f1 = find_optimal_validation_threshold(
            val_df['demand_surge'].values, val_probs
        )
        metrics = evaluate_predictions(
            val_df['demand_surge'].values, val_probs, threshold=threshold
        )
        candidate_name = (
            'Logistic Regression (balanced)' if class_weight else
            'Logistic Regression (unweighted)'
        )
        validation_candidates.append({
            'model_name': candidate_name,
            'class_weight': class_weight,
            'threshold': threshold,
            'validation_f1': validation_f1,
            'metrics': metrics,
        })
        candidate_pipelines[class_weight] = candidate

    train_majority_class = int(train_df['demand_surge'].mean() >= 0.5)
    majority_val_probs = np.full(len(val_df), train_majority_class, dtype=float)
    majority_val_metrics = evaluate_predictions(
        val_df['demand_surge'].values, majority_val_probs, threshold=0.5
    )
    validation_comparison = [{
        'model_name': 'Training-set majority-class baseline',
        'class_weight': None,
        'threshold': 0.5,
        'validation_f1': majority_val_metrics['f1_score'],
        'metrics': majority_val_metrics,
    }] + validation_candidates

    selected_candidate = max(
        validation_candidates,
        key=lambda item: (
            item['validation_f1'],
            item['metrics']['precision'],
            item['threshold'],
        )
    )
    selected_weight = selected_candidate['class_weight']
    pipeline = candidate_pipelines[selected_weight]
    optimal_th = selected_candidate['threshold']
    val_f1 = selected_candidate['validation_f1']
    validation_metrics = selected_candidate['metrics']
    train_duration = time.time() - t_start

    # 6. Evaluate only the validation-selected configuration on the holdout test set.
    test_probs = pipeline.predict_proba(test_df[ALL_MODEL_FEATURES])[:, 1]
    y_test = test_df['demand_surge'].values
    test_metrics = evaluate_predictions(y_test, test_probs, threshold=optimal_th)
    majority_test_probs = np.full(len(test_df), train_majority_class, dtype=float)
    majority_test_metrics = evaluate_predictions(
        y_test, majority_test_probs, threshold=0.5
    )

    # Evaluation curves
    curves = compute_evaluation_curves(y_test, test_probs)

    # Naive baseline comparison
    naive_results = evaluate_naive_baseline(
        val_df['sales_vs_7d_avg'].values,
        val_df['demand_surge'].values,
        test_df['sales_vs_7d_avg'].values,
        y_test
    )

    # 8. Extract Model Coefficients and Odds Ratios
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']

    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = list(NUMERIC_FEATURES) + list(cat_names)
    coefficients = classifier.coef_[0].tolist()
    odds_ratios = np.exp(classifier.coef_[0]).tolist()

    coef_list = []
    for f, c, o in zip(all_feature_names, coefficients, odds_ratios):
        coef_list.append({
            'feature': f,
            'coefficient': round(float(c), 4),
            'odds_ratio': round(float(o), 4),
            'direction': 'Increases Surge Odds' if c > 0 else 'Decreases Surge Odds'
        })

    # 7. Build metadata, then publish the matching versioned pipeline atomically.
    train_positive = int(train_df['demand_surge'].sum())
    val_positive = int(val_df['demand_surge'].sum())
    test_positive = int(test_df['demand_surge'].sum())
    evaluation_warnings = []
    if val_positive < 50:
        evaluation_warnings.append(
            f'Validation threshold tuning used only {val_positive} surge examples; '
            'the selected threshold may be unstable.'
        )
    if test_positive < 50:
        evaluation_warnings.append(
            f'The final test set contains only {test_positive} surge examples; '
            'test metrics have substantial sampling uncertainty.'
        )
    if not test_metrics['has_both_classes']:
        evaluation_warnings.append(
            'The final test set contains only one class, so ROC-AUC and PR-AUC are unavailable.'
        )

    metadata = {
        'model_name': 'DemandWise Logistic Regression Pipeline',
        'algorithm': (
            f"Logistic Regression (L-BFGS, class_weight={selected_weight}, C=1.0)"
        ),
        'selected_class_weight': selected_weight,
        'version': '1.1.0',
        'training_timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_note': (
            'This model was trained on the datasets listed in dataset_files. '
            'Schema compatibility with a store\'s data does not imply validated predictive accuracy '
            'for that store. Performance metrics below apply only to the held-out test split '
            'drawn from the training dataset. Independent validation on new retailer data '
            'would be required to assess generalisability.'
        ),
        'dataset_files': [m['file'] for m in file_metas],
        'dataset_file_metas': file_metas,
        'dataset_date_min': preflight.get('date_min'),
        'dataset_date_max': preflight.get('date_max'),
        'dataset_total_rows': preflight.get('total_rows'),
        'surge_threshold_multiplier': surge_multiplier,
        'surge_threshold_pct': f"+{int((surge_multiplier - 1.0) * 100)}%",
        'target_definition': (
            'For each store and category, baseline is mean daily sales over days t-6 through t; '
            'future demand is mean daily sales over t+1 through t+7. A surge requires a complete '
            'window, a positive baseline, and future mean >= the configured multiplier times baseline.'
        ),
        'optimal_decision_threshold': optimal_th,
        'threshold_selection_objective': (
            'Maximize validation F1; ties prefer higher precision, then higher threshold '
            'to reduce false alarms without sacrificing F1.'
        ),
        'selected_model_reason': (
            f"{selected_candidate['model_name']} had the highest validation F1 among the "
            'majority baseline and Logistic Regression candidates.'
        ),
        'probability_calibration_note': (
            'Logistic Regression probabilities have not been explicitly calibrated '
            '(e.g., via Platt scaling or isotonic regression). The reported probability '
            'is the model\'s estimated likelihood of a demand surge, not a guaranteed '
            'frequency estimate. Treat it as a ranked risk score rather than a precise probability.'
        ),
        'validation_f1': round(val_f1, 4),
        'validation_metrics': validation_metrics,
        'validation_model_comparison': validation_comparison,
        'split_info': split_meta,
        'test_metrics': test_metrics,
        'test_majority_baseline': majority_test_metrics,
        'naive_baseline': naive_results,
        'evaluation_warnings': evaluation_warnings,
        'evaluation_reliability': 'limited' if evaluation_warnings else 'adequate_positive_count',
        'feature_history_days_required': MIN_HISTORY_DAYS,
        'curves': curves,
        'features': {
            'numeric': NUMERIC_FEATURES,
            'categorical': CATEGORICAL_FEATURES,
            'all_transformed_count': len(all_feature_names)
        },
        'coefficients': sorted(coef_list, key=lambda x: abs(x['coefficient']), reverse=True),
        'train_duration_seconds': round(train_duration, 2),
        'dataset_provenance_note': (
            'Only the explicitly listed workbook files were used for training. Uploaded '
            'store data is used for inference unless explicitly selected in the training dashboard.'
        )
    }
    return _save_model_artifacts(pipeline, metadata, target_models_dir)


def load_trained_model(models_dir: Optional[str] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Loads saved model pipeline and metadata if available.
    Returns (pipeline, metadata_dict).
    """
    target_models_dir = models_dir or DEFAULT_MODELS_DIR
    meta_path = os.path.join(target_models_dir, 'model_metadata.json')

    if not os.path.exists(meta_path):
        if os.path.exists(os.path.join(target_models_dir, 'model_pipeline.joblib')):
            raise RuntimeError(
                "A model pipeline exists without its metadata file. Retrain the model "
                "to restore a consistent pipeline and threshold."
            )
        return None, None

    try:
        with open(meta_path, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read model metadata at '{meta_path}': {exc}") from exc

    artifact_name = os.path.basename(
        metadata.get('pipeline_artifact', 'model_pipeline.joblib')
    )
    pipe_path = os.path.join(target_models_dir, artifact_name)
    if not os.path.isfile(pipe_path):
        raise RuntimeError(
            f"Model metadata references a missing pipeline artifact: '{pipe_path}'. "
            "Retrain the model to create a matching artifact."
        )
    try:
        pipeline = joblib.load(pipe_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load model pipeline at '{pipe_path}': {exc}") from exc
    return pipeline, metadata


if __name__ == '__main__':
    print("Starting DemandWise model training...")
    meta = run_training_pipeline()
    print("Training finished successfully!")
    print(f"Optimal Threshold (tau*): {meta['optimal_decision_threshold']}")
    print(f"Test Accuracy: {meta['test_metrics']['accuracy'] * 100:.2f}%")
    print(f"Test Precision: {meta['test_metrics']['precision'] * 100:.2f}%")
    print(f"Test Recall: {meta['test_metrics']['recall'] * 100:.2f}%")
    print(f"Test F1: {meta['test_metrics']['f1_score']:.4f}")
    if meta['test_metrics']['roc_auc'] is not None:
        print(f"Test ROC-AUC: {meta['test_metrics']['roc_auc']:.4f}")
