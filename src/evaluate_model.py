"""
Evaluation Module for DemandWise
================================
Calculates genuine classification metrics, confusion matrices, evaluation curves,
and naive baseline comparisons on held-out test splits.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from sklearn.metrics import brier_score_loss


def find_optimal_validation_threshold(
    y_val: np.ndarray,
    y_val_probs: np.ndarray,
    threshold_range: Tuple[float, float] = (0.0, 1.0),
    steps: int = 71
) -> Tuple[float, float]:
    """
    Maximizes validation F1; ties prefer higher precision, then a higher threshold.
    The higher-threshold tie-break limits false alarms without reducing F1.
    """
    y_val = np.asarray(y_val, dtype=int)
    y_val_probs = np.asarray(y_val_probs, dtype=float)
    thresholds = np.unique(np.concatenate((
        np.linspace(threshold_range[0], threshold_range[1], steps),
        y_val_probs[
            (y_val_probs >= threshold_range[0]) &
            (y_val_probs <= threshold_range[1])
        ]
    )))
    best_th = 0.50
    best_metrics = (-1.0, -1.0, -1.0)
    
    for th in thresholds:
        preds = (y_val_probs >= th).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        precision = precision_score(y_val, preds, zero_division=0)
        candidate_metrics = (float(f), float(precision), float(th))
        if candidate_metrics > best_metrics:
            best_metrics = candidate_metrics
            best_th = float(th)
            
    return round(best_th, 4), float(best_metrics[0])


def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates predictions at a given decision threshold.
    Safely handles edge cases (e.g., single-class test slices) without fabricating scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)
    
    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) > 1
    
    # Basic classification metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Area under curves (only valid when both classes are present)
    if has_both_classes:
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = None
            
        try:
            pr_auc = float(average_precision_score(y_true, y_probs))
        except Exception:
            pr_auc = None
    else:
        roc_auc = None
        pr_auc = None
        
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    total = len(y_true)
    surge_count = int(np.sum(y_true == 1))
    non_surge_count = int(np.sum(y_true == 0))
    surge_rate = float(surge_count / total * 100.0) if total > 0 else 0.0
    
    return {
        'threshold': round(float(threshold), 4),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': float(brier_score_loss(y_true, y_probs)) if len(y_true) else None,
        'has_both_classes': has_both_classes,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'matrix': cm.tolist()
        },
        'class_distribution': {
            'total': total,
            'surge_count': surge_count,
            'non_surge_count': non_surge_count,
            'surge_rate_pct': round(surge_rate, 2)
        }
    }


def compute_evaluation_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray
) -> Dict[str, Any]:
    """
    Generates ROC curve coordinates, Precision-Recall curve coordinates,
    and a threshold sweep dataframe.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    
    if len(np.unique(y_true)) < 2:
        return {'available': False}
        
    fpr, tpr, roc_thresh = roc_curve(y_true, y_probs)
    precision_pts, recall_pts, pr_thresh = precision_recall_curve(y_true, y_probs)
    
    # Threshold sweep from 0.05 to 0.95
    sweep_rows = []
    for th in np.linspace(0.05, 0.95, 37):
        p = (y_probs >= th).astype(int)
        sweep_rows.append({
            'threshold': round(th, 2),
            'precision': float(precision_score(y_true, p, zero_division=0)),
            'recall': float(recall_score(y_true, p, zero_division=0)),
            'f1': float(f1_score(y_true, p, zero_division=0))
        })
        
    return {
        'available': True,
        'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': roc_thresh.tolist()},
        'pr': {'precision': precision_pts.tolist(), 'recall': recall_pts.tolist(), 'thresholds': pr_thresh.tolist()},
        'sweep': sweep_rows
    }


def evaluate_naive_baseline(
    val_momentum: np.ndarray,
    y_val: np.ndarray,
    test_momentum: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates a simple rule-based naive momentum baseline:
    Predict surge if recent momentum (sales_today - past_7d_avg) / past_7d_avg exceeds a threshold.
    The threshold is selected strictly on the validation slice.
    """
    clean_val = np.nan_to_num(val_momentum, nan=0.0, posinf=10.0, neginf=-10.0)
    clean_test = np.nan_to_num(test_momentum, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Search threshold on validation
    candidates = np.linspace(-0.5, 1.5, 41)
    best_th = 0.30
    best_f1 = -1.0
    for th in candidates:
        preds = (clean_val >= th).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_th = th
            
    test_preds = (clean_test >= best_th).astype(int)
    
    roc = None
    if len(np.unique(y_test)) > 1:
        try:
            roc = float(roc_auc_score(y_test, clean_test))
        except Exception:
            roc = None
            
    return {
        'model_name': 'Naive Momentum Rule',
        'threshold_used': round(float(best_th), 2),
        'accuracy': float(accuracy_score(y_test, test_preds)),
        'precision': float(precision_score(y_test, test_preds, zero_division=0)),
        'recall': float(recall_score(y_test, test_preds, zero_division=0)),
        'f1_score': float(f1_score(y_test, test_preds, zero_division=0)),
        'roc_auc': roc,
        'description': "Rule: Flag surge if sales_today is >= threshold above past_7d_avg"
    }
