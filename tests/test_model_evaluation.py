"""Tests for validation-only model selection and model artifact consistency."""

import tempfile
import unittest
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.data_loader import build_complete_store_calendar
from src.evaluate_model import evaluate_predictions, find_optimal_validation_threshold
from src.train_model import (
    _save_model_artifacts,
    load_trained_model,
    prepare_chronological_splits,
)


class TestModelEvaluation(unittest.TestCase):

    def test_threshold_selection_uses_validation_scores_and_maximizes_f1(self):
        y_validation = np.array([0, 0, 1, 1])
        validation_probabilities = np.array([0.1, 0.4, 0.45, 0.9])

        threshold, validation_f1 = find_optimal_validation_threshold(
            y_validation, validation_probabilities
        )

        self.assertEqual(threshold, 0.45)
        self.assertEqual(validation_f1, 1.0)

    def test_metrics_report_absent_positive_class_without_auc(self):
        metrics = evaluate_predictions(
            np.array([0, 0, 0]),
            np.array([0.1, 0.2, 0.3]),
            threshold=0.5,
        )

        self.assertEqual(metrics['class_distribution']['surge_count'], 0)
        self.assertIsNone(metrics['roc_auc'])
        self.assertIsNone(metrics['pr_auc'])
        self.assertAlmostEqual(metrics['brier_score'], (0.01 + 0.04 + 0.09) / 3)
        self.assertEqual(metrics['confusion_matrix']['matrix'], [[3, 0], [0, 0]])

    def test_metrics_preserve_selected_threshold_precision(self):
        metrics = evaluate_predictions(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            threshold=0.164,
        )

        self.assertEqual(metrics['threshold'], 0.164)

    def test_temporal_splits_keep_seven_day_purge_boundaries(self):
        dates = pd.date_range('2025-01-01', periods=220, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'demand_surge': np.arange(len(dates)) % 2,
            'past_30d_avg': 10.0,
            'sales_30d_ago': 10.0,
        })

        train, validation, test, metadata = prepare_chronological_splits(
            df, test_days=40, val_days=30, purge_gap_days=7
        )

        self.assertLess(
            (validation['date'].min() - train['date'].max()).days,
            9
        )
        self.assertEqual(
            (test['date'].min() - validation['date'].max()).days,
            8
        )
        self.assertEqual(metadata['val_rows'], 30)
        self.assertEqual(metadata['test_rows'], 40)

    def test_calendar_only_creates_observed_store_category_pairs(self):
        source = pd.DataFrame([
            {
                'date': pd.Timestamp('2025-01-01'),
                'store_name': 'Store A',
                'category': 'Bakery',
                'sales': 2.0,
                'revenue': 4.0,
                'unit_price': 2.0,
                'promotion': 0,
            },
            {
                'date': pd.Timestamp('2025-01-02'),
                'store_name': 'Store B',
                'category': 'Snacks',
                'sales': 3.0,
                'revenue': 6.0,
                'unit_price': 2.0,
                'promotion': 0,
            },
        ])

        calendar, _ = build_complete_store_calendar(source)

        self.assertEqual(len(calendar), 4)
        self.assertEqual(
            set(zip(calendar['store_name'], calendar['category'])),
            {('Store A', 'Bakery'), ('Store B', 'Snacks')}
        )

    def test_versioned_model_and_metadata_load_as_a_pair(self):
        with tempfile.TemporaryDirectory() as models_dir:
            pipeline = DummyClassifier(strategy='constant', constant=1)
            pipeline.fit(pd.DataFrame({'feature': [0, 1]}), [0, 1])
            metadata = _save_model_artifacts(
                pipeline,
                {'optimal_decision_threshold': 0.42},
                models_dir,
            )

            loaded_pipeline, loaded_metadata = load_trained_model(models_dir)
            compatibility_pipeline = joblib.load(
                os.path.join(models_dir, 'model_pipeline.joblib')
            )

        self.assertEqual(loaded_metadata['model_id'], metadata['model_id'])
        self.assertEqual(
            loaded_metadata['pipeline_artifact'],
            metadata['pipeline_artifact']
        )
        np.testing.assert_array_equal(
            loaded_pipeline.predict(pd.DataFrame({'feature': [1]})),
            pipeline.predict(pd.DataFrame({'feature': [1]}))
        )
        np.testing.assert_array_equal(
            compatibility_pipeline.predict(pd.DataFrame({'feature': [1]})),
            pipeline.predict(pd.DataFrame({'feature': [1]}))
        )


if __name__ == '__main__':
    unittest.main()
