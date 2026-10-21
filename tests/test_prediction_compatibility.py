"""Prediction tests for matching training history and feature schemas."""

import unittest

import numpy as np
import pandas as pd

from src.predict_demand import predict_category_surge


class MismatchedFeaturePipeline:
    feature_names_in_ = np.array(['unexpected_feature'])


class TestPredictionCompatibility(unittest.TestCase):

    def setUp(self):
        self.dates = pd.date_range('2025-01-01', periods=40, freq='D')
        self.history = pd.DataFrame({
            'date': self.dates,
            'store_name': 'Test Store',
            'category': 'Bakery',
            'sales': 10.0,
            'revenue': 20.0,
            'unit_price': 2.0,
            'promotion': 0,
        })

    def test_requires_same_feature_complete_history_as_training(self):
        result = predict_category_surge(
            self.history.iloc[:30],
            'Test Store',
            'Bakery',
            pipeline=MismatchedFeaturePipeline(),
            metadata={'optimal_decision_threshold': 0.5},
        )

        self.assertFalse(result['success'])
        self.assertIn('31 consecutive calendar days', result['error'])

    def test_rejects_pipeline_with_different_feature_names_or_order(self):
        result = predict_category_surge(
            self.history,
            'Test Store',
            'Bakery',
            pipeline=MismatchedFeaturePipeline(),
            metadata={'optimal_decision_threshold': 0.5},
        )

        self.assertFalse(result['success'])
        self.assertIn('feature names or order do not match', result['error'])

    def test_requires_saved_threshold_in_model_metadata(self):
        from sklearn.dummy import DummyClassifier
        from src.feature_engineering import ALL_MODEL_FEATURES, compute_features_and_target

        complete_features = compute_features_and_target(self.history).dropna(
            subset=['past_30d_avg', 'sales_30d_ago']
        )[ALL_MODEL_FEATURES]
        classifier = DummyClassifier(strategy='prior')
        classifier.fit(complete_features.iloc[:2], [0, 1])

        result = predict_category_surge(
            self.history,
            'Test Store',
            'Bakery',
            pipeline=classifier,
            metadata={},
        )

        self.assertFalse(result['success'])
        self.assertIn('validated decision threshold', result['error'])


if __name__ == '__main__':
    unittest.main()
