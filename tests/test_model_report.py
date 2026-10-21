import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from src.model_report import build_model_report_html, create_model_bundle


class TestModelReport(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "algorithm": "Logistic Regression",
            "optimal_decision_threshold": 0.6,
            "surge_threshold_pct": "+30%",
            "training_timestamp": "2026-01-01 10:00:00",
            "data_type": "Retail sales data",
            "split_info": {"test_start": "2025-10-01", "test_end": "2025-11-01", "test_rows": 100},
            "test_metrics": {
                "accuracy": 0.9,
                "precision": 0.8,
                "recall": 0.7,
                "f1_score": 0.75,
                "roc_auc": 0.85,
                "pr_auc": 0.6,
                "confusion_matrix": {"matrix": [[80, 10], [5, 5]]},
            },
            "curves": {
                "available": True,
                "roc": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0]},
            },
            "coefficients": [{"feature": "past_7d_avg", "coefficient": 0.4}],
        }
        self.pipeline = SimpleNamespace(
            named_steps={
                "classifier": SimpleNamespace(
                    C=1.0,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                )
            }
        )

    def test_report_contains_performance_settings_and_graphs(self):
        report = build_model_report_html(self.metadata, self.pipeline)
        self.assertIn("Test performance", report)
        self.assertIn("Regularization C", report)
        self.assertIn('aria-label="ROC curve"', report)
        self.assertIn('aria-label="Confusion matrix"', report)
        self.assertIn("train_model.py", report)

    def test_bundle_contains_model_report_metadata_and_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "model_pipeline_versioned.joblib").write_bytes(b"trained-model")
            metadata = dict(self.metadata, pipeline_artifact="model_pipeline_versioned.joblib")
            bundle_data = create_model_bundle(metadata, self.pipeline, temp_dir)

        with zipfile.ZipFile(io.BytesIO(bundle_data)) as bundle:
            self.assertIn("model_pipeline.joblib", bundle.namelist())
            self.assertIn("model_metadata.json", bundle.namelist())
            self.assertIn("Model_Report.html", bundle.namelist())
            self.assertIn("src/train_model.py", bundle.namelist())
            self.assertIn(b"ROC curve", bundle.read("Model_Report.html"))
            bundled_metadata = json.loads(bundle.read("model_metadata.json"))
            self.assertEqual(bundled_metadata["pipeline_artifact"], "model_pipeline.joblib")


if __name__ == "__main__":
    unittest.main()
