"""
Unit Tests for Model Workflow & Database Operations
==================================================
Verifies authentication hashing, database transactions, model reloading,
and inference prediction schemas.
"""

import os
import tempfile
import unittest
import pandas as pd
import numpy as np

from src.auth import hash_password, verify_password, register_user, authenticate_user
from src.database import init_db, get_connection, add_store, get_stores_for_user
from src.train_model import load_trained_model
from src.predict_demand import predict_category_surge


class TestModelWorkflow(unittest.TestCase):

    def setUp(self):
        # Create temporary database for testing
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        init_db(self.temp_db_path)

    def tearDown(self):
        try:
            os.close(self.temp_db_fd)
            if os.path.exists(self.temp_db_path):
                os.remove(self.temp_db_path)
        except Exception:
            pass

    def test_password_hashing_and_verification(self):
        pwd = "SecretPassword123"
        h, salt = hash_password(pwd)
        self.assertNotEqual(pwd, h)
        self.assertTrue(verify_password(pwd, h, salt))
        self.assertFalse(verify_password("WrongPassword", h, salt))

    def test_user_registration_and_auth_workflow(self):
        ok, msg, user = register_user(
            username="alice",
            email="alice@retail.com",
            password="StrongPassword123",
            confirm_password="StrongPassword123",
            business_name="Alice Groceries",
            owner_name="Alice Smith",
            seed_demo=False,
            db_path=self.temp_db_path
        )
        self.assertTrue(ok)
        self.assertIsNotNone(user)
        self.assertEqual(user['username'], 'alice')

        # Successful auth
        auth_ok, auth_msg, u, b = authenticate_user("alice", "StrongPassword123", db_path=self.temp_db_path)
        self.assertTrue(auth_ok)
        self.assertEqual(u['username'], 'alice')
        self.assertEqual(b['business_name'], 'Alice Groceries')

        # Invalid auth
        auth_bad, _, _, _ = authenticate_user("alice", "WrongPwd", db_path=self.temp_db_path)
        self.assertFalse(auth_bad)

    def test_store_isolation_per_user(self):
        # User 1
        _, _, u1 = register_user("user1", "u1@test.com", "pass123", "pass123", "B1", "O1", seed_demo=False, db_path=self.temp_db_path)
        # User 2
        _, _, u2 = register_user("user2", "u2@test.com", "pass123", "pass123", "B2", "O2", seed_demo=False, db_path=self.temp_db_path)

        add_store(u1['id'], "Store U1", "S1", "Urban", "Downtown", None, 0, db_path=self.temp_db_path)
        add_store(u2['id'], "Store U2", "S2", "Suburban", "Uptown", None, 0, db_path=self.temp_db_path)

        stores_u1 = get_stores_for_user(u1['id'], db_path=self.temp_db_path)
        stores_u2 = get_stores_for_user(u2['id'], db_path=self.temp_db_path)

        self.assertEqual(len(stores_u1), 1)
        self.assertEqual(stores_u1[0]['store_name'], "Store U1")
        self.assertEqual(len(stores_u2), 1)
        self.assertEqual(stores_u2[0]['store_name'], "Store U2")

    def test_model_loading_and_prediction(self):
        pipe, meta = load_trained_model()
        self.assertIsNotNone(pipe, "Trained pipeline must exist on disk.")
        self.assertIsNotNone(meta, "Trained metadata must exist on disk.")
        self.assertIn('optimal_decision_threshold', meta)

        # Create dummy store history of 40 days
        dates = pd.date_range('2025-01-01', periods=40, freq='D')
        rows = []
        for d in dates:
            rows.append({
                'date': d,
                'store_name': 'Test_Shop',
                'category': 'Snacks',
                'sales': 50.0,
                'revenue': 100.0,
                'unit_price': 2.0,
                'promotion': 0
            })
        test_df = pd.DataFrame(rows)

        pred = predict_category_surge(test_df, 'Test_Shop', 'Snacks', pipeline=pipe, metadata=meta)
        self.assertTrue(pred['success'])
        self.assertIn('probability', pred)
        self.assertIn('is_surge', pred)
        self.assertIn('advisory', pred)
        self.assertIn('forecast_horizon_label', pred)


if __name__ == '__main__':
    unittest.main()
