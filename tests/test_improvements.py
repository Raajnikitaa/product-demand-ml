"""
Tests for new features: column aliases, revenue fallback, training validation,
recommendations advisory text, and database metadata columns.
"""

import io
import os
import tempfile
import unittest
import pandas as pd

from src.data_validation import validate_excel_file, apply_column_aliases, COLUMN_ALIASES
from src.recommendations import generate_recommendation_and_explanation
from src.database import init_db, update_store_dataset_meta, get_stores_for_user, add_store
from src.auth import register_user


class TestColumnAliasMapping(unittest.TestCase):
    """Tests for alias-based column name detection in data validation."""

    def _make_excel_bytes(self, df, sheet='Sales_Data'):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df.to_excel(w, sheet_name=sheet, index=False)
        buf.seek(0)
        return buf

    def test_units_sold_alias_mapped_to_sales(self):
        """'units_sold' should be auto-mapped to 'sales'."""
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        rows = []
        for d in dates:
            rows.append({
                'date': d,
                'store_name': 'TestStore',
                'product': 'Widget',
                'category': 'Electronics',
                'units_sold': 30.0,   # alias for 'sales'
                'unit_price': 5.0,
                'revenue': 150.0,
                'promotion': 0
            })
        df = pd.DataFrame(rows)
        buf = self._make_excel_bytes(df)
        res = validate_excel_file(buf)
        # Validation should pass and a warning about the alias mapping should exist
        self.assertTrue(res['is_valid'], msg=f"Errors: {res['errors']}")
        alias_warnings = [w for w in res['warnings'] if 'units_sold' in w and 'sales' in w]
        self.assertTrue(len(alias_warnings) > 0, "Expected alias mapping warning for 'units_sold'")

    def test_qty_alias_mapped_to_sales(self):
        """'qty' should be auto-mapped to 'sales'."""
        dates = pd.date_range('2025-02-01', periods=20, freq='D')
        rows = [{'date': d, 'store_name': 'S', 'product': 'P', 'category': 'C',
                 'qty': 10.0, 'unit_price': 2.0, 'revenue': 20.0, 'promotion': 0}
                for d in dates]
        df = pd.DataFrame(rows)
        buf = self._make_excel_bytes(df)
        res = validate_excel_file(buf)
        self.assertTrue(res['is_valid'], msg=f"Errors: {res['errors']}")

    def test_revenue_calculated_from_sales_and_price(self):
        """When 'revenue' column is missing, it should be computed from sales × unit_price."""
        dates = pd.date_range('2025-03-01', periods=20, freq='D')
        rows = [{'date': d, 'store_name': 'S', 'product': 'P', 'category': 'C',
                 'sales': 10.0, 'unit_price': 3.0, 'promotion': 0}
                for d in dates]
        df = pd.DataFrame(rows)
        buf = self._make_excel_bytes(df)
        res = validate_excel_file(buf)
        self.assertTrue(res['is_valid'], msg=f"Errors: {res['errors']}")
        self.assertTrue(res['revenue_calculated'], "Expected revenue_calculated=True")
        rev_warnings = [w for w in res['warnings'] if 'Revenue' in w and 'calculated' in w]
        self.assertTrue(len(rev_warnings) > 0, "Expected a revenue-calculated warning")

    def test_missing_column_without_alias_still_fails(self):
        """A column with no alias should still fail validation."""
        dates = pd.date_range('2025-01-01', periods=20, freq='D')
        rows = [{'date': d, 'store_name': 'S', 'product': 'P', 'category': 'C',
                 'sales': 10.0, 'unit_price': 2.0, 'revenue': 20.0, 'promotion': 0}
                for d in dates]
        df = pd.DataFrame(rows).drop(columns=['category'])  # no alias for 'category'
        buf = self._make_excel_bytes(df)
        res = validate_excel_file(buf)
        self.assertFalse(res['is_valid'])
        self.assertTrue(any('category' in e for e in res['errors']))

    def test_apply_column_aliases_does_not_overwrite_existing(self):
        """If the canonical column already exists, alias should NOT rename it."""
        df = pd.DataFrame({'sales': [1, 2], 'units_sold': [3, 4], 'other': [5, 6]})
        renamed, notes = apply_column_aliases(df)
        # 'sales' already exists, so 'units_sold' should NOT be renamed to 'sales'
        self.assertIn('sales', renamed.columns)
        self.assertIn('units_sold', renamed.columns)  # original alias stays
        self.assertEqual(len(notes), 0)


class TestRecommendations(unittest.TestCase):
    """Tests that advisories do not contain fabricated claims."""

    def _build_pred(self, badge, prob, threshold=0.5, sales_vs_7d=0.0):
        return {
            'is_surge': badge == 'SURGE',
            'badge_status': badge,
            'probability': prob,
            'threshold_used': threshold,
            'category': 'Beverages',
            'feature_values': {
                'sales_today': 50.0,
                'past_7d_avg': 48.0,
                'past_30d_avg': 46.0,
                'sales_vs_7d_avg': sales_vs_7d,
                'onpromotion': 0
            }
        }

    def test_normal_advisory_no_inventory_claim(self):
        """Normal advisory must NOT claim inventory is aligned or sufficient."""
        pred = self._build_pred('NORMAL', 0.10)
        result = generate_recommendation_and_explanation(pred)
        advisory = result['advisory'].lower()
        self.assertNotIn('inventory coverage is aligned', advisory)
        self.assertNotIn('sufficient', advisory)
        self.assertNotIn('aligned with standard', advisory)

    def test_surge_advisory_mentions_review(self):
        """Surge advisory should suggest reviewing, not claim specific quantities."""
        pred = self._build_pred('SURGE', 0.75, sales_vs_7d=0.4)
        result = generate_recommendation_and_explanation(pred)
        advisory = result['advisory']
        self.assertIn('review', advisory.lower())
        # Must not claim specific inventory levels
        self.assertNotIn('coverage is aligned', advisory.lower())

    def test_watch_advisory_mentions_monitoring(self):
        """Watch advisory should mention monitoring."""
        pred = self._build_pred('WATCH', 0.48)
        result = generate_recommendation_and_explanation(pred)
        advisory = result['advisory'].lower()
        self.assertIn('monitor', advisory)

    def test_explanation_references_actual_values(self):
        """Explanation must reference actual computed feature values."""
        pred = self._build_pred('SURGE', 0.8, sales_vs_7d=0.5)
        result = generate_recommendation_and_explanation(pred)
        explanation = result['explanation']
        # Should mention the actual numbers from feature_values
        self.assertIn('50', explanation)  # sales_today


class TestDatabaseMetadataColumns(unittest.TestCase):
    """Tests for new dataset metadata columns on the stores table."""

    def setUp(self):
        import tempfile
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        init_db(self.temp_db_path)

    def tearDown(self):
        try:
            os.close(self.temp_db_fd)
            if os.path.exists(self.temp_db_path):
                os.remove(self.temp_db_path)
        except Exception:
            pass

    def test_update_store_dataset_meta_persists_fields(self):
        """update_store_dataset_meta should persist all metadata fields atomically."""
        ok, _, user = register_user(
            'metauser', 'meta@test.com', 'pass123', 'pass123', 'Meta Biz', 'Meta Owner',
            seed_demo=False, db_path=self.temp_db_path
        )
        self.assertTrue(ok)
        store_id = add_store(user['id'], 'Meta Store', db_path=self.temp_db_path)
        updated = update_store_dataset_meta(
            store_id=store_id,
            user_id=user['id'],
            dataset_path='/some/path.xlsx',
            dataset_filename='path.xlsx',
            dataset_upload_date='2026-10-18 12:00',
            dataset_row_count=500,
            dataset_date_min='2025-01-01',
            dataset_date_max='2025-12-31',
            dataset_revenue_calculated=1,
            dataset_column_warnings='qty → sales',
            db_path=self.temp_db_path
        )
        self.assertTrue(updated)

        stores = get_stores_for_user(user['id'], db_path=self.temp_db_path)
        s = next(s for s in stores if s['id'] == store_id)
        self.assertEqual(s['dataset_filename'], 'path.xlsx')
        self.assertEqual(s['dataset_row_count'], 500)
        self.assertEqual(s['dataset_date_min'], '2025-01-01')
        self.assertEqual(s['dataset_date_max'], '2025-12-31')
        self.assertEqual(s['dataset_revenue_calculated'], 1)
        self.assertEqual(s['dataset_column_warnings'], 'qty → sales')

    def test_update_wrong_user_does_not_modify(self):
        """update_store_dataset_meta with wrong user_id should not modify the store."""
        ok, _, user = register_user(
            'metauser2', 'meta2@test.com', 'pass123', 'pass123', 'B', 'O',
            seed_demo=False, db_path=self.temp_db_path
        )
        self.assertTrue(ok)
        store_id = add_store(user['id'], 'Secure Store', db_path=self.temp_db_path)
        updated = update_store_dataset_meta(
            store_id=store_id,
            user_id=9999,  # wrong user_id
            dataset_path='/hack.xlsx',
            dataset_filename='hack.xlsx',
            dataset_upload_date='2026-01-01',
            dataset_row_count=1,
            dataset_date_min=None,
            dataset_date_max=None,
            db_path=self.temp_db_path
        )
        self.assertFalse(updated, "Should not update when user_id does not match")


if __name__ == '__main__':
    unittest.main()
