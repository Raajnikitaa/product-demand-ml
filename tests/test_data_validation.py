"""
Unit Tests for Data Validation Module
====================================
Verifies schema enforcement, missing value detection, range checks, and error reporting.
"""

import os
import io
import unittest
import pandas as pd
import numpy as np

from src.data_validation import validate_excel_file, REQUIRED_COLUMNS


class TestDataValidation(unittest.TestCase):

    def setUp(self):
        # Create a valid synthetic DataFrame
        dates = pd.date_range('2025-01-01', periods=35, freq='D')
        rows = []
        for d in dates:
            for cat in ['Groceries', 'Beverages']:
                rows.append({
                    'date': d,
                    'store_name': 'Test_Store',
                    'product': f"{cat} Assortment",
                    'category': cat,
                    'sales': 45.0,
                    'unit_price': 3.5,
                    'revenue': 157.5,
                    'promotion': 0
                })
        self.valid_df = pd.DataFrame(rows)

    def _df_to_excel_bytes(self, df: pd.DataFrame, sheet_name: str = 'Sales_Data') -> io.BytesIO:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        buf.seek(0)
        return buf

    def test_valid_excel_workbook(self):
        buf = self._df_to_excel_bytes(self.valid_df)
        res = validate_excel_file(buf)
        self.assertTrue(res['is_valid'])
        self.assertEqual(len(res['errors']), 0)
        self.assertEqual(res['unique_days'], 35)
        self.assertEqual(res['category_count'], 2)

    def test_valid_excel_bytes_input(self):
        buf = self._df_to_excel_bytes(self.valid_df)
        raw_bytes = buf.getvalue()
        res = validate_excel_file(raw_bytes)
        self.assertTrue(res['is_valid'])
        self.assertEqual(len(res['errors']), 0)

    def test_missing_required_column(self):
        incomplete = self.valid_df.drop(columns=['sales'])
        buf = self._df_to_excel_bytes(incomplete)
        res = validate_excel_file(buf)
        self.assertFalse(res['is_valid'])
        self.assertTrue(any('sales' in err for err in res['errors']))

    def test_negative_sales_rejected(self):
        neg_df = self.valid_df.copy()
        neg_df.loc[0, 'sales'] = -15.0
        buf = self._df_to_excel_bytes(neg_df)
        res = validate_excel_file(buf)
        self.assertFalse(res['is_valid'])
        self.assertTrue(any('negative' in err for err in res['errors']))

    def test_missing_sheet_name_handled(self):
        # Workbook with wrong sheet name
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            self.valid_df.to_excel(writer, sheet_name='Sheet1', index=False)
            self.valid_df.to_excel(writer, sheet_name='Sheet2', index=False)
        buf.seek(0)
        res = validate_excel_file(buf)
        self.assertFalse(res['is_valid'])
        self.assertTrue(any('Sales_Data' in err for err in res['errors']))

    def test_nonexistent_file(self):
        res = validate_excel_file("data/non_existent_file_xyz.xlsx")
        self.assertFalse(res['is_valid'])
        self.assertTrue(any('not found' in err for err in res['errors']))


if __name__ == '__main__':
    unittest.main()
