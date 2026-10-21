"""Tests for configured store dataset loading and date filtering."""

import io
import os
import tempfile
import unittest

import pandas as pd

from src.data_loader import filter_dataset_by_date_range, load_store_dataset


class TestStoreDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.source_store = "Workbook Store"
        dates = pd.date_range("2025-01-01", periods=35, freq="D")
        self.df = pd.DataFrame([
            {
                " Date ": date,
                "STORE": self.source_store,
                "Product_Name": f"{category} assortment",
                "Product_Category": f"  {category}   ",
                "Units_Sold": 10 + index,
                "Price": 2.5,
                "Total_Revenue": (10 + index) * 2.5,
                "Promo": index % 2,
            }
            for index, date in enumerate(dates)
            for category in ("Bakery", "Snacks")
        ])

    def _excel_bytes(self, df=None):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            (df if df is not None else self.df).to_excel(
                writer, sheet_name="Sales_Data", index=False
            )
        return buffer.getvalue()

    def test_loads_bytes_and_maps_workbook_store_to_configured_name(self):
        calendar, metadata = load_store_dataset(
            self._excel_bytes(),
            configured_store_name="  DemandWise   Store  "
        )

        self.assertEqual(len(calendar), 70)
        self.assertEqual(calendar["store_name"].unique().tolist(), ["DemandWise Store"])
        self.assertEqual(calendar["category"].unique().tolist(), ["Bakery", "Snacks"])
        self.assertEqual(calendar["date"].min(), pd.Timestamp("2025-01-01"))
        self.assertEqual(calendar["date"].max(), pd.Timestamp("2025-02-04"))
        self.assertEqual(metadata["row_count"], 70)
        self.assertEqual(metadata["store_names"], ["DemandWise Store"])

    def test_loads_uploaded_file_like_and_file_path(self):
        workbook_bytes = self._excel_bytes()
        uploaded_file = io.BytesIO(workbook_bytes)
        uploaded_file.name = "upload.xlsx"
        from_uploaded, _ = load_store_dataset(uploaded_file, "Configured Store")
        from_bytes, _ = load_store_dataset(workbook_bytes, "Configured Store")

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "dataset.xlsx")
            with open(file_path, "wb") as workbook:
                workbook.write(workbook_bytes)
            from_path, _ = load_store_dataset(file_path, "Configured Store")

        pd.testing.assert_frame_equal(from_uploaded, from_bytes)
        pd.testing.assert_frame_equal(from_path, from_bytes)

    def test_date_filter_includes_both_calendar_day_boundaries(self):
        calendar, _ = load_store_dataset(self._excel_bytes(), "Configured Store")

        selected = filter_dataset_by_date_range(
            calendar, "2025-01-03", "2025-01-05"
        )

        self.assertEqual(selected["date"].nunique(), 3)
        self.assertEqual(selected["date"].min(), pd.Timestamp("2025-01-03"))
        self.assertEqual(selected["date"].max(), pd.Timestamp("2025-01-05"))

    def test_short_history_remains_available_for_reporting(self):
        short_history = self.df[self.df[" Date "].isin(
            pd.date_range("2025-01-01", periods=7)
        )]

        calendar, metadata = load_store_dataset(
            self._excel_bytes(short_history), "Configured Store"
        )

        self.assertEqual(len(calendar), 14)
        self.assertEqual(metadata["unique_days"], 7)
        self.assertTrue(any("at least 14 days" in warning for warning in metadata["warnings"]))

    def test_invalid_workbook_raises_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "Could not read Excel workbook"):
            load_store_dataset(b"not an Excel workbook", "Configured Store")

    def test_empty_workbook_data_raises_instead_of_returning_empty_frame(self):
        empty = self.df.iloc[0:0]

        with self.assertRaisesRegex(ValueError, "contains no data rows"):
            load_store_dataset(self._excel_bytes(empty), "Configured Store")


if __name__ == "__main__":
    unittest.main()
