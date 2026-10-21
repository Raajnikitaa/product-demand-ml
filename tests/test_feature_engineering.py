"""
Unit Tests for Feature Engineering & Target Calculation
======================================================
Verifies exact surge mathematical formula, incomplete window handling,
and strict temporal leakage prevention.
"""

import unittest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    compute_features_and_target,
    SURGE_THRESHOLD_MULTIPLIER,
    NUMERIC_FEATURES
)


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # 60 days of synthetic data with a known surge spike
        dates = pd.date_range('2025-01-01', periods=60, freq='D')
        rows = []
        for i, d in enumerate(dates):
            # Days 0..39 have steady 50 sales, Days 40..46 surge to 100 sales, Days 47..59 return to 50
            sales = 100.0 if (40 <= i <= 46) else 50.0
            rows.append({
                'date': d,
                'store_name': 'Store_Alpha',
                'category': 'Snacks',
                'sales': sales,
                'revenue': sales * 2.0,
                'unit_price': 2.0,
                'promotion': 1 if (40 <= i <= 46) else 0
            })
        self.df = pd.DataFrame(rows)

    def test_target_formula_and_incomplete_window(self):
        featured = compute_features_and_target(self.df)
        
        self.assertTrue(featured['demand_surge'].iloc[:6].isna().all())
        # Verify final 7 days have NaN target
        max_date = featured['date'].max()
        last_7_days = featured[featured['date'] > (max_date - pd.Timedelta(days=7))]
        self.assertTrue(last_7_days['demand_surge'].isna().all())
        
        # Verify interior rows match exact formula
        valid = featured.dropna(subset=['demand_surge', 'future_7d_avg', 'past_7d_avg'])
        for _, row in valid.iterrows():
            expected_surge = int(
                (row['future_7d_avg'] >= 1.30 * row['past_7d_avg']) and
                (row['future_7d_avg'] > row['past_7d_avg']) and
                (row['past_7d_avg'] > 0.0)
            )
            self.assertEqual(int(row['demand_surge']), expected_surge)

    def test_non_positive_baseline_is_excluded_from_target(self):
        df = self.df.copy()
        df.loc[df['date'] <= pd.Timestamp('2025-01-07'), 'sales'] = 0.0

        featured = compute_features_and_target(df)
        zero_baseline = featured[featured['date'] == pd.Timestamp('2025-01-07')]

        self.assertTrue(zero_baseline['past_7d_avg'].eq(0).all())
        self.assertTrue(zero_baseline['demand_surge'].isna().all())

    def test_feature_and_target_windows_do_not_mix_store_category_series(self):
        other_series = self.df.copy()
        other_series['store_name'] = 'Store_Beta'
        other_series['category'] = 'Bakery'
        other_series['sales'] = 500.0
        featured = compute_features_and_target(
            pd.concat([self.df, other_series], ignore_index=True)
        )

        alpha = featured[
            (featured['store_name'] == 'Store_Alpha') &
            (featured['date'] == pd.Timestamp('2025-02-05'))
        ].iloc[0]
        beta = featured[
            (featured['store_name'] == 'Store_Beta') &
            (featured['date'] == pd.Timestamp('2025-02-05'))
        ].iloc[0]
        self.assertEqual(alpha['past_7d_avg'], 50.0)
        self.assertEqual(beta['past_7d_avg'], 500.0)

    def test_strict_temporal_leakage_prevention(self):
        """
        Critical Leakage Verification:
        Modifying sales strictly in the future (day t + 1 .. t + 7) MUST NOT
        change any feature computed for day t.
        """
        featured_orig = compute_features_and_target(self.df)
        
        # Pick decision day t = Day 35 (where past 30-day window is fully populated)
        t_date = pd.to_datetime('2025-02-05')
        features_at_t_orig = featured_orig[featured_orig['date'] == t_date][NUMERIC_FEATURES].iloc[0].to_dict()
        
        # Alter future sales from day 36 onwards (massive 10x spike)
        df_altered = self.df.copy()
        future_mask = df_altered['date'] > t_date
        df_altered.loc[future_mask, 'sales'] = df_altered.loc[future_mask, 'sales'] * 10.0
        df_altered.loc[future_mask, 'promotion'] = 1
        
        featured_altered = compute_features_and_target(df_altered)
        features_at_t_altered = featured_altered[featured_altered['date'] == t_date][NUMERIC_FEATURES].iloc[0].to_dict()
        
        # Every single retrospective feature at day t must be identical
        for feat in NUMERIC_FEATURES:
            orig_val = features_at_t_orig[feat]
            altered_val = features_at_t_altered[feat]
            if pd.isna(orig_val) and pd.isna(altered_val):
                continue
            self.assertAlmostEqual(
                orig_val, altered_val, places=4,
                msg=f"Temporal Leakage Detected! Feature '{feat}' changed on day t when future was altered."
            )
            
        # Target for day t SHOULD change (because target looks forward to t+1..t+7)
        orig_target = featured_orig[featured_orig['date'] == t_date]['demand_surge'].iloc[0]
        altered_target = featured_altered[featured_altered['date'] == t_date]['demand_surge'].iloc[0]
        self.assertEqual(altered_target, 1.0)


if __name__ == '__main__':
    unittest.main()
