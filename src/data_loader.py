"""
Data Loader Module for DemandWise
=================================
Discovers Excel workbooks, parses daily sales data, aggregates to store-category level,
and constructs continuous daily calendars for reliable rolling window calculations.
"""

import os
import glob
import io
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union, IO

from src.data_validation import validate_excel_file, REQUIRED_COLUMNS


def get_default_data_dir() -> str:
    """Returns absolute path to the data directory, dynamically locating data/ or Data/."""
    candidates = [
        os.path.abspath('data'),
        os.path.abspath('Data'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data'),
        os.path.abspath(os.path.join('..', 'data')),
        os.path.abspath(os.path.join('..', 'Data'))
    ]
    for c in candidates:
        if os.path.exists(c):
            xlsx_files = []
            for pattern in (os.path.join(c, "*.xlsx"), os.path.join(c, "**", "*.xlsx")):
                xlsx_files.extend(
                    f for f in glob.glob(pattern, recursive=True)
                    if not os.path.basename(f).startswith("~$")
                )
            if xlsx_files:
                return c
    return candidates[0]


def get_user_data_dir(user_id: int) -> str:
    """
    Returns the per-user isolated upload directory: data/users/{user_id}/
    Created on first call. Uploads for different users are fully separate.
    """
    base = os.path.abspath('data')
    # Try to resolve to the project root
    project_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
        os.path.abspath('data'),
    ]
    for c in project_candidates:
        if os.path.exists(c):
            base = c
            break
    user_dir = os.path.join(base, 'users', str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def discover_excel_files(data_dir: Optional[str] = None) -> List[str]:
    """
    Discovers all .xlsx files in the configured data directory.
    Includes nested user-upload folders such as data/users/<user_id>/...
    Returns sorted list of paths.
    """
    target_dir = data_dir or get_default_data_dir()
    if not os.path.exists(target_dir):
        return []

    patterns = [
        os.path.join(target_dir, "*.xlsx"),
        os.path.join(target_dir, "**", "*.xlsx"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    unique_files = []
    seen = set()
    for f in files:
        if os.path.basename(f).startswith("~$"):
            continue
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    return sorted(unique_files)


def load_single_store_excel(
    file_path_or_buffer: Union[str, os.PathLike, bytes, IO[bytes]],
    target_sheet: Optional[str] = 'Sales_Data'
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Loads and validates a single store Excel workbook.
    Returns (cleaned_df, validation_meta).
    """
    val_result = validate_excel_file(file_path_or_buffer, target_sheet=target_sheet)
    if not val_result['is_valid']:
        return None, val_result
        
    df = val_result['cleaned_df']
    return df, val_result


def resolve_dataset_path(file_path: Union[str, os.PathLike]) -> str:
    """Resolve configured relative paths from either the current directory or project root."""
    configured_path = os.fspath(file_path)
    candidates = [os.path.abspath(configured_path)]
    if not os.path.isabs(configured_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates.append(os.path.abspath(os.path.join(project_root, configured_path)))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def load_store_dataset(
    file_source: Union[str, os.PathLike, bytes, IO[bytes]],
    configured_store_name: str,
    target_sheet: Optional[str] = 'Sales_Data'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and normalize one store's dataset into daily store-category calendar rows.

    The configured store name is authoritative because uploaded workbook store names
    may differ from the name used by the application and its database.
    """
    source = (
        resolve_dataset_path(file_source)
        if isinstance(file_source, (str, os.PathLike))
        else file_source
    )
    source_label = os.fspath(source) if isinstance(source, os.PathLike) else (
        source if isinstance(source, str) else "uploaded workbook"
    )

    store_name = " ".join(str(configured_store_name).split())
    if not store_name:
        raise ValueError("A configured store name is required to load this dataset.")

    raw_df, validation = load_single_store_excel(source, target_sheet=target_sheet)
    if raw_df is None or raw_df.empty:
        errors = validation.get('errors') or ["The workbook contains no usable records."]
        raise ValueError(
            f"Could not load dataset '{source_label}': {' '.join(errors)}"
        )

    df = raw_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    if df['date'].isna().any():
        invalid_count = int(df['date'].isna().sum())
        raise ValueError(f"Dataset contains {invalid_count} rows with invalid dates.")

    for column in ('store_name', 'product', 'category'):
        df[column] = (
            df[column]
            .astype('string')
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
        )
    if df['category'].isna().any() or df['category'].eq('').any():
        raise ValueError("Dataset contains rows without a category.")

    df['store_name'] = store_name
    for column in ('sales', 'unit_price', 'revenue'):
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if not np.isfinite(df[column].dropna()).all() or df[column].isna().any():
            raise ValueError(f"Dataset contains missing or invalid numeric values in '{column}'.")
    df['promotion'] = pd.to_numeric(df['promotion'], errors='coerce').fillna(0).astype(int)

    daily = aggregate_daily_category_sales(df)
    calendar, _ = build_complete_store_calendar(daily)
    if calendar.empty:
        raise ValueError("The workbook contains no usable daily sales records.")

    metadata = {
        key: value
        for key, value in validation.items()
        if key != 'cleaned_df'
    }
    metadata['store_names'] = [store_name]
    metadata['categories'] = sorted(calendar['category'].dropna().unique().tolist())
    metadata['category_count'] = len(metadata['categories'])
    metadata['row_count'] = len(df)
    metadata['date_min'] = df['date'].min().strftime('%Y-%m-%d')
    metadata['date_max'] = df['date'].max().strftime('%Y-%m-%d')
    return calendar, metadata


def filter_dataset_by_date_range(
    df: pd.DataFrame,
    start_date: Any,
    end_date: Any
) -> pd.DataFrame:
    """Return rows within an inclusive, calendar-day date range."""
    if 'date' not in df.columns:
        raise ValueError("Cannot filter dataset: required 'date' column is missing.")

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    return df.loc[dates.between(start, end)].copy()


def aggregate_daily_category_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates granular sales records to daily store-category observations.
    Combines sales and revenue as sums, promotion as binary indicator (max).
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    grouped = df.groupby(['store_name', 'category', 'date'], as_index=False).agg({
        'sales': 'sum',
        'revenue': 'sum',
        'unit_price': 'mean',
        'promotion': 'max'
    })
    
    # Sort strictly chronologically
    return grouped.sort_values(['store_name', 'category', 'date']).reset_index(drop=True)


def build_complete_store_calendar(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds a continuous daily calendar for every (store_name, category) pair in the data.
    Ensures missing calendar days are explicitly imputed as 0 sales, eliminating rolling gaps.
    """
    if df.empty:
        return df, {'missing_dates_filled': 0}
        
    min_date = df['date'].min()
    max_date = df['date'].max()
    all_dates = pd.date_range(min_date, max_date, freq='D')
    
    observed_pairs = df[['store_name', 'category']].drop_duplicates()
    grid = observed_pairs.merge(
        pd.DataFrame({'date': all_dates}),
        how='cross'
    )
    
    # Merge with actual aggregated daily records
    merged = pd.merge(grid, df, on=['date', 'store_name', 'category'], how='left')
    missing_count = int(merged['sales'].isna().sum())
    
    # Impute missing non-operational days with 0
    merged['sales'] = merged['sales'].fillna(0.0).astype(float)
    merged['revenue'] = merged['revenue'].fillna(0.0).astype(float)
    merged['unit_price'] = merged['unit_price'].fillna(0.0).astype(float)
    merged['promotion'] = merged['promotion'].fillna(0).astype(int)
    
    # Sort chronologically by store, category, and date
    sorted_df = merged.sort_values(['store_name', 'category', 'date']).reset_index(drop=True)
    
    meta = {
        'min_date': min_date,
        'max_date': max_date,
        'total_calendar_days': len(all_dates),
        'missing_dates_filled': missing_count,
        'total_series': len(observed_pairs)
    }
    return sorted_df, meta


def load_all_demo_stores(data_dir: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], List[Dict[str, Any]]]:
    """
    Discovers and loads all demonstration Excel workbooks found in data/.
    Returns (combined_daily_calendar_df, list_of_store_metadatas).
    """
    files = discover_excel_files(data_dir)
    if not files:
        return None, []
        
    dfs = []
    store_metas = []
    for f in files:
        df, meta = load_single_store_excel(f)
        if df is not None and not df.empty:
            dfs.append(df)
            store_metas.append({
                'file_name': os.path.basename(f),
                'file_path': f,
                'store_names': meta.get('store_names', []),
                'row_count': meta.get('row_count', 0),
                'date_min': meta.get('date_min'),
                'date_max': meta.get('date_max'),
                'categories': meta.get('categories', [])
            })
            
    if not dfs:
        return None, store_metas
        
    raw_combined = pd.concat(dfs, ignore_index=True)
    daily_agg = aggregate_daily_category_sales(raw_combined)
    calendar_df, _ = build_complete_store_calendar(daily_agg)
    return calendar_df, store_metas
