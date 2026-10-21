"""
Data Validation Module for DemandWise
=====================================
Validates uploaded and loaded Excel datasets before use in analysis or ML pipeline.
Enforces schema requirements, data types, missing values, duplicates, and range constraints.

Column alias mapping: Common alternative column names are detected and mapped automatically.
Revenue fallback: If revenue is absent but sales and unit_price are present, revenue is
calculated as sales × unit_price and flagged as a warning (not an error).
"""

import io
import os
import openpyxl
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, IO

REQUIRED_COLUMNS: List[str] = [
    'date',
    'store_name',
    'product',
    'category',
    'sales',
    'unit_price',
    'revenue',
    'promotion'
]

RECOMMENDED_SHEET_NAME: str = 'Sales_Data'

# ---------------------------------------------------------------------------
# Column alias mapping
# Maps alternative column names (lowercase) → canonical column name
# Longer / more specific aliases are matched first to avoid conflicts.
# ---------------------------------------------------------------------------
COLUMN_ALIASES: Dict[str, str] = {
    # date variants
    'date_of_sale': 'date',
    'sale_date': 'date',
    'transaction_date': 'date',
    'order_date': 'date',
    'day': 'date',

    # store_name variants
    'store': 'store_name',
    'store_id': 'store_name',
    'store_identifier': 'store_name',
    'branch': 'store_name',
    'branch_name': 'store_name',
    'location_name': 'store_name',

    # product variants
    'product_name': 'product',
    'item': 'product',
    'item_name': 'product',
    'sku': 'product',
    'product_id': 'product',

    # category variants
    'product_category': 'category',
    'cat': 'category',
    'dept': 'category',
    'department': 'category',
    'section': 'category',

    # sales (units) variants
    'units_sold': 'sales',
    'qty': 'sales',
    'quantity': 'sales',
    'quantity_sold': 'sales',
    'units': 'sales',
    'vol': 'sales',
    'volume': 'sales',
    'sold_qty': 'sales',

    # unit_price variants
    'price': 'unit_price',
    'selling_price': 'unit_price',
    'price_per_unit': 'unit_price',
    'cost': 'unit_price',
    'unit_cost': 'unit_price',

    # revenue variants
    'total_revenue': 'revenue',
    'total_sales': 'revenue',
    'amount': 'revenue',
    'sales_amount': 'revenue',
    'gross_revenue': 'revenue',
    'total': 'revenue',
    'value': 'revenue',

    # promotion variants
    'promo': 'promotion',
    'on_promo': 'promotion',
    'on_promotion': 'promotion',
    'is_promotion': 'promotion',
    'discount': 'promotion',
    'is_promo': 'promotion',
    'promotional': 'promotion',
}


def apply_column_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    Attempts to map alternative column names to canonical column names.
    Returns the renamed DataFrame and a list of applied mapping messages.

    Only renames a column if the alias target is not already present.
    Does not rename if the alias target already exists in the DataFrame.
    """
    mapping_notes: List[str] = []
    rename_map: Dict[str, str] = {}
    current_cols = set(df.columns)

    for alias, canonical in COLUMN_ALIASES.items():
        if alias in current_cols and canonical not in current_cols:
            rename_map[alias] = canonical
            current_cols.add(canonical)
            current_cols.discard(alias)
            mapping_notes.append(
                f"Column '{alias}' was automatically mapped to '{canonical}'."
            )

    if rename_map:
        df = df.rename(columns=rename_map)

    return df, mapping_notes


def validate_excel_file(
    file_source: Union[str, os.PathLike, bytes, IO[bytes]],
    target_sheet: Optional[str] = RECOMMENDED_SHEET_NAME
) -> Dict[str, Any]:
    """
    Validates an Excel file from a path, uploaded bytes, or binary file-like object.

    Returns a dictionary:
    - is_valid: bool
    - errors: List[str]
    - warnings: List[str]  (includes alias mappings, calculated revenue note)
    - sheet_name: str
    - row_count: int
    - date_min: Optional[str]
    - date_max: Optional[str]
    - unique_days: int
    - categories: List[str]
    - store_names: List[str]
    - revenue_calculated: bool  — True when revenue was derived from sales × unit_price
    - cleaned_df: Optional[pd.DataFrame]
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Open Workbook
    try:
        if isinstance(file_source, (str, os.PathLike)):
            if not os.path.exists(file_source):
                return {
                    'is_valid': False,
                    'errors': [f"File not found: {file_source}"],
                    'warnings': [],
                    'cleaned_df': None,
                    'revenue_calculated': False
                }
            wb = openpyxl.load_workbook(file_source, read_only=True)
        else:
            wb = openpyxl.load_workbook(
                io.BytesIO(file_source) if isinstance(file_source, bytes) else file_source,
                read_only=True
            )
        sheet_names = wb.sheetnames
        wb.close()
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Could not read Excel workbook: {str(e)}"],
            'warnings': [],
            'cleaned_df': None,
            'revenue_calculated': False
        }

    # 2. Identify Target Sheet
    chosen_sheet = None
    if target_sheet and target_sheet in sheet_names:
        chosen_sheet = target_sheet
    elif RECOMMENDED_SHEET_NAME in sheet_names:
        chosen_sheet = RECOMMENDED_SHEET_NAME
    elif len(sheet_names) == 1:
        chosen_sheet = sheet_names[0]
        warnings.append(
            f"Worksheet '{RECOMMENDED_SHEET_NAME}' was not found; using '{chosen_sheet}' instead."
        )
    else:
        errors.append(
            f"Expected worksheet '{RECOMMENDED_SHEET_NAME}' not found. "
            f"Available sheets: {', '.join(sheet_names)}"
        )
        return {
            'is_valid': False,
            'errors': errors,
            'warnings': warnings,
            'cleaned_df': None,
            'revenue_calculated': False
        }

    # 3. Read DataFrame
    try:
        if isinstance(file_source, (str, os.PathLike)):
            df = pd.read_excel(file_source, sheet_name=chosen_sheet)
        else:
            if isinstance(file_source, bytes):
                file_obj = io.BytesIO(file_source)
            else:
                file_obj = file_source
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
            df = pd.read_excel(file_obj, sheet_name=chosen_sheet)
    except Exception as e:
        errors.append(f"Failed to read sheet '{chosen_sheet}': {str(e)}")
        return {
            'is_valid': False,
            'errors': errors,
            'warnings': warnings,
            'cleaned_df': None,
            'revenue_calculated': False
        }

    if df.empty:
        errors.append("The selected worksheet contains no data rows.")
        return {
            'is_valid': False,
            'errors': errors,
            'warnings': warnings,
            'cleaned_df': None,
            'revenue_calculated': False
        }

    # Standardize column names (strip whitespace, lowercase)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # 4. Apply column alias mapping BEFORE checking required columns
    df, alias_notes = apply_column_aliases(df)
    warnings.extend(alias_notes)

    # 5. Revenue fallback: compute from sales × unit_price if revenue is missing
    revenue_calculated = False
    if 'revenue' not in df.columns and 'sales' in df.columns and 'unit_price' in df.columns:
        df['revenue'] = pd.to_numeric(df['sales'], errors='coerce') * pd.to_numeric(df['unit_price'], errors='coerce')
        revenue_calculated = True
        warnings.append(
            "Revenue column was not found. Revenue has been calculated as 'sales × unit_price'. "
            "Revenue-dependent figures in this dataset are calculated, not directly recorded."
        )

    # 6. Check Required Columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        # Produce a helpful message showing what we looked for and what alternatives exist
        err_lines = [
            f"Missing required columns: {', '.join(missing_cols)}.",
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}.",
            "If your column names differ, rename them to match the expected names, "
            "or check whether a column alias listed in the dataset guide applies.",
        ]
        errors.append(" ".join(err_lines))
        return {
            'is_valid': False,
            'errors': errors,
            'warnings': warnings,
            'cleaned_df': None,
            'revenue_calculated': revenue_calculated
        }

    # 7. Validate Date Column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    null_dates = int(df['date'].isna().sum())
    if null_dates > 0:
        errors.append(f"Found {null_dates} rows with invalid or blank dates.")

    # 8. Validate Categorical Fields
    for col in ['store_name', 'product', 'category']:
        normalized = df[col].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True)
        missing_count = int((normalized.isna() | normalized.eq('')).sum())
        if missing_count > 0:
            errors.append(f"Column '{col}' has {missing_count} missing or blank entries.")
        df[col] = normalized

    # 9. Validate Numeric Fields
    for num_col in ['sales', 'unit_price', 'revenue']:
        df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
        nan_num = int(df[num_col].isna().sum())
        if nan_num > 0:
            if num_col == 'revenue' and revenue_calculated:
                # Calculated revenue NaNs come from invalid source columns — warn, don't error
                warnings.append(
                    f"Calculated 'revenue' has {nan_num} missing values because 'sales' or "
                    f"'unit_price' were missing for those rows. Those rows will be excluded from "
                    f"revenue calculations."
                )
            else:
                errors.append(f"Column '{num_col}' contains {nan_num} non-numeric or blank values.")
        else:
            non_finite_count = int((~np.isfinite(df[num_col])).sum())
            if non_finite_count > 0:
                errors.append(
                    f"Column '{num_col}' contains {non_finite_count} infinite numeric values."
                )
            neg_count = int((df[num_col] < 0).sum())
            if neg_count > 0:
                if num_col in ('sales', 'unit_price'):
                    errors.append(
                        f"Column '{num_col}' contains {neg_count} negative values, which are invalid."
                    )
                else:
                    warnings.append(
                        f"Column '{num_col}' contains {neg_count} negative values. "
                        "This may indicate returns or adjustments."
                    )

    # Promotion check
    promotion = pd.to_numeric(df['promotion'], errors='coerce')
    invalid_promos = int(
        (promotion.isna() | ~np.isfinite(promotion.fillna(0)) | ~promotion.isin([0, 1])).sum()
    )
    if invalid_promos > 0:
        warnings.append(
            f"Column 'promotion' contains {invalid_promos} values other than 0 and 1; "
            "clipped to binary flag."
        )
    df['promotion'] = (promotion.fillna(0).replace([np.inf, -np.inf], 0) > 0).astype(int)

    # 10. Check Duplicates
    dup_mask = df.duplicated(subset=['date', 'store_name', 'category'], keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        warnings.append(
            f"Found {dup_count} duplicate records for the same date, store, and category. "
            "These will be aggregated to daily category totals during import."
        )

    # 11. History checks
    valid_dates = df['date'].dropna()
    unique_days = int(valid_dates.dt.normalize().nunique())
    if unique_days < 14:
        warnings.append(
            f"Limited history ({unique_days} days). Demand predictions require at least "
            "14 days of sales history; uploaded sales remain available for reporting."
        )
    elif unique_days < 30:
        warnings.append(
            f"Limited history ({unique_days} days). At least 30 days are recommended for "
            "optimal 14-day and 30-day features."
        )

    is_valid = len(errors) == 0
    categories = sorted(df['category'].dropna().unique().tolist()) if 'category' in df else []
    stores = sorted(df['store_name'].dropna().unique().tolist()) if 'store_name' in df else []
    date_min = valid_dates.min().strftime('%Y-%m-%d') if not valid_dates.empty else None
    date_max = valid_dates.max().strftime('%Y-%m-%d') if not valid_dates.empty else None

    return {
        'is_valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'sheet_name': chosen_sheet,
        'row_count': len(df),
        'date_min': date_min,
        'date_max': date_max,
        'unique_days': unique_days,
        'categories': categories,
        'category_count': len(categories),
        'store_names': stores,
        'revenue_calculated': revenue_calculated,
        'cleaned_df': df if is_valid else None
    }
