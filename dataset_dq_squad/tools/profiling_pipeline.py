from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from ..models import DatasetProfile, ColumnProfile, CorrelationPair


def _infer_logical_dtype(series: pd.Series, distinct_count: int, row_count: int) -> str:
    """Infer the logical data type of a column."""
    # Try datetime first
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Try to parse as datetime if object
    if series.dtype == 'object' and distinct_count > 2:
        try:
            pd.to_datetime(series.dropna().head(100), errors='raise')
            return "datetime"
        except:
            pass
    
    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        # Boolean-like numeric (only 0/1 or True/False)
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            return "boolean"
        # Categorical if low cardinality
        if distinct_count <= 20:
            return "categorical"
        return "numeric"
    
    # Boolean
    if series.dtype == 'bool':
        return "boolean"
    
    # Object/string types
    if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
        # Boolean-like strings
        unique_vals_lower = set(str(v).lower() for v in series.dropna().unique()[:10])
        if unique_vals_lower.issubset({'true', 'false', 'yes', 'no', '0', '1', 't', 'f', 'y', 'n'}):
            return "boolean"
        # Categorical if low cardinality
        if distinct_count <= 20:
            return "categorical"
        # Text if average length > 50
        avg_length = series.dropna().astype(str).str.len().mean()
        if avg_length > 50:
            return "text"
        return "categorical"
    
    return "unknown"


def run_profiling_pipeline(dataset_path: str) -> DatasetProfile:
    """
    Enhanced profiling pipeline with:
    - Logical dtype inference
    - Sample values and top values for categorical
    - Duplicate detection
    - Time-series gap analysis
    - Basic stats, outliers, correlations
    """
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    df = pd.read_csv(path)
    row_count = int(len(df))

    # --- Duplicate analysis ------------------------------------------------
    duplicate_count = int(df.duplicated().sum())
    duplicate_ratio = float(duplicate_count) / float(row_count) if row_count > 0 else 0.0

    # --- Datetime detection ------------------------------------------------
    datetime_cols = []
    datetime_info: Dict[str, Any] = {}

    # --- Per-column profiling ----------------------------------------------
    columns: List[ColumnProfile] = []

    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        distinct_count = int(s.nunique(dropna=True))

        # Logical dtype
        logical_dtype = _infer_logical_dtype(s, distinct_count, row_count)

        # Sample values (first 5 unique non-null)
        sample_values = [str(v) for v in s.dropna().unique()[:5]]

        # Top values for categorical
        top_values = None
        if logical_dtype in ["categorical", "boolean"]:
            value_counts = s.value_counts().head(5)
            top_values = {str(k): int(v) for k, v in value_counts.items()}

        # Numeric stats
        col_min = col_max = col_mean = col_std = col_skew = None
        z_out_ratio = iqr_out_ratio = None

        if pd.api.types.is_numeric_dtype(s):
            s_clean = s.dropna()
            if not s_clean.empty:
                col_min = float(s_clean.min())
                col_max = float(s_clean.max())
                col_mean = float(s_clean.mean())
                col_std = float(s_clean.std(ddof=1)) if len(s_clean) > 1 else 0.0
                col_skew = float(s_clean.skew()) if len(s_clean) > 2 else 0.0

                n_clean = len(s_clean)

                # Z-score outliers
                if col_std and col_std > 0:
                    z_scores = (s_clean - col_mean) / col_std
                    z_outliers = (z_scores.abs() > 3.0).sum()
                    z_out_ratio = float(z_outliers) / float(n_clean) if n_clean > 0 else 0.0
                else:
                    z_out_ratio = 0.0

                # IQR outliers
                q1 = float(s_clean.quantile(0.25))
                q3 = float(s_clean.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    iqr_outliers = ((s_clean < lower) | (s_clean > upper)).sum()
                    iqr_out_ratio = float(iqr_outliers) / float(n_clean) if n_clean > 0 else 0.0
                else:
                    iqr_out_ratio = 0.0

        # Datetime analysis
        if logical_dtype == "datetime":
            datetime_cols.append(col)
            try:
                dt_series = pd.to_datetime(s.dropna(), errors='coerce')
                if not dt_series.empty:
                    dt_min = str(dt_series.min())
                    dt_max = str(dt_series.max())
                    # Check for gaps (simplified - just count vs expected)
                    dt_range = dt_series.max() - dt_series.min()
                    expected_days = dt_range.days if hasattr(dt_range, 'days') else 0
                    actual_count = len(dt_series.unique())
                    datetime_info[col] = {
                        "min": dt_min,
                        "max": dt_max,
                        "unique_dates": actual_count,
                        "range_days": expected_days
                    }
            except:
                pass

        columns.append(
            ColumnProfile(
                name=col,
                dtype=str(s.dtype),
                null_count=null_count,
                distinct_count=distinct_count,
                min=col_min,
                max=col_max,
                mean=col_mean,
                std=col_std,
                skew=col_skew,
                zscore_outlier_ratio=z_out_ratio,
                iqr_outlier_ratio=iqr_out_ratio,
                logical_dtype=logical_dtype,
                sample_values=sample_values if sample_values else None,
                top_values=top_values,
            )
        )

    # --- Numeric correlations ----------------------------------------------
    corr_pairs: List[CorrelationPair] = []
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        for i, col1 in enumerate(corr.columns):
            for j, col2 in enumerate(corr.columns):
                if j <= i:
                    continue
                pearson = corr.loc[col1, col2]
                if pd.isna(pearson):
                    continue
                corr_pairs.append(
                    CorrelationPair(col1=col1, col2=col2, pearson=float(pearson))
                )

    return DatasetProfile(
        dataset_name=path.stem,
        row_count=row_count,
        column_count=len(df.columns),
        columns=columns,
        correlations=corr_pairs or None,
        duplicate_count=duplicate_count,
        duplicate_ratio=duplicate_ratio,
        has_datetime_columns=len(datetime_cols) > 0,
        datetime_info=datetime_info if datetime_info else None,
    )
