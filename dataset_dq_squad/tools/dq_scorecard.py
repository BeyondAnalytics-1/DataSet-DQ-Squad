from __future__ import annotations

import json
from typing import Any, Dict

from ..models import (
    DatasetProfile,
    DQScorecard,
    ColumnScore,
    DQIssuesByCategory,
)


def build_dq_scorecard(dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a richer Data Quality scorecard from a DatasetProfile JSON-like dict.

    Uses:
      - Missingness severity tiers (low / medium / high)
      - Uniqueness classification (ID-like, categorical, constant)
      - Constant / near-constant columns
      - Skewness issues for heavily skewed numeric columns
      - Outlier risk based on (max - min) vs std
      - Outlier ratios (Z-score, IQR) from profiling
      - Correlation-based redundancy flags

    Returns a plain dict suitable for ADK tool output.
    """
    # Handle string input (if LLM passes JSON string)
    if isinstance(dataset_profile, str):
        try:
            dataset_profile = json.loads(dataset_profile)
        except json.JSONDecodeError:
            print(f"[build_dq_scorecard] Failed to parse dataset_profile string: {dataset_profile[:100]}...")
            raise ValueError("dataset_profile must be a valid JSON object or string")

    print(f"[build_dq_scorecard] called for dataset: {dataset_profile.get('dataset_name')}", flush=True)
    
    # Validate / normalize input using pydantic
    try:
        # Patch missing top-level keys if necessary (LLM resilience)
        if isinstance(dataset_profile, dict):
            if "dataset_name" not in dataset_profile:
                print("[build_dq_scorecard] WARNING: 'dataset_name' missing, defaulting to 'unknown'")
                dataset_profile["dataset_name"] = "unknown"
            if "row_count" not in dataset_profile:
                print("[build_dq_scorecard] WARNING: 'row_count' missing, defaulting to 0")
                dataset_profile["row_count"] = 0
            if "column_count" not in dataset_profile:
                cols = dataset_profile.get("columns", [])
                print(f"[build_dq_scorecard] WARNING: 'column_count' missing, inferred {len(cols)}")
                dataset_profile["column_count"] = len(cols)
            
            if "columns" not in dataset_profile:
                print("[build_dq_scorecard] WARNING: 'columns' missing, defaulting to empty list")
                dataset_profile["columns"] = []
            
            # Patch columns if it's a dict instead of a list (LLM hallucination)
            if "columns" in dataset_profile and isinstance(dataset_profile["columns"], dict):
                print("[build_dq_scorecard] WARNING: 'columns' is a dict, converting to list")
                new_cols = []
                for col_name, col_data in dataset_profile["columns"].items():
                    if isinstance(col_data, dict):
                        col_data["name"] = col_name  # Ensure name is present
                        new_cols.append(col_data)
                dataset_profile["columns"] = new_cols

            # Patch missing column keys
            if "columns" in dataset_profile and isinstance(dataset_profile["columns"], list):
                for col in dataset_profile["columns"]:
                    if isinstance(col, dict):
                        if "name" not in col:
                            col["name"] = "unknown_col"
                        if "dtype" not in col:
                            col["dtype"] = "unknown"
                        if "null_count" not in col:
                            col["null_count"] = 0
                        if "distinct_count" not in col:
                            col["distinct_count"] = 0

        profile = DatasetProfile.model_validate(dataset_profile)
    except Exception as e:
        print(f"[build_dq_scorecard] Validation error: {e}")
        # Try to be lenient if possible, or just re-raise
        raise e

    row_count = max(1, profile.row_count)
    columns = profile.columns

    issues_by_category = DQIssuesByCategory()
    column_scores: list[ColumnScore] = []
    total_score = 0.0

    for col in columns:
        null_ratio = col.null_count / row_count
        distinct_ratio = col.distinct_count / row_count

        score = 100.0
        col_issues: list[str] = []

        # --- Missingness severity -----------------------------------------
        if null_ratio > 0:
            if null_ratio < 0.05:
                severity = "low"
                penalty = 10
            elif null_ratio < 0.20:
                severity = "medium"
                penalty = 25
            else:
                severity = "high"
                penalty = 50

            score -= penalty
            issues_by_category.missing_values.append(
                f"{col.name}: {severity} missingness ({null_ratio:.1%} missing)"
            )
            col_issues.append(
                f"{severity.capitalize()} missing values ({null_ratio:.1%})"
            )

        # --- Uniqueness & cardinality -------------------------------------
        if distinct_ratio < 0.01:
            issues_by_category.uniqueness.append(
                f"{col.name}: near-constant values (distinct_ratio={distinct_ratio:.2%})"
            )
            col_issues.append("Near-constant column; may be uninformative")
            score -= 10
        elif distinct_ratio < 0.05:
            issues_by_category.uniqueness.append(
                f"{col.name}: very low distinct ratio (distinct_ratio={distinct_ratio:.2%})"
            )
            col_issues.append("Low cardinality; likely categorical")
        elif distinct_ratio > 0.95:
            col_issues.append("High uniqueness; likely identifier")

        # --- Numeric validity & derived metrics ----------------------------
        is_numeric = col.dtype.startswith("float") or col.dtype.startswith("int")
        if is_numeric:
            # Missing min/max is suspicious
            if col.min is None or col.max is None:
                issues_by_category.validity.append(
                    f"{col.name}: numeric column with no min/max computed"
                )
                col_issues.append("Numeric column without min/max")
                score -= 10

            # Constant numeric column (std ≈ 0)
            if col.std is not None and col.std == 0:
                issues_by_category.validity.append(
                    f"{col.name}: constant numeric column (std=0)"
                )
                col_issues.append("Constant numeric column (std=0)")
                score -= 15

            # Skewness
            if col.skew is not None:
                abs_skew = abs(col.skew)
                if abs_skew > 2.0:
                    issues_by_category.validity.append(
                        f"{col.name}: heavily skewed distribution (skew={col.skew:.2f})"
                    )
                    col_issues.append(
                        f"Heavily skewed distribution (skew={col.skew:.2f}); "
                        "may require log/Box-Cox transformation or winsorization"
                    )
                    score -= 15
                elif abs_skew > 1.0:
                    issues_by_category.validity.append(
                        f"{col.name}: moderately skewed distribution (skew={col.skew:.2f})"
                    )
                    col_issues.append(
                        f"Moderately skewed distribution (skew={col.skew:.2f})"
                    )
                    score -= 5

            # Outlier risk (range vs std)
            if (
                col.min is not None
                and col.max is not None
                and col.std is not None
                and col.std > 0
            ):
                value_range = col.max - col.min
                if value_range > 20 * col.std:
                    issues_by_category.validity.append(
                        f"{col.name}: very wide range compared to std "
                        f"(range={value_range:.2f}, std={col.std:.2f})"
                    )
                    col_issues.append(
                        "Very wide range vs std; potential outliers present"
                    )
                    score -= 10

            # --- NEW: Outlier ratios (Z-score & IQR) ----------------------
            # Thresholds are heuristic; tweak if needed.
            if col.zscore_outlier_ratio is not None:
                z_ratio = col.zscore_outlier_ratio
                if z_ratio > 0.10:  # >10% of values are |z|>3
                    issues_by_category.validity.append(
                        f"{col.name}: high Z-score outlier ratio ({z_ratio:.1%})"
                    )
                    col_issues.append(
                        f"High proportion of Z-score outliers ({z_ratio:.1%}); "
                        "consider winsorizing, clipping, or removing extreme values"
                    )
                    score -= 10
                elif z_ratio > 0.05:
                    issues_by_category.validity.append(
                        f"{col.name}: moderate Z-score outlier ratio ({z_ratio:.1%})"
                    )
                    col_issues.append(
                        f"Non-trivial proportion of Z-score outliers ({z_ratio:.1%})"
                    )
                    score -= 5

            if col.iqr_outlier_ratio is not None:
                iqr_ratio = col.iqr_outlier_ratio
                if iqr_ratio > 0.10:  # >10% outside 1.5*IQR
                    issues_by_category.validity.append(
                        f"{col.name}: high IQR outlier ratio ({iqr_ratio:.1%})"
                    )
                    col_issues.append(
                        f"High proportion of IQR outliers ({iqr_ratio:.1%}); "
                        "distribution has heavy tails or strong outliers"
                    )
                    score -= 10
                elif iqr_ratio > 0.05:
                    issues_by_category.validity.append(
                        f"{col.name}: moderate IQR outlier ratio ({iqr_ratio:.1%})"
                    )
                    col_issues.append(
                        f"Non-trivial proportion of IQR outliers ({iqr_ratio:.1%})"
                    )
                    score -= 5

        # --- High-cardinality text (consistency-ish) ----------------------
        if not is_numeric:
            if distinct_ratio > 0.9 and row_count > 100:
                issues_by_category.consistency.append(
                    f"{col.name}: text column with very high distinct ratio "
                    f"({distinct_ratio:.1%}) — likely an ID or free-form text."
                )
                col_issues.append(
                    "High-cardinality text; may not be suitable as categorical feature"
                )

        # Clamp score
        score = max(0.0, min(100.0, score))
        total_score += score

        column_scores.append(
            ColumnScore(
                name=col.name,
                dtype=col.dtype,
                score=int(round(score)),
                null_ratio=null_ratio,
                distinct_ratio=distinct_ratio,
                issues=col_issues,
            )
        )

    # --- Correlation-based issues -----------------------------------------
    if profile.correlations:
        for pair in profile.correlations:
            if abs(pair.pearson) >= 0.8:
                issues_by_category.correlations.append(
                    f"{pair.col1} and {pair.col2} are highly correlated "
                    f"(pearson={pair.pearson:.2f}); consider dropping or combining one."
                )

    dataset_score = int(round(total_score / max(1, len(column_scores))))

    scorecard = DQScorecard(
        dataset_name=profile.dataset_name,
        row_count=profile.row_count,
        column_count=profile.column_count,
        dataset_score=dataset_score,
        column_scores=column_scores,
        issues_by_category=issues_by_category,
    )

    # Return plain dict for ADK/tool compatibility
    return scorecard.model_dump()
