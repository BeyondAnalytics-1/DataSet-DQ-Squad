# dataset_dq_squad/tools/notebook_builder.py

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _markdown_cell(text: str) -> Dict[str, Any]:
    """Create a Jupyter markdown cell."""
    lines = text.splitlines()
    source = [line + "\n" for line in lines] if lines else [""]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def _code_cell(code: str) -> Dict[str, Any]:
    """Create a Jupyter code cell."""
    lines = code.splitlines()
    source = [line + "\n" for line in lines] if lines else [""]
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def _create_readme(
    output_dir: Path,
    dataset_name: str,
    row_count: int,
    col_count: int,
    dataset_score: int,
    notebook_filename: str,
    dataset_filename: str
) -> None:
    """Generate a README.md file in the project folder."""
    readme_content = f"""# Data Quality Project: {dataset_name}

## Overview
This project contains the Data Quality assessment for the **{dataset_name}** dataset.

- **Rows**: {row_count}
- **Columns**: {col_count}
- **Overall DQ Score**: {dataset_score}/100

## Contents
- **{dataset_filename}**: The raw dataset file.
- **{notebook_filename}**: The interactive Data Quality Report notebook.

## Getting Started
1. Open `{notebook_filename}` in Jupyter Notebook or JupyterLab.
2. Run all cells to see the profiling results, scorecard, and proposed fixes.
"""
    readme_path = output_dir / "README.md"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme_content)


def build_notebook(
    dataset_profile: Dict[str, Any],
    dq_scorecard: Dict[str, Any],
    dq_fixes: Dict[str, Any],
    profiling_markdown: str,
    scorecard_markdown: str,
    fixes_markdown: str,
    dataset_path: str,
    output_path: str = "dq_report.ipynb",
) -> str:
    """
    Build a reorganized, comprehensive Jupyter notebook report.
    """
    print(f"[build_notebook] called for dataset: {dataset_path}", flush=True)
    
    # Helper to parse JSON if string
    def ensure_dict(obj, name):
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                print(f"[build_notebook] Failed to parse {name} string")
                return {}
        return obj

    dataset_profile = ensure_dict(dataset_profile, "dataset_profile")
    dq_scorecard = ensure_dict(dq_scorecard, "dq_scorecard")
    dq_fixes = ensure_dict(dq_fixes, "dq_fixes")

    
    exec_summary = f"""## Executive Summary

**Overall Data Quality Score: {dataset_score}/100**

This interactive notebook provides a comprehensive data quality assessment including:
- Dataset overview with duplicate detection
- Column-level analysis with logical type detection
- Quality scorecard with visualizations
- Correlation analysis
- Actionable fixes with executable code

{profiling_markdown if profiling_markdown.strip() else ''}

{scorecard_markdown if scorecard_markdown.strip() else ''}

{fixes_markdown if fixes_markdown.strip() else ''}
"""
    cells.append(_markdown_cell(exec_summary))

    # ==================================================================
    # SECTION 2: DATASET OVERVIEW
    # ==================================================================
    cells.append(_markdown_cell("## 1. Dataset Overview"))

    overview_table = f"""### 1.1 Basic Statistics

| Metric | Value |
|--------|-------|
| **Dataset Name** | `{ds_name}` |
| **Total Rows** | {row_count:,} |
| **Total Columns** | {col_count} |
| **Duplicate Rows** | {duplicate_count:,} ({duplicate_ratio:.2%}) |
| **Overall DQ Score** | **{dataset_score}/100** |
"""
    cells.append(_markdown_cell(overview_table))

    # Duplicate analysis with code
    if duplicate_count > 0:
        cells.append(_markdown_cell("### 1.2 Duplicate Row Analysis"))
        dup_code = f"""# Inspect duplicate rows
import pandas as pd

df = pd.read_csv('{dataset_filename}')
duplicates = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))
print(f"Found {{len(duplicates)}} duplicate rows (including all occurrences)")
duplicates.head(20)
"""
        cells.append(_code_cell(dup_code))

    # ==================================================================
    # SECTION 3: COLUMN ANALYSIS  
    # ==================================================================
    cells.append(_markdown_cell("## 2. Column Analysis"))

    # 2.1 Column overview table
    col_overview_lines = [
        "### 2.1 Column Overview",
        "",
        "| Column | Type | Logical Type | Nulls | Distinct | Sample Values |",
        "|--------|------|--------------|-------|----------|---------------|",
    ]
    
    for col in columns:
        name = col.get("name", "unknown")
        dtype = col.get("dtype", "unknown")
        logical_dtype = col.get("logical_dtype", "unknown")
        null_count = int(col.get("null_count", 0))
        null_pct = (null_count / row_count * 100) if row_count > 0 else 0
        distinct_count = int(col.get("distinct_count", 0))
        sample_values = col.get("sample_values", [])
        samples_str = ", ".join(str(v)[:20] for v in sample_values[:2]) if sample_values else ""
        
        col_overview_lines.append(
            f"| `{name}` | {dtype} | **{logical_dtype}** | {null_count} ({null_pct:.1f}%) | {distinct_count:,} | {samples_str} |"
        )
    
    cells.append(_markdown_cell("\n".join(col_overview_lines)))

    # 2.2 Categorical distributions
    cells.append(_markdown_cell("### 2.2 Categorical Value Distributions"))
    cat_dist_code = f"""# Value distributions for categorical columns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('{dataset_filename}')
cat_cols = [col for col in df.columns if 5 < df[col].nunique() <= 20]

if cat_cols:
    n_cols = len(cat_cols)
    fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(14, 4 * ((n_cols + 1) // 2)))
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(cat_cols):
        df[col].value_counts().head(10).plot(kind='barh', ax=axes[idx])
        axes[idx].set_title(f'{{col}} - Top Values')
        axes[idx].set_xlabel('Count')
    
    for idx in range(len(cat_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("No categorical columns with 5-20 unique values")
"""
    cells.append(_code_cell(cat_dist_code))

    # 2.3 Outlier analysis table
    outlier_summary = []
    for col in columns:
        name = col.get("name", "unknown")
        dtype = col.get("dtype", "unknown")
        z_ratio = col.get("zscore_outlier_ratio", None)
        iqr_ratio = col.get("iqr_outlier_ratio", None)
        
        if z_ratio and z_ratio > 0.01:  # Only show if >1% outliers
            outlier_summary.append((name, "Z-score", z_ratio))
        if iqr_ratio and iqr_ratio > 0.01:
            outlier_summary.append((name, "IQR", iqr_ratio))
    
    if outlier_summary:
        outlier_lines = [
            "### 2.3 Outlier Detection Summary",
            "",
            "| Column | Method | Outlier Ratio |",
            "|--------|--------|---------------|",
        ]
        for name, method, ratio in sorted(outlier_summary, key=lambda x: x[2], reverse=True):
            outlier_lines.append(f"| `{name}` | {method} | {ratio:.1%} |")
        
        cells.append(_markdown_cell("\n".join(outlier_lines)))

        # Boxplots for outliers
        cells.append(_markdown_cell("### 2.4 Outlier Visualizations"))
        outlier_cols = list(set([name for name, _, _ in outlier_summary]))
        boxplot_code = f"""# Boxplots for columns with outliers
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('{dataset_filename}')
outlier_cols = {outlier_cols}

n_cols = len(outlier_cols)
fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(12, 4 * ((n_cols + 1) // 2)))
if n_cols == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, col in enumerate(outlier_cols):
    if col in df.columns:
        df.boxplot(column=col, ax=axes[idx])
        axes[idx].set_title(f'{{col}} - Outlier Detection')

for idx in range(len(outlier_cols), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
"""
        cells.append(_code_cell(boxplot_code))

    # 2.5  Time-series if applicable
    if has_datetime and datetime_info:
        ts_table_lines = [
            "### 2.5 Time-Series Columns",
            "",
            "| Column | Date Range | Unique Dates | Span (days) |",
            "|--------|------------|--------------|-------------|",
        ]
        
        for col_name, info in datetime_info.items():
            dt_min = info.get('min', 'N/A')
            dt_max = info.get('max', 'N/A')
            unique_dates = info.get('unique_dates', 'N/A')
            range_days = info.get('range_days', 'N/A')
            ts_table_lines.append(f"| `{col_name}` | {dt_min} to {dt_max} | {unique_dates} | {range_days} |")
        
        cells.append(_markdown_cell("\n".join(ts_table_lines)))

        # Time-series plots
        cells.append(_markdown_cell("### 2.6 Time-Series Visualizations"))
        dt_cols_list = list(datetime_info.keys())
        ts_code = f"""# Time-series plots
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('{dataset_filename}')
dt_cols = {dt_cols_list}

for col in dt_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df.set_index(col).resample('D').size().plot(figsize=(12, 4))
        plt.title(f'Daily Records for {{col}}')
        plt.ylabel('Count')
        plt.show()
"""
        cells.append(_code_cell(ts_code))

    # ==================================================================
    # SECTION 4: DATA QUALITY SCORECARD
    # ==================================================================
    cells.append(_markdown_cell("## 3. Data Quality Scorecard"))

    # 3.1 Overall score with gauge
    score_md = f"""### 3.1 Overall Score: {dataset_score}/100

The data quality score is calculated based on:
- **Missing values** (nullability analysis)
- **Validity** (outliers, skewness checks)
- **Consistency** (correlation analysis)
- **Uniqueness** (distinct value ratios)
"""
    cells.append(_markdown_cell(score_md))

    gauge_code = f"""# DQ Score Visualization
import matplotlib.pyplot as plt

score = {dataset_score}
colors = ['#d32f2f', '#ff9800', '#ffc107', '#8bc34a', '#4caf50']
bounds = [0, 20, 40, 60, 80, 100]

fig, ax = plt.subplots(figsize=(8, 2))
for i in range(len(colors)):
    ax.barh(0, bounds[i+1] - bounds[i], left=bounds[i], height=0.5, color=colors[i], alpha=0.3)

ax.barh(0, score, height=0.5, color='#1976d2', label=f'Score: {{score}}/100')
ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('Score')
ax.set_yticks([])
ax.legend()
plt.title('Data Quality Score Gauge')
plt.tight_layout()
plt.show()
"""
    cells.append(_code_cell(gauge_code))

    # 3.2 Per-column scores as table
    score_table_lines = [
        "### 3.2 Column Quality Scores",
        "",
        "| Column | Type | Score | Null % | Distinct % | Issues |",
        "|--------|------|-------|--------|------------|--------|",
    ]
    
    for cs in col_scores:
        name = cs.get("name", "unknown")
        dtype = cs.get("dtype", "unknown")
        score = cs.get("score", 0)
        null_ratio = cs.get("null_ratio", 0.0)
        distinct_ratio = cs.get("distinct_ratio", 0.0)
        issues = cs.get("issues", [])
        
        score_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        issue_text = "; ".join(issues[:2]) if issues else "None"
        if len(issues) > 2:
            issue_text += f" (+{len(issues)-2} more)"
        
        score_table_lines.append(
            f"| `{name}` | {dtype} | {score_emoji} **{score}** | {null_ratio:.1%} | {distinct_ratio:.1%} | {issue_text} |"
        )
    
    cells.append(_markdown_cell("\n".join(score_table_lines)))

    # 3.3 Visual comparison
    cells.append(_markdown_cell("### 3.3 Column Score Comparison"))
    col_names_chart = [cs.get("name", "unknown") for cs in col_scores]
    col_scores_chart = [cs.get("score", 0) for cs in col_scores]
    
    bar_chart_code = f"""# Column scores bar chart
import matplotlib.pyplot as plt

col_names = {col_names_chart}
scores = {col_scores_chart}
colors = ['#4caf50' if s >= 80 else '#ff9800' if s >= 60 else '#f44336' for s in scores]

plt.figure(figsize=(10, max(6, len(col_names) * 0.4)))
plt.barh(col_names, scores, color=colors, alpha=0.7)
plt.xlabel('DQ Score')
plt.title('Quality Scores by Column')
plt.xlim(0, 100)
plt.axvline(80, color='green', linestyle='--', alpha=0.5, label='Good')
plt.axvline(60, color='orange', linestyle='--', alpha=0.5, label='Fair')
plt.legend()
plt.tight_layout()
plt.show()
"""
    cells.append(_code_cell(bar_chart_code))

    # 3.4 Issues by category
    if issues_by_category:
        issue_lines = ["### 3.4 Issues by Category", ""]
        for cat, items in issues_by_category.items():
            if items:
                issue_lines.append(f"**{cat.replace('_', ' ').title()}** ({len(items)} issues):")
                for item in items[:5]:
                    issue_lines.append(f"- {item}")
                if len(items) > 5:
                    issue_lines.append(f"- ... and {len(items) - 5} more")
                issue_lines.append("")
        
        cells.append(_markdown_cell("\n".join(issue_lines)))

    # ==================================================================
    # SECTION 5: CORRELATION ANALYSIS
    # ==================================================================
    if correlations:
        cells.append(_markdown_cell("## 4. Correlation Analysis"))
        
        # 4.1 Top correlations table
        sorted_corrs = sorted(correlations, key=lambda x: abs(x.get("pearson", 0)), reverse=True)
        corr_table_lines = [
            "### 4.1 Strongest Correlations",
            "",
            "| Column 1 | Column 2 | Pearson Correlation | Strength |",
            "|----------|----------|---------------------|----------|",
        ]
        
        for corr in sorted_corrs[:15]:
            col1 = corr.get("col1", "")
            col2 = corr.get("col2", "")
            pearson = corr.get("pearson", 0.0)
            abs_p = abs(pearson)
            strength = "Very Strong" if abs_p > 0.9 else "Strong" if abs_p > 0.7 else "Moderate" if abs_p > 0.5 else "Weak"
            corr_table_lines.append(f"| `{col1}` | `{col2}` | {pearson:.3f} | {strength} |")
        
        cells.append(_markdown_cell("\n".join(corr_table_lines)))

        # 4.2 Heatmap
        cells.append(_markdown_cell("### 4.2 Correlation Matrix Heatmap"))
        heatmap_code = f"""# Correlation heatmap
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('{dataset_filename}')
numeric_df = df.select_dtypes(include=['number'])

if not numeric_df.empty:
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    
    plt.title('Correlation Matrix', pad=20)
    plt.tight_layout()
    plt.show()
"""
        cells.append(_code_cell(heatmap_code))

        # 4.3 Scatter plots for high correlations
        high_corrs = [c for c in sorted_corrs if abs(c.get("pearson", 0)) > 0.7][:6]
        if high_corrs:
            cells.append(_markdown_cell("### 4.3 Relationship Scatter Plots"))
            scatter_pairs = [(c['col1'], c['col2']) for c in high_corrs]
            scatter_code = f"""# Scatter plots for highly correlated pairs
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('{dataset_filename}')
pairs = {scatter_pairs}

n_pairs = len(pairs)
fig, axes = plt.subplots((n_pairs + 1) // 2, 2, figsize=(12, 4 * ((n_pairs + 1) // 2)))
if n_pairs == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, (col1, col2) in enumerate(pairs):
    if col1 in df.columns and col2 in df.columns:
        axes[idx].scatter(df[col1], df[col2], alpha=0.5)
        axes[idx].set_xlabel(col1)
        axes[idx].set_ylabel(col2)
        axes[idx].set_title(f'{{col1}} vs {{col2}}')

for idx in range(len(pairs), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
"""
            cells.append(_code_cell(scatter_code))

    # ==================================================================
    # SECTION 6: DATA QUALITY FIXES
    # ==================================================================
    cells.append(_markdown_cell("## 5. Data Quality Fixes"))

    # 5.1 Fix summary
    priority_counts = {}
    for fix in fixes:
        priority = fix.get("priority", "unspecified").lower()
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    fix_summary_lines = [
        "### 5.1 Fix Overview",
        "",
        fixes_summary if fixes_summary else "Proposed fixes to improve data quality:",
        "",
        "| Priority | Count |",
        "|----------|-------|",
    ]
    
    for priority in ["must-have", "should-have", "could-have", "nice-to-have"]:
        count = priority_counts.get(priority, 0)
        if count > 0:
            emoji = {"must-have": "🔴", "should-have": "🟡", "could-have": "🔵", "nice-to-have": "⚪"}
            fix_summary_lines.append(f"| {emoji.get(priority, '')} {priority.title()} | {count} |")
    
    fix_summary_lines.append("")
    cells.append(_markdown_cell("\n".join(fix_summary_lines)))

    # 5.2 Individual fixes
    cells.append(_markdown_cell("### 5.2 Detailed Fixes with Executable Code"))
    
    for idx, fix in enumerate(fixes, start=1):
        column = fix.get("column") or "dataset-level"
        issue_category = fix.get("issue_category", "unknown")
        priority = fix.get("priority", "unspecified").lower()
        description = fix.get("description", "").strip()
        code_snippet = fix.get("pandas_code_snippet", "").strip()

        priority_emoji = {"must-have": "🔴", "should-have": "🟡", "could-have": "🔵", "nice-to-have": "⚪"}.get(priority, "⚫")

        fix_md = f"""#### Fix {idx}: `{column}` {priority_emoji}

| Attribute | Value |
|-----------|-------|
| **Column** | `{column}` |
| **Category** | {issue_category} |
| **Priority** | {priority} |

{description if description else 'No description provided.'}
"""
        cells.append(_markdown_cell(fix_md))

        if code_snippet:
            fix_code = f"""# Apply Fix {idx}: {column}
import pandas as pd

df = pd.read_csv('{dataset_filename}')

# Apply fix
{code_snippet}

# Preview
print(f"Dataset shape: {{df.shape}}")
df.head()
"""
            cells.append(_code_cell(fix_code))

            validation_code = f"""# Validate Fix {idx}
if '{column}' in df.columns:
    print(f"└─ Null count: {{df['{column}'].isna().sum()}} ({{df['{column}'].isna().mean():.2%}})")
    print(f"└─ Distinct: {{df['{column}'].nunique()}}")
else:
    print(f"└─ Total rows: {{len(df)}}")
"""
            cells.append(_code_cell(validation_code))

    # ==================================================================
    # ASSEMBLE & WRITE NOTEBOOK
    # ==================================================================
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    _create_readme(
        output_dir=project_dir,
        dataset_name=ds_name,
        row_count=row_count,
        col_count=col_count,
        dataset_score=dataset_score,
        notebook_filename=Path(output_path).name,
        dataset_filename=dataset_filename
    )

    return str(nb_path)