from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# --- Dataset profile -------------------------------------------------------

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    distinct_count: int
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    skew: Optional[float] = None
    zscore_outlier_ratio: Optional[float] = None
    iqr_outlier_ratio: Optional[float] = None
    # New fields for enhanced profiling
    logical_dtype: Optional[str] = None  # categorical, numeric, datetime, text, boolean
    sample_values: Optional[List[str]] = None  # First 5 unique non-null values
    top_values: Optional[Dict[str, int]] = None  # Top 5 most frequent values for categorical

class CorrelationPair(BaseModel):
    col1: str
    col2: str
    pearson: float

class DatasetProfile(BaseModel):
    dataset_name: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile]
    correlations: Optional[List[CorrelationPair]] = None
    # New fields for enhanced profiling
    duplicate_count: Optional[int] = None
    duplicate_ratio: Optional[float] = None
    has_datetime_columns: Optional[bool] = None
    datetime_info: Optional[Dict[str, Any]] = None  # Column name -> {min, max, gaps}

# --- DQ scorecard ----------------------------------------------------------

class ColumnScore(BaseModel):
    name: str
    dtype: str
    score: int = Field(ge=0, le=100)
    null_ratio: float
    distinct_ratio: float
    issues: List[str]

class DQIssuesByCategory(BaseModel):
    missing_values: List[str] = Field(default_factory=list)
    validity: List[str] = Field(default_factory=list)
    consistency: List[str] = Field(default_factory=list)
    uniqueness: List[str] = Field(default_factory=list)
    correlations: List[str] = Field(default_factory=list)

class DQScorecard(BaseModel):
    dataset_name: str
    row_count: int
    column_count: int
    dataset_score: int = Field(ge=0, le=100)
    column_scores: List[ColumnScore]
    issues_by_category: DQIssuesByCategory

# --- DQ fixes --------------------------------------------------------------

class DQFix(BaseModel):
    column: Optional[str] = None
    issue_category: str
    priority: str  # must-have / should-have / could-have / nice-to-have
    description: str
    pandas_code_snippet: str

class DQFixes(BaseModel):
    summary: str
    fixes: List[DQFix]

# --- Notebook summary (optional) ------------------------------------------

class DQNotebookSummary(BaseModel):
    notebook_path: str
    message: str