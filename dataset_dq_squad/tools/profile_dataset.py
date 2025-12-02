from pathlib import Path
from typing import Any, Dict
import pandas as pd

def profile_dataset(dataset_path: str) -> Dict[str, Any]:
    print(f"[profile_dataset] called with: {dataset_path}", flush=True)

    path = Path(dataset_path)
    df = pd.read_csv(path)

    profile: Dict[str, Any] = {
        "dataset_name": path.stem,
        "dataset_path": str(path.resolve()),  # Store absolute path
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [],
    }

    for col in df.columns:
        s = df[col]
        col_profile: Dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "distinct_count": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            col_profile["min"] = float(s.min()) if not s.dropna().empty else None
            col_profile["max"] = float(s.max()) if not s.dropna().empty else None
        else:
            col_profile["min"] = None
            col_profile["max"] = None

        profile["columns"].append(col_profile)

    print(f"[profile_dataset] finished, {len(profile['columns'])} columns.", flush=True)
    return profile
