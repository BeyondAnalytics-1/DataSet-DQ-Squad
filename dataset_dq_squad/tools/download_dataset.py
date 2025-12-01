from __future__ import annotations

from pathlib import Path
from typing import Optional

import hashlib
import requests


def download_dataset(url: str, target_subdir: str = "data") -> str:
    """
    Download a CSV file from an HTTP(S) URL into the project.

    Parameters
    ----------
    url : str
        HTTP(S) URL to a CSV file.
    target_subdir : str, optional
        Subdirectory under the `dataset_dq_squad/` package root where the
        file will be stored. Defaults to "data".

    Returns
    -------
    str
        Absolute path to the downloaded CSV file.

    Notes
    -----
    - If you are running on Kaggle and the file is already provided by
      Kaggle as a local path, you should NOT call this function; pass the
      local path directly to the profiling tools instead.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"download_dataset only supports http(s) URLs, got: {url}")

    project_root = Path(__file__).resolve().parents[1]  # -> dataset_dq_squad/
    data_dir = project_root / target_subdir
    data_dir.mkdir(parents=True, exist_ok=True)

    # Stable-ish filename from URL hash
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    filename = f"remote_{h}.csv"
    out_path = data_dir / filename

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    out_path.write_bytes(resp.content)

    return str(out_path.resolve())
