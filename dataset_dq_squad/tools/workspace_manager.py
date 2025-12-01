"""
Workspace management for dataset analysis.
Creates project folders and manages dataset paths.
"""

from pathlib import Path
import shutil
from typing import Dict, Any


def prepare_dataset_workspace(dataset_path: str) -> Dict[str, Any]:
    """
    Prepare a workspace for dataset analysis.
    
    This function:
    1. Creates a dedicated folder for the dataset
    2. Copies the dataset to that folder
    3. Returns workspace information for use by other agents
    
    Args:
        dataset_path: Path to the dataset file (CSV)
        
    Returns:
        Dictionary with workspace information:
        - workspace_path: Path to the workspace directory
        - dataset_path: Path to the dataset file in workspace
        - dataset_name: Name of the dataset
    """
    src_path = Path(dataset_path)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Get dataset name
    dataset_name = src_path.stem
    
    # Create workspace directory
    project_root = Path(__file__).resolve().parents[1]
    workspace_dir = project_root / "data" / dataset_name
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy dataset to workspace
    dataset_filename = src_path.name
    workspace_dataset_path = workspace_dir / dataset_filename
    
    try:
        shutil.copy2(src_path, workspace_dataset_path)
    except shutil.SameFileError:
        # Already in the workspace
        pass
    
    return {
        "workspace_path": str(workspace_dir.absolute()),
        "dataset_path": str(workspace_dataset_path.absolute()),
        "dataset_name": dataset_name,
        "dataset_filename": dataset_filename,
        "message": f"Workspace prepared at: {workspace_dir}"
    }
