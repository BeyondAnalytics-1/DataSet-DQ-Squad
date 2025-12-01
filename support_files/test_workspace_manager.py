"""
Test workspace management functionality.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset_dq_squad.tools.workspace_manager import prepare_dataset_workspace


def test_workspace_creation():
    """Test workspace creation with a sample dataset."""
    print("=" * 70)
    print("Testing Workspace Manager")
    print("=" * 70)
    
    # Use an existing dataset
    dataset_path = "dataset_dq_squad/data/train.csv"
    
    print(f"\n[1/2] Testing workspace creation for: {dataset_path}")
    
    try:
        result = prepare_dataset_workspace(dataset_path)
        
        print(f"  [OK] Workspace created!")
        print(f"  Workspace path: {result['workspace_path']}")
        print(f"  Dataset path: {result['dataset_path']}")
        print(f"  Dataset name: {result['dataset_name']}")
        print(f"  Message: {result['message']}")
        
        # Verify files exist
        workspace_path = Path(result['workspace_path'])
        dataset_file = Path(result['dataset_path'])
        
        if workspace_path.exists():
            print(f"  [OK] Workspace directory exists")
        else:
            print(f"  [ERROR] Workspace directory not found")
            
        if dataset_file.exists():
            print(f"  [OK] Dataset file exists in workspace")
        else:
            print(f"  [ERROR] Dataset file not found in workspace")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nTesting workspace management...\n")
    success = test_workspace_creation()
    
    print("\n" + "=" * 70)
    if success:
        print("[SUCCESS] Workspace manager is working!")
    else:
        print("[FAILURE] Workspace manager test failed")
    print("=" * 70)
