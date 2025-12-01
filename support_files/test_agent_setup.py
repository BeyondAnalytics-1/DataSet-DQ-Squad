"""
Quick test to verify the conversational agent is working.
This tests the basic setup without making actual API calls.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work."""
    print("[1/4] Testing imports...")
    try:
        from dataset_dq_squad.conversational_agent import (
            conversational_dq_agent,
            create_dq_agent_runner,
            run_dq_agent_async,
            run_dq_agent,
            start_interactive_session
        )
        print("  [OK] All imports successful")
        return True
    except Exception as e:
        print(f"  [ERROR] Import failed: {e}")
        return False


def test_agent_creation():
    """Test that the agent is properly configured."""
    print("\n[2/4] Testing agent creation...")
    try:
        from dataset_dq_squad.conversational_agent import conversational_dq_agent
        
        assert conversational_dq_agent.name == "ConversationalDQAgent"
        assert len(conversational_dq_agent.tools) == 4
        
        print(f"  [OK] Agent created: {conversational_dq_agent.name}")
        print(f"  [OK] Tools available: {len(conversational_dq_agent.tools)}")
        return True
    except Exception as e:
        print(f"  [ERROR] Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_runner_creation():
    """Test that runner can be created."""
    print("\n[3/4] Testing runner creation...")
    try:
        from dataset_dq_squad.conversational_agent import create_dq_agent_runner
        
        runner = create_dq_agent_runner()
        
        assert runner is not None
        assert runner.agent is not None
        
        print("  [OK] Runner created successfully")
        print(f"  [OK] Agent: {runner.agent.name}")
        return True
    except Exception as e:
        print(f"  [ERROR] Runner creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structure():
    """Test that the agent has the expected structure."""
    print("\n[4/4] Testing agent structure...")
    try:
        from dataset_dq_squad.conversational_agent import conversational_dq_agent
        
        # Check tools
        tool_names = [tool.name if hasattr(tool, 'name') else str(tool) for tool in conversational_dq_agent.tools]
        print(f"  [OK] Available tools:")
        for name in tool_names:
            print(f"    - {name}")
        
        # Check model
        print(f"  [OK] Model: {conversational_dq_agent.model}")
        
        return True
    except Exception as e:
        print(f"  [ERROR] Structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("Conversational Agent Setup Test")
    print("=" * 70)
    print("\nThis test verifies the agent is properly configured.")
    print("It does NOT make any API calls.\n")
    
    results = []
    
    results.append(test_imports())
    results.append(test_agent_creation())
    results.append(test_runner_creation())
    results.append(test_structure())
    
    print("\n" + "=" * 70)
    if all(results):
        print("[SUCCESS] All tests passed!")
        print("\nThe conversational agent is ready to use.")
        print("\nNext steps:")
        print("  1. Set GOOGLE_API_KEY environment variable")
        print("  2. Run: python dq_agent_cli.py (for interactive mode)")
        print("  3. Or: python demo_agent.py (for programmatic demo)")
    else:
        print("[FAILURE] Some tests failed")
        print("\nPlease check the error messages above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
