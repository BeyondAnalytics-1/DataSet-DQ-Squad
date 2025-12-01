"""
Test script to verify LoggingPlugin integration.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataset_dq_squad.conversational_agent import (
    create_dq_agent_runner,
    run_dq_agent_async
)


async def test_logging():
    """Test that LoggingPlugin is working."""
    print("=" * 70)
    print("LoggingPlugin Integration Test")
    print("=" * 70)
    
    log_file = "dq_agent_test.log"
    
    print(f"\n[1/3] Creating runner with logging enabled...")
    print(f"  Log file: {log_file}")
    
    try:
        runner = create_dq_agent_runner(
            enable_logging=True,
            log_file=log_file
        )
        print("  [OK] Runner created with LoggingPlugin")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[2/3] Testing agent with a simple query...")
    
    try:
        message = "Hello! Can you tell me what you can do?"
        print(f"  Query: '{message}'")
        
        response = await run_dq_agent_async(
            message,
            runner,
            session_id="test_logging"
        )
        
        print(f"  [OK] Got response (length: {len(response)} chars)")
        print(f"  Response preview: {response[:100]}...")
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n[3/3] Checking log file...")
    
    log_path = Path(log_file)
    if log_path.exists():
        log_size = log_path.stat().st_size
        print(f"  [OK] Log file created: {log_file}")
        print(f"  [OK] Log file size: {log_size} bytes")
        
        # Show first few lines
        with open(log_path, 'r') as f:
            lines = f.readlines()[:10]
        
        print(f"\n  Log preview (first {len(lines)} lines):")
        for line in lines:
            print(f"    {line.rstrip()}")
        
        return True
    else:
        print(f"  [ERROR] Log file not found: {log_file}")
        return False


async def test_without_logging():
    """Test runner without logging."""
    print("\n" + "=" * 70)
    print("Testing Without Logging")
    print("=" * 70)
    
    print(f"\n[1/2] Creating runner with logging disabled...")
    
    try:
        runner = create_dq_agent_runner(
            enable_logging=False
        )
        print("  [OK] Runner created without LoggingPlugin")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False
    
    print(f"\n[2/2] Testing agent query...")
    
    try:
        response = await run_dq_agent_async(
            "What can you help me with?",
            runner,
            session_id="test_no_logging"
        )
        print(f"  [OK] Got response without logging")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


async def main():
    """Run all tests."""
    print("\nTesting LoggingPlugin integration...\n")
    
    # Test with logging
    result1 = await test_logging()
    
    # Test without logging
    result2 = await test_without_logging()
    
    print("\n" + "=" * 70)
    if result1 and result2:
        print("[SUCCESS] All logging tests passed!")
        print("\nLogging is working correctly.")
        print("Check dq_agent_test.log for logged interactions.")
    else:
        print("[FAILURE] Some tests failed")
    print("=" * 70)


if __name__ == "__main__":
    print("Make sure GOOGLE_API_KEY environment variable is set.\n")
    asyncio.run(main())
