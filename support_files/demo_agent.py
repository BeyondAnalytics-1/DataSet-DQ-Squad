"""
Demo script showing how to use the conversational DQ agent programmatically.

This demonstrates:
1. Single-turn interactions
2. Multi-turn conversations with context
3. Different types of requests (profile, scorecard, full report)
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


async def demo_single_turn():
    """Demonstrate a single-turn interaction."""
    print("=" * 70)
    print("DEMO 1: Single-turn interaction")
    print("=" * 70)
    
    dataset_path = "dataset_dq_squad/data/train.csv"
    message = f"Please profile this dataset: {dataset_path}"
    
    print(f"\nUser: {message}")
    response = await run_dq_agent_async(message)
    print(f"\nAgent: {response}\n")


async def demo_multi_turn():
    """Demonstrate a multi-turn conversation with context."""
    print("=" * 70)
    print("DEMO 2: Multi-turn conversation")
    print("=" * 70)
    
    # Create a runner for persistent conversation
    runner = create_dq_agent_runner()
    session_id = "demo_session"
    
    # Turn 1: Profile dataset
    message1 = "I have a dataset at dataset_dq_squad/data/fast_food_ordering_dataset.csv"
    print(f"\nUser: {message1}")
    response1 = await run_dq_agent_async(message1, runner, session_id)
    print(f"\nAgent: {response1}\n")
    
    # Turn 2: Ask about data quality
    message2 = "What's the data quality score?"
    print(f"User: {message2}")
    response2 = await run_dq_agent_async(message2, runner, session_id)
    print(f"\nAgent: {response2}\n")
    
    # Turn 3: Request full report
    message3 = "Can you generate a full report with a notebook?"
    print(f"User: {message3}")
    response3 = await run_dq_agent_async(message3, runner, session_id)
    print(f"\nAgent: {response3}\n")


async def demo_specific_requests():
    """Demonstrate specific types of requests."""
    print("=" * 70)
    print("DEMO 3: Specific request types")
    print("=" * 70)
    
    runner = create_dq_agent_runner()
    session_id = "specific_demo"
    
    dataset_path = "dataset_dq_squad/data/train.csv"
    
    # Profile only
    print("\n--- Request: Profile only ---")
    message = f"Just profile {dataset_path}, don't do anything else"
    print(f"User: {message}")
    response = await run_dq_agent_async(message, runner, session_id)
    print(f"Agent: {response}\n")
    
    # Scorecard only
    print("\n--- Request: Scorecard (should use existing profile) ---")
    message = "Now give me the DQ scorecard"
    print(f"User: {message}")
    response = await run_dq_agent_async(message, runner, session_id)
    print(f"Agent: {response}\n")


async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("Data Quality Agent - Programmatic Demo")
    print("=" * 70)
    print("\nThis demo shows how to use the conversational agent in code.\n")
    
    try:
        # Run demos
        await demo_single_turn()
        await asyncio.sleep(1)
        
        await demo_multi_turn()
        await asyncio.sleep(1)
        
        await demo_specific_requests()
        
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Make sure GOOGLE_API_KEY environment variable is set.\n")
    asyncio.run(main())
