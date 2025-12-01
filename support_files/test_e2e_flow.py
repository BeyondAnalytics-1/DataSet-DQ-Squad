"""
End-to-End Test Script for Data Quality Agent

This script simulates the user's requested workflow:
1. Profile dataset
2. Get scorecard
3. Create notebook
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Force UTF-8 encoding for stdout/stderr to handle emojis in logs
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Verify API key
if not os.environ.get("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not set")
    sys.exit(1)

from dataset_dq_squad.conversational_agent import (
    create_dq_agent_runner,
    run_dq_agent_async
)

async def run_e2e_test():
    print("=" * 80)
    print("End-to-End Test: Profile -> Scorecard -> Notebook")
    print("=" * 80)
    
    dataset_path = "dataset_dq_squad/data/synthetic_medical_symptoms_dataset.csv"
    
    # Create runner with logging enabled
    runner = create_dq_agent_runner(enable_logging=True)
    session_id = "e2e_test_session"
    
    # Step 1: Profile
    print("\n" + "-" * 40)
    print("STEP 1: Profile Dataset")
    print("-" * 40)
    msg1 = f"Here is a dataset located here: {dataset_path}. Profile this dataset and provide a human readable overview."
    print(f"User: {msg1}\n")
    
    response1 = await run_dq_agent_async(msg1, runner, session_id)
    print(f"Agent:\n{response1}")
    
    # Step 2: Scorecard
    print("\n" + "-" * 40)
    print("STEP 2: Get Scorecard")
    print("-" * 40)
    msg2 = "Provide a scorecard for this dataset"
    print(f"User: {msg2}\n")
    
    response2 = await run_dq_agent_async(msg2, runner, session_id)
    print(f"Agent:\n{response2}")
    
    # Step 3: Notebook
    print("\n" + "-" * 40)
    print("STEP 3: Create Notebook")
    print("-" * 40)
    msg3 = "Create the notebook for this dataset together with the readme file."
    print(f"User: {msg3}\n")
    
    response3 = await run_dq_agent_async(msg3, runner, session_id)
    print(f"Agent:\n{response3}")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_e2e_test())
