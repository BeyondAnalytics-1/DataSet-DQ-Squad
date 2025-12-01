#!/usr/bin/env python3
"""
Interactive Data Quality Agent CLI

Usage:
    python dq_agent_cli.py [--log-file LOG_FILE]

Options:
    --log-file LOG_FILE    Optional log file path for session logging

This starts an interactive conversation with the DQ agent.
You can ask to profile datasets, get scorecards, or generate reports.
"""

import sys
import os
import argparse
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Verify API key is set
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY environment variable is not set!")
    print("Please set it in your .env file or environment.")
    sys.exit(1)

# Ensure the key is available to google-genai
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataset_dq_squad.conversational_agent import start_interactive_session

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive Data Quality Conversational Agent"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path for session logging"
    )
    
    args = parser.parse_args()
    
    print("Starting Data Quality Agent...")
    print("Make sure you have set GOOGLE_API_KEY environment variable.")
    
    if args.log_file:
        print(f"Logging to: {args.log_file}")
    
    print()
    
    try:
        start_interactive_session(log_file=args.log_file)
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nError starting agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
