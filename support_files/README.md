# Support Files Directory

This directory contains test scripts, demo code, and other support files that are not part of the core agent functionality.

## Contents

### Test Scripts
- `test_agent_setup.py` - Verifies agent configuration and runner setup
- `test_logging.py` - Tests LoggingPlugin integration
- `test_output_project/` - Legacy test scripts

### Demo & Examples
- `demo_agent.py` - Programmatic usage examples showing:
  - Single-turn interactions
  - Multi-turn conversations
  - Different request types (profile, scorecard, reports)

## Usage

### Test Agent Setup
```bash
cd support_files
python test_agent_setup.py
```

### Test Logging
```bash
cd support_files
python test_logging.py
```

### Run Demos
```bash
cd support_files
python demo_agent.py
```

## Note

These files are for testing, demonstration, and development purposes only. They are not required for running the main DQ agent.

For production usage, see the main project files:
- `dq_agent_cli.py` - Interactive CLI
- `run.py` - Simple runner
- `run_with_logging.py` - Runner with logging
- `README.md` - Full documentation
