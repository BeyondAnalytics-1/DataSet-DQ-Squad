<p align="center">
  <img src="/DQSquad.jpg" alt="Data Quality Squad Conversational Agent Logo" width="300"/>
</p>

# Data Quality Squad - Conversational Agent

This directory contains an enhanced data quality assessment pipeline with a conversational agent interface.

## Features

### Core Profiling & Analysis
- ✅ **Enhanced Profiling**: Logical dtype detection, duplicate analysis, sample values
- ✅ **DQ Scorecard**: Column-level and dataset-level quality scores (0-100)
- ✅ **Automated Fixes**: AI-generated pandas code to improve data quality
- ✅ **Interactive Notebooks**: Comprehensive Jupyter notebooks with visualizations

### Conversational Agent
- 🤖 **Natural Language Interface**: Ask questions about your data in plain English
- 💬 **Multi-turn Conversations**: Maintains context across questions
- 🔧 **Tool Integration**: Automatically uses profiling, scoring, and notebook tools
- 📊 **Smart Summaries**: Concise, actionable insights

## Quick Start

### 1. Interactive CLI Mode

Start a conversational session:

```bash
python dq_agent_cli.py
```

Example conversation:
```
You: I have a dataset at dataset_dq_squad/data/train.csv
Agent: I'll profile that dataset for you...

You: What's the data quality score?
Agent: The overall DQ score is 83/100...

You: Generate a full report
Agent: Creating comprehensive notebook report...
```

### 2. Programmatic Usage

```python
import asyncio
from dataset_dq_squad.conversational_agent import (
    create_dq_agent_runner,
    run_dq_agent_async
)

async def analyze_dataset():
    runner = create_dq_agent_runner()
    session_id = "my_session"
    
    # Ask the agent to profile
    response = await run_dq_agent_async(
        "Profile dataset_dq_squad/data/train.csv",
        runner,
        session_id
    )
    print(response)
    
    # Ask follow-up questions
    response = await run_dq_agent_async(
        "What are the main quality issues?",
        runner,
        session_id
    )
    print(response)

asyncio.run(analyze_dataset())
```

### 3. Direct Tool Usage

You can also call the tools directly without the agent:

```python
from dataset_dq_squad.tools import (
    run_profiling_pipeline,
    build_dq_scorecard,
    build_notebook
)

# Profile
profile = run_profiling_pipeline("path/to/dataset.csv")

# Scorecard
scorecard = build_dq_scorecard(profile.model_dump())

# Notebook
notebook_path = build_notebook(
    dataset_profile=profile.model_dump(),
    dq_scorecard=scorecard,
    dq_fixes=fixes,
    # ... other params
)
```

## Agent Capabilities

The conversational agent can:

### 📊 Profile Datasets
```
"Profile my dataset at data/customers.csv"
"Show me profiling results"
"What columns does this dataset have?"
```

### 🎯 Generate Scorecards
```
"What's the data quality score?"
"Give me a DQ scorecard"
"Which columns have the most issues?"
```

### 📓 Create Notebooks
```
"Build a notebook"
"Generate a full DQ report"
"Create a comprehensive analysis"
```

### 💡 Answer Questions
```
"How many duplicates are there?"
"Which columns have missing values?"
"Tell me about the correlations"
```

## Architecture

```
conversational_agent.py
├── ConversationalDQAgent (LlmAgent)
│   ├── Tools:
│   │   ├── download_dataset()
│   │   ├── run_profiling_pipeline()
│   │   ├── build_dq_scorecard()
│   │   └── dq_pipeline_tool (SequentialAgent)
│   │       ├── DataProfilerAgent
│   │       ├── DQScorecardAgent
│   │       ├── DQFixerAgent
│   │       └── NotebookBuilderAgent
│   └── Runner:
│       ├── InMemoryRunner
│       └── InMemorySessionService
```

## Configuration

The agent uses:
- **Model**: `gemini-2.5-flash-lite`
- **Retry Logic**: 5 attempts with exponential backoff
- **Session Management**: In-memory sessions for conversation continuity

## Examples

See:
- `dq_agent_cli.py` - Interactive CLI interface
- `demo_agent.py` - Programmatic usage examples

## Requirements

- Python 3.10+
- Google AI API key (set as `GOOGLE_API_KEY` environment variable)
- Required packages: `google-adk`, `pandas`, `numpy`, `pydantic`

## Tips

1. **Multi-turn conversations**: Use the same `runner` and `session_id` for context
2. **Specific requests**: Be clear about what you want (profile only, scorecard only, full report)
3. **Dataset paths**: Provide relative or absolute paths
4. **Follow-up questions**: The agent remembers previous outputs in the session
