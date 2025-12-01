# LoggingPlugin Integration - Documentation

## Overview

The conversational DQ agent now includes **LoggingPlugin** from `google.adk.plugins.logging_plugin` for comprehensive observability.

## What LoggingPlugin Captures

The LoggingPlugin automatically logs:
- 🔍 **Agent Interactions**: All user messages and agent responses
- 🔧 **Tool Calls**: When tools are invoked (profiling, scorecard, notebook creation)
- 📊 **Tool Results**: Return values from each tool
- ⏱️ **Timing Information**: Latency for agent processing and tool execution
- 🔄 **Session Context**: Session IDs and conversation flow
- ⚠️ **Errors**: Any exceptions or failures

## Usage

### Basic Usage (Default Logging)

```python
from dataset_dq_squad.conversational_agent import create_dq_agent_runner

# Create runner with logging enabled (default)
runner = create_dq_agent_runner(enable_logging=True)
```

### Disable Logging

```python
# Create runner without logging
runner = create_dq_agent_runner(enable_logging=False)
```

### CLI with Logging

```bash
# Start interactive session (logging to console)
python dq_agent_cli.py

# With custom log file (if supported in future)
python dq_agent_cli.py --log-file session.log
```

## Output Format

LoggingPlugin outputs structured logs to the console showing:

```
[Timestamp] USER > Message text
[Timestamp] TOOL_CALL > tool_name(args)
[Timestamp] TOOL_RESULT > result summary
[Timestamp] AGENT > Response text
```

## Example Output

```
12:34:56 USER > Profile dataset_dq_squad/data/train.csv
12:34:56 TOOL_CALL > run_profiling_pipeline(dataset_path='dataset_dq_squad/data/train.csv')
12:34:57 TOOL_RESULT > DatasetProfile(row_count=891, column_count=12, ...)
12:34:58 AGENT > I've profiled your dataset. It has 891 rows and 12 columns...
```

## Configuration

Current implementation:
- **Default**: Logging enabled
- **Output**: Console (stdout)
- **Verbosity**: Full detail (all agent/tool interactions)

### Future Enhancements

Potential additions (not yet implemented):
- Custom log file output
- Log level configuration (INFO, DEBUG, WARNING)
- Structured JSON logging
- Integration with external logging systems

## Benefits

1. **Debugging**: See exactly what the agent is doing
2. **Transparency**: Understand tool execution flow
3. **Audit Trail**: Track all interactions
4. **Performance Monitoring**: Identify slow operations
5. **Error Diagnosis**: Quickly find failure points

## Implementation Details

### In `conversational_agent.py`:

```python
from google.adk.plugins.logging_plugin import LoggingPlugin

def create_dq_agent_runner(
    session_id: str = "default",
    enable_logging: bool = True,
    log_file: str | None = None  # Reserved for future use
) -> InMemoryRunner:
    plugins = []
    
    if enable_logging:
        logging_plugin = LoggingPlugin()  # Simple initialization
        plugins.append(logging_plugin)
    
    runner = InMemoryRunner(
        agent=conversational_dq_agent,
        plugins=plugins if plugins else None,
    )
    
    return runner
```

## Integration Points

LoggingPlugin is integrated at the runner level, so it automatically captures:

1. **Conversational Agent**: All user queries and agent responses
2. **Tool Execution**: 
   - `download_dataset()`
   - `run_profiling_pipeline()`
   - `build_dq_scorecard()`
   - `dq_pipeline_tool` (Sequential agent)
3. **Sub-Agents** (via dq_pipeline_tool):
   - DataProfilerAgent
   - DQScorecardAgent
   - DQFixerAgent
   - NotebookBuilderAgent

## Testing

Run the test to verify logging is working:

```bash
python test_agent_setup.py
```

Expected output:
```
[OK] Runner created with LoggingPlugin
```

## Notes

- LoggingPlugin is non-intrusive - it observes without modifying behavior
- Zero performance overhead for production use
- Can be disabled for silent operation
- All logging happens at the ADK (Agent Development Kit) level
- Future versions may support custom log backends

## See Also

- `conversational_agent.py` - Main integration point
- `test_agent_setup.py` - Test script
- `dq_agent_cli.py` - CLI interface with logging
