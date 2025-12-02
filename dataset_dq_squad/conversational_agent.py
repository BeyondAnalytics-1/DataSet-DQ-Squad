from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import FunctionTool, AgentTool
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types
from .dq_pipeline import dq_pipeline_agent

from .tools import (
    run_profiling_pipeline,
    build_dq_scorecard,
    build_notebook,
    download_dataset,
    prepare_dataset_workspace
)

# ----------------------------------------------------------------------
# Shared retry configuration
# ----------------------------------------------------------------------
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# ----------------------------------------------------------------------
# Conversational Data Quality Agent
# ----------------------------------------------------------------------
conversational_dq_agent = LlmAgent(
    name="ConversationalDQAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    description=(
        "Conversational data quality assistant with workspace management. "
        "Prepares dataset workspaces, runs profiling and DQ scoring, "
        "proposes fixes, and generates Jupyter notebook reports."
    ),
    instruction="""
You have four tools:

1) prepare_dataset_workspace(dataset_path: str) -> workspace info
   - **ALWAYS call this FIRST when a new dataset is provided**
   - Creates a dedicated workspace folder for the dataset
   - Copies the dataset to the workspace
   - Returns: workspace_path, dataset_path (in workspace), dataset_name
   - **IMPORTANT**: After calling this, USE the returned dataset_path for all subsequent operations
   - Store the dataset_path in your context for the entire conversation

2) run_profiling_pipeline(dataset_path: str) -> dataset_profile JSON
   - Use for quick profiling only
   - Use the dataset_path from the workspace (from prepare_dataset_workspace)
   - Returns detailed column statistics and analysis

3) dq_pipeline_agent(dataset_path: str) -> complete analysis
   - Use for full analysis: scorecard, fixes, and notebook
   - Use the dataset_path from the workspace
   - Runs complete pipeline and generates notebook

4) download_dataset(url: str) -> local CSV path
   - Use ONLY for HTTP(S) URLs
   - After downloading, call prepare_dataset_workspace with the returned path

CRITICAL WORKFLOW:

When a user provides a NEW dataset:
1. Call prepare_dataset_workspace(dataset_path) FIRST
2. Remember the returned dataset_path for the entire session
3. Use that dataset_path for ALL subsequent operations

When a user asks for analysis on the CURRENT dataset:
1. Check conversation history for the workspace dataset_path
2. Use that path (don't ask for it again)
3. Call the appropriate tool

EXAMPLES:

User: "Analyze data/train.csv"
You: 
  - Call prepare_dataset_workspace("data/train.csv")
  - Store returned dataset_path (e.g., "dataset_dq_squad/data/train/train.csv")
  - Call dq_pipeline_tool(dataset_path) with the workspace path
  
User: "Show me the scorecard" (after dataset already loaded)
You:
  - Look in conversation history for the workspace dataset_path
  - Call dq_pipeline_tool(dataset_path) with that path
  
User: "Now analyze another_file.csv"  
You:
  - Call prepare_dataset_workspace("another_file.csv") 
  - Update your stored dataset_path
  - Call dq_pipeline_tool with the new workspace path

STATE MANAGEMENT:

- **Always maintain the current workspace dataset_path in context**
- When user says "the dataset" or "this dataset", use the stored path
- Only ask for a new path if no dataset has been loaded yet
- If user provides a new file path, that becomes the new current dataset

CONVERSATION STYLE:
- Confirm workspace creation
- Be concise with findings
- Direct users to the generated notebook for details
- Remember the dataset context across the conversation
""",
    tools=[
        FunctionTool(prepare_dataset_workspace),
        FunctionTool(download_dataset),
        FunctionTool(run_profiling_pipeline),
        AgentTool(dq_pipeline_agent),
    ],
)

# ----------------------------------------------------------------------
# Runner and Session Service Setup
# ----------------------------------------------------------------------

def create_dq_agent_runner(
    session_id: str = "default",
    enable_logging: bool = True,
    log_file: str | None = None
) -> InMemoryRunner:
    """
    Create a runner instance for the conversational DQ agent.
    
    Args:
        session_id: Optional session identifier for conversation continuity
        enable_logging: Whether to enable LoggingPlugin for observability
        log_file: Optional log file path. If None, logs to console only.
        
    Returns:
        InMemoryRunner configured with the conversational agent
    """
    # Create plugins list
    plugins = []
    
    if enable_logging:
        # Create logging plugin for observability
        # LoggingPlugin logs to console by default
        # It captures all agent interactions, tool calls, and responses
        logging_plugin = LoggingPlugin()
        plugins.append(logging_plugin)
    
    # Create runner with plugins
    runner = InMemoryRunner(
        agent=conversational_dq_agent,
        plugins=plugins if plugins else None,
    )
    
    return runner


async def run_dq_agent_async(
    message: str,
    runner: InMemoryRunner | None = None,
    session_id: str = "default",
    user_id: str = "local-user"
) -> str:
    """
    Run the conversational DQ agent with a message asynchronously.
    
    Args:
        message: User message/question
        runner: Optional existing runner (for multi-turn conversations)
        session_id: Session identifier
        user_id: User identifier
        
    Returns:
        Agent's response text
    """
    if runner is None:
        runner = create_dq_agent_runner(session_id)
    
    # Get session service
    session_service = runner.session_service
    app_name = runner.app_name
    
    # Create or get session
    try:
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception:
        session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    
    # Create message content
    msg = types.Content(role="user", parts=[types.Part(text=message)])
    
    # Run agent and collect response
    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=msg,
    ):
        if event.content and event.content.parts:
            text = event.content.parts[0].text
            if text and text != "None":
                response_text += text
    
    return response_text.strip()


def run_dq_agent(
    message: str,
    runner: InMemoryRunner | None = None,
    session_id: str = "default",
    user_id: str = "local-user"
) -> str:
    """
    Run the conversational DQ agent with a message (synchronous wrapper).
    
    Args:
        message: User message/question
        runner: Optional existing runner (for multi-turn conversations)
        session_id: Session identifier
        user_id: User identifier
        
    Returns:
        Agent's response text
    """
    import asyncio
    return asyncio.run(run_dq_agent_async(message, runner, session_id, user_id))


# ----------------------------------------------------------------------
# Interactive CLI Function
# ----------------------------------------------------------------------

async def interactive_dq_session(log_file: str | None = None):
    """
    Start an interactive CLI session with the DQ agent.
    Type 'exit' or 'quit' to end the session.
    
    Args:
        log_file: Optional path to log file for session logging
    """
    print("=" * 70)
    print("Data Quality Conversational Agent")
    print("=" * 70)
    print("\nI can help you analyze datasets! You can:")
    print("  - Provide a dataset path to profile")
    print("  - Ask for a DQ scorecard")
    print("  - Request a full report with notebook")
    print("  - Ask questions about your data")
    print("\nType 'exit' or 'quit' to end the session.")
    
    if log_file:
        print(f"\nLogging enabled: {log_file}")
    print()
    
    runner = create_dq_agent_runner(
        enable_logging=True,
        log_file=log_file
    )
    session_id = "interactive"
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nAgent: Goodbye! Have a great day!")
                break
            
            print("\nAgent: ", end="", flush=True)
            response = await run_dq_agent_async(user_input, runner, session_id)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nAgent: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def start_interactive_session(log_file: str | None = None):
    """
    Synchronous entry point for interactive session.
    
    Args:
        log_file: Optional path to log file for session logging
    """
    import asyncio
    asyncio.run(interactive_dq_session(log_file))
