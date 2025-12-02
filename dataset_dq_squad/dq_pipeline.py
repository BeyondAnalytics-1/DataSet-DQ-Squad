# dataset_dq_squad/dq_pipeline.py

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.tools import FunctionTool

from .tools import (
    profile_dataset,
    build_dq_scorecard,
    build_notebook
)

from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)  # <---- 1. Import the Plugin
from google.genai import types
import asyncio



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
# 1. Data profiler agent: path -> JSON profile
# ----------------------------------------------------------------------

data_profiler = LlmAgent(
    name="DataProfilerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    description=(
        "Takes a local CSV path, loads it, and produces a compact JSON "
        "dataset profile using a Python profiling tool."
    ),
    instruction=(
        "You are the first step in a DQ pipeline.\\n"
        "Input: the user provides a single string which is a local CSV path.\\n\\n"
        "You MUST always:\\n"
        "1) Take that input string as dataset_path.\\n"
        "2) Call the profile_dataset tool with dataset_path.\\n"
        "3) Return the EXACT JSON object returned by the tool.\\n"
        "   - Do NOT summarize it.\\n"
        "   - Do NOT restructure it.\\n"
        "   - Do NOT wrap it in markdown blocks.\\n"
        "   - Do NOT use 'categorical_columns' or 'numerical_columns' keys.\\n"
        "   - Do NOT use pandas describe() format.\\n"
        "   - Just output the raw JSON string as is.\\n\\n"
        "EXAMPLE OUTPUT FORMAT (do not change keys):\\n"
        "{\\n"
        "  \"dataset_name\": \"train\",\\n"
        "  \"dataset_path\": \"/absolute/path/to/train.csv\",\\n"
        "  \"row_count\": 100,\\n"
        "  \"column_count\": 2,\\n"
        "  \"columns\": [\\n"
        "    {\"name\": \"age\", \"dtype\": \"int64\", \"null_count\": 0, \"distinct_count\": 50, \"min\": 1, \"max\": 100}\\n"
        "  ]\\n"
        "}\\n"
    ),
    tools=[FunctionTool(profile_dataset)],
    output_key="dataset_profile_json",
)

# ----------------------------------------------------------------------
# 2. DQ scorecard agent: JSON profile -> JSON scorecard
# ----------------------------------------------------------------------

dq_scorecard_agent = LlmAgent(
    name="DQScorecardAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    description=(
        "Builds a data quality scorecard (0–100 per column and overall) "
        "from a dataset profile using a Python scoring tool."
    ),
    instruction=(
        "You are the second step in a DQ pipeline.\\n\\n"
        "The previous agent has produced a dataset profile JSON, which appears "
        "in the conversation history as the last JSON object.\\n\\n"
        "Your job:\\n"
        "1) Read that dataset profile JSON from the previous agent's message.\\n"
        "2) Call the build_dq_scorecard tool exactly once, passing the parsed "
        "   dataset profile as the dataset_profile argument.\\n"
        "3) Return ONLY the JSON scorecard produced by build_dq_scorecard, "
        "   with no additional explanation around it.\\n"
    ),
    tools=[FunctionTool(build_dq_scorecard)],
    output_key="dq_scorecard_json",
)

# ----------------------------------------------------------------------
# 3. Fixer agent: JSON scorecard -> JSON fixes
# ----------------------------------------------------------------------

fixer_agent = LlmAgent(
    name="DQFixerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config,
    ),
    description=(
        "Reads a DQ scorecard and proposes prioritized fixes with pandas code."
    ),
    instruction=(
        "You are a data quality fixer.\\n\\n"
        "Input: a DQ scorecard JSON from the previous step. It appears in the "
        "conversation history as the last JSON object.\\n\\n"
        "Your job:\\n"
        "1) Read the DQ scorecard JSON from the previous agent.\\n"
        "2) Identify the most important issues by category: missing_values, "
        "   validity, consistency, uniqueness, correlations.\\n"
        "3) Propose concrete transformations to improve the score.\\n"
        "   For each transformation, provide:\\n"
        "     - column (or 'dataset')\\n"
        "     - issue_category\\n"
        "     - priority: one of 'must-have', 'should-have', "
        "       'could-have', 'nice-to-have'\\n"
        "     - description (short text)\\n"
        "     - pandas_code_snippet: a short pandas code snippet that applies "
        "       the fix (assume the DataFrame is named `df`).\\n"
        "4) Return a single JSON object with:\\n"
        "     - summary: short natural-language overview\\n"
        "     - fixes: list of fix objects as described above.\\n"
        "Do NOT return the original scorecard JSON.\\n"
    ),
    tools=[],
    output_key="dq_fixes_json",
)

# ----------------------------------------------------------------------
# 4. Notebook builder agent: profile + scorecard + fixes -> notebook path
# ----------------------------------------------------------------------

notebook_builder_agent = LlmAgent(
    name="NotebookBuilderAgent",
    model=Gemini(
        model="gemini-2.0-flash",
        retry_options=retry_config,
    ),
    description=(
        "Final step of the DQ pipeline. Builds a Jupyter notebook report from the "
        "dataset profile JSON, DQ scorecard JSON, and DQ fixes JSON produced by "
        "previous agents."
    ),
    instruction=(
        "You are the final agent in the DQ pipeline.\n\n"
        "Your task:\n"
        "1) Look at the conversation history and find the three JSON objects:\n"
        "   - Dataset profile JSON (from DataProfilerAgent)\n"
        "   - DQ scorecard JSON (from DQScorecardAgent)\n"
        "   - DQ fixes JSON (from DQFixerAgent)\n\n"
        "2) From the dataset profile JSON, extract the 'dataset_path' field value.\n\n"
        "3) Call build_notebook with these exact arguments:\n"
        "   - dataset_profile: the dataset profile JSON object\n"
        "   - dq_scorecard: the DQ scorecard JSON object\n"
        "   - dq_fixes: the DQ fixes JSON object\n"
        "   - profiling_markdown: 'Profiling summary'\n"
        "   - scorecard_markdown: 'Scorecard summary'\n"
        "   - fixes_markdown: 'Proposed fixes summary'\n"
        "   - dataset_path: the dataset_path value you extracted from step 2\n"
        "   - output_path: 'dq_report.ipynb'\n\n"
        "4) Return only the file path string that build_notebook returns.\n\n"
        "IMPORTANT: Pass the JSON objects directly as dictionaries, not as strings."
    ),
    tools=[FunctionTool(build_notebook)],
    output_key="dq_notebook_path",
)

# ----------------------------------------------------------------------
# 5. Sequential pipeline agent
# ----------------------------------------------------------------------

dq_pipeline_agent = SequentialAgent(
    name="DQPipelineAgent",
    sub_agents=[
        data_profiler,
        dq_scorecard_agent,
        fixer_agent,
        notebook_builder_agent,
    ],
    description=(
        "Executes a sequence of data profiling, DQ scorecard creation, "
        "DQ fix suggestion, notebook building."
    ),
)
