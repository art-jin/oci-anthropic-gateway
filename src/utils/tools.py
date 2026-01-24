"""Tool calling conversion utilities."""

import logging
import oci
from typing import Union, List

logger = logging.getLogger("oci-gateway")


def anthropic_to_oci_tools(tools: list) -> list:
    """Convert Anthropic tools to OCI ToolDefinition format (for Generic API).

    Args:
        tools: List of Anthropic tool definitions

    Returns:
        List of OCI-formatted tool definitions
    """
    oci_tools = []
    for tool in tools:
        oci_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", tool.get("parameters", {}))
            }
        }
        oci_tools.append(oci_tool)
    return oci_tools


def anthropic_to_cohere_tools(tools: list) -> list:
    """Convert Anthropic tools to CohereTool format (for Cohere API).

    Args:
        tools: List of Anthropic tool definitions

    Returns:
        List of CohereTool objects
    """
    cohere_tools = []
    for tool in tools:
        # Get input schema
        input_schema = tool.get("input_schema", tool.get("parameters", {}))
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # Convert to Cohere parameter definitions
        parameter_definitions = {}
        for param_name, param_info in properties.items():
            param_def = oci.generative_ai_inference.models.CohereParameterDefinition(
                description=param_info.get("description", ""),
                type=_convert_json_type_to_python(param_info.get("type", "str")),
                is_required=param_name in required
            )
            parameter_definitions[param_name] = param_def

        cohere_tool = oci.generative_ai_inference.models.CohereTool(
            name=tool["name"],
            description=tool.get("description", ""),
            parameter_definitions=parameter_definitions
        )
        cohere_tools.append(cohere_tool)
    return cohere_tools


def _convert_json_type_to_python(json_type: str) -> str:
    """Convert JSON Schema type to Python type string for Cohere.

    Args:
        json_type: JSON Schema type string

    Returns:
        Python type string
    """
    type_mapping = {
        "string": "str",
        "str": "str",
        "integer": "int",
        "number": "float",
        "float": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict"
    }
    return type_mapping.get(json_type, "str")


def _build_tool_use_instruction(tools: list, tool_choice: str) -> str:
    """
    Build tool use instruction prompt for Generic format models.

    This creates a system prompt that instructs the model on how to use tools
    when it doesn't have native function calling support.

    Args:
        tools: List of tool definitions
        tool_choice: Tool choice strategy

    Returns:
        System prompt instruction string
    """
    if not tools:
        return ""

    # Build tool descriptions
    tool_list = []
    for tool in tools:
        # Handle different tool formats
        if isinstance(tool, dict):
            name = tool.get("name")
            desc = tool.get("description", "")
            # Support both input_schema and parameters
            schema = tool.get("input_schema") or tool.get("parameters") or {}
        else:
            logger.warning(f"Unexpected tool format: {type(tool)}")
            continue

        if not name:
            logger.warning(f"Tool missing 'name' field: {tool}")
            continue

        # Build parameter descriptions with examples
        params_desc = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        if properties:
            for param, info in properties.items():
                req_mark = "**(required)**" if param in required else "(optional)"
                param_desc = info.get('description', '')
                param_type = info.get('type', 'string')
                
                # Add type-specific examples for complex parameters
                example_hint = ""
                if param_type == 'array':
                    items_type = info.get('items', {}).get('type', 'object')
                    if items_type == 'object':
                        example_hint = " [array of objects]"
                    else:
                        example_hint = f" [array of {items_type}]"
                elif param_type == 'object':
                    example_hint = " [object with properties]"
                
                params_desc.append(f"  - `{param}` {req_mark} ({param_type}{example_hint}): {param_desc}")

        params_str = "\n".join(params_desc) if params_desc else "  (no parameters)"

        tool_list.append(f"""**{name}**: {desc}
Parameters:
{params_str}""")

    # Tool choice strategy description
    strategy_map = {
        "auto": "Use tools when the user's request requires external data, searches, file operations, or actions. Answer directly for simple questions.",
        "required": "You MUST use at least one tool for EVERY request. NEVER answer without calling a tool first.",
        "none": "Do not use any tools. Answer directly.",
        "any": "You should use tools whenever they would help answer the user's request."
    }
    strategy = strategy_map.get(tool_choice, "auto")

    # Enhanced examples with more variations
    examples = """## Examples of CORRECT Tool Usage

**Example 1 - Single tool (weather query):**
User: "What's the weather in Tokyo?"

Your response:
<TOOL_CALL>
{"name": "get_weather", "input": {"location": "Tokyo", "unit": "celsius"}}
</TOOL_CALL>

**Example 2 - Multiple tools sequentially:**
User: "Search for Python tutorials and then read the README.md file"

Your response:
<TOOL_CALL>
{"name": "web_search", "input": {"query": "Python tutorials"}}
</TOOL_CALL>
<TOOL_CALL>
{"name": "read_file", "input": {"file_path": "README.md"}}
</TOOL_CALL>

**Example 3 - Tool with complex parameters:**
User: "Create a user with name John Doe, email john@example.com, and age 30"

Your response:
<TOOL_CALL>
{"name": "create_user", "input": {"name": "John Doe", "email": "john@example.com", "age": 30}}
</TOOL_CALL>

**Example 4 - Direct answer (no tool needed) for tool_choice="auto":**
User: "Hello, how are you?"

Your response:
I'm doing well, thank you for asking! How can I help you today?

**Example 5 - Tool with minimal parameters:**
User: "Get the current time"

Your response:
<TOOL_CALL>
{"name": "get_time", "input": {}}
</TOOL_CALL>

**Example 6 - Tool with array of objects parameter (TodoWrite):**
User: "Create a todo list with tasks: finish report, review code"

Your response:
<TOOL_CALL>
{"name": "TodoWrite", "input": {"todos": [{"content": "finish report", "status": "pending", "activeForm": "sessionTrackerTasks"}, {"content": "review code", "status": "pending", "activeForm": "sessionTrackerTasks"}]}}
</TOOL_CALL>

Note: TodoWrite requires each todo object to have: content, status, and activeForm fields.

**Example 7 - Tool with nested object structure:**
User: "Add user preferences for dark mode and email notifications"

Your response:
<TOOL_CALL>
{"name": "update_preferences", "input": {"settings": {"theme": "dark", "notifications": {"email": true, "push": false}}}}
</TOOL_CALL>

## COMMON MISTAKES TO AVOID

❌ WRONG - Including explanatory text with tool calls:
"I'll search for that information for you.
<TOOL_CALL>
{"name": "web_search", "input": {"query": "information"}}
</TOOL_CALL>
Let me know if you need anything else!"

✓ CORRECT - Only tool call, no extra text:
<TOOL_CALL>
{"name": "web_search", "input": {"query": "information"}}
</TOOL_CALL>

❌ WRONG - Malformed JSON:
<TOOL_CALL>
{name: "search", input: {query: "test"}}
</TOOL_CALL>

✓ CORRECT - Properly quoted JSON:
<TOOL_CALL>
{"name": "search", "input": {"query": "test"}}
</TOOL_CALL>

❌ WRONG - Missing closing tag:
<TOOL_CALL>
{"name": "search", "input": {"query": "test"}}

✓ CORRECT - Complete tag structure:
<TOOL_CALL>
{"name": "search", "input": {"query": "test"}}
</TOOL_CALL>

❌ WRONG - Using wrong parameter name for array (e.g., "content" instead of "todos"):
<TOOL_CALL>
{"name": "TodoWrite", "input": {"content": "Task description"}}
</TOOL_CALL>

✓ CORRECT - Using correct array parameter with all required fields:
<TOOL_CALL>
{"name": "TodoWrite", "input": {"todos": [{"content": "Task description", "status": "pending", "activeForm": "sessionTrackerTasks"}]}}
</TOOL_CALL>

❌ WRONG - Passing string when array is required:
<TOOL_CALL>
{"name": "TodoWrite", "input": {"todos": "Task 1, Task 2"}}
</TOOL_CALL>

✓ CORRECT - Passing proper array structure with all required fields:
<TOOL_CALL>
{"name": "TodoWrite", "input": {"todos": [{"content": "Task 1", "status": "pending", "activeForm": "sessionTrackerTasks"}, {"content": "Task 2", "status": "pending", "activeForm": "sessionTrackerTasks"}]}}
</TOOL_CALL>

❌ WRONG - Missing required fields (activeForm) in todo objects:
<TOOL_CALL>
{"name": "TodoWrite", "input": {"todos": [{"content": "Task 1", "status": "pending"}]}}
</TOOL_CALL>

✓ CORRECT - All required fields included (content, status, activeForm):
<TOOL_CALL>
{"name": "TodoWrite", "input": {"todos": [{"content": "Task 1", "status": "pending", "activeForm": "sessionTrackerTasks"}]}}
</TOOL_CALL>"""

    return f"""# TOOL USE INSTRUCTIONS - READ CAREFULLY

You have access to specialized tools to help answer user requests. Your task is to decide when to use them and format tool calls correctly.

## Tool Selection Strategy
{strategy}

## Available Tools

{chr(10).join(tool_list)}

## Tool Call Format (STRICTLY FOLLOW THIS)

When you decide to use a tool, you MUST output ONLY the tool call in this EXACT format:

<TOOL_CALL>
{{"name": "tool_name", "input": {{"param1": "value1", "param2": "value2"}}}}
</TOOL_CALL>

**CRITICAL FORMATTING RULES:**
1. Start with opening tag: <TOOL_CALL>
2. Include valid JSON with "name" and "input" fields
3. Use double quotes (") for all JSON keys and string values
4. End with closing tag: </TOOL_CALL>
5. Do NOT include ANY text before or after the tool call tags
6. For multiple tools, place tool call blocks one after another
7. Ensure proper JSON formatting (commas, braces, brackets)

{examples}

## Tool Results Format

When you receive tool results, they will be provided in this format:

<TOOL_RESULT id='toolu_xxx' status='success'>
[actual result data from the tool]
</TOOL_RESULT>

Or for errors:

<TOOL_RESULT id='toolu_xxx' status='error'>
[error message]
</TOOL_RESULT>

**IMPORTANT**: Tool results contain actual data from the tools. Use this information to answer the user's question. Do NOT say the tool returned empty or no results if there is data in the <TOOL_RESULT> block.

## Final Reminders
- When you need to use a tool: Output ONLY <TOOL_CALL> blocks, nothing else
- When you receive a tool result: Read and use the data from the <TOOL_RESULT> block
- When you don't need a tool: Respond naturally in conversational language
- Always validate your JSON is properly formatted before output
- Never explain what you're doing when making a tool call - just output the tool call
- Never ignore or dismiss tool results - they contain real data to help answer the user"""


def convert_tool_choice(tool_choice: Union[str, dict, None]) -> dict:
    """Convert Anthropic tool_choice to OCI format (for Generic API).

    Args:
        tool_choice: Anthropic tool_choice value

    Returns:
        OCI-formatted tool_choice dict
    """
    if tool_choice is None or tool_choice == "auto":
        return {"type": "auto"}
    elif tool_choice == "none":
        return {"type": "none"}
    elif tool_choice == "required":
        return {"type": "required"}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return {"type": "auto"}
