# MCP Tools for OpenHands

This directory contains custom MCP (Model Context Protocol) tools that extend OpenHands agent capabilities without forking the repository.

## Quick Start

**No setup needed!** Just run tasks normally:

```bash
bash run.sh --tasks-file tasks.txt --model anthropic/claude-sonnet-4-20250514 --use-s3
```

The agent will automatically have access to custom MCP tools.

## Available Tools

### ask_user

Allows the agent to ask the user a clarifying question to get more information about the task.

**Parameters:**
- `question` (required): The clarifying question to ask the user
- `context` (optional): Optional additional context (e.g., conversation history summary)

**When the agent uses it:**
- Needing clarification on ambiguous requirements
- Getting specific details that weren't included in the task
- Understanding preferences or priorities
- Asking about expected behavior or edge cases
- Requiring external resources or credentials
- Getting domain-specific knowledge or context

**Example:**
```python
ask_user(
    question="What specific metrics should I use to evaluate the model?",
    context="I'm building the ML pipeline and need to know which regression metrics to report."
)
```

## Monitoring Help Requests

Help requests appear in two places:

1. **Docker logs** (real-time):
   ```bash
   docker logs -f <container-name>
   ```
   Look for the 🆘 emoji and "HELP REQUEST" banners

2. **Log file** (inside container):
   ```
   /tmp/openhands_help_requests.jsonl
   ```

Example output:
```
================================================================================
🆘 HELP REQUEST [HIGH]
================================================================================
Time: 2026-01-06T21:30:00
Question: Getting 403 Forbidden from API. Need credentials?
Context: Tried accessing https://api.example.com/data...
================================================================================
```

## How It Works

```
task.py runs
  ↓
1. Copies mcp_server.py to workspace/.openhands/
2. Mounts config.toml into OpenHands container
3. Starts OpenHands with --config-file flag
  ↓
OpenHands container starts
  ↓
1. Reads config.toml
2. Sees enable_mcp = true
3. Installs mcp package (if needed)
4. Starts mcp_server.py as subprocess
5. Connects via stdio
  ↓
Agent sees new tools and can call them
```

## Files Structure

```
openhands/
├── config.toml              # MCP configuration
├── task.py                  # Modified to support MCP
└── mcp_tools/
    ├── README.md            # This file
    ├── mcp_server.py        # The MCP server with tools
    └── mcp_requirements.txt # Dependencies
```

## Adding New Tools

Add a new tool to `mcp_server.py` using the `FastMCP` decorator:

```python
@mcp.tool()
async def your_new_tool(param1: str, param2: str = "") -> str:
    """What your tool does.

    Args:
        param1: Description of param1.
        param2: Optional description.
    """
    return f"Tool executed with {param1}"
```

`FastMCP` auto-generates the JSON schema from the function signature and docstring.

## Configuration

### Enabling/Disabling MCP Tools

Edit `config.toml`:
```toml
[agent]
enable_mcp = true  # Set to false to disable MCP tools
```

### Modifying Tool Behavior

Edit `mcp_server.py` to:
- Change tool descriptions or parameters
- Add new tools
- Modify logging behavior
- Customize responses

Changes apply to new task runs automatically.

## Troubleshooting

| Issue | Check | Fix |
|-------|-------|-----|
| Tool not appearing | Config.toml mounted? | Verify docker -v mount in task.py |
| MCP server error | Python errors in logs? | Check mcp_server.py syntax |
| Help not logged | File permissions? | Check /tmp access in container |
| Config not used | --config-file arg? | Verify docker command in task.py |

**Common issues:**

1. **Tool not appearing**
   - Check that `config.toml` has `enable_mcp = true`
   - Verify the MCP server was copied to workspace (check logs for "Copied MCP server to workspace")
   - Ensure `mcp` Python package is installed in runtime

2. **MCP server errors**
   - Check Docker logs for Python errors
   - Verify `mcp_server.py` syntax
   - Check that `/workspace/.openhands/mcp_server.py` exists in container

3. **Help requests not logged**
   - Check file permissions on `/tmp/openhands_help_requests.jsonl`
   - Look for errors in stderr output
   - The tool will still work even if logging fails

## Testing

To test the setup:

1. **Quick test with a simple task:**
   ```bash
   echo "admin_test_task" > test_tasks.txt
   python main.py --tasks-file test_tasks.txt --model gpt-4o --max-iterations 10
   ```

2. **Check the logs for:**
   - "Copied MCP server to workspace"
   - MCP server startup messages
   - Tool availability messages

3. **Verify the tool is available:**
   - Check the first agent message in the trajectory
   - Look for "ask_user" in the available tools

4. **Monitor help requests:**
   ```bash
   docker logs -f <container-name>
   ```

## Best Practices

1. **Use descriptive names**: Tool names should clearly indicate their purpose
2. **Provide context**: Include enough context in tool descriptions for the LLM to know when to use them
3. **Handle errors gracefully**: Tools should not crash the MCP server
4. **Log important events**: Help with debugging by logging tool usage
5. **Keep tools focused**: Each tool should do one thing well

## Benefits

✅ No forking OpenHands needed
✅ Easy to add more tools
✅ Works with all models
✅ Minimal changes to existing code
✅ Tools are automatically available to agent
✅ Clean separation of concerns
✅ Easy to maintain and update

## Dependencies

- `fastmcp` — FastMCP framework (installed in runtime)
- OpenHands 1.1+ — MCP support built-in
- Python 3.10+
