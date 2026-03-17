#!/usr/bin/env python3
"""Visualize MCP Atlas completion results as HTML for easy comparison."""

import argparse
import csv
import html
import json
import sys
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)


def parse_trajectory(traj_str: str) -> list:
    """Parse trajectory JSON string."""
    try:
        return json.loads(traj_str) if traj_str else []
    except Exception:
        return []


def format_gt_trajectory(traj: list) -> str:
    """Format ground truth trajectory as HTML."""
    if not traj:
        return '<div class="empty">No ground truth trajectory</div>'

    lines = []
    for i, msg in enumerate(traj):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "assistant":
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", "{}")
                    try:
                        args_pretty = json.dumps(json.loads(args), indent=2)
                    except Exception:
                        args_pretty = args
                    lines.append(
                        f'<div class="msg assistant tool-call">'
                        f'<span class="role">Assistant</span> calls '
                        f'<span class="tool-name">{html.escape(name)}</span>'
                        f'<pre class="args">{html.escape(args_pretty)}</pre>'
                        f"</div>"
                    )
            if content:
                assistant_content = content
                if isinstance(assistant_content, (list, dict)):
                    assistant_content = json.dumps(assistant_content, indent=2)
                lines.append(
                    f'<div class="msg assistant">'
                    f'<span class="role">Assistant</span>'
                    f'<div class="content">{html.escape(str(assistant_content))}</div>'
                    f"</div>"
                )
        elif role == "tool":
            # Content can be string, list, dict, or None
            if isinstance(content, (list, dict)):
                content_str = json.dumps(content, indent=2)
            else:
                content_str = str(content) if content else "null"
            content_preview = content_str[:500] + "..." if len(content_str) > 500 else content_str
            lines.append(
                f'<div class="msg tool">'
                f'<span class="role">Tool Response</span>'
                f'<pre class="content">{html.escape(content_preview)}</pre>'
                f"</div>"
            )
        elif role == "user":
            lines.append(
                f'<div class="msg user">'
                f'<span class="role">User</span>'
                f'<div class="content">{html.escape(content)}</div>'
                f"</div>"
            )

    return "\n".join(lines)


def format_model_conversation(conv: list) -> str:
    """Format model's raw conversation history as HTML."""
    if not conv:
        return '<div class="empty">No conversation history</div>'

    lines = []
    for i, msg in enumerate(conv):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if role == "assistant":
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", "{}")
                    try:
                        args_pretty = json.dumps(json.loads(args), indent=2)
                    except Exception:
                        args_pretty = args
                    lines.append(
                        f'<div class="msg assistant tool-call">'
                        f'<span class="role">Assistant</span> calls '
                        f'<span class="tool-name">{html.escape(name)}</span>'
                        f'<pre class="args">{html.escape(args_pretty)}</pre>'
                        f"</div>"
                    )
            if content:
                assistant_content = content
                if isinstance(assistant_content, (list, dict)):
                    assistant_content = json.dumps(assistant_content, indent=2)
                lines.append(
                    f'<div class="msg assistant">'
                    f'<span class="role">Assistant</span>'
                    f'<div class="content">{html.escape(str(assistant_content))}</div>'
                    f"</div>"
                )
        elif role == "tool":
            # Content can be string, list, dict, or None
            if isinstance(content, (list, dict)):
                content_str = json.dumps(content, indent=2)
            else:
                content_str = str(content) if content else "null"
            content_preview = content_str[:500] + "..." if len(content_str) > 500 else content_str
            lines.append(
                f'<div class="msg tool">'
                f'<span class="role">Tool Response</span>'
                f'<pre class="content">{html.escape(content_preview)}</pre>'
                f"</div>"
            )
        elif role == "user":
            user_content = content
            if isinstance(user_content, (list, dict)):
                user_content = json.dumps(user_content)
            lines.append(
                f'<div class="msg user">'
                f'<span class="role">User</span>'
                f'<div class="content">{html.escape(str(user_content) if user_content else "")}</div>'
                f"</div>"
            )

    return "\n".join(lines)


def format_claims(claims_str: str) -> str:
    """Format GTFA claims as HTML list."""
    try:
        claims = json.loads(claims_str) if claims_str else []
    except Exception:
        return f"<pre>{html.escape(claims_str[:500])}</pre>"

    if not claims:
        return '<div class="empty">No claims</div>'

    items = [f"<li>{html.escape(c)}</li>" for c in claims]
    return f'<ol class="claims">{"".join(items)}</ol>'


def format_enabled_tools(tools_str: str) -> str:
    """Format enabled tools as HTML chips."""
    if not tools_str:
        return '<div class="empty">No tools specified</div>'

    # Tools can be comma-separated or JSON array (strings or dicts)
    try:
        if tools_str.startswith("["):
            parsed = json.loads(tools_str)
            # Handle list of dicts with 'name' key
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                tools = [t.get("name", str(t)) for t in parsed]
            # Handle list of strings
            elif isinstance(parsed, list):
                tools = parsed
            else:
                tools = tools_str.split(",")
        else:
            tools = tools_str.split(",")
    except Exception:
        tools = tools_str.split(",")

    tools = [str(t).strip() for t in tools if t and str(t).strip()]

    if not tools:
        return '<div class="empty">No tools specified</div>'

    chips = [f'<span class="tool-chip">{html.escape(t)}</span>' for t in tools]
    return f'<div class="tools-list">{" ".join(chips)}</div>'


def generate_html(rows: list, output_path: str, min_score: float, max_score: float):
    """Generate HTML visualization."""

    # Filter rows by score range
    filtered = []
    for r in rows:
        try:
            score = float(r.get("coverage_score", 0))
        except Exception:
            score = 0
        if min_score <= score <= max_score:
            filtered.append((r, score))

    # Sort by score
    filtered.sort(key=lambda x: x[1])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MCP Atlas Results Viewer</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: #1a1a2e;
            color: white;
            padding: 20px;
            margin: -20px -20px 20px -20px;
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: rgba(255,255,255,0.1);
            padding: 10px 15px;
            border-radius: 5px;
        }}
        .stat-value {{ font-size: 24px; font-weight: bold; }}
        .stat-label {{ font-size: 12px; opacity: 0.8; }}
        .filters {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .task {{
            background: white;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .task-header {{
            background: #2d3748;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}
        .task-header:hover {{ background: #3d4758; }}
        .task-id {{ font-family: monospace; font-size: 14px; }}
        .score {{
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .score.pass {{ background: #48bb78; color: white; }}
        .score.fail {{ background: #f56565; color: white; }}
        .score.partial {{ background: #ed8936; color: white; }}
        .task-body {{ padding: 20px; display: none; }}
        .task.expanded .task-body {{ display: block; }}
        .prompt {{
            background: #edf2f7;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 14px;
            line-height: 1.5;
        }}
        .claims {{
            background: #fef3c7;
            padding: 15px 15px 15px 35px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 13px;
        }}
        .claims li {{ margin-bottom: 5px; }}
        .tools-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }}
        .tool-chip {{
            background: #e2e8f0;
            color: #4a5568;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-family: monospace;
        }}
        .enabled-tools {{
            background: #f0fff4;
            padding: 12px 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            border: 1px solid #c6f6d5;
        }}
        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .panel {{
            border: 1px solid #e2e8f0;
            border-radius: 5px;
            overflow: hidden;
        }}
        .panel-header {{
            background: #e2e8f0;
            padding: 10px 15px;
            font-weight: 600;
            font-size: 14px;
        }}
        .panel-header.gt {{ background: #c6f6d5; }}
        .panel-header.model {{ background: #bee3f8; }}
        .panel-body {{
            padding: 15px;
            max-height: 600px;
            overflow-y: auto;
        }}
        .msg {{
            margin-bottom: 12px;
            padding: 10px;
            border-radius: 5px;
            font-size: 13px;
        }}
        .msg.assistant {{ background: #e6fffa; border-left: 3px solid #38b2ac; }}
        .msg.assistant.tool-call {{ background: #ebf8ff; border-left: 3px solid #4299e1; }}
        .msg.tool {{ background: #f7fafc; border-left: 3px solid #a0aec0; }}
        .msg.user {{ background: #faf5ff; border-left: 3px solid #9f7aea; }}
        .role {{
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            color: #4a5568;
        }}
        .tool-name {{
            font-family: monospace;
            background: #4299e1;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .content {{ margin-top: 8px; white-space: pre-wrap; word-break: break-word; }}
        pre.content, pre.args {{
            margin: 8px 0 0 0;
            font-size: 11px;
            background: #f7fafc;
            padding: 8px;
            border-radius: 3px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .empty {{ color: #a0aec0; font-style: italic; padding: 20px; text-align: center; }}
        .toggle-all {{
            background: #4299e1;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
        }}
        .toggle-all:hover {{ background: #3182ce; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MCP Atlas Results Viewer</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(filtered)}</div>
                <div class="stat-label">Tasks (filtered)</div>
            </div>
            <div class="stat">
                <div class="stat-value">{sum(1 for r,s in filtered if s >= 0.75)}</div>
                <div class="stat-label">Passed (≥0.75)</div>
            </div>
            <div class="stat">
                <div class="stat-value">{min_score:.2f} - {max_score:.2f}</div>
                <div class="stat-label">Score Range</div>
            </div>
        </div>
    </div>

    <div class="filters">
        <button class="toggle-all" onclick="toggleAll(true)">Expand All</button>
        <button class="toggle-all" onclick="toggleAll(false)">Collapse All</button>
        <span style="margin-left: 20px; color: #718096;">Click task headers to expand/collapse</span>
    </div>
"""

    for row, score in filtered:
        task_id = row.get("TASK", "unknown")
        prompt = row.get("PROMPT", "")
        claims_str = row.get("GTFA_CLAIMS", "")
        enabled_tools_str = row.get("ENABLED_TOOLS", "")
        gt_traj = parse_trajectory(row.get("TRAJECTORY", "[]"))
        model_conv = parse_trajectory(row.get("raw_conversation_history", "[]"))

        score_class = "pass" if score >= 0.75 else "partial" if score > 0 else "fail"

        html_content += f"""
    <div class="task" data-score="{score}">
        <div class="task-header" onclick="this.parentElement.classList.toggle('expanded')">
            <div>
                <span class="task-id">{html.escape(task_id)}</span>
            </div>
            <span class="score {score_class}">{score:.2f}</span>
        </div>
        <div class="task-body">
            <div class="prompt"><strong>Prompt:</strong> {html.escape(prompt)}</div>
            <div class="enabled-tools">
                <strong>Enabled Tools:</strong>
                {format_enabled_tools(enabled_tools_str)}
            </div>
            <div style="margin-bottom: 15px;">
                <strong>Claims to verify:</strong>
                {format_claims(claims_str)}
            </div>
            <div class="comparison">
                <div class="panel">
                    <div class="panel-header gt">Ground Truth Trajectory</div>
                    <div class="panel-body">{format_gt_trajectory(gt_traj)}</div>
                </div>
                <div class="panel">
                    <div class="panel-header model">Model Conversation</div>
                    <div class="panel-body">{format_model_conversation(model_conv)}</div>
                </div>
            </div>
        </div>
    </div>
"""

    html_content += """
    <script>
        function toggleAll(expand) {
            document.querySelectorAll('.task').forEach(task => {
                if (expand) {
                    task.classList.add('expanded');
                } else {
                    task.classList.remove('expanded');
                }
            });
        }
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated: {output_path}")
    print(f"Tasks: {len(filtered)} (score range: {min_score:.2f} - {max_score:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MCP Atlas results as HTML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View all results
  python visualize_results.py results.csv -o results.html

  # View only failing tasks (score 0-0.5)
  python visualize_results.py results.csv --min 0 --max 0.5

  # View passing tasks only
  python visualize_results.py results.csv --min 0.75 --max 1.0
        """,
    )
    parser.add_argument("csv_file", help="Input CSV file (completions or scored)")
    parser.add_argument("-o", "--output", default="results.html", help="Output HTML file")
    parser.add_argument(
        "--min", type=float, default=0.0, help="Minimum coverage score (default: 0.0)"
    )
    parser.add_argument(
        "--max", type=float, default=1.0, help="Maximum coverage score (default: 1.0)"
    )

    args = parser.parse_args()

    # Load CSV
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Error: No rows in CSV")
        sys.exit(1)

    # Check if this is a scored CSV (has coverage_score) or completions CSV
    if "coverage_score" not in rows[0]:
        print("Warning: No coverage_score column found, using 0 for all tasks")
        for r in rows:
            r["coverage_score"] = "0"

    generate_html(rows, args.output, args.min, args.max)


if __name__ == "__main__":
    main()
