"""
Generate an HTML viewer for exploring underspecification tasks by ambiguity class.

Usage:
    python underspec_viewer.py --input /path/to/filtered_underspec.csv --output viewer.html
"""

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import safe_csv_read


def parse_json_or_string(value: str) -> List[str]:
    """
    Parse a value that may be a JSON array or a plain string.

    Args:
        value: String like 'goal' or '["goal", "goal"]'

    Returns:
        List of values (single-element list for plain strings)
    """
    if not value or value.strip() == "":
        return []

    value = value.strip()
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass

    return [value]


def parse_variant_id(variant_local_id: str) -> Dict[str, Any]:
    """
    Parse variant_local_id to extract segment count and severity.

    Args:
        variant_local_id: String like 'V_S1_S2_delete' or 'V_S1_delete'

    Returns:
        Dict with 'segments' (int) and 'severity' (str)
    """
    segment_matches = re.findall(r"S\d+", variant_local_id)
    segments = len(segment_matches)

    parts = variant_local_id.split("_")
    severity = parts[-1] if parts else "unknown"

    return {"segments": segments, "severity": severity}


def load_prompts(prompts_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load prompts CSV and build lookup by full variant_id.

    Args:
        prompts_path: Path to underspec_prompts.csv

    Returns:
        Dict mapping '{TASK}::{variant_id}' to prompt data
    """
    if not prompts_path.exists():
        return {}

    rows = safe_csv_read(prompts_path)
    lookup = {}
    for row in rows:
        task = row.get("TASK", "")
        variant_id = row.get("variant_id", "")
        if task and variant_id:
            key = f"{task}::{variant_id}"
            lookup[key] = {
                "original_prompt": row.get("original_prompt", ""),
                "underspecified_prompt": row.get("underspecified_prompt", ""),
            }
    return lookup


def find_prompts_file(input_path: Path) -> Optional[Path]:
    """
    Auto-detect underspec_prompts.csv relative to input file.

    Args:
        input_path: Path to filtered_underspec.csv

    Returns:
        Path to underspec_prompts.csv or None if not found
    """
    # Try underspec_generation subdirectory
    prompts_path = input_path.parent / "underspec_generation" / "underspec_prompts.csv"
    if prompts_path.exists():
        return prompts_path

    # Try same directory
    prompts_path = input_path.parent / "underspec_prompts.csv"
    if prompts_path.exists():
        return prompts_path

    return None


def escape_html(text: str) -> str:
    """HTML-escape text and preserve newlines."""
    if not text:
        return ""
    return html.escape(str(text)).replace("\n", "<br>")


def find_completion_files(input_path: Path) -> List[Path]:
    """
    Find completion CSV files for underspec trials.
    Prefers scored files (with coverage_score) over raw completion files.

    Args:
        input_path: Path to filtered_underspec.csv

    Returns:
        List of paths to completion/scored files, sorted by trial number
    """
    dataset_dir = input_path.parent

    # Look for run directories
    run_dirs = list(dataset_dir.glob("run_*"))
    if not run_dirs:
        return []

    # Use the most recent run directory
    run_dir = sorted(run_dirs)[-1]

    # Prefer scored files (they have coverage_score)
    scored_files = list(run_dir.glob("scores_*/scored_*.csv"))
    if scored_files:
        # Sort by trial number (extract from parent directory name)
        def extract_trial_num_scored(p: Path) -> int:
            match = re.search(r"_(\d+)_\d{8}_\d{6}$", p.parent.name)
            return int(match.group(1)) if match else 0
        return sorted(scored_files, key=extract_trial_num_scored)

    # Fall back to completion files
    completion_files = list(run_dir.glob("completions_*.csv"))

    # Sort by trial number (extract from filename)
    def extract_trial_num(p: Path) -> int:
        match = re.search(r"_(\d+)_\d{8}_\d{6}\.csv$", p.name)
        return int(match.group(1)) if match else 0

    return sorted(completion_files, key=extract_trial_num)


def load_completions(completion_files: List[Path], prompts_lookup: Dict[str, Dict[str, str]]) -> Dict[str, List[Dict]]:
    """
    Load completion files and build lookup by variant_id.

    Args:
        completion_files: List of completion CSV paths
        prompts_lookup: Dict mapping variant_id to prompt data

    Returns:
        Dict mapping variant_id to list of trial data (one per file)
    """
    # Build reverse lookup: (task_id, underspec_prompt_normalized) -> variant_id
    prompt_to_variant: Dict[tuple, str] = {}
    for variant_id, prompts in prompts_lookup.items():
        task_id = variant_id.split("::")[0]
        underspec = prompts.get("underspecified_prompt", "").strip().lower()
        prompt_to_variant[(task_id, underspec)] = variant_id

    # Load each completion file
    traces_by_variant: Dict[str, List[Dict]] = {}

    for trial_idx, comp_file in enumerate(completion_files):
        rows = safe_csv_read(comp_file)
        for row in rows:
            task_id = row.get("TASK", "")
            prompt = row.get("PROMPT", "").strip().lower()

            # Look up variant_id
            variant_id = prompt_to_variant.get((task_id, prompt))
            if not variant_id:
                continue

            if variant_id not in traces_by_variant:
                traces_by_variant[variant_id] = []

            # Ensure list is long enough
            while len(traces_by_variant[variant_id]) <= trial_idx:
                traces_by_variant[variant_id].append(None)

            traces_by_variant[variant_id][trial_idx] = {
                "raw_conversation_history": row.get("raw_conversation_history", ""),
                "coverage_score": row.get("coverage_score", ""),
            }

    return traces_by_variant


def parse_conversation(conv_str: str) -> List[Dict]:
    """Parse conversation JSON string."""
    try:
        return json.loads(conv_str) if conv_str else []
    except:
        return []


def format_conversation(conv: List[Dict]) -> str:
    """Format conversation as HTML (reused from visualize_results.py)."""
    if not conv:
        return '<div class="empty">No conversation history</div>'

    lines = []
    for msg in conv:
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
                    except:
                        args_pretty = args
                    lines.append(
                        f'<div class="msg assistant tool-call">'
                        f'<span class="role">Assistant</span> calls '
                        f'<span class="tool-name">{html.escape(name)}</span>'
                        f'<pre class="args">{html.escape(args_pretty)}</pre>'
                        f"</div>"
                    )
            if content:
                if isinstance(content, (list, dict)):
                    content = json.dumps(content, indent=2)
                lines.append(
                    f'<div class="msg assistant">'
                    f'<span class="role">Assistant</span>'
                    f'<div class="content">{html.escape(str(content))}</div>'
                    f"</div>"
                )
        elif role == "tool":
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
            if isinstance(content, (list, dict)):
                content = json.dumps(content)
            lines.append(
                f'<div class="msg user">'
                f'<span class="role">User</span>'
                f'<div class="content">{html.escape(str(content) if content else "")}</div>'
                f"</div>"
            )

    return "\n".join(lines)


def generate_html(
    rows: List[Dict],
    prompts_lookup: Dict[str, Dict[str, str]],
    traces_lookup: Dict[str, List[Dict]],
    output_path: Path,
) -> None:
    """Generate the HTML viewer."""

    # Group rows by ambiguity class
    by_class: Dict[str, List[Dict]] = {}
    for row in rows:
        cls = row.get("ambiguity_class", "unknown")
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(row)

    # Sort each class by task ID for consistency
    for cls in by_class:
        by_class[cls].sort(key=lambda x: x.get("variant_id", ""))

    # Class configuration: (key, css_class, display_name, color)
    class_config = [
        ("benign", "benign", "Benign", "#4caf50"),
        ("new_task", "new-task", "New Task", "#ff9800"),
        ("outcome-critical", "outcome-critical", "Outcome Critical", "#f44336"),
        ("divergent", "divergent", "Divergent", "#9c27b0"),
    ]

    # Build HTML
    html_parts = [
        f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Underspecification Task Viewer</title>
    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
            padding-left: 220px;
            background: #f5f5f5;
        }}

        h1 {{
            color: #1a237e;
            border-bottom: 3px solid #1a237e;
            padding-bottom: 10px;
            margin-top: 0;
        }}

        /* Navigation sidebar */
        .nav {{
            position: fixed;
            top: 0;
            left: 0;
            width: 200px;
            height: 100vh;
            background: white;
            padding: 20px;
            box-shadow: 2px 0 8px rgba(0,0,0,0.1);
            overflow-y: auto;
            z-index: 100;
        }}

        .nav h3 {{
            margin: 0 0 16px 0;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
        }}

        .nav a {{
            display: block;
            padding: 10px 14px;
            color: #333;
            text-decoration: none;
            border-radius: 6px;
            margin: 4px 0;
            font-size: 14px;
            border-left: 4px solid transparent;
        }}

        .nav a:hover {{
            background: #f5f5f5;
        }}

        .nav a.benign {{ border-left-color: #4caf50; }}
        .nav a.new-task {{ border-left-color: #ff9800; }}
        .nav a.outcome-critical {{ border-left-color: #f44336; }}
        .nav a.divergent {{ border-left-color: #9c27b0; }}

        .nav .count {{
            float: right;
            background: #eee;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
        }}

        /* Summary cards */
        .summary {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 40px;
        }}

        .summary-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .summary-card.benign {{ border-top: 4px solid #4caf50; }}
        .summary-card.new-task {{ border-top: 4px solid #ff9800; }}
        .summary-card.outcome-critical {{ border-top: 4px solid #f44336; }}
        .summary-card.divergent {{ border-top: 4px solid #9c27b0; }}

        .summary-card h3 {{
            margin: 0 0 8px 0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
        }}

        .summary-card .count {{
            font-size: 36px;
            font-weight: 700;
        }}

        .summary-card.benign .count {{ color: #2e7d32; }}
        .summary-card.new-task .count {{ color: #e65100; }}
        .summary-card.outcome-critical .count {{ color: #c62828; }}
        .summary-card.divergent .count {{ color: #7b1fa2; }}

        /* Section headers */
        .section-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 40px 0 20px 0;
            padding: 16px 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .section-header.benign {{ border-left: 6px solid #4caf50; }}
        .section-header.new-task {{ border-left: 6px solid #ff9800; }}
        .section-header.outcome-critical {{ border-left: 6px solid #f44336; }}
        .section-header.divergent {{ border-left: 6px solid #9c27b0; }}

        .section-header h2 {{
            margin: 0;
            flex: 1;
        }}

        .section-header .badge {{
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            color: white;
        }}

        .section-header.benign .badge {{ background: #4caf50; }}
        .section-header.new-task .badge {{ background: #ff9800; }}
        .section-header.outcome-critical .badge {{ background: #f44336; }}
        .section-header.divergent .badge {{ background: #9c27b0; }}

        /* Task cards */
        .task-card {{
            background: white;
            border-radius: 12px;
            margin-bottom: 20px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .task-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px 24px;
            cursor: pointer;
            user-select: none;
        }}

        .task-header:hover {{
            opacity: 0.95;
        }}

        .task-header .task-id {{
            font-family: monospace;
            font-size: 14px;
        }}

        .task-header .variant-id {{
            font-size: 12px;
            opacity: 0.8;
            margin-top: 4px;
        }}

        .task-header .toggle {{
            float: right;
            font-size: 20px;
        }}

        .task-header .class-label {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-left: 12px;
            vertical-align: middle;
        }}

        .task-header .class-label.benign {{ background: #4caf50; }}
        .task-header .class-label.new-task {{ background: #ff9800; }}
        .task-header .class-label.outcome-critical {{ background: #f44336; }}
        .task-header .class-label.divergent {{ background: #9c27b0; }}

        .task-content {{
            padding: 24px;
            display: none;
        }}

        .task-content.expanded {{
            display: block;
        }}

        /* Metadata row */
        .metadata-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }}

        .metadata-item {{
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
        }}

        .metadata-item label {{
            display: block;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
            margin-bottom: 4px;
        }}

        .metadata-item .value {{
            font-weight: 500;
            color: #333;
        }}

        /* Removed value */
        .removed-value {{
            margin-bottom: 20px;
            padding: 16px;
            background: #fff3e0;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
        }}

        .removed-value label {{
            display: block;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
            margin-bottom: 8px;
        }}

        .removed-value .value {{
            font-size: 14px;
            color: #333;
            line-height: 1.6;
        }}

        /* Prompts comparison */
        .prompts-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .prompt-box {{
            padding: 16px;
            border-radius: 8px;
            max-height: 400px;
            overflow-y: auto;
        }}

        .prompt-box.original {{
            background: #e3f2fd;
            border-left: 4px solid #1976d2;
        }}

        .prompt-box.underspec {{
            background: #fce4ec;
            border-left: 4px solid #c2185b;
        }}

        .prompt-box h4 {{
            margin: 0 0 12px 0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .prompt-box.original h4 {{ color: #1976d2; }}
        .prompt-box.underspec h4 {{ color: #c2185b; }}

        .prompt-box .prompt-text {{
            font-size: 14px;
            line-height: 1.7;
            color: #333;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        /* Judge reasoning */
        .judge-reasoning {{
            padding: 16px;
            background: #f3e5f5;
            border-radius: 8px;
            border-left: 4px solid #9c27b0;
        }}

        .judge-reasoning label {{
            display: block;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #7b1fa2;
            margin-bottom: 8px;
        }}

        .judge-reasoning .value {{
            font-size: 14px;
            color: #333;
            line-height: 1.6;
        }}

        /* Trial traces */
        .trials-section {{
            margin-top: 20px;
            border-top: 1px solid #e0e0e0;
            padding-top: 16px;
        }}

        .trials-section > label {{
            display: block;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
            margin-bottom: 12px;
        }}

        .trial-dropdown {{
            margin-bottom: 12px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}

        .trial-header {{
            background: #f8f9fa;
            padding: 10px 16px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        }}

        .trial-header:hover {{
            background: #f0f0f0;
        }}

        .trial-header .trial-title {{
            font-weight: 500;
            font-size: 13px;
        }}

        .trial-header .trial-score {{
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}

        .trial-header .trial-score.pass {{ background: #c8e6c9; color: #2e7d32; }}
        .trial-header .trial-score.fail {{ background: #ffcdd2; color: #c62828; }}
        .trial-header .trial-score.partial {{ background: #ffe0b2; color: #e65100; }}

        .trial-header .toggle-icon {{
            font-size: 16px;
            color: #666;
        }}

        .trial-body {{
            display: none;
            padding: 16px;
            max-height: 500px;
            overflow-y: auto;
            background: #fafafa;
        }}

        .trial-body.expanded {{
            display: block;
        }}

        /* Conversation message styles */
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

        .msg .role {{
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            color: #4a5568;
        }}

        .msg .tool-name {{
            font-family: monospace;
            background: #4299e1;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }}

        .msg .content {{ margin-top: 8px; white-space: pre-wrap; word-break: break-word; }}

        .msg pre.content, .msg pre.args {{
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

        /* Timestamp */
        .timestamp {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 40px;
            padding-bottom: 20px;
        }}

        /* Responsive */
        @media (max-width: 1200px) {{
            .summary {{
                grid-template-columns: repeat(2, 1fr);
            }}
            .metadata-row {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 900px) {{
            body {{
                padding-left: 20px;
            }}
            .nav {{
                display: none;
            }}
            .prompts-comparison {{
                grid-template-columns: 1fr;
            }}
            .summary {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <nav class="nav">
        <h3>Jump to</h3>
"""
    ]

    # Navigation links
    for cls_key, css_class, display_name, _ in class_config:
        count = len(by_class.get(cls_key, []))
        if count > 0:
            html_parts.append(
                f'        <a href="#{css_class}" class="{css_class}">{display_name}<span class="count">{count}</span></a>\n'
            )

    html_parts.append(
        """    </nav>

    <h1>Underspecification Task Viewer</h1>

    <div class="summary">
"""
    )

    # Summary cards
    for cls_key, css_class, display_name, _ in class_config:
        count = len(by_class.get(cls_key, []))
        html_parts.append(
            f"""        <div class="summary-card {css_class}">
            <h3>{display_name}</h3>
            <div class="count">{count}</div>
        </div>
"""
        )

    html_parts.append("    </div>\n")

    # Sections for each class
    for cls_key, css_class, display_name, _ in class_config:
        tasks = by_class.get(cls_key, [])
        if not tasks:
            continue

        html_parts.append(
            f"""
    <section id="{css_class}">
        <div class="section-header {css_class}">
            <h2>{display_name}</h2>
            <span class="badge">{len(tasks)} tasks</span>
        </div>
"""
        )

        for i, row in enumerate(tasks):
            variant_id = row.get("variant_id", "")
            task_id = row.get("task", "")
            variant_local_id = row.get("variant_local_id", "")

            # Parse metadata
            dimensions = parse_json_or_string(row.get("dimension", ""))
            subdimensions = parse_json_or_string(row.get("subdimension", ""))
            removed_values = parse_json_or_string(row.get("removed_value", ""))
            parsed_variant = parse_variant_id(variant_local_id)

            # Get prompts from lookup
            prompts = prompts_lookup.get(variant_id, {})
            original_prompt = prompts.get("original_prompt", "")
            underspec_prompt = prompts.get("underspecified_prompt", "")

            # Judge reasoning
            judge_reasoning = row.get("judge_reasoning", "")

            # Build card
            card_id = f"{css_class}-{i}"
            html_parts.append(
                f"""        <div class="task-card">
            <div class="task-header" onclick="toggleCard('{card_id}')">
                <span class="toggle" id="toggle-{card_id}">+</span>
                <div class="task-id">Task: {escape_html(task_id)}<span class="class-label {css_class}">{display_name}</span></div>
                <div class="variant-id">Variant: {escape_html(variant_local_id)}</div>
            </div>
            <div class="task-content" id="content-{card_id}">
                <div class="metadata-row">
                    <div class="metadata-item">
                        <label>Dimension</label>
                        <div class="value">{escape_html(", ".join(dimensions)) or "—"}</div>
                    </div>
                    <div class="metadata-item">
                        <label>Subdimension</label>
                        <div class="value">{escape_html(", ".join(subdimensions)) or "—"}</div>
                    </div>
                    <div class="metadata-item">
                        <label>Segments Removed</label>
                        <div class="value">{parsed_variant['segments']}</div>
                    </div>
                    <div class="metadata-item">
                        <label>Severity</label>
                        <div class="value">{escape_html(parsed_variant['severity'])}</div>
                    </div>
                </div>
"""
            )

            # Removed value
            if removed_values:
                removed_display = " | ".join(removed_values)
                html_parts.append(
                    f"""                <div class="removed-value">
                    <label>Removed Value</label>
                    <div class="value">{escape_html(removed_display)}</div>
                </div>
"""
                )

            # Prompts comparison
            if original_prompt or underspec_prompt:
                html_parts.append(
                    f"""                <div class="prompts-comparison">
                    <div class="prompt-box original">
                        <h4>Original Prompt</h4>
                        <div class="prompt-text">{escape_html(original_prompt) or "—"}</div>
                    </div>
                    <div class="prompt-box underspec">
                        <h4>Underspecified Prompt</h4>
                        <div class="prompt-text">{escape_html(underspec_prompt) or "—"}</div>
                    </div>
                </div>
"""
                )

            # Judge reasoning
            if judge_reasoning:
                html_parts.append(
                    f"""                <div class="judge-reasoning">
                    <label>Judge Reasoning</label>
                    <div class="value">{escape_html(judge_reasoning)}</div>
                </div>
"""
                )

            # Trial traces
            trials = traces_lookup.get(variant_id, [])
            if trials:
                html_parts.append(
                    """                <div class="trials-section">
                    <label>Underspecified Prompt Trials</label>
"""
                )
                for trial_idx, trial_data in enumerate(trials):
                    if trial_data is None:
                        continue
                    trial_id = f"{card_id}-trial-{trial_idx}"
                    conv_str = trial_data.get("raw_conversation_history", "")
                    score_str = trial_data.get("coverage_score", "")

                    # Determine score class
                    try:
                        score = float(score_str)
                        if score >= 0.75:
                            score_class = "pass"
                        elif score > 0:
                            score_class = "partial"
                        else:
                            score_class = "fail"
                        score_display = f"{score:.2f}"
                    except:
                        score_class = "fail"
                        score_display = "N/A"

                    conv = parse_conversation(conv_str)
                    conv_html = format_conversation(conv)

                    html_parts.append(
                        f"""                    <div class="trial-dropdown">
                        <div class="trial-header" onclick="toggleTrial('{trial_id}')">
                            <span class="trial-title">Trial {trial_idx + 1}</span>
                            <span class="trial-score {score_class}">{score_display}</span>
                            <span class="toggle-icon" id="toggle-{trial_id}">+</span>
                        </div>
                        <div class="trial-body" id="body-{trial_id}">
                            {conv_html}
                        </div>
                    </div>
"""
                    )
                html_parts.append("                </div>\n")

            html_parts.append(
                """            </div>
        </div>
"""
            )

        html_parts.append("    </section>\n")

    # Close HTML
    html_parts.append(
        f"""
    <div class="timestamp">
        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total tasks: {len(rows)}
    </div>

    <script>
        function toggleCard(cardId) {{
            const content = document.getElementById('content-' + cardId);
            const toggle = document.getElementById('toggle-' + cardId);
            if (content.classList.contains('expanded')) {{
                content.classList.remove('expanded');
                toggle.textContent = '+';
            }} else {{
                content.classList.add('expanded');
                toggle.textContent = '−';
            }}
        }}

        function toggleTrial(trialId) {{
            const body = document.getElementById('body-' + trialId);
            const toggle = document.getElementById('toggle-' + trialId);
            if (body.classList.contains('expanded')) {{
                body.classList.remove('expanded');
                toggle.textContent = '+';
            }} else {{
                body.classList.add('expanded');
                toggle.textContent = '−';
            }}
        }}
    </script>
</body>
</html>
"""
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"Viewer generated: {output_path}")
    print(f"  Total tasks: {len(rows)}")
    for cls_key, css_class, display_name, _ in class_config:
        count = len(by_class.get(cls_key, []))
        if count > 0:
            print(f"  - {display_name}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML viewer for underspecification tasks"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to filtered_underspec.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="underspec_viewer.html",
        help="Output HTML file path (default: underspec_viewer.html)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = input_path.parent / output_path

    # Load main data
    rows = safe_csv_read(input_path)

    # Auto-detect and load prompts
    prompts_path = find_prompts_file(input_path)
    prompts_lookup = {}
    if prompts_path:
        print(f"Loading prompts from: {prompts_path}")
        prompts_lookup = load_prompts(prompts_path)
        print(f"  Loaded {len(prompts_lookup)} prompt entries")
    else:
        print("Warning: underspec_prompts.csv not found, prompts will not be displayed")

    # Auto-detect and load completion files for traces
    traces_lookup = {}
    completion_files = find_completion_files(input_path)
    if completion_files:
        print(f"Loading traces from {len(completion_files)} completion files:")
        for f in completion_files:
            print(f"  - {f.name}")
        traces_lookup = load_completions(completion_files, prompts_lookup)
        print(f"  Loaded traces for {len(traces_lookup)} variants")
    else:
        print("Warning: No completion files found, traces will not be displayed")

    generate_html(rows, prompts_lookup, traces_lookup, output_path)


if __name__ == "__main__":
    main()
