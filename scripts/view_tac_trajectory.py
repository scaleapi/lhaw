#!/usr/bin/env python3
"""
Trajectory viewer for user-simulator experiments.

Usage:
    python scripts/view_tac_trajectory.py "pattern*"    # Generate HTML
    python scripts/view_tac_trajectory.py --list        # List recent runs

Output: experiments/agentcompany/runs/trajectories.html
"""

import argparse
import html as html_module
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import PATHS  # noqa: E402

RUNS_DIR = REPO_ROOT / PATHS["runs_dir"]
TASKS_DIR = REPO_ROOT / PATHS["tac_tasks"]

PROMPT_DESCRIPTIONS = {
    "baseline": "Baseline (no user simulator)",
    "user_sim": "User simulator (no explicit prompt)",
    "none": "No prompt",
    "default": "Prompted (paper default)",
}

from task_completion_agentcompany import CLARIFICATION_PROMPTS  # noqa: E402


def find_run_dirs(pattern: str) -> list[Path]:
    """Find run directories matching pattern."""
    if Path(pattern).is_dir():
        return [Path(pattern)]
    full_path = RUNS_DIR / pattern
    if full_path.is_dir():
        return [full_path]
    if "*" in pattern or "?" in pattern:
        return sorted(RUNS_DIR.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return sorted(RUNS_DIR.glob(f"*{pattern}*"), key=lambda x: x.stat().st_mtime, reverse=True)


def load_trajectory(run_dir: Path, task_filter: str = None) -> dict:
    """Load trajectory data from a run directory - returns ALL trials."""
    result = {"run_dir": run_dir.name, "run_path": str(run_dir), "prompt_key": None, "trials": []}

    # Detect experiment type from run name
    run_name = run_dir.name

    # First check if it's a baseline run (no user simulator, no clarification prompt)
    if "_baseline_" in run_name:
        result["prompt_key"] = "baseline"
    else:
        # Detect prompt variant from run name (check longer keys first to avoid partial matches)
        # e.g., "ctx_minimal" should match before "minimal"
        for key in sorted(CLARIFICATION_PROMPTS.keys(), key=len, reverse=True):
            if f"_{key}_" in run_name or run_name.endswith(f"_{key}"):
                result["prompt_key"] = key
                break

        # If still no match but it's a user_sim run, default to "none" (no explicit prompt)
        if result["prompt_key"] is None and "_user_sim_" in run_name:
            result["prompt_key"] = "user_sim"

    # Find tasks
    local_results = run_dir / "local_results"
    if not local_results.exists():
        return result

    for task_dir in sorted(local_results.iterdir()):
        if not task_dir.is_dir() or (task_filter and task_dir.name != task_filter):
            continue

        # Load prompts from run's local_tasks (has variant-specific info)
        prompts = {}

        # Try run's local_tasks first (has primary_task.json with variant info)
        for lt_dir in run_dir.glob("local_tasks_*"):
            run_task_json = lt_dir / task_dir.name / "primary_task.json"
            if run_task_json.exists():
                try:
                    task_def = json.loads(run_task_json.read_text().strip())
                    prompts = {
                        "primary": task_def.get("primary_task", ""),
                        "underspec": task_def.get("underspecified_task", ""),
                        "variant_info": {
                            "variant_id": task_def.get("_variant_id", ""),
                            "removed_segments": task_def.get("removed_segments", []),
                        },
                    }
                    break
                except Exception:
                    pass  # Fall back to next source

        # Fallback to base task.json if no local_tasks found
        if not prompts:
            task_json = TASKS_DIR / task_dir.name / "task.json"
            if task_json.exists():
                try:
                    task_def = json.loads(task_json.read_text().strip())
                    prompts = {
                        "primary": task_def.get("primary_task", ""),
                        "underspec": task_def.get("underspecified_task", ""),
                        "variant_info": {
                            "variant_id": task_def.get("_variant_id", ""),
                            "removed_segments": task_def.get("removed_segments", []),
                        },
                    }
                except Exception:
                    pass  # Fall back to next source

        # Load all evaluations for this task (indexed by trial number)
        evals = {}
        for eval_file in run_dir.glob(f"final_results/*/{task_dir.name}/*/evaluation_*.json"):
            try:
                idx = int(eval_file.stem.split("_")[-1])  # evaluation_0.json -> 0
                evals[idx] = json.loads(eval_file.read_text().strip())
            except Exception:
                continue

        # Collect ALL trials (sessions) for this task
        trial_idx = 0
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for date_dir in sorted(model_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                sessions_dir = date_dir / "state" / "sessions"
                if not sessions_dir.exists():
                    continue

                for session_dir in sorted(sessions_dir.iterdir()):
                    events_dir = session_dir / "events"
                    if not events_dir.exists():
                        continue

                    # Load events for this trial
                    events = []
                    event_files = sorted(
                        [f for f in events_dir.glob("*.json") if f.stem.isdigit()],
                        key=lambda x: int(x.stem),
                    )
                    for ev_file in event_files:
                        try:
                            ev = json.loads(ev_file.read_text().strip())
                            ev["_id"] = int(ev_file.stem)
                            events.append(ev)
                        except Exception:
                            continue

                    # Add this trial with matching eval
                    trial_id = f"{session_dir.name[:8]}"
                    result["trials"].append(
                        {
                            "task": task_dir.name,
                            "trial_id": trial_id,
                            "events": events,
                            "eval": evals.get(trial_idx),
                            "prompts": prompts,
                        }
                    )
                    trial_idx += 1

    return result


def format_event(ev: dict) -> dict:
    """Format an event for display. Returns dict with type, content, is_clarification."""
    action = ev.get("action", "")
    observation = ev.get("observation", "")
    source = ev.get("source", "")
    step = ev.get("_id", 0)

    result = {"step": step, "type": action, "content": "", "is_clar": False, "show": True}

    if action == "call_tool_mcp":
        args = ev.get("args", {})
        result["type"] = "mcp_call"
        # Tool name can be in args.name or tool_call_metadata.function_name
        tool_name = args.get("name", "") or ev.get("tool_call_metadata", {}).get(
            "function_name", ""
        )
        result["tool"] = tool_name
        # Arguments in args.arguments (dict) not args.function_arguments
        func_args = args.get("arguments", {})
        if isinstance(func_args, str):
            try:
                func_args = json.loads(func_args)
            except json.JSONDecodeError:
                pass
        result["content"] = func_args.get(
            "question",
            json.dumps(func_args, indent=2) if isinstance(func_args, dict) else str(func_args),
        )
        result["is_clar"] = "ask_user" in tool_name.lower() or "clarification" in tool_name.lower()

    elif observation == "mcp":
        # MCP response - observation type, not action type
        result["type"] = "mcp_response"
        tool_name = ev.get("tool_call_metadata", {}).get("function_name", "") or ev.get(
            "extras", {}
        ).get("name", "")
        result["tool"] = tool_name
        # Extract response text from JSON message
        msg = ev.get("message", "") or ev.get("content", "")
        if isinstance(msg, str):
            try:
                msg_data = json.loads(msg)
                result["content"] = msg_data.get("structuredContent", {}).get("result", msg)
            except Exception:
                result["content"] = msg
        else:
            result["content"] = str(msg)
        result["is_clar"] = "ask_user" in tool_name.lower() or "clarification" in tool_name.lower()

    elif action == "run":
        result["type"] = "bash"
        result["content"] = ev.get("args", {}).get("command", "")

    elif action == "read":
        result["type"] = "read"
        result["content"] = ev.get("args", {}).get("path", "")

    elif action == "edit":
        result["type"] = "edit"
        result["content"] = ev.get("args", {}).get("path", "")

    elif action == "write":
        result["type"] = "write"
        result["content"] = ev.get("args", {}).get("path", "")

    elif action == "finish":
        result["type"] = "finish"
        result["content"] = ev.get("args", {}).get("outputs", {}).get("content", "")

    elif action == "message" and source == "user":
        result["type"] = "user_msg"
        result["content"] = ev.get("message", "")

    elif action is None or action == "":
        # Observation events (command output, file content, etc.)
        obs = ev.get("observation", "")
        content = ev.get("content", "")
        if obs or content:
            result["type"] = "output"
            text = str(content) if not isinstance(content, str) else content
            result["content"] = text[:500] + "..." if len(text) > 500 else text
        else:
            result["show"] = False  # Skip empty observations

    else:
        result["show"] = False  # Skip system, recall, agent thinking, etc.

    return result


def generate_html(runs: list[dict], output_path: Path = None) -> str:
    """Generate combined HTML for all runs with all trials."""
    if output_path is None:
        reports_dir = RUNS_DIR / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / "trajectories.html"

    def esc(s):
        return html_module.escape(str(s)) if s else ""

    sections = []
    nav_items = []
    seen_keys = set()
    total_trials = 0

    for data in runs:
        prompt_key = data.get("prompt_key") or "unknown"
        trials = data.get("trials", [])
        total_trials += len(trials)

        if prompt_key not in seen_keys:
            seen_keys.add(prompt_key)
            nav_items.append(f'<a href="#{esc(prompt_key)}">{esc(prompt_key)} ({len(trials)})</a>')

        # Group trials by this prompt variant
        trials_html = ""
        for i, trial in enumerate(trials):
            prompts = trial.get("prompts", {})
            events = trial.get("events", [])
            ev_data = trial.get("eval", {})

            # Process events
            clar_used = False
            clar_q, clar_r = [], []
            events_html = ""
            action_count = 0

            # Use original step IDs for proper trajectory counting
            for ev in events:
                f = format_event(ev)
                if not f.get("show", True):
                    continue
                if f["is_clar"]:
                    clar_used = True

                step = f["step"]  # Original event ID

                if f["type"] == "mcp_call":
                    label = "ASK USER" if f["is_clar"] else "MCP"
                    if f["is_clar"]:
                        clar_q.append(f["content"])
                    events_html += f'<div class="ev mcp"><b>[{step}] {label}</b><pre>{esc(f["content"])}</pre></div>'
                    action_count += 1
                elif f["type"] == "mcp_response":
                    label = "USER RESPONSE" if f["is_clar"] else "MCP RESULT"
                    if f["is_clar"]:
                        clar_r.append(f["content"])
                    events_html += f'<div class="ev resp"><b>[{step}] {label}</b><pre>{esc(str(f["content"]))}</pre></div>'
                    action_count += 1
                elif f["type"] == "bash":
                    events_html += f'<div class="ev bash"><b>[{step}] RUN</b><pre>{esc(f["content"])}</pre></div>'
                    action_count += 1
                elif f["type"] == "read":
                    events_html += f'<div class="ev read"><b>[{step}] READ</b><pre>{esc(f["content"])}</pre></div>'
                    action_count += 1
                elif f["type"] == "edit":
                    events_html += f'<div class="ev edit"><b>[{step}] EDIT</b><pre>{esc(f["content"])}</pre></div>'
                    action_count += 1
                elif f["type"] == "write":
                    events_html += f'<div class="ev write"><b>[{step}] WRITE</b><pre>{esc(f["content"])}</pre></div>'
                    action_count += 1
                elif f["type"] == "output":
                    events_html += f'<div class="ev output"><b>[{step}] OUTPUT</b><pre>{esc(f["content"])}</pre></div>'
                elif f["type"] == "finish":
                    events_html += f'<div class="ev finish"><b>[{step}] FINISH</b></div>'
                    action_count += 1

            # Eval - use final_score for accurate pass/fail, checkpoints for breakdown
            fs = ev_data.get("final_score", {}) if ev_data else {}
            cps = ev_data.get("checkpoints", []) if ev_data else []

            # Use final_score (authoritative) if available, else sum checkpoint results
            if fs:
                passed = int(fs.get("result", 0) or 0)
                total = int(fs.get("total", len(cps)) or len(cps))
            else:
                passed = sum(int(cp.get("result", 0) or 0) for cp in cps)
                total = sum(int(cp.get("total", 1) or 1) for cp in cps)

            # Build trajectory state tuple showing pass/fail per checkpoint (1=pass, 0=fail)
            # A checkpoint passes if result >= total (or result > 0 for binary checkpoints)
            def cp_pass(cp):
                r = int(cp.get("result", 0) or 0)
                t = int(cp.get("total", 1) or 1)
                return 1 if r >= t else 0

            state_tuple = "(" + ", ".join(str(cp_pass(cp)) for cp in cps) + ")" if cps else "()"

            clar_class = "yes" if clar_used else "no"
            score_class = "yes" if passed == total and total > 0 else "no"

            trials_html += f"""
            <div class="trial">
                <div class="trial-header">
                    <b>Trial {i+1}</b> ({trial['trial_id']}) 
                    <span class="badge {clar_class}">Clar: {"Y" if clar_used else "N"}</span>
                    <span class="badge {score_class}">Score: {passed}/{total}</span>
                    <span class="state">{state_tuple}</span>
                </div>
                <div class="trial-events">{events_html if events_html else '<span class="empty">No actions</span>'}</div>
            </div>"""

        # Variant info (same for all trials)
        prompts = trials[0].get("prompts", {}) if trials else {}
        vi = prompts.get("variant_info", {})
        removed = vi.get("removed_segments", [])
        removed_str = (
            ", ".join(f"[{s.get('dimension', '?')}] {s.get('value', '')}" for s in removed)
            if removed
            else "N/A"
        )

        # Get prompt description and actual text
        prompt_desc = PROMPT_DESCRIPTIONS.get(prompt_key, "Unknown") if prompt_key else "Unknown"
        prompt_text = CLARIFICATION_PROMPTS.get(prompt_key, "") if prompt_key else ""

        sections.append(
            f"""
<div class="run" id="{esc(prompt_key)}">
    <h2>{esc(prompt_key.upper())} <span class="count">({len(trials)} trials)</span></h2>
    <div class="meta">
        <div><b>Prompt:</b> <code>{esc(prompt_key)}</code> — {esc(prompt_desc)}</div>
        <div><b>Variant:</b> {esc(vi.get('variant_id', 'N/A'))}</div>
        <div><b>Removed:</b> {esc(removed_str)}</div>
    </div>
    <details class="prompt-text">
        <summary>Clarification Prompt Text</summary>
        <pre>{esc(prompt_text.strip()) if prompt_text.strip() else "(none - baseline)"}</pre>
    </details>
    <details>
        <summary>Task Prompts</summary>
        <div class="prompts">
            <div><b>AGENT SAW (underspecified):</b><pre>{esc(prompts.get('underspec', 'N/A'))}</pre></div>
            <div><b>GROUND TRUTH (primary):</b><pre>{esc(prompts.get('primary', 'N/A'))}</pre></div>
        </div>
    </details>
    <div class="trials">{trials_html}</div>
</div>"""
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Trajectories ({total_trials} trials)</title>
<style>
body {{ font-family: system-ui; padding: 20px; background: #0a0a14; color: #e0e0e0; }}
h1 {{ color: #4299e1; }}
h2 {{ color: #fff; border-bottom: 2px solid #4299e1; padding-bottom: 8px; margin-top: 30px; }}
.count {{ color: #718096; font-size: 14px; font-weight: normal; }}
.nav {{ background: #1a1a2e; padding: 12px 15px; border-radius: 8px; margin-bottom: 20px; }}
.nav a {{ color: #63b3ed; margin: 0 8px; text-decoration: none; }}
.run {{ background: #1a1a2e; padding: 20px; border-radius: 12px; margin-bottom: 25px; }}
.badge {{ background: #2d3748; padding: 2px 8px; border-radius: 8px; font-size: 11px; margin-left: 6px; }}
.badge.yes {{ background: #276749; color: #9ae6b4; }}
.badge.no {{ background: #742a2a; color: #feb2b2; }}
.state {{ color: #a0aec0; font-family: monospace; font-size: 11px; margin-left: 8px; }}
.meta {{ color: #718096; font-size: 12px; margin: 8px 0; }}
.meta code {{ background: #2d3748; padding: 2px 6px; border-radius: 4px; color: #ecc94b; }}
details {{ margin: 10px 0; }}
summary {{ cursor: pointer; color: #63b3ed; }}
.prompts pre {{ background: #0f0f1a; padding: 10px; border-radius: 6px; font-size: 11px; white-space: pre-wrap; max-height: 150px; overflow-y: auto; margin: 6px 0; }}
.prompt-text {{ margin: 8px 0; }}
.prompt-text pre {{ background: #1a2332; border: 1px solid #4299e1; padding: 10px; border-radius: 6px; font-size: 11px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; margin: 6px 0; color: #e2e8f0; }}
.trials {{ margin-top: 15px; }}
.trial {{ background: rgba(0,0,0,0.2); border-radius: 8px; padding: 15px; margin-bottom: 15px; }}
.trial-header {{ margin-bottom: 10px; font-size: 14px; }}
.trial-events {{ max-height: 600px; overflow-y: auto; }}
.ev {{ margin: 6px 0; padding: 8px 12px; border-radius: 4px; font-size: 13px; }}
.ev pre {{ background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px; font-size: 12px; white-space: pre-wrap; max-height: 200px; overflow-y: auto; margin: 6px 0 0 0; }}
.ev.mcp {{ background: rgba(236,201,75,0.1); border-left: 3px solid #ecc94b; }}
.ev.resp {{ background: rgba(56,178,172,0.12); border-left: 3px solid #38b2ac; }}
.ev.bash {{ background: rgba(160,174,192,0.06); border-left: 3px solid #718096; }}
.ev.read {{ background: rgba(99,179,237,0.06); border-left: 3px solid #4299e1; }}
.ev.edit {{ background: rgba(237,137,54,0.06); border-left: 3px solid #ed8936; }}
.ev.write {{ background: rgba(72,187,120,0.06); border-left: 3px solid #38a169; }}
.ev.output {{ background: rgba(0,0,0,0.25); border-left: 3px solid #4a5568; font-size: 12px; }}
.ev.finish {{ background: rgba(159,122,234,0.06); border-left: 3px solid #805ad5; }}
.empty {{ color: #718096; font-style: italic; }}
</style>
</head>
<body>
<h1>Trajectories ({total_trials} trials across {len(runs)} variants)</h1>
<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
<div class="nav"><b>Jump:</b> {" | ".join(nav_items)}</div>
{"".join(sections)}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"✅ Generated: {output_path}")
    return str(output_path)


def list_runs(n: int = 15):
    """List recent runs."""
    if not RUNS_DIR.exists():
        print(f"❌ Runs directory not found: {RUNS_DIR}")
        return

    runs = sorted(RUNS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    print("\n📁 Recent runs:\n")

    for i, r in enumerate(runs[:n]):
        if r.is_dir() and r.name.startswith("run_"):
            local_results = r / "local_results"
            n_tasks = len(list(local_results.iterdir())) if local_results.exists() else 0
            prompt_key = next(
                (
                    k
                    for k in sorted(CLARIFICATION_PROMPTS.keys(), key=len, reverse=True)
                    if f"_{k}_" in r.name
                ),
                None,
            )
            print(
                f"  {i+1:2}. {r.name} ({n_tasks} tasks){f' [{prompt_key}]' if prompt_key else ''}"
            )


def main():
    parser = argparse.ArgumentParser(description="Generate HTML trajectory view for user-sim runs")
    parser.add_argument("pattern", nargs="?", help="Run directory or glob pattern")
    parser.add_argument("--task", "-t", help="Filter to specific task")
    parser.add_argument("--output", "-o", help="Output path for HTML")
    parser.add_argument("--list", "-l", action="store_true", help="List recent runs")
    parser.add_argument("--limit", type=int, default=None, help="Max run directories to scan")
    args = parser.parse_args()

    if args.list:
        list_runs()
        return

    if not args.pattern:
        parser.print_help()
        print("\n💡 Example:")
        print('  python scripts/view_tac_trajectory.py "run_underspecified_*"')
        return

    # Find matching runs
    run_dirs = find_run_dirs(args.pattern)
    if not run_dirs:
        print(f"❌ No runs found matching: {args.pattern}")
        list_runs(5)
        return

    if args.limit and len(run_dirs) > args.limit:
        print(f"📂 Found {len(run_dirs)} matching runs, limiting to {args.limit}")
        run_dirs = run_dirs[: args.limit]
    else:
        print(f"📂 Found {len(run_dirs)} matching runs")

    # Load all runs
    runs = []
    total_trials = 0
    unique_variants = set()
    for rd in run_dirs:
        data = load_trajectory(rd, args.task)
        if data["trials"]:
            runs.append(data)
            total_trials += len(data["trials"])
            # Extract variant_id from first trial's prompts
            for trial in data["trials"]:
                prompts = trial.get("prompts", {})
                vi = prompts.get("variant_info", {})
                vid = vi.get("variant_id", "")
                if vid:
                    unique_variants.add(vid)
                    break

    if not runs:
        print("❌ No trials found in matching runs")
        return

    n_variants = len(unique_variants) if unique_variants else len(runs)
    print(f"📊 Total: {total_trials} trials across {n_variants} variants ({len(runs)} run dirs)")
    output_path = Path(args.output) if args.output else None
    generate_html(runs, output_path)


if __name__ == "__main__":
    main()
