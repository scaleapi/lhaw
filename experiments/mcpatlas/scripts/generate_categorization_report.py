"""
Generate an HTML report for investigating categorized underspecification tasks.

Uses the consistency method results to display tasks in three categories:
1. Originally Overspecified - success rate unchanged after underspecification
2. New Tasks - consistent interpretation despite failing automated eval  
3. Underspecified - inconsistent responses indicating true ambiguity

Usage:
    uv run generate_categorization_report.py \
        --dataset-dir experiments/underspec_datasets/intersection_delete_v1_20260120_110000 \
        --output categorization_report.html
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def generate_html_report(dataset_dir: Path, output_path: Path) -> None:
    """Generate an HTML report from categorization results."""
    
    # Load consistency results
    results_path = dataset_dir / "categorization_consistency.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Categorization results not found: {results_path}")
    
    with open(results_path) as f:
        results = json.load(f)
    
    summary = results["summary"]
    task_details = results["task_details"]
    
    # Group tasks by category
    categories = {
        "originally_overspecified": [],
        "new_task": [],
        "underspecified": [],
    }
    
    for task_id, details in task_details.items():
        category = details.get("category", "unknown")
        if category in categories:
            categories[category].append((task_id, details))
    
    # Sort each category by task_id for consistency
    for cat in categories:
        categories[cat].sort(key=lambda x: x[0])
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Underspecification Categorization Report</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        h1 {{
            color: #1a237e;
            border-bottom: 3px solid #1a237e;
            padding-bottom: 10px;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .summary-card {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .summary-card.overspecified {{
            border-top: 4px solid #4caf50;
        }}
        
        .summary-card.new-task {{
            border-top: 4px solid #ff9800;
        }}
        
        .summary-card.underspecified {{
            border-top: 4px solid #f44336;
        }}
        
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #666;
        }}
        
        .summary-card .count {{
            font-size: 48px;
            font-weight: 700;
            margin: 10px 0;
        }}
        
        .summary-card.overspecified .count {{ color: #2e7d32; }}
        .summary-card.new-task .count {{ color: #e65100; }}
        .summary-card.underspecified .count {{ color: #c62828; }}
        
        .summary-card .percent {{
            font-size: 18px;
            color: #666;
        }}
        
        .category-section {{
            margin-bottom: 40px;
        }}
        
        .category-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding: 16px 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .category-header.overspecified {{ border-left: 6px solid #4caf50; }}
        .category-header.new-task {{ border-left: 6px solid #ff9800; }}
        .category-header.underspecified {{ border-left: 6px solid #f44336; }}
        
        .category-header h2 {{
            margin: 0;
            flex: 1;
        }}
        
        .category-header .badge {{
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            color: white;
        }}
        
        .category-header.overspecified .badge {{ background: #4caf50; }}
        .category-header.new-task .badge {{ background: #ff9800; }}
        .category-header.underspecified .badge {{ background: #f44336; }}
        
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
        }}
        
        .task-header .task-id {{
            font-family: monospace;
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .task-content {{
            padding: 24px;
        }}
        
        .removed-segments {{
            margin-bottom: 20px;
            padding: 16px;
            background: #fafafa;
            border-radius: 8px;
        }}
        
        .removed-segments h4 {{
            margin: 0 0 10px 0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
        }}
        
        .segment-tag {{
            display: inline-block;
            background: #f5f5f5;
            border: 1px solid #ddd;
            padding: 6px 14px;
            border-radius: 4px;
            margin: 4px;
            font-size: 14px;
            color: #111;
            font-weight: 500;
        }}
        
        .prompts-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .prompt-box {{
            padding: 16px;
            border-radius: 8px;
        }}
        
        .prompt-box.original {{
            background: #f8f9fa;
            border-left: 4px solid #1976d2;
        }}
        
        .prompt-box.underspec {{
            background: #fff8f8;
            border-left: 4px solid #d32f2f;
        }}
        
        .prompt-box h4 {{
            margin: 0 0 12px 0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .prompt-box.original h4 {{ color: #1976d2; }}
        .prompt-box.underspec h4 {{ color: #d32f2f; }}
        
        .prompt-box p {{
            margin: 0;
            font-size: 14px;
            line-height: 1.7;
            color: #333;
        }}
        
        .metadata {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        .metadata-box {{
            padding: 16px;
            background: #fafafa;
            border-radius: 8px;
        }}
        
        .metadata-box h4 {{
            margin: 0 0 10px 0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
        }}
        
        .metadata-box p {{
            margin: 0;
            font-size: 14px;
            color: #555;
            line-height: 1.6;
        }}
        
        .judge-result {{
            margin-top: 20px;
            padding: 16px;
            border-radius: 8px;
        }}
        
        .judge-result.consistent {{
            background: #e8f5e9;
            border: 1px solid #4caf50;
        }}
        
        .judge-result.inconsistent {{
            background: #ffebee;
            border: 1px solid #f44336;
        }}
        
        .judge-result h4 {{
            margin: 0 0 10px 0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .judge-result.consistent h4 {{ color: #2e7d32; }}
        .judge-result.inconsistent h4 {{ color: #c62828; }}
        
        .judge-result p {{
            margin: 0;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        .nav {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            z-index: 100;
        }}
        
        .nav a {{
            display: block;
            padding: 8px 16px;
            color: #333;
            text-decoration: none;
            border-radius: 6px;
            margin: 4px 0;
            font-size: 14px;
        }}
        
        .nav a:hover {{
            background: #f5f5f5;
        }}
        
        .nav a.overspecified {{ border-left: 3px solid #4caf50; }}
        .nav a.new-task {{ border-left: 3px solid #ff9800; }}
        .nav a.underspecified {{ border-left: 3px solid #f44336; }}
        
        .timestamp {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 40px;
        }}
        
        @media (max-width: 900px) {{
            .prompts-comparison, .metadata {{
                grid-template-columns: 1fr;
            }}
            .summary {{
                grid-template-columns: 1fr;
            }}
            .nav {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <nav class="nav">
        <strong>Jump to:</strong>
        <a href="#overspecified" class="overspecified">Overspecified ({len(categories['originally_overspecified'])})</a>
        <a href="#new-task" class="new-task">New Tasks ({len(categories['new_task'])})</a>
        <a href="#underspecified" class="underspecified">Underspecified ({len(categories['underspecified'])})</a>
    </nav>

    <h1>Underspecification Categorization Report</h1>
    <p>Dataset: <code>{dataset_dir.name}</code> | Method: Consistency | Model: {summary.get('judge_model', 'N/A')}</p>
    
    <div class="summary">
        <div class="summary-card overspecified">
            <h3>Originally Overspecified</h3>
            <div class="count">{summary['originally_overspecified']}</div>
            <div class="percent">{100*summary['originally_overspecified']/summary['total_tasks']:.1f}%</div>
            <p style="font-size: 12px; color: #666; margin-top: 10px;">Success rate unchanged</p>
        </div>
        <div class="summary-card new-task">
            <h3>New Tasks</h3>
            <div class="count">{summary['new_tasks']}</div>
            <div class="percent">{100*summary['new_tasks']/summary['total_tasks']:.1f}%</div>
            <p style="font-size: 12px; color: #666; margin-top: 10px;">Consistent interpretation</p>
        </div>
        <div class="summary-card underspecified">
            <h3>Truly Underspecified</h3>
            <div class="count">{summary['underspecified']}</div>
            <div class="percent">{100*summary['underspecified']/summary['total_tasks']:.1f}%</div>
            <p style="font-size: 12px; color: #666; margin-top: 10px;">Inconsistent responses</p>
        </div>
    </div>
"""
    
    # Generate sections for each category
    category_configs = [
        ("originally_overspecified", "overspecified", "Originally Overspecified", 
         "Tasks where the success rate after underspecification equals the baseline - the removed information was redundant."),
        ("new_task", "new-task", "New Tasks",
         "Tasks that fail automated evaluation but show consistent behavior across trials - the agent has a clear interpretation."),
        ("underspecified", "underspecified", "Truly Underspecified",
         "Tasks with inconsistent responses across trials - genuine ambiguity that causes different interpretations."),
    ]
    
    for cat_key, cat_class, cat_title, cat_desc in category_configs:
        tasks = categories[cat_key]
        html += f"""
    <section class="category-section" id="{cat_class}">
        <div class="category-header {cat_class}">
            <h2>{cat_title}</h2>
            <span class="badge">{len(tasks)} tasks</span>
        </div>
        <p style="color: #666; margin-bottom: 20px;">{cat_desc}</p>
"""
        
        for task_id, details in tasks:
            segments = details.get("removed_segments", [])
            segments_html = "".join([f'<span class="segment-tag">{s}</span>' for s in segments])
            
            original_prompt = details.get("original_prompt", "N/A")
            underspec_prompt = details.get("underspec_prompt", "N/A")
            expected_failure = details.get("expected_failure_mode", "N/A")
            judge_verdict = details.get("judge_verdict", "N/A")
            judge_reasoning = details.get("judge_reasoning", "N/A")
            
            # Determine judge result styling
            if cat_key == "originally_overspecified":
                judge_class = "consistent"
                judge_label = "All Trials Passed"
            elif cat_key == "new_task":
                judge_class = "consistent"
                judge_label = f"Verdict: {judge_verdict}"
            else:
                judge_class = "inconsistent"
                judge_label = f"Verdict: {judge_verdict}"
            
            html += f"""
        <div class="task-card">
            <div class="task-header">
                <div class="task-id">Task ID: {task_id}</div>
            </div>
            <div class="task-content">
                <div class="removed-segments">
                    <h4>Removed Segments</h4>
                    {segments_html if segments_html else '<span style="color: #999;">None</span>'}
                </div>
                
                <div class="prompts-comparison">
                    <div class="prompt-box original">
                        <h4>Original Prompt</h4>
                        <p>{original_prompt}</p>
                    </div>
                    <div class="prompt-box underspec">
                        <h4>Underspecified Prompt</h4>
                        <p>{underspec_prompt}</p>
                    </div>
                </div>
                
                <div class="metadata">
                    <div class="metadata-box">
                        <h4>Expected Failure Mode</h4>
                        <p>{expected_failure}</p>
                    </div>
                    <div class="metadata-box">
                        <h4>Evaluation Stats</h4>
                        <p>Baseline Pass@3: {details.get('baseline_pass_at_3', 'N/A')}<br>
                        Underspec Successes: {details.get('underspec_n_success', 'N/A')}/3</p>
                    </div>
                </div>
"""
            
            # Add judge result for judged tasks
            if cat_key != "originally_overspecified" and judge_reasoning:
                html += f"""
                <div class="judge-result {judge_class}">
                    <h4>{judge_label}</h4>
                    <p>{judge_reasoning}</p>
                </div>
"""
            
            html += """
            </div>
        </div>
"""
        
        html += """
    </section>
"""
    
    # Close HTML
    html += f"""
    <div class="timestamp">
        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""
    
    # Write output
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"Report generated: {output_path}")
    print(f"  - Originally Overspecified: {len(categories['originally_overspecified'])}")
    print(f"  - New Tasks: {len(categories['new_task'])}")
    print(f"  - Underspecified: {len(categories['underspecified'])}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML categorization report")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing categorization_consistency.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="categorization_report.html",
        help="Output HTML file path (default: categorization_report.html)",
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = dataset_dir / output_path
    
    generate_html_report(dataset_dir, output_path)


if __name__ == "__main__":
    main()
