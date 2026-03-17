"""
Categorize underspecified tasks into three categories:
1. Originally overspecified - success rate after underspecification is at least as high as before
   (for intersection tasks with 100% baseline, this means ALL trials must still pass)
2. New task - success rate dropped, but LLM judge thinks task is complete as specified
3. Underspecified task - success rate dropped, and LLM judge agrees task is incomplete

Two algorithms for identifying "new tasks" (--new-task-identifier):
- "completion": Judge if task was completed based on underspec prompt (single response)
- "consistency": Judge if all trials gave consistent responses (compares all trials)

Usage:
    # Using completion-based judging (default)
    uv run categorize_underspec_tasks.py \
        --dataset-dir experiments/mcpatlas/experiments/underspec_datasets/intersection_delete_v1_20260120_110000 \
        --run-dir run_underspec_larger_gemini-3-pro-preview_20260121_014321 \
        --output categorization_results.json \
        --judge-model gemini/gemini-3-flash-preview
    
    # Using consistency-based judging (compares all trials)
    uv run categorize_underspec_tasks.py \
        --dataset-dir experiments/mcpatlas/experiments/underspec_datasets/intersection_delete_v1_20260120_110000 \
        --run-dir run_underspec_larger_gemini-3-pro-preview_20260121_014321 \
        --output categorization_results_consistency.json \
        --judge-model gemini/gemini-3-flash-preview \
        --new-task-identifier consistency
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Increase CSV field size limit for large responses
csv.field_size_limit(10 * 1024 * 1024)

import litellm  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tqdm.asyncio import tqdm as async_tqdm  # noqa: E402

# Load environment variables (override=True to prefer .env over shell env)
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# Priority: LLM_API_KEY > OPENAI_API_KEY > LITELLM_API_KEY
api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")
if api_key:
    litellm.api_key = api_key
    logger.info("API key configured (source: env)")

# Priority: LLM_BASE_URL > LITELLM_BASE_URL
api_base = os.getenv("LLM_BASE_URL") or os.getenv("LITELLM_BASE_URL")
if api_base:
    litellm.api_base = api_base
    logger.info(f"Using API base URL: {api_base}")


class TaskCategory(str, Enum):
    ORIGINALLY_OVERSPECIFIED = "originally_overspecified"
    NEW_TASK = "new_task"
    UNDERSPECIFIED = "underspecified"


@dataclass
class TaskData:
    """Container for all data about a task."""

    task_id: str
    original_prompt: str
    underspec_prompt: str
    original_checkpoints: List[str]
    removed_segments: List[Dict[str, Any]]
    expected_failure_mode: str
    expected_questions: List[str]

    # Baseline results (original prompt)
    baseline_pass_at_3: float

    # Underspec results
    underspec_pass_at_3: float
    underspec_n_success: int

    # Agent outputs for judging
    agent_responses: List[str]  # From each trial

    # Categorization results
    category: Optional[TaskCategory] = None
    judge_verdict: Optional[str] = None
    judge_reasoning: Optional[str] = None


@dataclass
class CategorizationResult:
    """Results of the categorization process."""

    originally_overspecified: List[str]
    new_tasks: List[str]
    underspecified: List[str]
    task_details: Dict[str, Dict[str, Any]]
    summary: Dict[str, Any]


def load_task_json(json_dir: Path, task_id: str) -> Optional[Dict]:
    """Load the task JSON file with segments, variants, etc."""
    json_path = json_dir / f"{task_id}.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def load_baseline_pass_k(baseline_dir: Path) -> Dict[str, float]:
    """Load baseline pass@3 results for all models, return average or best."""
    pass_k_results = {}

    for model in ["gpt", "gemini", "claude"]:
        csv_path = baseline_dir / f"{model}_pass_k_per_task.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_id = row["task_id"]
                    pass_at_3 = float(row["pass@3"])
                    if task_id not in pass_k_results:
                        pass_k_results[task_id] = []
                    pass_k_results[task_id].append(pass_at_3)

    # Average across models (they should all pass since these are intersection tasks)
    return {task_id: sum(scores) / len(scores) for task_id, scores in pass_k_results.items()}


def load_underspec_evaluation(run_dir: Path) -> Dict[str, Dict]:
    """Load the underspec evaluation results."""
    eval_files = list(run_dir.glob("evaluation_*.json"))
    if not eval_files:
        raise FileNotFoundError(f"No evaluation files found in {run_dir}")

    # Use the first (should be only one)
    with open(eval_files[0]) as f:
        data = json.load(f)

    return data.get("tasks", {})


def load_completions(run_dir: Path) -> Dict[str, List[str]]:
    """Load agent responses from completion CSV files."""
    completions = {}

    for csv_path in sorted(run_dir.glob("completions_*.csv")):
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = row.get("TASK", "")
                response = row.get("script_model_response", "")

                if task_id:
                    if task_id not in completions:
                        completions[task_id] = []
                    completions[task_id].append(response)

    return completions


def get_judge_prompt(task_data: TaskData, response_idx: int = 0) -> str:
    """
    Create the prompt for LLM judge.

    Important: The judge should NOT see the original checkpoints since those
    are based on the original (fully specified) prompt.
    """
    response = ""
    if task_data.agent_responses:
        # Try to find a non-empty response
        for resp in task_data.agent_responses:
            if resp and resp.strip():
                response = resp
                break
        if not response:
            response = (
                task_data.agent_responses[response_idx]
                if response_idx < len(task_data.agent_responses)
                else ""
            )

    if not response or not response.strip():
        response = "[No response recorded - agent may have encountered an error]"

    return f"""You are an impartial judge evaluating whether an AI assistant has successfully completed a task.

## Task Given to Assistant
{task_data.underspec_prompt}

## Assistant's Final Response
{response}

## Your Evaluation Task
Determine whether the assistant has successfully completed the task AS SPECIFIED in the prompt above.

Important guidelines:
- Judge based ONLY on what was asked in the prompt
- If the prompt was ambiguous or underspecified, consider whether the assistant made a reasonable interpretation
- If the assistant asked clarifying questions or noted ambiguity, that's acceptable
- The task is complete if the assistant provided a reasonable answer to what was asked

## Response Format
Provide your evaluation in the following JSON format:
{{
    "task_completed": true/false,
    "reasoning": "Brief explanation of your verdict",
    "notes": "Any observations about ambiguity or underspecification in the prompt"
}}

Respond ONLY with the JSON object, no other text."""


def extract_json_from_response(content: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract JSON from LLM response, handling various formats.

    Handles:
    - Plain JSON
    - JSON in markdown code blocks (```json or ```)
    - JSON with trailing text
    - JSON with leading text
    """
    content = content.strip()

    # Try 1: Direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from markdown code blocks
    if "```" in content:
        # Try to find JSON in code blocks
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        matches = re.findall(code_block_pattern, content)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Try 3: Find JSON object with regex (handles leading/trailing text)
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, content, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try 4: More aggressive - find anything between first { and last }
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = content[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


async def judge_task_completion(
    task_data: TaskData,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Use LLM to judge whether a task was completed based on underspec prompt."""
    async with semaphore:
        last_error = None
        raw_content = None

        for attempt in range(max_retries + 1):
            try:
                # Use the best response (first successful one, or just first)
                prompt = get_judge_prompt(task_data)

                # Add retry hint if this is a retry
                if attempt > 0:
                    prompt += "\n\nIMPORTANT: Your response MUST be valid JSON only. No markdown, no extra text."

                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0 if attempt == 0 else 0.1,  # Slight temp increase on retry
                    max_tokens=1000,
                    response_format={"type": "json_object"},  # Force JSON output
                )

                raw_content = response.choices[0].message.content.strip()

                # Use robust JSON extraction (still useful as fallback)
                result = extract_json_from_response(raw_content)

                if result is not None:
                    return {
                        "task_completed": result.get("task_completed", False),
                        "reasoning": result.get("reasoning", ""),
                        "notes": result.get("notes", ""),
                    }
                else:
                    last_error = f"Could not parse JSON from response: {raw_content[:200]}"
                    logger.warning(
                        f"Attempt {attempt + 1}: Failed to parse JSON for task {task_data.task_id}: {raw_content[:100]}"
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}: Error for task {task_data.task_id}: {e}")

        # All retries failed
        logger.error(f"All attempts failed for task {task_data.task_id}: {last_error}")
        logger.debug(f"Raw response was: {raw_content[:500] if raw_content else 'None'}...")
        return {
            "task_completed": None,
            "reasoning": f"Error: {last_error}",
            "notes": f"Raw response: {raw_content[:200] if raw_content else 'None'}...",
        }


def get_consistency_judge_prompt(task_data: TaskData) -> str:
    """
    Create a prompt for the LLM to judge whether all trial responses are consistent.

    If all trials converge on the same answer/approach, the agent has a clear
    interpretation of the ambiguous prompt, suggesting it's a "new task".
    If trials diverge, it suggests true underspecification.
    """
    responses = task_data.agent_responses or []

    # Format each trial's response (no truncation - show full responses)
    trial_sections = []
    for i, resp in enumerate(responses):
        if resp and resp.strip():
            trial_sections.append(f"## Trial {i + 1} Final Response\n{resp}")
        else:
            trial_sections.append(f"## Trial {i + 1} Final Response\n[No response recorded]")

    trials_text = "\n\n".join(trial_sections) if trial_sections else "[No trials recorded]"

    return f"""You are an impartial judge evaluating whether an AI assistant gave consistent responses across multiple attempts at the same task.

## Task Given to Assistant
{task_data.underspec_prompt}

## Multiple Attempts
The assistant attempted this task {len(responses)} times. Here are the final responses from each attempt:

{trials_text}

## Your Evaluation Task
Determine whether the assistant's responses across all trials are **substantively consistent** - i.e., do they arrive at the same conclusion/answer, even if worded differently?

Consider:
- Do all trials reach the same final answer or conclusion?
- Do all trials interpret any ambiguities in the prompt the same way?
- Minor differences in wording or formatting don't matter - focus on the substance

If the trials are consistent, it suggests the assistant has a clear, confident interpretation of the task (even if different from what was originally intended).

If the trials diverge significantly, it suggests the task is genuinely ambiguous and the assistant is uncertain.

## Response Format
Provide your evaluation in the following JSON format:
{{
    "responses_consistent": true/false,
    "reasoning": "Brief explanation comparing the trial responses",
    "interpretation_used": "If consistent, briefly describe what interpretation/answer the agent converged on"
}}

Respond ONLY with the JSON object, no other text."""


async def judge_response_consistency(
    task_data: TaskData,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Use LLM to judge whether all trial responses are consistent with each other."""
    async with semaphore:
        last_error = None
        raw_content = None

        for attempt in range(max_retries + 1):
            try:
                prompt = get_consistency_judge_prompt(task_data)

                # Add retry hint if this is a retry
                if attempt > 0:
                    prompt += "\n\nIMPORTANT: Your response MUST be valid JSON only. No markdown, no extra text."

                response = await litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0 if attempt == 0 else 0.1,  # Slight temp increase on retry
                    max_tokens=1500,
                    response_format={"type": "json_object"},  # Force JSON output
                )

                raw_content = response.choices[0].message.content.strip()

                # Use robust JSON extraction (still useful as fallback)
                result = extract_json_from_response(raw_content)

                if result is not None:
                    return {
                        "responses_consistent": result.get("responses_consistent", False),
                        "reasoning": result.get("reasoning", ""),
                        "interpretation_used": result.get("interpretation_used", ""),
                    }
                else:
                    last_error = f"Could not parse JSON from response: {raw_content[:200]}"
                    logger.warning(
                        f"Attempt {attempt + 1}: Failed to parse JSON for task {task_data.task_id}: {raw_content[:100]}"
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}: Error for task {task_data.task_id}: {e}")

        # All retries failed
        logger.error(f"All attempts failed for consistency task {task_data.task_id}: {last_error}")
        logger.debug(f"Raw response was: {raw_content[:500] if raw_content else 'None'}...")
        return {
            "responses_consistent": None,
            "reasoning": f"Error: {last_error}",
            "interpretation_used": f"Raw response: {raw_content[:200] if raw_content else 'None'}...",
        }


async def categorize_tasks(
    dataset_dir: Path,
    run_dir: Path,
    judge_model: str,
    concurrency: int = 5,
    new_task_identifier: str = "completion",
) -> CategorizationResult:
    """
    Main categorization logic.

    Args:
        dataset_dir: Path to the underspec dataset directory
        run_dir: Name of the run directory within dataset_dir
        judge_model: Model to use for judging
        concurrency: Number of concurrent LLM requests
        new_task_identifier: Algorithm for identifying new tasks:
            - "completion": Judge if task was completed based on underspec prompt
            - "consistency": Judge if all trials gave consistent responses
    """

    json_dir = dataset_dir / "json"
    baseline_dir = dataset_dir / "baseline_results"
    full_run_dir = dataset_dir / run_dir

    # Load all data
    logger.info("Loading baseline pass@k results...")
    baseline_pass_k = load_baseline_pass_k(baseline_dir)

    logger.info("Loading underspec evaluation results...")
    underspec_eval = load_underspec_evaluation(full_run_dir)

    logger.info("Loading agent completions...")
    completions = load_completions(full_run_dir)

    # Build task data objects
    tasks: Dict[str, TaskData] = {}

    for task_id in underspec_eval.keys():
        task_json = load_task_json(json_dir, task_id)
        if not task_json:
            logger.warning(f"No JSON file for task {task_id}")
            continue

        eval_data = underspec_eval[task_id]
        n_success = eval_data.get("n_success", 0)
        n_trials = eval_data.get("n_trials", 3)

        # Get the variant used (first variant)
        variants = task_json.get("variants", [])
        variant = variants[0] if variants else {}

        tasks[task_id] = TaskData(
            task_id=task_id,
            original_prompt=task_json.get("task", {}).get("prompt", ""),
            underspec_prompt=variant.get("underspecified_prompt", ""),
            original_checkpoints=task_json.get("task", {}).get("checkpoints", []),
            removed_segments=variant.get("removed_segments", []),
            expected_failure_mode=variant.get("expected_failure_mode", ""),
            expected_questions=variant.get("expected_questions", []),
            baseline_pass_at_3=baseline_pass_k.get(
                task_id, 1.0
            ),  # Default to 1.0 for intersection tasks
            underspec_pass_at_3=1.0 if n_success >= 1 else 0.0,  # pass@3 = 1 if any trial succeeded
            underspec_n_success=n_success,
            agent_responses=completions.get(task_id, []),
        )

    # Categorize tasks
    originally_overspecified = []
    needs_judging = []

    for task_id, task_data in tasks.items():
        # Get the number of trials (usually 3)
        eval_data = underspec_eval.get(task_id, {})
        n_trials = eval_data.get("n_trials", 3)

        # Calculate success rates
        underspec_success_rate = task_data.underspec_n_success / n_trials if n_trials > 0 else 0.0
        baseline_success_rate = task_data.baseline_pass_at_3  # Typically 1.0 for intersection tasks

        # Originally overspecified = success rate after underspec is at least as high as before
        # For intersection tasks where baseline = 1.0, this means ALL trials must pass
        if underspec_success_rate >= baseline_success_rate:
            task_data.category = TaskCategory.ORIGINALLY_OVERSPECIFIED
            originally_overspecified.append(task_id)
        else:
            # Task success rate dropped - needs LLM judge to determine if new task or underspecified
            needs_judging.append(task_id)

    logger.info(f"Found {len(originally_overspecified)} originally overspecified tasks")
    logger.info(f"Need to judge {len(needs_judging)} tasks")
    logger.info(f"Using new task identifier: {new_task_identifier}")

    # Judge tasks that now fail
    new_tasks = []
    underspecified = []

    if needs_judging:
        semaphore = asyncio.Semaphore(concurrency)

        if new_task_identifier == "completion":
            # Algorithm 1: Judge if task was completed based on underspec prompt
            async def judge_and_categorize(task_id: str):
                task_data = tasks[task_id]
                result = await judge_task_completion(task_data, judge_model, semaphore)

                task_data.judge_verdict = "complete" if result["task_completed"] else "incomplete"
                task_data.judge_reasoning = result["reasoning"]

                if result["task_completed"]:
                    task_data.category = TaskCategory.NEW_TASK
                    return ("new", task_id)
                else:
                    task_data.category = TaskCategory.UNDERSPECIFIED
                    return ("underspec", task_id)

        elif new_task_identifier == "consistency":
            # Algorithm 2: Judge if all trials gave consistent responses
            async def judge_and_categorize(task_id: str):
                task_data = tasks[task_id]
                result = await judge_response_consistency(task_data, judge_model, semaphore)

                task_data.judge_verdict = (
                    "consistent" if result["responses_consistent"] else "inconsistent"
                )
                task_data.judge_reasoning = result["reasoning"]

                # Store the interpretation if consistent
                if result.get("interpretation_used"):
                    task_data.judge_reasoning += (
                        f"\nInterpretation: {result['interpretation_used']}"
                    )

                if result["responses_consistent"]:
                    # Consistent responses = agent has clear interpretation = new task
                    task_data.category = TaskCategory.NEW_TASK
                    return ("new", task_id)
                else:
                    # Inconsistent responses = genuine ambiguity = underspecified
                    task_data.category = TaskCategory.UNDERSPECIFIED
                    return ("underspec", task_id)

        else:
            raise ValueError(
                f"Unknown new_task_identifier: {new_task_identifier}. Use 'completion' or 'consistency'"
            )

        logger.info(f"Judging {len(needs_judging)} tasks with {judge_model}...")
        results = await async_tqdm.gather(
            *[judge_and_categorize(task_id) for task_id in needs_judging], desc="Judging tasks"
        )

        for category, task_id in results:
            if category == "new":
                new_tasks.append(task_id)
            else:
                underspecified.append(task_id)

    # Build result
    task_details = {}
    for task_id, task_data in tasks.items():
        task_details[task_id] = {
            "category": task_data.category.value if task_data.category else None,
            "original_prompt": task_data.original_prompt,
            "underspec_prompt": task_data.underspec_prompt,
            "removed_segments": [s.get("value", "") for s in task_data.removed_segments],
            "expected_failure_mode": task_data.expected_failure_mode,
            "baseline_pass_at_3": task_data.baseline_pass_at_3,
            "underspec_n_success": task_data.underspec_n_success,
            "judge_verdict": task_data.judge_verdict,
            "judge_reasoning": task_data.judge_reasoning,
        }

    # Analyze removed segment dimensions
    def get_segment_dimensions(task_ids: List[str]) -> Dict[str, int]:
        dimensions = {}
        for task_id in task_ids:
            if task_id in tasks:
                for seg in tasks[task_id].removed_segments:
                    dim = seg.get("dimension", "unknown")
                    dimensions[dim] = dimensions.get(dim, 0) + 1
        return dimensions

    summary = {
        "total_tasks": len(tasks),
        "originally_overspecified": len(originally_overspecified),
        "new_tasks": len(new_tasks),
        "underspecified": len(underspecified),
        "judge_model": judge_model,
        "new_task_identifier": new_task_identifier,
        "segment_analysis": {
            "originally_overspecified": get_segment_dimensions(originally_overspecified),
            "new_tasks": get_segment_dimensions(new_tasks),
            "underspecified": get_segment_dimensions(underspecified),
        },
    }

    return CategorizationResult(
        originally_overspecified=originally_overspecified,
        new_tasks=new_tasks,
        underspecified=underspecified,
        task_details=task_details,
        summary=summary,
    )


def main():
    parser = argparse.ArgumentParser(description="Categorize underspecified tasks")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the underspec dataset directory",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Name of the run directory within dataset-dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="categorization_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.getenv("LLM_EVAL_MODEL", "gemini/gemini-3-flash-preview"),
        help="Model to use for judging task completion",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent LLM requests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM judging, just categorize based on pass@3",
    )
    parser.add_argument(
        "--new-task-identifier",
        type=str,
        choices=["completion", "consistency"],
        default="completion",
        help="Algorithm for identifying new tasks: "
        "'completion' judges if task was completed based on underspec prompt (default), "
        "'consistency' judges if all trials gave consistent responses",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    run_dir = Path(args.run_dir)
    full_run_dir = dataset_dir / run_dir
    if not full_run_dir.exists():
        logger.error(f"Run directory not found: {full_run_dir}")
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run mode - skipping LLM judging")
        # Just load and report basic stats
        baseline_dir = dataset_dir / "baseline_results"

        baseline_pass_k = load_baseline_pass_k(baseline_dir)
        underspec_eval = load_underspec_evaluation(full_run_dir)

        overspec = 0
        needs_judge = 0

        for task_id, eval_data in underspec_eval.items():
            n_success = eval_data.get("n_success", 0)
            n_trials = eval_data.get("n_trials", 3)
            underspec_success_rate = n_success / n_trials if n_trials > 0 else 0.0
            baseline_success_rate = baseline_pass_k.get(task_id, 1.0)

            # Originally overspecified = success rate is at least as high as before
            if underspec_success_rate >= baseline_success_rate:
                overspec += 1
            else:
                needs_judge += 1

        print("\n=== Dry Run Results ===")
        print(f"Total tasks: {len(underspec_eval)}")
        print(f"Originally overspecified (still pass@3): {overspec}")
        print(f"Need LLM judging (fail pass@3): {needs_judge}")
        return

    # Run categorization
    result = asyncio.run(
        categorize_tasks(
            dataset_dir=dataset_dir,
            run_dir=run_dir,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
            new_task_identifier=args.new_task_identifier,
        )
    )

    # Save results
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = dataset_dir / output_path

    with open(output_path, "w") as f:
        json.dump(
            {
                "originally_overspecified": result.originally_overspecified,
                "new_tasks": result.new_tasks,
                "underspecified": result.underspecified,
                "task_details": result.task_details,
                "summary": result.summary,
            },
            f,
            indent=2,
        )

    # Print summary
    print("\n=== Categorization Results ===")
    print(f"Total tasks: {result.summary['total_tasks']}")
    print(
        f"Originally overspecified: {result.summary['originally_overspecified']} ({100*result.summary['originally_overspecified']/result.summary['total_tasks']:.1f}%)"
    )
    print(
        f"New tasks (complete despite fail): {result.summary['new_tasks']} ({100*result.summary['new_tasks']/result.summary['total_tasks']:.1f}%)"
    )
    print(
        f"Truly underspecified: {result.summary['underspecified']} ({100*result.summary['underspecified']/result.summary['total_tasks']:.1f}%)"
    )
    print(f"\nResults saved to: {output_path}")

    # Print examples of each category
    print("\n=== Examples ===")

    if result.originally_overspecified:
        print("\n--- Originally Overspecified (example) ---")
        task_id = result.originally_overspecified[0]
        details = result.task_details[task_id]
        print(f"Task ID: {task_id}")
        print(f"Removed: {details['removed_segments']}")
        print("Still passes because task was overspecified")

    if result.new_tasks:
        print("\n--- New Task (example) ---")
        task_id = result.new_tasks[0]
        details = result.task_details[task_id]
        print(f"Task ID: {task_id}")
        print(f"Removed: {details['removed_segments']}")
        print(f"Judge reasoning: {details['judge_reasoning']}")

    if result.underspecified:
        print("\n--- Underspecified (example) ---")
        task_id = result.underspecified[0]
        details = result.task_details[task_id]
        print(f"Task ID: {task_id}")
        print(f"Removed: {details['removed_segments']}")
        print(f"Expected failure: {details['expected_failure_mode'][:200]}...")
        print(f"Judge reasoning: {details['judge_reasoning']}")


if __name__ == "__main__":
    main()
