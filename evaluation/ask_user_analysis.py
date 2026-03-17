"""
Core extraction and judging logic for ask_user analysis.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader
from synthetic.llm import get_openai_client
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from .ask_user_types import (
    FLAG_CATEGORIES,
    AskUserAnalysis,
    AskUserCall,
    AskUserJudgment,
    AskUserOutcome,
    AskUserSequence,
)

logger = logging.getLogger(__name__)

# Template directory
TEMPLATE_DIR = Path(__file__).parent


def extract_ask_user_calls(
    conv_json: str,
    task_id: str,
    variant_id: str,
    trial_index: int,
    model: str,
) -> AskUserSequence:
    """Extract all ask_user calls from raw_conversation_history JSON.

    Args:
        conv_json: JSON string of the conversation history
        task_id: Task identifier
        variant_id: Variant identifier
        trial_index: Trial number
        model: Model name

    Returns:
        AskUserSequence containing all ask_user calls
    """
    calls = []

    if not conv_json or conv_json == "nan":
        return AskUserSequence(
            task_id=task_id,
            variant_id=variant_id,
            trial_index=trial_index,
            model=model,
            calls=calls,
        )

    try:
        conversation = json.loads(conv_json)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse conversation JSON: {e}")
        return AskUserSequence(
            task_id=task_id,
            variant_id=variant_id,
            trial_index=trial_index,
            model=model,
            calls=calls,
        )

    if not isinstance(conversation, list):
        return AskUserSequence(
            task_id=task_id,
            variant_id=variant_id,
            trial_index=trial_index,
            model=model,
            calls=calls,
        )

    # Build a map of tool_call_id -> response content
    tool_responses: Dict[str, str] = {}
    for msg in conversation:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            tool_responses[msg["tool_call_id"]] = msg.get("content", "")

    # Find all ask_user tool calls
    call_index = 0
    for msg_index, msg in enumerate(conversation):
        if msg.get("role") != "assistant":
            continue

        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            continue

        for tc in tool_calls:
            func = tc.get("function", {})
            if func.get("name") != "ask_user":
                continue

            tool_call_id = tc.get("id", "")

            # Parse the question from arguments
            try:
                args = json.loads(func.get("arguments", "{}"))
                question_text = args.get("question", "")
            except json.JSONDecodeError:
                question_text = func.get("arguments", "")

            # Get the response
            user_response = tool_responses.get(tool_call_id)

            # Determine outcome
            if user_response is None:
                outcome = AskUserOutcome.UNKNOWN
            elif user_response.strip() == "":
                outcome = AskUserOutcome.REFUSED
            else:
                outcome = AskUserOutcome.ANSWERED

            calls.append(
                AskUserCall(
                    call_index=call_index,
                    tool_call_id=tool_call_id,
                    question_text=question_text,
                    user_response=user_response,
                    outcome=outcome,
                    message_index=msg_index,
                )
            )
            call_index += 1

    return AskUserSequence(
        task_id=task_id,
        variant_id=variant_id,
        trial_index=trial_index,
        model=model,
        calls=calls,
    )


def render_judge_prompt(
    calls: List[AskUserCall],
    removed_segments: List[str],
    original_prompt: str,
    underspec_prompt: str,
    coverage_score: float,
) -> str:
    """Render the judge prompt template.

    Args:
        calls: List of AskUserCall objects
        removed_segments: List of removed segment strings
        original_prompt: Original fully-specified prompt
        underspec_prompt: Underspecified prompt given to agent
        coverage_score: Coverage score achieved

    Returns:
        Rendered prompt string
    """
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("ask_user_judge_template.j2")

    return template.render(
        calls=calls,
        removed_segments=removed_segments,
        original_prompt=original_prompt,
        underspec_prompt=underspec_prompt,
        coverage_score=coverage_score,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=60, exp_base=2),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call_judge_llm(model: str, messages: list) -> str:
    """Call the judge LLM with retries."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        logger.warning("Empty response from judge LLM")
        return "{}"
    return content


async def judge_ask_user_behavior(
    sequence: AskUserSequence,
    removed_segments: List[str],
    original_prompt: str,
    underspec_prompt: str,
    coverage_score: float,
    judge_model: str = "gpt-4o",
) -> Optional[AskUserJudgment]:
    """Use LLM to evaluate ask_user behavior against 21 flags.

    Args:
        sequence: AskUserSequence with extracted calls
        removed_segments: List of removed segment strings
        original_prompt: Original fully-specified prompt
        underspec_prompt: Underspecified prompt given to agent
        coverage_score: Coverage score achieved
        judge_model: Model to use for judging

    Returns:
        AskUserJudgment or None if judging failed
    """
    prompt = render_judge_prompt(
        calls=sequence.calls,
        removed_segments=removed_segments,
        original_prompt=original_prompt,
        underspec_prompt=underspec_prompt,
        coverage_score=coverage_score,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        content = await asyncio.to_thread(_call_judge_llm, judge_model, messages)

        # Try to extract JSON from markdown code blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()

        data = json.loads(content)

        # Always recompute failure_categories from flags to ensure consistent naming
        failure_cats = []
        for cat_name, flags in FLAG_CATEGORIES.items():
            if any(data.get(f, False) for f in flags):
                failure_cats.append(cat_name)
        data["failure_categories"] = failure_cats

        return AskUserJudgment.from_dict(data)

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse judge response: {e}")
        logger.debug(f"Raw response: {content[:500] if content else 'None'}...")
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")

    return None


def parse_removed_value(removed_value: Any) -> List[str]:
    """Parse removed_value field into a list of strings.

    Args:
        removed_value: Could be a JSON string, list, or plain string

    Returns:
        List of removed segment strings
    """
    if removed_value is None or (isinstance(removed_value, float) and str(removed_value) == "nan"):
        return []

    if isinstance(removed_value, list):
        return [str(v) for v in removed_value]

    if isinstance(removed_value, str):
        # Try parsing as JSON
        try:
            parsed = json.loads(removed_value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            return [removed_value]

    return [str(removed_value)]


async def analyze_row(
    row: Dict[str, Any],
    trial_index: int,
    judge_model: Optional[str] = None,
    skip_judging: bool = False,
) -> AskUserAnalysis:
    """Analyze a single row from a scored CSV.

    Args:
        row: Dictionary with CSV row data
        trial_index: Trial index
        judge_model: Model for judging (if not skipping)
        skip_judging: Whether to skip LLM judging

    Returns:
        AskUserAnalysis object
    """
    task_id = row.get("TASK", "")
    variant_id = row.get("variant_id", "")
    model = row.get("model", "")
    conv_json = row.get("raw_conversation_history", "")

    # Extract ask_user calls
    sequence = extract_ask_user_calls(
        conv_json=str(conv_json) if conv_json else "",
        task_id=task_id,
        variant_id=variant_id,
        trial_index=trial_index,
        model=model,
    )

    # Get metadata
    coverage_score = row.get("coverage_score")
    if coverage_score is not None:
        try:
            coverage_score = float(coverage_score)
        except (ValueError, TypeError):
            coverage_score = None

    ambiguity_class = row.get("ambiguity_class")
    removed_value = row.get("removed_value")
    dimension = row.get("dimension")

    # Judge if requested and there are calls
    judgment = None
    if not skip_judging and judge_model and sequence.count > 0:
        removed_segments = parse_removed_value(removed_value)
        original_prompt = str(row.get("original_prompt", ""))
        underspec_prompt = str(row.get("PROMPT", ""))

        judgment = await judge_ask_user_behavior(
            sequence=sequence,
            removed_segments=removed_segments,
            original_prompt=original_prompt,
            underspec_prompt=underspec_prompt,
            coverage_score=coverage_score or 0.0,
            judge_model=judge_model,
        )

    return AskUserAnalysis(
        sequence=sequence,
        judgment=judgment,
        coverage_score=coverage_score,
        ambiguity_class=ambiguity_class,
        removed_value=removed_value,
        dimension=dimension,
    )


async def analyze_scored_csv(
    csv_path: Path,
    trial_index: int,
    judge_model: Optional[str] = None,
    concurrency: int = 5,
    skip_judging: bool = False,
    skip_no_calls: bool = False,
    progress_bar: Optional[Any] = None,
) -> List[AskUserAnalysis]:
    """Analyze all rows in a scored CSV.

    Args:
        csv_path: Path to the scored CSV
        trial_index: Trial index for this CSV
        judge_model: Model for judging
        concurrency: Max concurrent judge calls
        skip_judging: Skip LLM judging entirely
        skip_no_calls: Skip rows with no ask_user calls
        progress_bar: Optional tqdm progress bar to update

    Returns:
        List of AskUserAnalysis objects
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def analyze_with_limit(row: Dict) -> Optional[AskUserAnalysis]:
        async with semaphore:
            analysis = await analyze_row(
                row=row,
                trial_index=trial_index,
                judge_model=judge_model,
                skip_judging=skip_judging,
            )
            if progress_bar is not None:
                progress_bar.update(1)
            if skip_no_calls and analysis.sequence.count == 0:
                return None
            return analysis

    tasks = [analyze_with_limit(row) for row in rows]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]
