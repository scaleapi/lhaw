"""
LHAW Constants
==============

Centralized defaults for all scripts. Import these instead of hardcoding values.

Usage:
    from constants import DEFAULTS, MODELS

    parser.add_argument("--num_trials", type=int, default=DEFAULTS["num_trials"])
"""

import os

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
# Canonical values used across all entrypoints. Change here to affect everything.

DEFAULTS = {
    # Trials
    "num_trials": 3,  # Number of rollouts per task/variant (for pass@k)
    # Parallelism
    "parallel_variants": 1,  # Sequential by default to avoid OOM; override with PARALLEL_VARIANTS=N
    # Agent limits
    "max_iterations": 100,  # Max agent steps per task
    # Underspec generation
    "max_level": 2,  # Segments to remove together (1=single, 2=pairs)
    "max_variants": None,  # None = no limit (use top_k_per_level in pipeline)
    "severity": "delete",  # Removal strategy: delete, vaguify, genericize
    # Default model for agent evaluation
    "model": "opus_4_5",
    # Docker cleanup
    "cleanup_interval": 10,  # Prune containers every N completed variants
    # Benchmark filtering
    "max_total": 100,  # Max total samples in filtered benchmark
    "max_per_task_tac": 10,  # Max samples per original task (TAC)
    "max_per_task_swebench": 8,  # Max samples per original instance (SWE-bench)
    "seed": 42,  # Random seed for reproducibility
    # Container timeouts (SWE-bench)
    "startup_timeout": 600,  # Container startup timeout in seconds
    "runtime_timeout": 900,  # Container runtime timeout in seconds
    # Parallelism
    "concurrency": 10,  # Parallel completions (MCP-Atlas, SWE-bench)
}

# =============================================================================
# MODEL SHORTCUTS
# =============================================================================
# Short names → full model identifiers for LiteLLM/OpenAI API

MODELS = {
    # Standard LiteLLM model identifiers (provider/model-name).
    # When LLM_BASE_URL is set (proxy mode), "litellm_proxy/" is auto-prepended
    # at runtime so LiteLLM routes through the proxy. For direct provider access,
    # leave LLM_BASE_URL unset and set provider-specific API keys instead
    # (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY).
    #
    # Anthropic
    "opus_4_5": "anthropic/claude-opus-4-5-20251101",
    "sonnet_4_5": "anthropic/claude-sonnet-4-5-20250929",
    "opus_4_6": "anthropic/claude-opus-4-6-20260205",
    "sonnet_4_6": "anthropic/claude-sonnet-4-6",
    "sonnet_4": "anthropic/claude-sonnet-4-20250514",
    "haiku_4_5": "anthropic/claude-haiku-4-5-20251001",
    # OpenAI
    "gpt_5_2": "openai/gpt-5.2-2025-12-11",
    "gpt_5_1": "openai/gpt-5.1-2025-11-13",
    "gpt_5": "openai/gpt-5-2025-08-07",
    "o3_pro": "openai/o3-pro-2025-06-10",
    "o3": "openai/o3-2025-04-16",
    "gpt_4_1_mini": "openai/gpt-4.1-mini",
    # Google
    "gemini_3_pro": "gemini/gemini-3-pro-preview",
    "gemini_3_flash": "gemini/gemini-3-flash-preview",
    "gemini_3_1_pro": "gemini/gemini-3.1-pro-preview",
    "gemini_3_1_flash_lite": "gemini/gemini-3.1-flash-lite-preview",
    # Other
    "kimi_k2": "fireworks_ai/kimi-k2-instruct-0905",
    "qwen3_235b": "fireworks_ai/qwen3-235b-a22b",
    "llama4_maverick": "fireworks_ai/llama4-maverick-instruct-basic",
    "glm_4p5_air": "fireworks_ai/glm-4p5-air",
    "nova_2_lite": "bedrock/global.amazon.nova-2-lite-v1:0",
}

# =============================================================================
# PATH CONSTANTS
# =============================================================================
# Relative to REPO_ROOT (computed at import time by each script)

PATHS = {
    "synthetic_outputs": "synthetic/outputs",
    "tac_tasks": "task_pairs_agentcompany",
    "runs_dir": "experiments/agentcompany/runs",
    "tac_golden_trajectories": "experiments/agentcompany/golden_trajectories",
}

# =============================================================================
# LHAW BENCHMARK TASKS
# =============================================================================
# 13 TAC tasks selected for LHAW (avg_ckpt_acc >= 0.5 across reference models)

SELECTED_TASKS = [
    "ds_answer_numerical_data_question",
    "ds_fix_table_values_and_missing_answers",
    "ds_format_excel_sheets",
    "ds_predictive_modeling",
    "ds_visualize_data_in_pie_and_bar_chart",
    "finance_budget_variance",
    "finance_check_attendance_payroll",
    "finance_expense_validation",
    "hr_check_attendance_multiple_days",
    "hr_check_attendance_one_day",
    "hr_create_employee_manual",
    "hr_new_grad_job_description",
    "sde_create_sqlite_database",
]


VARIANT_DELIMITER = "__V_"


def parse_variant_id(instance_id: str) -> tuple:
    """Parse a SWE-bench variant instance ID into (original_id, variant_suffix).

    Returns (original_id, variant_suffix) where variant_suffix is "" for originals.
    Example: "inst_foo__V_S1_delete" → ("inst_foo", "S1_delete")
             "inst_foo"              → ("inst_foo", "")
    """
    if VARIANT_DELIMITER in instance_id:
        original_id, variant_suffix = instance_id.split(VARIANT_DELIMITER, 1)
        return original_id, variant_suffix
    return instance_id, ""


def print_bash_models() -> None:
    """Print MODELS as a bash associative array declaration.

    Shell scripts source this via:
        eval "$(python3 constants.py bash_models)"
    """
    pairs = " ".join(f'["{k}"]="{v}"' for k, v in MODELS.items())
    print(f"declare -A MODELS=({pairs})")


def get_model(key: str) -> str:
    """Get full model identifier from shortcut, or return as-is if not found."""
    return MODELS.get(key, key)


def uses_litellm_proxy() -> bool:
    """Check if LLM_BASE_URL points to a LiteLLM proxy.

    Returns True when LLM_BASE_URL contains "litellm" (e.g.
    ``https://litellm-proxy.{DOMAIN}.com/v1``).  The ``litellm_proxy/``
    model prefix is only needed when OpenHands routes through such a proxy.
    """
    base_url = os.environ.get("LLM_BASE_URL", "").lower()
    return "litellm-proxy" in base_url


def resolve_model(key: str) -> str:
    """Get model identifier with litellm_proxy/ prefix when using a LiteLLM proxy.

    When LLM_BASE_URL points to a LiteLLM proxy, the model needs the
    ``litellm_proxy/`` prefix so the LiteLLM SDK routes through the proxy.
    Otherwise, returns the standard LiteLLM identifier for direct provider
    access (e.g. ``anthropic/claude-sonnet-4-6``).
    """
    model = get_model(key)
    if uses_litellm_proxy() and not model.startswith("litellm_proxy/"):
        return f"litellm_proxy/{model}"
    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "bash_models":
        print_bash_models()
    else:
        print(f"Usage: {sys.argv[0]} bash_models", file=sys.stderr)
        sys.exit(1)
