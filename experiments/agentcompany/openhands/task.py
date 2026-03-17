import json
import os
import queue
import shutil
import subprocess
import sys
from datetime import datetime

from utils import (
    download_task_data_from_s3,
    extract_llm_metrics,
    prepare_workspace,
    read_brief,
    upload_task_results_to_s3,
)


# Simple thread-safe counter class
class Counter:
    """Simple thread-safe counter class for tracking progress"""

    def __init__(self, initial_value=0):
        self.value = initial_value


def run_openhands_task(
    task_id,
    llm_model,
    base_url,
    api_key,
    results,
    progress_lock,
    completed,
    total,
    max_iterations,
    max_budget_per_task,
    use_s3=False,
    s3_bucket_name="",
    s3_prefix="",
    aws_profile=None,
    download_only=False,
    upload_only=False,
    local_tasks_dir=None,
    local_results_dir=None,
    temperature=None,
    top_p=None,
    max_output_tokens=None,
    task_context=None,
    model_config=None,
    include_user_help=False,
    user_simulator_model=None,
    agent_strategy=None,
):
    """Run an OpenHands task and store the result in the results dictionary."""
    start_time = datetime.now()

    with progress_lock:
        print(
            f"[{start_time.strftime('%H:%M:%S')}] Starting task {task_id} with model {llm_model}... ({completed.value}/{total} completed)"
        )
        if temperature is not None:
            print(f"Temperature: {temperature}")
        if top_p is not None:
            print(f"Top_p: {top_p}")
        if max_output_tokens is not None:
            print(f"Max output tokens: {max_output_tokens}")

    # Handle local tasks directory or S3 download
    if local_tasks_dir:
        # Setup task data from local directory
        from utils import setup_local_task_data

        local_task_setup_success = setup_local_task_data(task_id, local_tasks_dir)

        if not local_task_setup_success:
            error_msg = f"Failed to setup task data from local directory for task {task_id}"
            print(error_msg)
            results[f"{task_id}_{llm_model}"] = {
                "task_id": task_id,
                "model": llm_model,
                "success": False,
                "error": error_msg,
                "start_time": start_time,
                "end_time": datetime.now(),
                "duration": (datetime.now() - start_time).total_seconds(),
            }

            with progress_lock:
                completed.value += 1
                print(
                    f"Progress: {completed.value}/{total} tasks completed ({completed.value/total*100:.1f}%)"
                )

            return
    elif use_s3 and s3_bucket_name and not upload_only:
        # If S3 download is enabled and not in upload-only mode, download task data from S3
        data_download_success = download_task_data_from_s3(
            task_id, s3_bucket_name, s3_prefix, aws_profile=aws_profile
        )

        if not data_download_success:
            error_msg = f"Failed to download task data from S3 for task {task_id}"
            print(error_msg)
            results[f"{task_id}_{llm_model}"] = {
                "task_id": task_id,
                "model": llm_model,
                "success": False,
                "error": error_msg,
                "start_time": start_time,
                "end_time": datetime.now(),
                "duration": (datetime.now() - start_time).total_seconds(),
            }

            with progress_lock:
                completed.value += 1
                print(
                    f"Progress: {completed.value}/{total} tasks completed ({completed.value/total*100:.1f}%)"
                )

            return

    # Read the brief from the task directory
    task_prompt = read_brief(task_id)

    # Check if brief reading resulted in an error (error messages start with "ERROR:")
    if task_prompt.startswith("ERROR:"):
        error_msg = task_prompt
        print(f"Brief error: {error_msg}")
        results[f"{task_id}_{llm_model}"] = {
            "task_id": task_id,
            "model": llm_model,
            "success": False,
            "error": error_msg,
            "start_time": start_time,
            "end_time": datetime.now(),
            "duration": (datetime.now() - start_time).total_seconds(),
        }

        with progress_lock:
            completed.value += 1
            print(
                f"Progress: {completed.value}/{total} tasks completed ({completed.value/total*100:.1f}%)"
            )

        return

    # Prepare workspace with input files, including model name in the path
    workspace_dir, state_dir, has_input_files, timestamp, task_definition_dir = prepare_workspace(
        task_id, llm_model
    )

    # Get the model directory (parent of workspace_dir)
    model_dir = os.path.dirname(workspace_dir)

    # Setup MCP tools in workspace (only when user help is enabled to avoid
    # leaking experiment structure hints to the agent during baseline runs)
    mcp_tools_src = os.path.join(os.path.dirname(__file__), "mcp_tools")
    mcp_dest_dir = os.path.join(workspace_dir, ".openhands")

    if include_user_help and os.path.exists(mcp_tools_src):
        if os.path.exists(mcp_dest_dir):
            shutil.rmtree(mcp_dest_dir)

        shutil.copytree(mcp_tools_src, mcp_dest_dir)

        with progress_lock:
            print(f"Copied MCP tools directory to workspace: {mcp_dest_dir}")

    # Copy agent strategy to .openhands_instructions for OpenHands skill system
    # OpenHands reads .openhands_instructions and injects it as repo_instructions
    # This is the documented mechanism for project-specific agent customization
    # See: https://docs.openhands.dev/usage/prompting#workspace-context
    if agent_strategy and agent_strategy != "none":
        strategies_dir = os.path.join(os.path.dirname(__file__), "agent_strategies")
        strategy_file = os.path.join(strategies_dir, f"{agent_strategy}.md")

        if os.path.exists(strategy_file):
            # Copy strategy as .openhands_instructions at workspace root
            instructions_dest = os.path.join(workspace_dir, ".openhands_instructions")
            shutil.copy2(strategy_file, instructions_dest)

            with progress_lock:
                print(f"Installed agent strategy '{agent_strategy}' as OpenHands skill:")
                print(f"  - {instructions_dest}")

    # Add task context to the prompt
    if task_context and "inputs_note" in task_context:
        task_prompt += f"\n\n{task_context['inputs_note']}"

    if task_context and "output_note" in task_context:
        task_prompt += f"\n\n{task_context['output_note']}"

    # Get user ID
    user_id = subprocess.check_output(["id", "-u"]).decode().strip()

    # Generate unique container name that includes model info
    safe_model = llm_model.replace("/", "-").replace("\\", "-")
    container_name = f"openhands-task-{task_id}-{safe_model}-{timestamp}"

    # OpenHands version - configurable via env var for easy updates
    # Supports: Claude 4.5, GPT-5.x, Gemini 3, o3-pro, etc.
    # NOTE: v1.x uses new registry (docker.openhands.dev), v0.x used old (docker.all-hands.dev)
    openhands_version = os.environ.get("OPENHANDS_VERSION", "1.1")
    image_url = f"docker.openhands.dev/openhands/openhands:{openhands_version}"
    runtime_image = f"docker.openhands.dev/openhands/runtime:{openhands_version}-nikolaik"

    # Generate a dynamic config.toml with environment variables for MCP server
    # Build the config template, conditionally including MCP server
    config_template = f"""# OpenHands Configuration

[agent]
# Enable MCP tools
enable_mcp = {str(include_user_help).lower()}

# Enable prompt extensions (microagents, repo/runtime info)
# This allows OpenHands to load skills from .openhands/microagents/ and .openhands_instructions
enable_prompt_extensions = true

# Keep all standard tools enabled
enable_cmd = true
enable_think = true
enable_finish = true
enable_browsing = true
enable_jupyter = true
enable_editor = true
enable_llm_editor = false
"""

    # Only add MCP server configuration if user help is enabled
    if include_user_help:
        # Escape TOML basic string special characters in interpolated values
        def _toml_esc(s):
            if not s:
                return ""
            return s.replace("\\", "\\\\").replace('"', '\\"')

        _api_key = _toml_esc(api_key or "")
        _base_url = _toml_esc(base_url or "")
        _usim_model = _toml_esc(
            user_simulator_model
            or os.environ.get("USER_SIMULATOR_MODEL", "openai/gpt-4.1-2025-04-14")
        )

        # Load and base64-encode the primary task definition so it can be
        # passed to the MCP server via env var instead of a mounted volume.
        # This prevents the agent from reading the full task spec directly.
        import base64

        _primary_task_b64 = ""
        primary_task_path = os.path.join(task_definition_dir, "primary_task.json")
        if os.path.exists(primary_task_path):
            with open(primary_task_path, "r", encoding="utf-8") as f:
                _primary_task_b64 = base64.b64encode(f.read().encode("utf-8")).decode("ascii")

        config_template += f"""
[mcp]
# Configure stdio MCP servers (run as subprocesses inside the runtime container)
[[mcp.stdio_servers]]
name = "outside-user"
command = "python3"
args = ["/workspace/.openhands/mcp_server.py"]
# Pass environment variables for LLM API calls and task definition.
# These env vars are only visible to the MCP server subprocess, not the agent.
[mcp.stdio_servers.env]
LLM_API_KEY = "{_api_key}"
LLM_BASE_URL = "{_base_url}"
USER_SIMULATOR_MODEL = "{_usim_model}"
PRIMARY_TASK_B64 = "{_primary_task_b64}"
"""

    # Write dynamic config to a temp file in the model directory
    config_file_path = os.path.join(model_dir, "config.toml")
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.write(config_template)

    # Build docker command with host networking
    docker_cmd = [
        "docker",
        "run",
        # "--net=host",  # Use host networking to avoid port binding issues
        "--pull=missing",
        "-e",
        f"SANDBOX_RUNTIME_CONTAINER_IMAGE={runtime_image}",
        "-e",
        f"SANDBOX_USER_ID={user_id}",
        "-e",
        f"SANDBOX_VOLUMES={workspace_dir}:/workspace",
        "-e",
        f"LLM_API_KEY={api_key}",
        "-e",
        f"LLM_MODEL={llm_model}",
        "-e",
        f"LLM_BASE_URL={base_url}",
        "-e",
        "LOG_ALL_EVENTS=true",
        "-v",
        f"{state_dir}:/.openhands:rw,Z",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "-v",
        f"{config_file_path}:/app/config.toml:ro",  # Mount config.toml
        "--add-host",
        "host.docker.internal:host-gateway",
        "--name",
        container_name,
        image_url,
        "python",
        "-m",
        "openhands.core.main",
        "--config-file",
        "/app/config.toml",  # Use the mounted config
        "-t",
        task_prompt,
        "-i",
        str(max_iterations),
        "-b",
        str(max_budget_per_task),
    ]

    # Build optional LLM environment variables from model config
    if model_config is None:
        model_config = {}

    # Claude 4.5 models don't support both temperature and top_p
    is_claude_45 = "claude-sonnet-4-5" in llm_model or "claude-opus-4-5" in llm_model

    # Collect env vars to inject (order matters for some providers)
    llm_env_vars = {}
    if temperature is not None and not is_claude_45:
        llm_env_vars["LLM_TEMPERATURE"] = temperature
    if top_p is not None and not is_claude_45:
        llm_env_vars["LLM_TOP_P"] = top_p
    if max_output_tokens is not None:
        llm_env_vars["LLM_MAX_OUTPUT_TOKENS"] = max_output_tokens
    if model_config.get("timeout"):
        llm_env_vars["LLM_TIMEOUT"] = model_config["timeout"]
    llm_env_vars["LLM_NUM_RETRIES"] = model_config.get("num_retries", 3)

    # Insert env flags BEFORE the image name (so docker run sees them)
    if llm_env_vars:
        try:
            insert_at = docker_cmd.index(image_url)
        except ValueError:
            insert_at = len(docker_cmd)
        for key, value in llm_env_vars.items():
            docker_cmd[insert_at:insert_at] = ["-e", f"{key}={value}"]
            insert_at += 2

    # Execute docker command
    with progress_lock:
        print(f"Starting task {task_id} with model {llm_model}")
        print(f"Workspace: {workspace_dir}")
        print(f"State directory: {state_dir}")
        print(
            f"Brief: {task_prompt[:100]}..." if len(task_prompt) > 100 else f"Brief: {task_prompt}"
        )
        print(f"Input files available: {'Yes' if has_input_files else 'No'}")

    # Task timeout: 30 minutes per task (rate-limited agents will be killed)
    TASK_TIMEOUT_SECONDS = int(os.environ.get("TASK_TIMEOUT_SECONDS", 30 * 60))

    try:
        process = subprocess.run(
            docker_cmd, capture_output=True, text=True, timeout=TASK_TIMEOUT_SECONDS
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Extract LLM metrics from the trajectory files
        llm_metrics = extract_llm_metrics(state_dir, task_id, llm_model)

        # Create a run report JSON
        run_report = {
            "task_id": task_id,
            "model": llm_model,
            "timestamp": timestamp,
            "success": process.returncode == 0,
            "error": (
                None
                if process.returncode == 0
                else f"Docker process failed with return code {process.returncode}"
            ),
            "stderr": (
                process.stderr[-500:]
                if process.returncode != 0 and len(process.stderr) > 500
                else process.stderr if process.returncode != 0 else None
            ),
            "workspace": workspace_dir,
            "state_dir": state_dir,
            "has_input_files": has_input_files,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": duration,
            "accumulated_cost": llm_metrics.get("accumulated_cost", 0.0),
            "accumulated_token_usage": llm_metrics.get("accumulated_token_usage", 0),
            "metrics_found": llm_metrics.get("metrics_found", False),
        }

        # Save the run report to the model directory
        run_report_path = os.path.join(model_dir, "run_report.json")
        with open(run_report_path, "w") as f:
            json.dump(run_report, f, indent=2)

        print(f"Saved run report to {run_report_path}")

        # Update run_info with completion data
        run_info_path = os.path.join(model_dir, "run_info.json")
        try:
            with open(run_info_path, "r") as f:
                run_info = json.load(f)

            run_info.update(
                {
                    "end_time": datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "exit_code": process.returncode,
                    "accumulated_cost": llm_metrics.get("accumulated_cost", 0.0),
                    "accumulated_token_usage": llm_metrics.get("accumulated_token_usage", 0),
                }
            )

            with open(run_info_path, "w") as f:
                json.dump(run_info, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not update run info: {str(e)}")

        # Check return code for success/failure
        if process.returncode == 0:
            with progress_lock:
                print(
                    f"[{end_time.strftime('%H:%M:%S')}] Task {task_id} with model {llm_model} completed successfully in {duration:.1f} seconds"
                )

            # Check if anything was saved to the output directory
            output_dir = os.path.join(workspace_dir, "output")
            output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []

            results[f"{task_id}_{llm_model}_{timestamp}"] = {
                "task_id": task_id,
                "model": llm_model,
                "timestamp": timestamp,
                "success": True,
                "workspace": workspace_dir,
                "state_dir": state_dir,
                "has_input_files": has_input_files,
                "output_files": len(output_files),
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "accumulated_cost": llm_metrics.get("accumulated_cost", 0.0),
                "accumulated_token_usage": llm_metrics.get("accumulated_token_usage", 0),
                "metrics_found": llm_metrics.get("metrics_found", False),
            }

            # Handle local results directory or S3 upload
            if local_results_dir:
                with progress_lock:
                    print(
                        f"[{end_time.strftime('%H:%M:%S')}] Copying task {task_id} results to local directory..."
                    )

                from utils import copy_results_to_local_dir

                copy_success = copy_results_to_local_dir(
                    task_id, llm_model, timestamp, workspace_dir, state_dir, local_results_dir
                )

                # Add copy status to results
                results[f"{task_id}_{llm_model}_{timestamp}"]["local_copy"] = copy_success

                if copy_success:
                    with progress_lock:
                        print(
                            f"[{end_time.strftime('%H:%M:%S')}] Task {task_id} results copied to local directory successfully"
                        )
                else:
                    with progress_lock:
                        print(
                            f"[{end_time.strftime('%H:%M:%S')}] Failed to copy task {task_id} results to local directory"
                        )
            elif use_s3 and s3_bucket_name and not download_only:
                # If S3 upload is enabled and not in download-only mode, upload results to S3
                with progress_lock:
                    print(
                        f"[{end_time.strftime('%H:%M:%S')}] Uploading task {task_id} results to S3..."
                    )

                upload_success = upload_task_results_to_s3(
                    task_id,
                    llm_model,
                    timestamp,
                    workspace_dir,
                    state_dir,
                    s3_bucket_name,
                    s3_prefix,
                    aws_profile,
                )

                # Add upload status to results
                results[f"{task_id}_{llm_model}_{timestamp}"]["s3_upload"] = upload_success

                if upload_success:
                    with progress_lock:
                        print(
                            f"[{end_time.strftime('%H:%M:%S')}] Task {task_id} results uploaded to S3 successfully"
                        )
                else:
                    with progress_lock:
                        print(
                            f"[{end_time.strftime('%H:%M:%S')}] Failed to upload task {task_id} results to S3"
                        )

        else:
            with progress_lock:
                print(
                    f"[{end_time.strftime('%H:%M:%S')}] Task {task_id} with model {llm_model} failed with return code {process.returncode} after {duration:.1f} seconds"
                )

            results[f"{task_id}_{llm_model}_{timestamp}"] = {
                "task_id": task_id,
                "model": llm_model,
                "timestamp": timestamp,
                "success": False,
                "error": f"Docker process failed with return code {process.returncode}",
                "stderr": process.stderr[-500:] if len(process.stderr) > 500 else process.stderr,
                "workspace": workspace_dir,
                "state_dir": state_dir,
                "has_input_files": has_input_files,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "accumulated_cost": llm_metrics.get("accumulated_cost", 0.0),
                "accumulated_token_usage": llm_metrics.get("accumulated_token_usage", 0),
                "metrics_found": llm_metrics.get("metrics_found", False),
            }
    except subprocess.TimeoutExpired:
        # Task timed out - kill container and record failure
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        error_msg = f"Task timed out after {TASK_TIMEOUT_SECONDS}s"

        # Kill the container
        try:
            subprocess.run(["docker", "stop", container_name], capture_output=True, timeout=30)
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=30)
        except Exception:
            print(f"Warning: failed to stop/remove container {container_name}", file=sys.stderr)

        with progress_lock:
            print(
                f"[{end_time.strftime('%H:%M:%S')}] Task {task_id} with model {llm_model} TIMED OUT after {duration:.1f} seconds"
            )

        results[f"{task_id}_{llm_model}_{timestamp}"] = {
            "task_id": task_id,
            "model": llm_model,
            "timestamp": timestamp,
            "success": False,
            "error": error_msg,
            "workspace": workspace_dir,
            "state_dir": state_dir,
            "has_input_files": has_input_files,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "accumulated_cost": 0.0,
            "accumulated_token_usage": 0,
            "metrics_found": False,
        }
    except Exception as e:
        error_msg = str(e)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        with progress_lock:
            print(
                f"[{end_time.strftime('%H:%M:%S')}] Error in task {task_id} with model {llm_model}: {error_msg} after {duration:.1f} seconds"
            )

        results[f"{task_id}_{llm_model}_{timestamp}"] = {
            "task_id": task_id,
            "model": llm_model,
            "timestamp": timestamp,
            "success": False,
            "error": error_msg,
            "workspace": workspace_dir,
            "state_dir": state_dir,
            "has_input_files": has_input_files,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "accumulated_cost": 0.0,
            "accumulated_token_usage": 0,
            "metrics_found": False,
        }
    finally:
        try:
            if os.path.exists(config_file_path):
                os.remove(config_file_path)
            subprocess.run(["chmod", "-R", "u+w", workspace_dir], capture_output=True, check=False)
        except Exception:
            print("Warning: cleanup failed for workspace", file=sys.stderr)

    # Update progress counter
    with progress_lock:
        completed.value += 1
        print(
            f"Progress: {completed.value}/{total} tasks completed ({completed.value/total*100:.1f}%)"
        )


def worker(
    task_queue,
    llm_model,
    base_url,
    api_key,
    results,
    progress_lock,
    completed,
    total,
    max_iterations,
    max_budget_per_task,
    use_s3=False,
    s3_bucket_name="",
    s3_prefix="",
    aws_profile=None,
    download_only=False,
    upload_only=False,
    local_tasks_dir=None,
    local_results_dir=None,
    temperature=None,
    top_p=None,
    max_output_tokens=None,
    task_context=None,
    model_config=None,
    include_user_help=False,
    user_simulator_model=None,
    agent_strategy=None,
):
    """Worker function that processes tasks from the queue."""
    while not task_queue.empty():
        found_item = False
        try:
            task_id = task_queue.get(block=False)
            found_item = True
            run_openhands_task(
                task_id,
                llm_model,
                base_url,
                api_key,
                results,
                progress_lock,
                completed,
                total,
                max_iterations,
                max_budget_per_task,
                use_s3,
                s3_bucket_name,
                s3_prefix,
                aws_profile,
                download_only,
                upload_only,
                local_tasks_dir,
                local_results_dir,
                temperature,
                top_p,
                max_output_tokens,
                task_context,
                model_config,
                include_user_help,
                user_simulator_model,
                agent_strategy,
            )
        except queue.Empty:
            break
        finally:
            if found_item:
                task_queue.task_done()
