import logging
import os
import shutil
import subprocess

# LHAW imports
import sys
import time
from subprocess import DEVNULL
from typing import List

sys.path.append("../../../")
from evaluation.common import grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402


def get_risingwave_path():
    """Return the risingwave binary path, checking multiple possible locations.

    The task brief is ambiguous:
    - It says "Navigate to /inputs/THE OWNCLOUD FOLDER" for reading the PDF
    - Then says "Install risingwave under /THE OWNCLOUD FOLDER/"

    Agents often misinterpret this and install to inputs/THE OWNCLOUD FOLDER/
    instead of the intended output/ directory. We check both locations.
    """
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    workspace_root = os.path.dirname(output_root)  # /workspace

    # Possible paths where the agent might have installed risingwave
    possible_paths = [
        # Correct location (output directory)
        os.path.join(output_root, "risingwave"),
        # Common agent mistake: interprets /THE OWNCLOUD FOLDER/ as inputs path
        os.path.join(workspace_root, "inputs", "THE OWNCLOUD FOLDER", "risingwave"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            logging.info(f"Found risingwave at: {path}")
            return path

    logging.warning(f"risingwave not found in any expected location: {possible_paths}")
    return None


def run_command(command):
    # In LHAW, command might be relative to workspace
    if not shutil.which(command[0]) and not shutil.which(command[0], path="/workspace"):
        logging.warning(f"Command '{command[0]}' not found in PATH")
        # Try absolute path if relative failed
        if not command[0].startswith("/"):
            cmd_abs = "/workspace/" + command[0]
            if shutil.which(cmd_abs):
                command[0] = cmd_abs
            else:
                return None
    try:
        result = subprocess.run(command, check=False, timeout=0.1, stdout=DEVNULL)
        if result.returncode != 0:
            logging.warning(
                f"Command '{' '.join(command)}' failed with return code {result.returncode}"
            )
            return None
        return result.stdout.strip() if result.stdout else ""
    except subprocess.TimeoutExpired:
        print("Successfully terminated after 0.1 second. No need to run indefinitely for testing")
        return True
    except Exception as e:
        print(type(e))
        logging.error(f"Error running command '{' '.join(command)}': {e}")
        return None


@grader
def grade_checkpoint1():
    try:
        # Check if binary exists
        path = get_risingwave_path()
        if path:
            # Attempt to run it briefly
            output = run_command([path])
            return output is not None
        return False
    except Exception as e:
        logging.error(f"Fail to grade checkpoint 1: {e}")
        return False


def grade_checkpoint2():
    path = get_risingwave_path()
    if not path:
        return None, False

    try:
        # Try starting with just the binary name
        p = subprocess.Popen([path, "playground"], stdout=DEVNULL, stderr=DEVNULL)
    except Exception:
        return None, False

    try:
        # Wait for 8 seconds (matching TAC)
        time.sleep(8)

        # Connect to it
        pipe = subprocess.run(
            [
                "psql",
                "-h",
                "localhost",
                "-p",
                "4566",
                "-d",
                "dev",
                "-U",
                "root",
                "-c",
                "SELECT * FROM average_exam_scores",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("---------QUERY OUTPUT---------")
        output = pipe.stdout.decode("utf-8")
        print(output)
        print("---------QUERY ERROR---------")
        print(pipe.stderr.decode("utf-8"))
    except Exception as e:
        logging.error(f"Fail to grade checkpoint 2: {e}")
        p.terminate()
        return p, False
    else:
        p.terminate()
        # Check for expected data matching TAC verification logic
        # TAC checks: "exam_id", "average_score", "total_scores", "101", "102", "87.7", "88.6", "5"
        expected_items = [
            "exam_id",
            "average_score",
            "total_scores",
            "101",
            "102",
            "87.7",
            "88.6",
            "5",
        ]
        return p, all(item in output for item in expected_items)


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)

    checkpoints.append(Checkpoint(1, int(grade_checkpoint1())))

    process, passed_checkpoint2 = grade_checkpoint2()
    checkpoints.append(Checkpoint(1, int(passed_checkpoint2)))

    if process is not None:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()

    return result
