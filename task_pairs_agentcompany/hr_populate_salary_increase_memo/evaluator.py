import logging
import os

# Import from LHAW evaluation system
import sys
import zipfile
from typing import Dict, List

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result

logging.basicConfig(level=logging.INFO)


def _paths():
    workspace_dir = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")

    def _first_existing(paths, predicate):
        return next((p for p in paths if predicate(p)), None)

    # Because the output directory is /workspace and the output location is unclear (e.g. task instruction says to output in /Documents/Human Resources Team),
    # we will check both variants.
    roots = [
        workspace_dir,
        os.path.join(
            workspace_dir,
            "Documents",
            "Human Resources Team",
        ),
        os.path.join(
            workspace_dir,
            "THE OWNCLOUD FOLDER",
            "Documents",
            "Human Resources Team",
        ),
    ]

    template_candidates = [os.path.join(root, "Salary_Increase_MEMO.odt") for root in roots]
    memos_candidates = [os.path.join(root, "salary_increase_notice") for root in roots]

    template_path = _first_existing(template_candidates, os.path.isfile)
    memos_dir = _first_existing(memos_candidates, os.path.isdir)
    return template_path, memos_dir


def _parse_salary_file(path: str) -> List[Dict[str, str]]:
    employees: List[Dict[str, str]] = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Expect lines like: Name: John Doe; Email: john@example.com; New Salary: 93500; Start: 2024-07-01; End: 2025-06-30
                parts = [p.strip() for p in line.split(";") if p.strip()]
                record: Dict[str, str] = {}
                for p in parts:
                    if ":" in p:
                        k, v = p.split(":", 1)
                        record[k.strip().lower()] = v.strip()
                if record:
                    employees.append(record)
        return employees
    except Exception:
        return []


def _list_odt_files(memos_dir: str) -> List[str]:
    if not os.path.isdir(memos_dir):
        return []
    return [os.path.join(memos_dir, f) for f in os.listdir(memos_dir) if f.lower().endswith(".odt")]


def _odt_contains_strings(odt_path: str, required_tokens: List[str]) -> bool:
    try:
        with zipfile.ZipFile(odt_path, "r") as zf:
            with zf.open("content.xml") as content_file:
                xml_bytes = content_file.read()
                text = xml_bytes.decode("utf-8", errors="ignore").lower()
                lower_tokens = [t.lower() for t in required_tokens]
                return all(token in text for token in lower_tokens)
    except Exception:
        return False


def _read_odt_text_lower(odt_path: str) -> str:
    try:
        with zipfile.ZipFile(odt_path, "r") as zf:
            with zf.open("content.xml") as content_file:
                xml_bytes = content_file.read()
                return xml_bytes.decode("utf-8", errors="ignore").lower()
    except Exception:
        return ""


@grader
def grade_checkpoint1(trajectory: str) -> bool:
    # Mirror TAC: look for encoded directory param in the trajectory
    urls = [
        "dir=/Documents/Human%20Resources%20Team",
        "/Documents/Human Resources Team",
    ]
    return any(url in trajectory for url in urls)


@grader
def grade_checkpoint2() -> float:
    # Mirror TAC: (up to 2 points)
    # 1) ODT template exists (1 point)
    # 2) Keyword coverage in the ODT content (0..1 point)
    template_path, _ = _paths()
    score: float = 0.0
    if not template_path or not os.path.isfile(template_path):
        return 0.0
    score += 1.0
    text = _read_odt_text_lower(template_path)
    keywords = [
        "name",
        "email",
        "salary amount",
        "assignment start and end date",
    ]
    found = [kw for kw in keywords if kw in text]
    keyword_score = len(found) / len(keywords) if keywords else 0.0
    score += keyword_score
    return score


@grader
def grade_checkpoint3() -> float:
    # Mirror TAC: (up to 4 points)
    # 1) salary_increase_notice directory exists (1 point)
    # 2) + up to 3 points based on file count: min(int(file_count/10 * 3), 3)
    _, memos_dir = _paths()
    if not memos_dir or not os.path.isdir(memos_dir):
        return 0.0
    score: float = 1.0
    try:
        file_count = len(
            [f for f in os.listdir(memos_dir) if os.path.isfile(os.path.join(memos_dir, f))]
        )
    except Exception:
        file_count = 0
    bonus = min(int((file_count / 10) * 3), 3)
    score += bonus
    return score


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(trajectory))))
    checkpoints.append(Checkpoint(2, int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(4, int(grade_checkpoint3())))
    return result
