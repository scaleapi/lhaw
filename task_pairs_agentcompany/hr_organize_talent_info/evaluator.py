import csv
import os

# LHAW imports
import sys
from typing import List

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result


def get_summary_file_content(name: str = None):
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    path = os.path.join(output_root, "summary.csv")
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                if not row:
                    continue
                if name is not None and name not in row[0]:
                    continue
                rows.append(row)
    except Exception:
        return []
    return rows


@grader
def check_exists_file():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    path = os.path.join(output_root, "summary.csv")
    return os.path.exists(path)


@grader
def check_emily_info():
    rows = get_summary_file_content(name="Emily Johnson")
    if len(rows) != 1:
        return 0
    name, education, work, skills = rows[0]
    if name != "Emily Johnson":
        return 0
    if "Berkeley" not in education:
        return 0
    if any([c not in work for c in ["Google", "Amazon"]]):
        return 0
    if any(
        [
            c not in skills
            for c in [
                "Python",
                "Java",
                "C++",
                "JavaScript",
                "React",
                "Node.js",
                "Flask",
                "Git",
                "Docker",
                "AWS",
                "MySQL",
            ]
        ]
    ):
        return 0
    return 1


@grader
def check_michael_info():
    rows = get_summary_file_content(name="Michael Davis")
    if len(rows) != 1:
        return 0
    name, education, work, skills = rows[0]
    if name != "Michael Davis":
        return 0
    if "New York University" not in education:
        return 0
    if any([c not in work for c in ["Meta", "Spotify"]]):
        return 0
    if any(
        [
            c not in skills
            for c in [
                "JavaScript",
                "Python",
                "Swift",
                "Ruby",
                "Django",
                "Angular",
                "React Native",
                "Git",
                "Firebase",
                "PostgreSQL",
                "Jenkins",
            ]
        ]
    ):
        return 0
    return 1


@grader
def check_sarah_info():
    rows = get_summary_file_content(name="Sarah Thompson")
    if len(rows) != 1:
        return 0
    name, education, work, skills = rows[0]
    if name != "Sarah Thompson":
        return 0
    if "University of Texas at Austin" not in education:
        return 0
    if any([c not in work for c in ["Microsoft", "Oracle"]]):
        return 0
    if any(
        [
            c not in skills
            for c in [
                "Python",
                "Java",
                "C#",
                "HTML/CSS",
                "Angular",
                "Spring Boot",
                "Express.js",
                "Git",
                "Docker",
                "MongoDB",
                "Kubernetes",
            ]
        ]
    ):
        return 0
    return 1


def grade_checkpoints(trajectory=""):
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(check_exists_file())))
    checkpoints.append(Checkpoint(1, check_emily_info()))
    checkpoints.append(Checkpoint(1, check_michael_info()))
    checkpoints.append(Checkpoint(1, check_sarah_info()))
    return result
