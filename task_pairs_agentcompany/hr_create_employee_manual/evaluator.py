import json
import logging
import os
import sys
from typing import List

from odf.element import Node
from odf.opendocument import load

# LHAW imports
sys.path.append("../../../")
from evaluation.common import grader, llm_complete  # noqa: E402
from evaluation.scoring import Checkpoint, Result, bonus_for_completing_any  # noqa: E402


def extract_text_from_odt(file_path):
    text_content = ""
    document = load(file_path)

    def get_text_from_node(node):
        text = ""
        for child in node.childNodes:
            if isinstance(child, Node):
                if hasattr(child, "data"):
                    text += child.data + "\n"
                text += get_text_from_node(child)
        return text

    for elem in document.text.childNodes:
        text_content += get_text_from_node(elem) + "\n"

    return text_content


def check_keywords_in_content(content, keywords):
    content = content.lower()
    found_keywords = [keyword for keyword in keywords if keyword in content]
    score = len(found_keywords) / len(keywords)
    return found_keywords, score


@grader
def grade_checkpoint1(trajectory):
    urls = ["dir=/Documents/Human%20Resources%20Team", "/Documents/Human Resources Team"]
    return any(url in trajectory for url in urls)


@grader
def grade_checkpoint2():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    file_path = os.path.join(output_root, "Employee_Manual.odt")

    if not os.path.isfile(file_path):
        logging.warning(f"Employee_Manual.odt not found at {file_path}")
        return 0

    # We give 1 point for existence
    score = 1

    try:
        text_content = extract_text_from_odt(file_path)
        if not text_content:
            return score  # Return existence score if can't parse

        keywords = [
            "company values",
            "code of conduct",
            "operational procedures",
            "employee benefits",
            "company policies",
        ]

        found_keywords, keyword_score = check_keywords_in_content(text_content, keywords)

        # Add up to 1 point based on keywords
        score += keyword_score
    except Exception as e:
        logging.warning(f"Error parsing ODT: {e}")

    return score


@grader
def grade_checkpoint3():
    # Check if manual is formatted for readability using LLM (matching TAC)
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    file_path = os.path.join(output_root, "Employee_Manual.odt")

    if not os.path.isfile(file_path):
        logging.warning(f"Employee_Manual.odt not found at {file_path}")
        return 0

    try:
        text_content = extract_text_from_odt(file_path)

        response = llm_complete(
            messages=[
                {
                    "content": f"An Employee Manual is provided below: \"\"{text_content}\"\" \n Is the manual formatted for readability? Please answer 'yes' if it does, or 'no' if it doesn't. ",
                    "role": "user",
                }
            ]
        )
        if response:
            data = json.loads(response) if isinstance(response, str) else response
            response_content = data["choices"][0]["message"]["content"].lower().strip()
            if "yes" in response_content:
                return 1
        return 0
    except Exception as e:
        logging.error(f"Failed to evaluate checkpoint 3: {e}")
        return 0


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints, bonus_for_completing_any)

    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(trajectory))))
    checkpoints.append(Checkpoint(2, int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint3())))

    return result
