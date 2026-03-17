import logging
import os

# LHAW imports
import sys
from pathlib import Path
from typing import List

sys.path.append("../../../")
from evaluation.common import grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402


def _get_output_root() -> Path:
    """Return the output root path."""
    return Path(os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output"))


@grader
def run_checkpoint_1():
    """Check all PDFs are categorized (94 total)."""
    output_root = _get_output_root()
    subdirs = ["early-career", "mid-level", "senior", "unsure"]

    pdf_counts = {}
    total_pdfs = 0

    for subdir in subdirs:
        dir_path = output_root / subdir
        try:
            pdf_count = len(list(dir_path.glob("*.pdf")))
            pdf_counts[subdir] = pdf_count
            total_pdfs += pdf_count
        except FileNotFoundError:
            pdf_counts[subdir] = 0
            logging.warning(f"Warning: Directory {dir_path} not found")

    logging.info(f"Total PDF files: {total_pdfs}, correct answer: 94")
    return total_pdfs == 94


@grader
def run_checkpoint_2():
    """Check early-career folder has 58 PDFs with correct files."""
    dir_path = _get_output_root() / "early-career"
    try:
        resumes = list(dir_path.glob("*.pdf"))
        early_career_count = len(resumes)
        logging.info(
            f"Number of files in early-career folder: {early_career_count}, correct answer: 58"
        )

        expected_files = {
            "Aaliyah_Thompson.pdf",
            "Emma_Smith.pdf",
            "John_Smith.pdf",
            "Lu_Jian.pdf",
            "Taylor_Swift_Bootcamp.pdf",
            "Avery_Williams_Bootcamp.pdf",
            "Ethan_Anderson.pdf",
            "Jordan_Lee_Bootcamp.pdf",
            "Luo_Jie.pdf",
            "Wang_Hai_Yan.pdf",
            "Blake_Anderson_Bootcamp.pdf",
            "Feng_Feng_Ying.pdf",
            "Keisha_Washington.pdf",
            "Lv_Yang.pdf",
            "Xiao_Xia.pdf",
            "Charlie_Brown_Bootcamp.pdf",
            "Finley_Moore_Bootcamp.pdf",
            "Kendall_Wright_Bootcamp.pdf",
            "Mei_Chen.pdf",
            "Xu_Gui_Hua.pdf",
            "Chen_Tao.pdf",
            "Harper_Garcia_Bootcamp.pdf",
            "Lai_Li.pdf",
            "Noah_Garcia.pdf",
            "Xu_Yan.pdf",
            "Christopher_Brown.pdf",
            "He_Chen.pdf",
            "Li_Bing.pdf",
            "Olivia_Martinez.pdf",
            "Yang_Shu_Zhen.pdf",
            "Cui_Xue_Mei.pdf",
            "Hiroshi_Tanaka.pdf",
            "Li_Jie.pdf",
            "Peyton_Clark_Bootcamp.pdf",
            "Yao_Xiu_Ying.pdf",
            "DeShawn_Robinson.pdf",
            "Huang_Bin.pdf",
            "Li_Mei.pdf",
            "Reese_Campbell_Bootcamp.pdf",
            "Yuki_Nakamura.pdf",
            "Deng_Cheng.pdf",
            "Huang_Ting_Ting.pdf",
            "Li_Tao.pdf",
            "Skylar_White_Bootcamp.pdf",
            "Zhang_Gui_Lan.pdf",
            "Ding_Shu_Zhen.pdf",
            "Huang_Yu_Zhen.pdf",
            "Lin_Yan.pdf",
            "Song_Shu_Ying.pdf",
            "Zhang_Hong.pdf",
            "Drew_Martinez_Bootcamp.pdf",
            "Jamie_Davis_Bootcamp.pdf",
            "Liu_Wei.pdf",
            "Soo-Jin_Kim.pdf",
            "Emerson_Scott_Bootcamp.pdf",
            "Jessica_Wilson.pdf",
            "Lu_Hui.pdf",
            "Tan_Rui.pdf",
        }

        found_files = {r.name for r in resumes}
        return early_career_count == 58 and expected_files.issubset(found_files)
    except Exception:
        return False


@grader
def run_checkpoint_3():
    """Check mid-level folder has Alex_Chen.pdf."""
    dir_path = _get_output_root() / "mid-level"
    try:
        resumes = list(dir_path.glob("*.pdf"))
        mid_level_count = len(resumes)
        has_alex = any(r.name == "Alex_Chen.pdf" for r in resumes)
        return mid_level_count == 1 and has_alex
    except Exception:
        return False


@grader
def run_checkpoint_4():
    """Check senior folder has Emily_Zhang.pdf."""
    dir_path = _get_output_root() / "senior"
    try:
        resumes = list(dir_path.glob("*.pdf"))
        senior_count = len(resumes)
        has_emily = any(r.name == "Emily_Zhang.pdf" for r in resumes)
        return senior_count == 1 and has_emily
    except Exception:
        return False


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(run_checkpoint_1())))
    checkpoints.append(Checkpoint(1, int(run_checkpoint_2())))
    checkpoints.append(Checkpoint(1, int(run_checkpoint_3())))
    checkpoints.append(Checkpoint(1, int(run_checkpoint_4())))
    return result
