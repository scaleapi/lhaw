import logging
import os

# Import from LHAW evaluation system (only difference from original TAC)
import sys
from io import BytesIO
from typing import List

import pandas as pd
import PyPDF2
from PyPDF2.generic import ArrayObject

sys.path.append("../../../")
from evaluation.common import grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402


def extract_pdf_form_data(pdf_reader: PyPDF2.PdfReader) -> dict:
    """Return a dict of field name -> normalized string value from a PDF form.

    - Supports checkboxes/radios by falling back to /AS when /V is empty
    - Coerces NameObject (/MD) to 'MD'
    - Handles arrays and nested dictionaries
    """
    form_data = pdf_reader.get_fields() or {}

    def _val(fd):
        v = fd.get("/V")
        if (v is None or v == "") and fd.get("/FT") == "/Btn":
            v = fd.get("/AS")
        if isinstance(v, ArrayObject):
            return "" if not v else str(v[0]).lstrip("/")
        return "" if v is None else str(v)

    return {k: _val(fd) for k, fd in form_data.items()}


@grader
def run_checkpoint_1():
    # Resolve output root and eval assets root from environment
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", "/utils")

    # Agent output PDFs may be saved under either output_root/i_9_forms or output_root/Documents/Admin/i_9_forms
    candidates = [
        os.path.join(output_root, "i_9_forms"),
        os.path.join(output_root, "Documents", "Admin", "i_9_forms"),
    ]
    pdf_output_dir = next((p for p in candidates if os.path.isdir(p)), candidates[-1])
    # Golden CSV is provided via MRT_EVAL_ASSETS_ROOT or remains accessible via /utils mapping if runner patches
    csv_path = os.path.join(assets_root, "personell_data_golden.csv")

    df = pd.read_csv(csv_path, dtype=str)

    discrepancies = {}
    for name in df["Name"]:
        discrepancies[name] = []

    num_fields = 0
    for index, row in df.iterrows():
        file_name = f"i-9_{row['First Name (Given Name)']}_{row['Last Name (Family Name)']}.pdf"
        name = row["Name"]
        pdf_path = os.path.join(pdf_output_dir, file_name)
        logging.info(f"Checking PDF: {pdf_path}")
        if not os.path.exists(pdf_path):
            discrepancies[name].append((file_name, "File missing"))
            continue

        # Read bytes and pass a BytesIO stream so PyPDF2 can seek without a closed handle
        with open(pdf_path, "rb") as f:
            content = f.read()
        if not content:
            discrepancies[name].append((file_name, "File empty"))
            continue

        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        except Exception as e:
            discrepancies[name].append((file_name, f"Error reading PDF: {e}"))
            continue

        # Extract data from the PDF using helper
        pdf_data = extract_pdf_form_data(pdf_reader)

        # Compare extracted data with CSV row data
        golden_values = {
            "Last Name (Family Name)": row["Last Name (Family Name)"],
            "First Name Given Name": row["First Name (Given Name)"],
            "Employee Middle Initial (if any)": (
                row["Middle Initial (if any)"] if pd.notnull(row["Middle Initial (if any)"]) else ""
            ),
            "Employee Other Last Names Used (if any)": (
                row["Other Last Names Used (if any)"]
                if pd.notnull(row["Other Last Names Used (if any)"])
                else ""
            ),
            "Address Street Number and Name": row["Address (Street Number and Name)"],
            "Apt Number (if any)": (
                row["Apt. Number (if any)"] if pd.notnull(row["Apt. Number (if any)"]) else ""
            ),
            "City or Town": row["City or Town"],
            "ZIP Code": row["ZIP Code"],
            "Date of Birth mmddyyyy": row["Date of Birth (mm/dd/yyyy)"],
            "US Social Security Number": str(row["U.S. Social Security Number"]),
            "Employees E-mail Address": row["Employee's Email Address"],
            "Telephone Number": str(row["Employee's Telephone Number"]),
            "3 A lawful permanent resident Enter USCIS or ANumber": (
                str(row["USCIS A-Number"]) if pd.notnull(row["USCIS A-Number"]) else ""
            ),
            "Exp Date mmddyyyy": (
                row["Expiration Date (mm/dd/yyyy)"]
                if pd.notnull(row["Expiration Date (mm/dd/yyyy)"])
                else ""
            ),
            "USCIS ANumber": (
                str(row["USCIS A-Number"])
                if pd.notnull(row["USCIS A-Number"])
                and row["Citizenship/Immigration Status"] == "A noncitizen authorized to work"
                else ""
            ),
            "Form I94 Admission Number": (
                str(row["Form I-94 Admission Number"])
                if pd.notnull(row["Form I-94 Admission Number"])
                else ""
            ),
            "Foreign Passport Number and Country of IssuanceRow1": (
                f"{row['Foreign Passport Number']} {row['Country of Issuance']}"
                if pd.notnull(row["Foreign Passport Number"])
                else ""
            ),
            "Signature of Employee": row["Signature of Employee"],
            "Today's Date mmddyyy": row["Today's Date (mm/dd/yyyy)"],
            "State": f"{row['State']}",
            "CB_1": (
                "/On"
                if row["Citizenship/Immigration Status"] == "A citizen of the United States"
                else "/Off"
            ),
            "CB_2": (
                "/On"
                if row["Citizenship/Immigration Status"]
                == "A noncitizen national of the United States"
                else "/Off"
            ),
            "CB_3": (
                "/On"
                if row["Citizenship/Immigration Status"] == "A lawful permanent resident"
                else "/Off"
            ),
            "CB_4": (
                "/On"
                if row["Citizenship/Immigration Status"] == "A noncitizen authorized to work"
                else "/Off"
            ),
        }
        # add non-nan or non-empty or non-off fields to num_fields
        num_fields += len(
            [
                field
                for field in golden_values
                if golden_values[field] and golden_values[field] != "/Off"
            ]
        )
        for field, expected_value in golden_values.items():
            pdf_value = pdf_data.get(field, "")
            if expected_value == "/Off":
                if pdf_value and pdf_value not in ["", "/Off"]:
                    discrepancies[name].append(
                        f"{file_name}: {field} - Expected: {expected_value}, Found: {pdf_value}"
                    )
            elif str(pdf_value) != expected_value:
                discrepancies[name].append(
                    f"{file_name}: {field} - Expected: {expected_value}, Found: {pdf_value}"
                )

    num_forms = len(df)
    total_correct_forms = sum([1 for name in discrepancies if not discrepancies[name]])
    total_incorrect_fields = sum([len(discrepancies[name]) for name in discrepancies])
    # log the discrepancies
    for name in discrepancies:
        for discrepancy in discrepancies[name]:
            logging.warning(discrepancy)

    logging.info(f"Total correct forms: {total_correct_forms}")
    if total_correct_forms == num_forms:
        logging.info("All I-9 forms are correctly filled.")
        return 5
    else:
        logging.warning(
            f"Total incorrect fields: {total_incorrect_fields} out of {num_fields} fields, total incorrect forms: {num_forms - total_correct_forms} out of {num_forms} forms."
        )
        return min(
            round(5 * (num_fields - total_incorrect_fields) / num_fields),
            round(5 * total_correct_forms / num_forms),
        )


def grade_checkpoints(trajectory=""):
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(5, int(run_checkpoint_1())))
    return result
