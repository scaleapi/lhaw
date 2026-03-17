import logging
import os
import re

# Import from LHAW evaluation system
import sys
from typing import List

import numpy as np

sys.path.append("../../../")
import pandas as pd
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result

logging.basicConfig(level=logging.INFO)


def _paths():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    flagged_path = os.path.join(output_root, "flagged_payments.xlsx")
    invoices_xlsx = os.path.join(eval_assets_root, "invoices.xlsx")
    payments_xlsx = os.path.join(eval_assets_root, "payments.xlsx")
    return flagged_path, invoices_xlsx, payments_xlsx


@grader
def grade_checkpoint1(trajectory: str) -> bool:
    # Accept both TAC-style "&openfile=true" and plain path navigation for local runs
    payment_signals = [
        "/Documents/Financials/payments.xlsx",
        "/Documents/Financials/payments.xlsx&openfile=true",
    ]
    invoices_signals = [
        "/Documents/Financials/invoices_pdfs",
        "dir=/Documents/Financials/invoices_pdfs",
    ]
    cond_payment = any(sig in trajectory for sig in payment_signals)
    cond_invoices = any(sig in trajectory for sig in invoices_signals)
    return cond_payment and cond_invoices


@grader
def grade_checkpoint2() -> int:
    flagged_path, _, _ = _paths()
    if not os.path.isfile(flagged_path):
        return 0
    try:
        df = pd.read_excel(flagged_path)
    except Exception as e:
        logging.warning(f"Failed to read flagged_payments.xlsx: {e}")
        return 0
    required_cols = ["Payment_ID", "Issue"]
    if not all(c in df.columns for c in required_cols):
        return 1
    pattern = r"^Total amount mismatch: Invoices=\d+(\.\d{2})?, Payments=\d+(\.\d{2})?$"
    last = df.iloc[-1]
    if (last.get("Payment_ID") == "TOTAL") and (
        re.match(pattern, str(last.get("Issue"))) is not None
    ):
        return 2
    return 1


@grader
def grade_checkpoint3() -> int:
    flagged_path, invoices_xlsx, payments_xlsx = _paths()
    if not (
        os.path.isfile(flagged_path)
        and os.path.isfile(invoices_xlsx)
        and os.path.isfile(payments_xlsx)
    ):
        return 0
    try:
        flagged_df = pd.read_excel(flagged_path)
        invoices_df = pd.read_excel(invoices_xlsx)
        payments_df = pd.read_excel(payments_xlsx)
    except Exception:
        return 0

    invoice_amounts = {row["Invoice_ID"]: row["Amount"] for _, row in invoices_df.iterrows()}
    issues = []
    for _, payment in payments_df.iterrows():
        references = str(payment["Reference"]).replace(" ", "").split(",")
        payment_id = payment["Payment_ID"]
        payment_amount = float(payment["Amount"])
        matched_invoices = []
        total_matched = 0.0
        for ref in references:
            if ref.startswith("INV"):
                invoice_id = ref[:7]
                partial = ("(Partial)" in ref) or ("(Remaining)" in ref)
                if invoice_id not in invoice_amounts:
                    issues.append(
                        {
                            "Payment_ID": payment_id,
                            "Issue": f"Referenced invoice {invoice_id} not found",
                        }
                    )
                else:
                    matched_invoices.append(invoice_id)
                    total_matched += (
                        payment_amount if partial else float(invoice_amounts[invoice_id])
                    )
        if len(matched_invoices) > 1:
            if not np.isclose(total_matched, payment_amount, rtol=1e-5):
                issues.append(
                    {
                        "Payment_ID": payment_id,
                        "Issue": f"Combined payment amount mismatch for invoices {matched_invoices}",
                    }
                )
        elif len(matched_invoices) == 1 and "(Partial)" in str(payment["Reference"]):
            invoice_id = matched_invoices[0]
            remaining = float(invoice_amounts[invoice_id]) - payment_amount
            if remaining > 0:
                issues.append(
                    {
                        "Payment_ID": payment_id,
                        "Issue": f"Partial payment for invoice {invoice_id}, remaining amount {remaining:.2f}",
                    }
                )

    total_invoice_amount = float(invoices_df["Amount"].sum())
    total_payment_amount = float(payments_df["Amount"].sum())
    if not np.isclose(total_invoice_amount, total_payment_amount, rtol=1e-5):
        issues.append(
            {
                "Payment_ID": "TOTAL",
                "Issue": f"Total amount mismatch: Invoices={total_invoice_amount:.2f}, Payments={total_payment_amount:.2f}",
            }
        )

    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        return 0
    # compare last row
    if not issues_df.iloc[-1].equals(flagged_df.iloc[-1]):
        return 0
    # compare Payment_ID column
    if (
        not issues_df["Payment_ID"]
        .reset_index(drop=True)
        .equals(flagged_df["Payment_ID"].reset_index(drop=True))
    ):
        return 1
    return 2


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(trajectory))))
    checkpoints.append(Checkpoint(2, int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(2, int(grade_checkpoint3())))
    return result
