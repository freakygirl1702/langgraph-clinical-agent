"""
Generate a synthetic high-risk lab report PDF for demo/testing.
Output: samples/high-risk-report.pdf
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF


def _add_text(page: fitz.Page, text: str, x: float, y: float, size: float = 10, bold: bool = False):
    # Use base font to avoid missing font errors across environments.
    font = "helv"
    page.insert_text((x, y), text, fontname=font, fontsize=size + (1 if bold else 0), color=(0, 0, 0))


def _draw_table(page: fitz.Page, x: float, y: float, col_widths: list[float], row_h: float, rows: list[list[str]]):
    # Draw rows and columns with borders
    table_width = sum(col_widths)
    table_height = row_h * len(rows)

    # Outer border
    page.draw_rect(fitz.Rect(x, y, x + table_width, y + table_height), color=(0, 0, 0), width=0.8)

    # Horizontal lines
    for i in range(1, len(rows)):
        yy = y + i * row_h
        page.draw_line((x, yy), (x + table_width, yy), color=(0, 0, 0), width=0.5)

    # Vertical lines
    x_pos = x
    for w in col_widths[:-1]:
        x_pos += w
        page.draw_line((x_pos, y), (x_pos, y + table_height), color=(0, 0, 0), width=0.5)

    # Cell text
    for r_idx, row in enumerate(rows):
        yy = y + r_idx * row_h + row_h * 0.7
        x_pos = x + 2
        for c_idx, cell in enumerate(row):
            _add_text(page, cell, x_pos, yy, size=9, bold=(r_idx == 0))
            x_pos += col_widths[c_idx]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = root / "samples" / "high-risk-report.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    _add_text(page, "Synthetic High-Risk Lab Report (Demo)", 40, 50, size=16, bold=True)
    _add_text(page, "Patient: Demo Patient", 40, 75, size=10)
    _add_text(page, f"Report Date: {datetime.now().strftime('%d-%b-%Y')}", 40, 90, size=10)
    _add_text(page, "Note: This is a synthetic example for testing only.", 40, 110, size=9)

    headers = ["Test", "Result", "Unit", "Reference Range", "Flag", "Note"]
    data = [
        ["HbA1c", "10.8", "%", "4.0 - 5.6", "High (Critical)", "Suggests poor glucose control"],
        ["Fasting Glucose", "240", "mg/dL", "70 - 99", "High (Critical)", "Markedly elevated"],
        ["LDL Cholesterol", "190", "mg/dL", "< 130", "High", "Atherosclerosis risk"],
        ["Triglycerides", "420", "mg/dL", "< 150", "High (Critical)", "Very high TG"],
        ["Creatinine", "2.3", "mg/dL", "0.7 - 1.2", "High", "Reduced kidney function"],
        ["eGFR", "32", "mL/min/1.73m2", "> 90", "Low", "Stage 3b CKD range"],
        ["Potassium", "5.8", "mmol/L", "3.5 - 5.1", "High", "Hyperkalemia risk"],
        ["Sodium", "128", "mmol/L", "136 - 145", "Low", "Hyponatremia"],
        ["ALT (SGPT)", "145", "U/L", "7 - 56", "High", "Liver enzyme elevation"],
        ["AST (SGOT)", "120", "U/L", "10 - 40", "High", "Liver enzyme elevation"],
        ["Hemoglobin", "9.8", "g/dL", "12.0 - 16.0", "Low", "Anemia range"],
        ["WBC", "14.2", "x10^3/uL", "4.0 - 11.0", "High", "Leukocytosis"],
        ["CRP", "35", "mg/L", "< 3", "High", "Inflammation marker"],
        ["TSH", "8.5", "mIU/L", "0.4 - 4.0", "High", "Hypothyroid pattern"],
        ["Troponin I", "0.16", "ng/mL", "< 0.04", "High (Critical)", "Cardiac injury marker"],
    ]

    rows = [headers] + data
    col_widths = [130, 60, 70, 120, 80, 110]
    _draw_table(page, x=40, y=140, col_widths=col_widths, row_h=22, rows=rows)

    _add_text(
        page,
        "Disclaimer: Educational use only. Not a substitute for medical advice.",
        40,
        780,
        size=9,
    )

    doc.save(out_path.as_posix())
    doc.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
