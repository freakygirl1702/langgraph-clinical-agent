"""Parse raw lab report text into structured rows: test_name, value, unit, ref_low, ref_high."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Optional: default reference ranges for common tests (use when report has no range)
DEFAULT_REF_RANGES = {
    "hb": (12.0, 16.0),
    "haemoglobin": (12.0, 16.0),
    "hb a1c": (4.0, 5.6),
    "hba1c": (4.0, 5.6),
    "glucose fasting": (70.0, 100.0),
    "fasting glucose": (70.0, 100.0),
    "blood glucose": (70.0, 100.0),
    "creatinine": (0.7, 1.2),
    "total cholesterol": (0.0, 200.0),
    "hdl": (40.0, 60.0),
    "ldl": (0.0, 100.0),
    "triglycerides": (0.0, 150.0),
    "tsh": (0.4, 4.0),
    "rbc": (4.5, 5.5),
    "wbc": (4000.0, 11000.0),
    "platelet": (150000.0, 400000.0),
    "platelets": (150000.0, 400000.0),
}


def _normalize_test_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _parse_ref_range(s: str) -> tuple[float | None, float | None]:
    """Parse strings like '70-100', '4.0 - 5.6', '&lt; 5.6', '&gt; 12' into (low, high)."""
    s = s.strip()
    # Replace common HTML/encoding
    s = s.replace("&lt;", "<").replace("&gt;", ">")
    # Range: number - number or number – number
    m = re.search(r"([\d.]+)\s*[-–—]\s*([\d.]+)", s)
    if m:
        try:
            return (float(m.group(1)), float(m.group(2)))
        except ValueError:
            pass
    # Single bound: < 5.6 or > 12
    m_lt = re.search(r"<\s*([\d.]+)", s)
    m_gt = re.search(r">\s*([\d.]+)", s)
    if m_lt:
        try:
            return (None, float(m_lt.group(1)))
        except ValueError:
            pass
    if m_gt:
        try:
            return (float(m_gt.group(1)), None)
        except ValueError:
            pass
    return (None, None)


def _parse_value(s: str) -> float | str | None:
    """Parse numeric value or return string for qualitative (e.g. Positive)."""
    s = s.strip()
    m = re.match(r"^([\d.]+)\s*([a-zA-Z/%]*)$", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass
    if s and s.lower() not in ("na", "n/a", "-", ""):
        return s  # qualitative
    return None


def parse_lab_report(raw_text: str) -> list[dict[str, Any]]:
    """
    Parse raw lab report text into list of dicts:
    { test_name, value, unit, ref_low, ref_high }.
    value may be float or str for qualitative results.
    """
    logger.info("Parsing lab report: raw text length=%s chars", len(raw_text))
    rows = []
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    logger.debug("Parsing %s non-empty lines", len(lines))

    # Columnar PDF extraction: headers appear one per line, followed by values
    header_seq = ["test", "result", "unit", "reference range", "flag", "note"]
    lowered = [ln.lower() for ln in lines]
    header_start = None
    for i in range(0, len(lowered) - 5):
        if lowered[i:i + 6] == header_seq:
            header_start = i
            break
    if header_start is not None:
        rows = []
        data_lines = lines[header_start + 6:]
        for i in range(0, len(data_lines), 6):
            chunk = data_lines[i:i + 6]
            if len(chunk) < 6:
                break
            if chunk[0].lower().startswith("disclaimer"):
                break
            if chunk[0].lower().startswith("note"):
                continue
            test_name = _normalize_test_name(chunk[0])
            value = _parse_value(chunk[1])
            unit = chunk[2]
            ref_low, ref_high = _parse_ref_range(chunk[3])
            if value is None or not test_name:
                continue
            rows.append({
                "test_name": test_name,
                "value": value,
                "unit": unit or "",
                "ref_low": ref_low,
                "ref_high": ref_high,
            })
        logger.info("Parse complete (columnar): %s lab result row(s) extracted", len(rows))
        return rows

    # Pattern: test name, value, optional unit, optional reference
    # Many reports: "Test Name     Value   Unit   Ref Range"
    for i, line in enumerate(lines):
        # Skip headers / titles
        if re.match(r"^(lab|test|parameter|result|reference|unit)", line, re.I) and i < 3:
            continue
        if re.search(r"\btest\b.*\bresult\b.*\bunit\b.*\breference\b", line, re.I):
            continue

        # Try a strict single-line parse (useful when PDF text has single spaces between columns)
        m = re.match(
            r"^(?P<name>.+?)\s+"
            r"(?P<value>[-+]?\d+(?:\.\d+)?)\s+"
            r"(?P<unit>[A-Za-z0-9/%\.\-\^]+)\s+"
            r"(?P<ref>(?:<\s*\d+(?:\.\d+)?|>\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?))\b",
            line,
        )
        if m:
            test_name = _normalize_test_name(m.group("name"))
            value = _parse_value(m.group("value"))
            unit = m.group("unit")
            ref_low, ref_high = _parse_ref_range(m.group("ref"))
            if test_name and value is not None:
                rows.append({
                    "test_name": test_name,
                    "value": value,
                    "unit": unit or "",
                    "ref_low": ref_low,
                    "ref_high": ref_high,
                })
                continue
        # Split on multiple spaces or tabs
        parts = re.split(r"\s{2,}|\t", line)
        if len(parts) < 2:
            # Try: "Name: value" or "Name value unit ref"
            colon = line.find(":")
            if colon > 0:
                name = line[:colon].strip()
                rest = line[colon + 1 :].strip()
                parts = [name] + rest.split()
            else:
                # Single token value after last word
                tokens = line.split()
                if len(tokens) >= 2:
                    # Last token often value; before that name
                    value_str = tokens[-1]
                    name = " ".join(tokens[:-1])
                    parts = [name, value_str]

        if len(parts) < 2:
            continue

        test_name = _normalize_test_name(parts[0])
        if len(test_name) < 2:
            continue

        value = _parse_value(parts[1])
        if value is None:
            continue

        unit = ""
        ref_low, ref_high = None, None

        for j in range(2, min(5, len(parts))):
            p = parts[j]
            ref_lo, ref_hi = _parse_ref_range(p)
            if ref_lo is not None or ref_hi is not None:
                ref_low, ref_high = ref_lo, ref_hi
                break
            if p and not re.match(r"^[\d.]+$", p):
                unit = p

        # Apply default ref range if missing
        if ref_low is None and ref_high is None:
            for key, (lo, hi) in DEFAULT_REF_RANGES.items():
                if key in test_name:
                    ref_low, ref_high = lo, hi
                    break

        rows.append({
            "test_name": test_name,
            "value": value,
            "unit": unit or "",
            "ref_low": ref_low,
            "ref_high": ref_high,
        })

    logger.info("Parse complete: %s lab result row(s) extracted", len(rows))
    return rows
