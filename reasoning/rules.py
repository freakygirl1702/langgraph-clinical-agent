"""Rules engine: compare value to reference range and set status + fact per row."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _numeric_status(value: float, ref_low: float | None, ref_high: float | None) -> tuple[str, str]:
    """Return (status, fact) for numeric value vs reference range."""
    if ref_low is not None and value < ref_low:
        diff = ref_low - value
        unit_str = ""
        return "low", f"Result is {diff} below the normal lower limit."
    if ref_high is not None and value > ref_high:
        diff = value - ref_high
        return "high", f"Result is {diff} above the normal upper limit."
    if ref_low is not None and ref_high is not None:
        return "normal", "Result is within the normal range."
    if ref_low is not None and value >= ref_low:
        return "normal", "Result is at or above the normal lower limit."
    if ref_high is not None and value <= ref_high:
        return "normal", "Result is at or below the normal upper limit."
    return "unknown", "Reference range not available for comparison."


def apply_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    For each parsed row, add status and fact.
    Input: list of { test_name, value, unit, ref_low, ref_high }
    Output: same list with added status, fact (and optional critical).
    """
    logger.info("Applying rules: %s row(s)", len(rows))
    result = []
    for row in list(rows):
        r = dict(row)
        value = r.get("value")
        ref_low = r.get("ref_low")
        ref_high = r.get("ref_high")

        if isinstance(value, (int, float)):
            status, fact = _numeric_status(float(value), ref_low, ref_high)
            r["status"] = status
            r["fact"] = fact
            # Optional: mark critical if far out of range
            if ref_high is not None and float(value) > ref_high * 1.5:
                r["critical"] = True
            elif ref_low is not None and ref_low > 0 and float(value) < ref_low * 0.5:
                r["critical"] = True
            else:
                r["critical"] = False
        else:
            r["status"] = "non_numeric"
            r["fact"] = "Result is qualitative; please discuss with a doctor."
            r["critical"] = False

        # Log the generated rule for this row
        logger.info(
            "Rule generated: test_name=%s | value=%s %s | ref=%s-%s | status=%s | fact=%s | critical=%s",
            r.get("test_name", ""),
            r.get("value", ""),
            r.get("unit", ""),
            r.get("ref_low"),
            r.get("ref_high"),
            r.get("status", ""),
            r.get("fact", ""),
            r.get("critical", False),
        )
        result.append(r)
    normal = sum(1 for r in result if r.get("status") == "normal")
    high_low = sum(1 for r in result if r.get("status") in ("high", "low"))
    critical = sum(1 for r in result if r.get("critical"))
    logger.info("Rules complete: normal=%s, high/low=%s, critical=%s", normal, high_low, critical)
    return result
