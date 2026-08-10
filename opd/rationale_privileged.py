"""Privileged-information blocks for AD rationale OPD."""

from __future__ import annotations

import math
from typing import Any


PI_TEMPLATE = """Private reference information, for silent calibration only (do not mention, quote, or reveal any of this in your output):
The participant's assigned diagnostic group is {diagnosis}. Use this reference only to help you calibrate which transcript-supported language features should be emphasized and how carefully the summary should be worded, without treating the group label as observable evidence or citing it in the rationale.
The participant's MMSE-related severity level is {mmse_severity}. Use this reference only to help you calibrate how strongly you describe transcript-supported language difficulties, without treating MMSE or severity labels as observable evidence or citing them in the rationale."""


def format_diagnosis(label: str | None) -> str:
    normalized = str(label or "").strip().lower()
    if normalized in {"ad", "1", "1.0"}:
        return "AD"
    if normalized in {"control", "cn", "0", "0.0"}:
        return "Control"
    return "unknown"


def format_mmse_severity(mmse: Any) -> str:
    if mmse is None:
        return "unknown"
    if isinstance(mmse, str) and not mmse.strip():
        return "unknown"
    try:
        value = float(mmse)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(value):
        return "unknown"
    if 24 <= value <= 30:
        return "normal range (MMSE 24–30)"
    if 18 <= value <= 23:
        return "mild cognitive difficulty range (MMSE 18–23)"
    if 10 <= value <= 17:
        return "moderate cognitive difficulty range (MMSE 10–17)"
    if 0 <= value <= 9:
        return "marked cognitive difficulty range (MMSE 0–9)"
    return "unknown"


def build_rationale_privileged_context(
    *,
    diagnosis: str,
    mmse_severity: str,
) -> str:
    return PI_TEMPLATE.format(
        diagnosis=diagnosis,
        mmse_severity=mmse_severity,
    )


def build_privileged_context_from_record(
    record: dict[str, Any],
    *,
    use_labels: bool,
) -> str:
    if use_labels:
        diagnosis = format_diagnosis(record.get("label"))
        mmse_severity = format_mmse_severity(record.get("mmse"))
    else:
        diagnosis = "unknown"
        mmse_severity = "unknown"
    return build_rationale_privileged_context(
        diagnosis=diagnosis,
        mmse_severity=mmse_severity,
    )
