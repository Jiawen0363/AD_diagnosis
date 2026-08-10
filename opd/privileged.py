"""Privileged-information blocks for teacher-conditioned distillation.

Training objective (same dialogue state s_t):

    Teacher:  pi_teacher(a | s_t, z)   with privileged context z
    Student:  pi_theta(a | s_t)         learns from teacher signals only

At deployment the student tutor never receives z.
"""

from __future__ import annotations

import re
from typing import Any

REFERENCE_KNOWLEDGE_MARKER = "Reference Knowledge:"
GOAL_DESCRIPTION_MARKER = "Goal Description:"

STUDENT_LEVEL_LABELS = {
    "low_level": "low level (minimal prior knowledge)",
    "med_level": "medium level (partial dependency knowledge)",
    "high_level": "high level (broader dependency and partial solution knowledge)",
}


def student_prior_knowledge(element: dict[str, Any], student_level: str) -> str:
    """Ground-truth student prior / missing knowledge used only by the teacher."""
    if student_level == "low_level":
        return "The simulated student starts with no explicit dependency or solution prior."

    dependency = str(element.get("dependency_sampled", "")).strip()
    if student_level == "med_level":
        if not dependency:
            return "The simulated student is medium level but no dependency prior was recorded."
        return f"Known dependency prior:\n{dependency}"

    if student_level == "high_level":
        ref_steps = str(element.get("reference_steps", "")).split("2.")[0].strip()
        parts = [part for part in (dependency, ref_steps) if part]
        if not parts:
            return "The simulated student is high level but no prior knowledge was recorded."
        return "Known dependency and partial solution prior:\n" + "\n\n".join(parts)

    raise ValueError(f"Unsupported student level: {student_level}")


def reference_knowledge_block(element: dict[str, Any], *, contexts_above: str) -> str:
    """Format oracle reference knowledge for the teacher."""
    dependency_paths = str(element.get("dependency_all", "")).strip().split("\n\n")
    reference_steps = _format_kc_lines(dependency_paths, str(element.get("reference_steps", "")))
    return (
        f"{REFERENCE_KNOWLEDGE_MARKER}\n"
        f"- The contexts above the {element['function_name']} function:\n"
        f"```Python\n{contexts_above}\n```\n"
        f"- The dependency paths for the {element['function_name']} function:\n"
        f"{reference_steps['dependency']}\n"
        f"- The reference key solution steps:\n"
        f"{reference_steps['steps']}"
    )


def compose_privileged_context(
    *,
    reference_block: str,
    student_level: str | None = None,
    student_prior: str | None = None,
    extra: str | None = None,
) -> str:
    """Assemble teacher-only privileged context z."""
    sections: list[str] = []

    if student_level is not None:
        label = STUDENT_LEVEL_LABELS.get(student_level, student_level)
        sections.append(f"Privileged Student Profile:\n- Simulated student level: {label}")

    if student_prior:
        sections.append(f"Privileged Student Prior / Missing Knowledge:\n{student_prior.strip()}")

    if reference_block.strip():
        sections.append(reference_block.strip())

    if extra and extra.strip():
        sections.append(extra.strip())

    if not sections:
        raise ValueError("Privileged context must contain at least one non-empty section.")
    return "\n\n".join(sections)


def split_instruction_privileged(instruction: str) -> tuple[str, str | None]:
    """Split a legacy tutor instruction into student-visible text and reference knowledge."""
    if REFERENCE_KNOWLEDGE_MARKER not in instruction:
        return instruction.strip(), None

    before, after = instruction.split(REFERENCE_KNOWLEDGE_MARKER, 1)
    if GOAL_DESCRIPTION_MARKER in after:
        reference_block, goal_and_rest = after.split(GOAL_DESCRIPTION_MARKER, 1)
        student_instruction = (before + GOAL_DESCRIPTION_MARKER + goal_and_rest).strip()
        privileged_context = (REFERENCE_KNOWLEDGE_MARKER + reference_block.strip()).strip()
    else:
        student_instruction = before.strip()
        privileged_context = (REFERENCE_KNOWLEDGE_MARKER + after.strip()).strip()

    return student_instruction, privileged_context


def build_privileged_from_record(record: dict[str, Any]) -> str:
    """Resolve teacher privileged context from explicit or legacy record fields."""
    if record.get("privileged_context"):
        return str(record["privileged_context"]).strip()

    _, reference_only = split_instruction_privileged(str(record.get("instruction") or record.get("prompt") or ""))
    if not reference_only:
        raise ValueError("Record is missing privileged context and reference knowledge.")

    return compose_privileged_context(
        reference_block=reference_only,
        student_level=record.get("student_level"),
        student_prior=record.get("student_prior_knowledge"),
        extra=record.get("privileged_extra"),
    )


def student_observation(record: dict[str, Any]) -> str:
    """Resolve the student-visible instruction portion of s_t."""
    if record.get("student_instruction"):
        return str(record["student_instruction"]).strip()

    instruction = record.get("instruction") or record.get("prompt")
    if not instruction:
        raise ValueError("Record must contain `instruction`, `prompt`, or `student_instruction`.")
    student_instruction, _ = split_instruction_privileged(str(instruction))
    return student_instruction


def _format_kc_lines(dependency_paths: list[str], reference_steps_raw: str) -> dict[str, str]:
    kc_dependency, kc_reference = [], []
    idx = 1
    for dp in dependency_paths:
        dp = dp.replace("{", "[").replace("}", "]").strip()
        if dp:
            kc_dependency.append(f"KC-{idx}: {dp}")
            idx += 1

    steps = [step.strip() for step in re.split(r"\n(?=\d+\.)", reference_steps_raw.strip()) if step.strip()]
    for step in steps:
        step = step.replace("{", "[").replace("}", "]")
        kc_reference.append(f"KC-{idx}: {step}")
        idx += 1

    return {
        "dependency": "\n".join(kc_dependency),
        "steps": "\n".join(kc_reference),
    }
