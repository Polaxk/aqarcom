FORBIDDEN_PHRASES = [
    "loan approved",
    "loan rejected",
    "title verified",
    "legally compliant",
    "aqarcom approves",
    "guarantees ownership",
    "legal conclusion",
]


class GuardrailViolationError(ValueError):
    """Raised when output contains legally unsafe or non-compliant phrases."""

    pass


def validate_report_safety(text: str) -> str:
    """Scans output text and raises an error if non-compliant wording is detected."""
    lowered = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in lowered:
            raise GuardrailViolationError(
                f"Compliance Guardrail Intercept: Output contained forbidden phrase '{phrase}'."
            )
    return text