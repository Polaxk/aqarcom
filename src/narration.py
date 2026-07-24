from __future__ import annotations

from typing import Any


RISK_ACTIONS = {
    "LOW": (
        "No material policy conflicts were detected. "
        "Continue with standard human review."
    ),
    "MEDIUM": (
        "Targeted human review is required before the case can progress."
    ),
    "HIGH": (
        "Escalated human review is required due to material evidence conflicts."
    ),
}


def generate_report(evaluation: dict[str, Any]) -> str:
    """
    Format the deterministic policy evaluation into a presentation-ready report.
    No LLM is used. No new reasons are added.
    """
    risk_label = evaluation["risk_label"]
    risk_score = evaluation["risk_score"]
    findings = evaluation.get("evidence_log", [])
    action = RISK_ACTIONS[risk_label]
    disclaimer = evaluation["disclaimer"]

    lines = [
        "AQARCOM RISK TRIAGE REPORT",
        "=" * 30,
        f"Risk Level: {risk_label}",
        f"Risk Score: {risk_score}",
        f"Explanation Method: Deterministic Policy Trace",
        "",
    ]

    if not findings:
        lines.extend([
            "Evidence Summary:",
            "- No configured risk signals were triggered.",
            "",
        ])
    else:
        lines.extend([
            "Evidence Summary:",
        ])

        for index, finding in enumerate(findings, start=1):
            lines.append(
                f"{index}. {finding['title']} (+{finding['points']} point(s))"
            )
            lines.append(f"   Why: {finding['reason']}")
            lines.append(
                f"   Evidence: {finding['evidence']['source_field']} = "
                f"{finding['evidence']['observed_value']!r}"
            )
            lines.append(
                f"   Recommended action: {finding['recommended_action']}"
            )

        lines.append("")

    lines.extend([
        f"Human-review action: {action}",
        "",
        f"Disclaimer: {disclaimer}",
    ])

    return "\n".join(lines)