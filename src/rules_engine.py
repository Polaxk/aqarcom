from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "risk_policy.yaml"
)


def load_policy(policy_path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load the versioned Aqarcom risk-points policy from YAML."""
    with open(policy_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _as_bool(value: Any) -> bool:
    """Safely interpret values that may originate from JSON or OCR extraction."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}

    return bool(value)


def _as_non_negative_int(value: Any) -> int:
    """Convert missing-document count to a safe non-negative integer."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _evidence(
    *,
    rule_id: str,
    title: str,
    points: int,
    reason: str,
    action: str,
    source_field: str,
    observed_value: Any,
    severity: str,
) -> dict[str, Any]:
    """Create one structured, auditable evidence-log entry."""
    return {
        "rule_id": rule_id,
        "title": title,
        "points": points,
        "severity": severity,
        "reason": reason,
        "recommended_action": action,
        "evidence": {
            "source_field": source_field,
            "observed_value": observed_value,
        },
    }


def _risk_label(score: int, thresholds: dict[str, Any]) -> str:
    """Map a deterministic score to LOW, MEDIUM, or HIGH."""
    if score <= int(thresholds["low_max"]):
        return "LOW"

    if score <= int(thresholds["medium_max"]):
        return "MEDIUM"

    return "HIGH"


def evaluate_risk(
    features: dict[str, Any],
    policy_path: str | Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    """
    Evaluate an Aqarcom case against the YAML risk policy.

    Required feature keys:
      - zoning_mismatch_flag: bool
      - title_seller_match_flag: bool
      - foreign_buyer_flag: bool
      - foreign_approval_docs_present_flag: bool
      - missing_required_docs_count: int
      - account_status: str
      - debit_locked_flag: bool
      - balance_band: str  # expected: healthy, low, negative

    Returns:
      - risk_score
      - risk_label
      - evidence_log
      - checked_signals
      - policy metadata
    """
    policy = load_policy(policy_path)
    points = policy["points"]
    thresholds = policy["thresholds"]

    risk_score = 0
    evidence_log: list[dict[str, Any]] = []
    checked_signals: list[dict[str, Any]] = []

    # 1. Zoning / intended-use mismatch.
    zoning_mismatch = _as_bool(features.get("zoning_mismatch_flag", False))
    checked_signals.append({
        "signal": "zoning_mismatch_flag",
        "value": zoning_mismatch,
        "triggered": zoning_mismatch,
    })

    if zoning_mismatch:
        rule_points = int(points["zoning_mismatch"])
        risk_score += rule_points
        evidence_log.append(
            _evidence(
                rule_id="ZONING_MISMATCH",
                title="Zoning / intended-use mismatch",
                points=rule_points,
                severity="HIGH",
                reason=(
                    "The declared intended use conflicts with the "
                    "property's permitted zoning classification."
                ),
                action="Escalate to zoning and legal review.",
                source_field="zoning_mismatch_flag",
                observed_value=True,
            )
        )

    # 2. Seller must match the registered title holder.
    title_seller_match = _as_bool(
        features.get("title_seller_match_flag", True)
    )
    title_seller_mismatch = not title_seller_match

    checked_signals.append({
        "signal": "title_seller_match_flag",
        "value": title_seller_match,
        "triggered": title_seller_mismatch,
    })

    if title_seller_mismatch:
        rule_points = int(points["title_seller_mismatch"])
        risk_score += rule_points
        evidence_log.append(
            _evidence(
                rule_id="TITLE_SELLER_MISMATCH",
                title="Title / seller mismatch",
                points=rule_points,
                severity="HIGH",
                reason=(
                    "The seller named in the purchase agreement does not "
                    "match the registered title holder."
                ),
                action="Escalate to title and legal review.",
                source_field="title_seller_match_flag",
                observed_value=False,
            )
        )

    # 3. Foreign buyer requires the corresponding approval documents.
    foreign_buyer = _as_bool(features.get("foreign_buyer_flag", False))
    foreign_approval_docs_present = _as_bool(
        features.get("foreign_approval_docs_present_flag", False)
    )
    foreign_docs_missing = foreign_buyer and not foreign_approval_docs_present

    checked_signals.append({
        "signal": "foreign_buyer_requirements",
        "value": {
            "foreign_buyer_flag": foreign_buyer,
            "foreign_approval_docs_present_flag": foreign_approval_docs_present,
        },
        "triggered": foreign_docs_missing,
    })

    if foreign_docs_missing:
        rule_points = int(points["foreign_buyer_missing_docs"])
        risk_score += rule_points
        evidence_log.append(
            _evidence(
                rule_id="FOREIGN_BUYER_MISSING_APPROVAL",
                title="Foreign-buyer prerequisites missing",
                points=rule_points,
                severity="MEDIUM",
                reason=(
                    "The buyer is marked as foreign, but the required "
                    "foreign-buyer approval documents are not present."
                ),
                action="Request the required foreign-buyer approval documents.",
                source_field="foreign_approval_docs_present_flag",
                observed_value=False,
            )
        )

    # 4. Missing required files: one point for each missing document.
    missing_docs_count = _as_non_negative_int(
        features.get("missing_required_docs_count", 0)
    )
    missing_docs_points = (
        missing_docs_count
        * int(points["missing_required_doc_per_item"])
    )

    checked_signals.append({
        "signal": "missing_required_docs_count",
        "value": missing_docs_count,
        "triggered": missing_docs_count > 0,
    })

    if missing_docs_count > 0:
        risk_score += missing_docs_points
        evidence_log.append(
            _evidence(
                rule_id="MISSING_REQUIRED_DOCUMENTS",
                title="Incomplete document set",
                points=missing_docs_points,
                severity="MEDIUM",
                reason=(
                    f"{missing_docs_count} required document(s) are missing "
                    "from the financing case file."
                ),
                action="Request the missing document(s) before review continues.",
                source_field="missing_required_docs_count",
                observed_value=missing_docs_count,
            )
        )

    # 5. JOFS account status must be active.
    account_status = str(features.get("account_status", "unknown")).lower()
    account_not_active = account_status != "active"

    checked_signals.append({
        "signal": "account_status",
        "value": account_status,
        "triggered": account_not_active,
    })

    if account_not_active:
        rule_points = int(points["account_status_not_active"])
        risk_score += rule_points
        evidence_log.append(
            _evidence(
                rule_id="ACCOUNT_NOT_ACTIVE",
                title="Account status requires review",
                points=rule_points,
                severity="LOW",
                reason=(
                    f"The buyer's JOFS account status is '{account_status}', "
                    "not 'active'."
                ),
                action="Review JOFS account status and supporting context.",
                source_field="account_status",
                observed_value=account_status,
            )
        )

    # 6. Debit lock.
    debit_locked = _as_bool(features.get("debit_locked_flag", False))
    checked_signals.append({
        "signal": "debit_locked_flag",
        "value": debit_locked,
        "triggered": debit_locked,
    })

    if debit_locked:
        rule_points = int(points["debit_locked"])
        risk_score += rule_points
        evidence_log.append(
            _evidence(
                rule_id="DEBIT_LOCKED",
                title="Debit-locked account",
                points=rule_points,
                severity="LOW",
                reason="The buyer's JOFS account is marked as debit-locked.",
                action="Review the reason and current status of the debit lock.",
                source_field="debit_locked_flag",
                observed_value=True,
            )
        )

    # 7. Low / negative balance band.
    balance_band = str(features.get("balance_band", "unknown")).lower()
    poor_balance = balance_band in {"low", "negative"}

    checked_signals.append({
        "signal": "balance_band",
        "value": balance_band,
        "triggered": poor_balance,
    })

    if poor_balance:
        rule_points = int(points["low_or_negative_balance_band"])
        risk_score += rule_points
        evidence_log.append(
            _evidence(
                rule_id="LOW_OR_NEGATIVE_BALANCE",
                title="Low or negative balance band",
                points=rule_points,
                severity="LOW",
                reason=(
                    f"The buyer's JOFS balance band is marked as "
                    f"'{balance_band}'."
                ),
                action="Review available funds and source-of-funds context.",
                source_field="balance_band",
                observed_value=balance_band,
            )
        )

    label = _risk_label(risk_score, thresholds)

    return {
        "risk_label": label,
        "risk_score": risk_score,
        "thresholds": {
            "low_max": int(thresholds["low_max"]),
            "medium_max": int(thresholds["medium_max"]),
            "high_min": int(thresholds["medium_max"]) + 1,
        },
        "evidence_log": evidence_log,
        "checked_signals": checked_signals,
        "explanation_method": "deterministic_policy_trace",
        "policy_source": str(policy_path),
        "disclaimer": (
            "Aqarcom provides risk triage and evidence-linked review guidance "
            "only. It does not approve, reject, or recommend a financing decision. "
            "A qualified human reviewer makes the final determination."
        ),
    }