from datetime import datetime, timezone
from uuid import uuid4

import pytest

from self_cognition.core.contributions import (
    CognitiveContribution,
    CognitionType,
    ContributionOperation,
)
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.errors import ContractValidationError
from self_cognition.core.scopes import DataScope, DisclosureScope, SubjectScope


NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)
SUBJECT = SubjectScope.legacy_user("user-1")
SCOPE = DataScope(SUBJECT, DisclosureScope.PRIVATE)


def make_contribution(**overrides: object) -> CognitiveContribution:
    values = {
        "contribution_id": uuid4(),
        "target": SUBJECT,
        "target_field": "preferences.study_time",
        "operation": ContributionOperation.SET,
        "cognition_type": CognitionType.PREFERENCE,
        "value": "晚上",
        "confidence": 1.0,
        "evidence_refs": (EvidenceRef.for_event_id(uuid4(), SUBJECT),),
        "source_module": "semantic.preference_extractor",
        "module_version": "1",
        "scope": SCOPE,
        "created_at": NOW,
        "valid_from": NOW,
    }
    values.update(overrides)
    return CognitiveContribution(**values)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_rejects_confidence_outside_unit_interval(confidence: float) -> None:
    with pytest.raises(ContractValidationError):
        make_contribution(confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_accepts_confidence_interval_boundaries(confidence: float) -> None:
    assert make_contribution(confidence=confidence).confidence == confidence


@pytest.mark.parametrize("target_field", ["", "  \t"])
def test_rejects_blank_target_field(target_field: str) -> None:
    with pytest.raises(ContractValidationError):
        make_contribution(target_field=target_field)


@pytest.mark.parametrize("field", ["source_module", "module_version"])
def test_rejects_blank_source_metadata(field: str) -> None:
    with pytest.raises(ContractValidationError):
        make_contribution(**{field: " "})


def test_rejects_missing_or_invalid_evidence() -> None:
    with pytest.raises(ContractValidationError):
        make_contribution(evidence_refs=())
    with pytest.raises(ContractValidationError):
        make_contribution(evidence_refs=(uuid4(),))


def test_rejects_scope_owned_by_another_subject() -> None:
    foreign_scope = DataScope(
        SubjectScope.legacy_user("user-2"),
        DisclosureScope.PRIVATE,
    )

    with pytest.raises(ContractValidationError):
        make_contribution(scope=foreign_scope)


def test_rejects_naive_times_and_invalid_expiry() -> None:
    with pytest.raises(ContractValidationError):
        make_contribution(created_at=datetime(2026, 9, 1))
    with pytest.raises(ContractValidationError):
        make_contribution(expires_at=NOW)


def test_rejects_non_boolean_explicit_confirmation() -> None:
    with pytest.raises(ContractValidationError):
        make_contribution(explicitly_confirmed="yes")
