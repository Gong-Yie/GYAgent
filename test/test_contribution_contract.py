from uuid import uuid4

import pytest

from self_cognition.core.contributions import Contribution
from self_cognition.core.errors import ContractValidationError


def make_contribution(**overrides: object) -> Contribution:
    source_event_id = uuid4()
    values = {
        "contribution_id": uuid4(),
        "target_subject_id": "user-1",
        "target_field": "preferences.study_time",
        "value": "晚上",
        "confidence": 1.0,
        "evidence_event_ids": (source_event_id,),
        "source_event_id": source_event_id,
        "source_module": "semantic.preference_extractor",
    }
    values.update(overrides)
    return Contribution(**values)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_rejects_confidence_outside_unit_interval(confidence: float):
    with pytest.raises(ContractValidationError):
        make_contribution(confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_accepts_confidence_interval_boundaries(confidence: float):
    contribution = make_contribution(confidence=confidence)

    assert contribution.confidence == confidence


@pytest.mark.parametrize("target_subject_id", ["", "  \t"])
def test_rejects_blank_target_subject_id(target_subject_id: str):
    with pytest.raises(ContractValidationError):
        make_contribution(target_subject_id=target_subject_id)


@pytest.mark.parametrize("target_field", ["", "  \t"])
def test_rejects_blank_target_field(target_field: str):
    with pytest.raises(ContractValidationError):
        make_contribution(target_field=target_field)


def test_rejects_contribution_without_evidence():
    with pytest.raises(ContractValidationError):
        make_contribution(evidence_event_ids=())


@pytest.mark.parametrize("source_module", ["", "  \t"])
def test_rejects_blank_source_module(source_module: str):
    with pytest.raises(ContractValidationError):
        make_contribution(source_module=source_module)


def test_rejects_source_event_missing_from_evidence():
    with pytest.raises(ContractValidationError):
        make_contribution(source_event_id=uuid4())


def test_rejects_non_boolean_explicit_confirmation():
    with pytest.raises(ContractValidationError):
        make_contribution(explicitly_confirmed="yes")
