from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from self_cognition.core.errors import ContractValidationError


def new_event_id() -> UUID:
    return uuid4()


def new_run_id() -> UUID:
    return uuid4()


def new_correlation_id() -> UUID:
    return uuid4()


def contribution_id(
    event_id: UUID,
    source_module: str,
    discriminator: str,
) -> UUID:
    if not source_module.strip():
        raise ContractValidationError("source_module must not be blank")
    if not discriminator:
        raise ContractValidationError("discriminator must not be empty")
    return uuid5(
        NAMESPACE_URL,
        f"{event_id}:{source_module}:{discriminator}",
    )


def memory_id(contribution_id: UUID, encoder_version: str) -> UUID:
    if not isinstance(contribution_id, UUID):
        raise ContractValidationError("contribution_id must be a UUID")
    if not encoder_version.strip():
        raise ContractValidationError("encoder_version must not be blank")
    return uuid5(
        NAMESPACE_URL,
        f"memory:{contribution_id}:{encoder_version}",
    )


def memory_access_id() -> UUID:
    return uuid4()


def consolidated_memory_id(memory_ids: tuple[UUID, ...], policy_version: str) -> UUID:
    if not memory_ids or any(not isinstance(value, UUID) for value in memory_ids):
        raise ContractValidationError("memory_ids must contain UUID values")
    if not policy_version.strip():
        raise ContractValidationError("policy_version must not be blank")
    ordered = ":".join(str(value) for value in sorted(set(memory_ids), key=lambda item: item.int))
    return uuid5(NAMESPACE_URL, f"memory:consolidated:{policy_version}:{ordered}")
