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
