import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import TextIO

from self_cognition.application.results import ConverseResult, ProcessEventStatus
from self_cognition.bootstrap import ApplicationContainer, build_container
from self_cognition.core.dialogue import DialogueRequest, draft_to_dict
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.scopes import ConversationScope, SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.infrastructure.persistence.serialization import memory_to_dict, state_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-cognition local interface")
    parser.add_argument("subject_id", help="主体 ID")
    parser.add_argument("message", nargs="?", help="用户消息文本")
    parser.add_argument("--conversation-id")
    parser.add_argument("--data-dir", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None, *, container: ApplicationContainer | None = None) -> int:
    _configure_utf8_stream(sys.stdout)
    _configure_utf8_stream(sys.stderr)
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        command, args = _command_args(raw)
        return _chat(args if command == "chat" else raw, container) if command in {None, "chat"} else _run_command(command, args, container)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_code": _error_code(error), "error_type": type(error).__name__}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


def _command_args(argv: list[str]) -> tuple[str | None, list[str]]:
    aliases = {"chat", "memory", "memories", "correct", "export", "forget-dry-run", "forget", "replay", "doctor", "approve"}
    return (argv[0], argv[1:]) if argv and argv[0] in aliases else (None, argv)


def _chat(argv: list[str], container: ApplicationContainer | None) -> int:
    args = build_parser().parse_args(argv)
    if args.message is None:
        raise ValueError("message is required")
    dependencies = container or build_container(args.data_dir)
    with dependencies.lifecycle:
        from self_cognition.core.ids import new_correlation_id, new_run_id
        from self_cognition.runtime.run_context import RunContext

        context = RunContext(new_run_id(), new_correlation_id(), SYSTEM_CLOCK.now() + timedelta(seconds=30))
        conversation = ConversationScope(args.conversation_id) if args.conversation_id else None
        event = EventEnvelope.user_message(SubjectScope.legacy_user(args.subject_id), args.message, conversation=conversation, run_id=context.run_id, correlation_id=context.correlation_id)
        result = dependencies.converse.converse(DialogueRequest(event), context)
    print(json.dumps(_result_output(result), ensure_ascii=False, sort_keys=True))
    return 0 if result.status is ProcessEventStatus.SUCCEEDED else 1


def _run_command(command: str, argv: list[str], container: ApplicationContainer | None) -> int:
    parser = argparse.ArgumentParser(prog=f"self-cognition {command}")
    parser.add_argument("subject_id", nargs="?", default="user-1")
    parser.add_argument("--data-dir", default=None, type=Path)
    parser.add_argument("--target-field")
    parser.add_argument("--value")
    parser.add_argument("--cognition-type", default="preference")
    parser.add_argument("--action-id")
    args = parser.parse_args(argv)
    dependencies = container or build_container(args.data_dir)
    subject = SubjectScope.legacy_user(args.subject_id)
    with dependencies.lifecycle:
        if command in {"memory", "memories"}:
            payload: object = [memory_to_dict(item) for item in dependencies.memory_repository.read_by_subject(subject)]
        elif command == "correct":
            if not args.target_field or args.value is None:
                raise ValueError("--target-field and --value are required")
            result = dependencies.user_control.correct(subject, target_field=args.target_field, cognition_type=args.cognition_type, value=args.value, context=_context())
            payload = {"status": result.status.value, "run_id": str(result.run_id), "new_version": result.new_version, "error_type": result.error_type}
        elif command == "export":
            result = dependencies.user_control.export(subject)
            payload = {"export_id": str(result.export_id), "path": str(result.path), "counts": result.counts}
        elif command in {"forget-dry-run", "forget"}:
            from self_cognition.core.deletions import DeletionSelector
            plan = dependencies.forget.dry_run(DeletionSelector(subject, delete_subject=True), now=SYSTEM_CLOCK.now())
            if command == "forget":
                plan = dependencies.forget.execute(plan, now=SYSTEM_CLOCK.now())
            payload = {"plan_id": str(plan.plan_id), "status": plan.status.value, "event_count": len(plan.event_ids), "memory_count": len(plan.memory_ids)}
        elif command == "replay":
            payload = state_to_dict(dependencies.replay.replay(subject))
        elif command == "doctor":
            payload = dependencies.health.check().as_dict()
        elif command == "approve":
            if not args.action_id:
                raise ValueError("--action-id is required")
            result = dependencies.action.approve_and_execute(UUID(args.action_id), subject, _context())
            payload = {"status": result.status.value, "run_id": str(result.run_id), "result": _jsonable(result.result), "error_type": result.error_type, "reused": result.reused}
        else:
            raise ValueError(f"unsupported command: {command}")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _context():
    from self_cognition.core.ids import new_correlation_id, new_run_id
    from self_cognition.runtime.run_context import RunContext

    return RunContext(new_run_id(), new_correlation_id(), SYSTEM_CLOCK.now() + timedelta(seconds=30))


def _result_output(result: ConverseResult) -> dict[str, object]:
    output: dict[str, object] = {"status": result.status.value, "run_id": str(result.run_id), "correlation_id": str(result.correlation_id), "old_version": result.old_version, "new_version": result.new_version, "state_changed": result.state_changed, "event_saved": result.event_saved}
    if result.error_type is not None:
        output["error_type"] = result.error_type
    if result.response is not None:
        output.update({"response": result.response.text, "response_event_id": str(result.response_event_id), "claims": draft_to_dict(result.response)["claims"], "disclosure": draft_to_dict(result.response)["disclosure"], "evidence_refs": [_evidence_output(item) for item in result.evidence_refs]})
    return output


def _configure_utf8_stream(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def _evidence_output(evidence: EvidenceRef) -> dict[str, object]:
    return {"evidence_id": str(evidence.evidence_id), "source_kind": evidence.source_kind.value, "source_ref": evidence.source_ref, "locator": evidence.locator}


def _error_code(error: Exception) -> str:
    return {"ValueError": "invalid_request", "LookupError": "not_found", "PermissionError": "forbidden"}.get(type(error).__name__, "internal_error")


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value") and not hasattr(value, "__dataclass_fields__"):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
