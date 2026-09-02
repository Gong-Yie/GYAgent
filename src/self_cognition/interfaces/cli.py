import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import TextIO

from self_cognition.application.results import ProcessEventResult
from self_cognition.application.results import ProcessEventStatus
from self_cognition.bootstrap import ApplicationContainer, build_container
from self_cognition.core.evidence import EvidenceRef
from self_cognition.core.events import EventEnvelope
from self_cognition.core.ids import new_correlation_id, new_run_id
from self_cognition.core.scopes import SubjectScope
from self_cognition.core.time import SYSTEM_CLOCK
from self_cognition.runtime.run_context import RunContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process one self-cognition event")
    parser.add_argument("subject_id", help="主体 ID")
    parser.add_argument("message", help="用户消息文本")
    parser.add_argument(
        "--data-dir",
        default=None,
        type=Path,
        help="事件日志和状态快照目录（覆盖 .env 配置）",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    container: ApplicationContainer | None = None,
) -> int:
    _configure_utf8_stream(sys.stdout)
    _configure_utf8_stream(sys.stderr)
    args = build_parser().parse_args(argv)
    try:
        dependencies = container or build_container(args.data_dir)
        with dependencies.lifecycle:
            context = RunContext(
                run_id=new_run_id(),
                correlation_id=new_correlation_id(),
                deadline=SYSTEM_CLOCK.now() + timedelta(seconds=30),
            )
            event = EventEnvelope.user_message(
                SubjectScope.legacy_user(args.subject_id),
                args.message,
                run_id=context.run_id,
                correlation_id=context.correlation_id,
            )
            result = dependencies.process_event.process(event, context)
            output = _result_output(dependencies, args.message, result)
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0 if result.status is ProcessEventStatus.SUCCEEDED else 1
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(error).__name__},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


def _result_output(
    dependencies: ApplicationContainer,
    message: str,
    result: ProcessEventResult,
) -> dict[str, object]:
    output: dict[str, object] = {
        "status": result.status.value,
        "run_id": str(result.run_id),
        "correlation_id": str(result.correlation_id),
        "old_version": result.old_version,
        "new_version": result.new_version,
        "state_changed": result.state_changed,
        "event_saved": result.event_saved,
    }
    if result.error_type is not None:
        output["error_type"] = result.error_type

    if result.state is not None:
        response = dependencies.dialogue_model.respond(
            message,
            dependencies.workspace_builder.build(message, result.state),
        )
        output["response"] = response.text
        output["evidence_refs"] = [
            _evidence_output(evidence) for evidence in response.evidence_refs
        ]
    return output


def _configure_utf8_stream(stream: TextIO) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def _evidence_output(evidence: EvidenceRef) -> dict[str, object]:
    return {
        "evidence_id": str(evidence.evidence_id),
        "source_kind": evidence.source_kind.value,
        "source_ref": evidence.source_ref,
        "locator": evidence.locator,
    }


if __name__ == "__main__":
    raise SystemExit(main())
