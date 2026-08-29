import argparse
import time
from uuid import UUID

from self_cognition.core.indexes import WorkspaceIndex
from self_cognition.core.state import StateEntry, SubjectState


PREFIX = "episodic.experience."


def build_state(entry_count: int) -> SubjectState:
    entries = {
        f"{PREFIX}2026-08-13T00:00:{index % 60:02d}+00:00.{index}": StateEntry(
            value=f"经历 {index}",
            confidence=1.0,
            evidence_event_ids=(UUID(int=index + 1),),
            contribution_ids=(UUID(int=index + 1),),
        )
        for index in range(entry_count)
    }
    return SubjectState(
        subject_id="benchmark-user",
        version=entry_count,
        entries=entries,
        applied_contribution_ids=frozenset(
            UUID(int=index + 1) for index in range(entry_count)
        ),
        conflicts=frozenset(),
    )


def measure(entry_count: int, repeats: int) -> dict[str, object]:
    state = build_state(entry_count)

    started = time.perf_counter()
    index = WorkspaceIndex.build(state)
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scanned = ()
    for _ in range(repeats):
        scanned = tuple(
            sorted(field for field in state.entries if field.startswith(PREFIX))
        )
    scan_seconds = time.perf_counter() - started

    started = time.perf_counter()
    indexed = ()
    for _ in range(repeats):
        indexed = index.fields_for_prefix(PREFIX)
    query_seconds = time.perf_counter() - started

    saved_per_query = (scan_seconds - query_seconds) / repeats
    break_even_queries = (
        build_seconds / saved_per_query if saved_per_query > 0.0 else None
    )
    return {
        "entries": entry_count,
        "repeats": repeats,
        "build_seconds": build_seconds,
        "scan_seconds": scan_seconds,
        "index_query_seconds": query_seconds,
        "break_even_queries": break_even_queries,
        "same_fields": scanned == indexed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entries", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    print(measure(args.entries, args.repeats))


if __name__ == "__main__":
    main()
