from dataclasses import dataclass
from datetime import datetime

from self_cognition.core.errors import ScopeMismatchError
from self_cognition.core.state import SubjectState
from self_cognition.core.workspace import (
    RetrievalQuery,
    WorkspaceBuilder,
    WorkspacePacket,
    WorkspaceRunInfo,
)


@dataclass(frozen=True, slots=True)
class ReadOnlyCognitionContext:
    builder: WorkspaceBuilder
    state: SubjectState
    as_of: datetime
    run_info: WorkspaceRunInfo | None = None

    def query(self, query: RetrievalQuery) -> WorkspacePacket:
        if query.subject != self.state.subject_scope:
            raise ScopeMismatchError(
                "cognition context query cannot cross its subject scope"
            )
        return self.builder.build(
            query.task,
            self.state,
            as_of=self.as_of,
            query=query,
            run_info=self.run_info,
        )
