from dataclasses import dataclass

from self_cognition.core.dialogue import GroundingReview
from self_cognition.core.workspace import WorkspacePacket


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    accepted: bool
    reason: str
    emotion_modulated: bool = False


class ReviewPolicy:
    """Apply a bounded emotion modulation to social review decisions."""

    def decide(
        self,
        review: GroundingReview,
        workspace: WorkspacePacket,
    ) -> ReviewDecision:
        if review.supported:
            return ReviewDecision(True, "review supported")
        if review.unsupported_claims:
            return ReviewDecision(False, "unsupported factual claims cannot be accepted")
        if review.social_response and self._boredom_score(workspace) >= 0.5:
            return ReviewDecision(
                True,
                "boredom lowers the social acceptance threshold",
                emotion_modulated=True,
            )
        return ReviewDecision(False, "grounding review rejected the response")

    @staticmethod
    def _boredom_score(workspace: WorkspacePacket) -> float:
        score = 0.0
        for item in workspace.fixed_context.emotion:
            content = item.get("content")
            if not isinstance(content, dict):
                continue
            emotion = str(content.get("emotion", ""))
            valence = str(content.get("valence", ""))
            try:
                arousal = float(content.get("arousal", 0.5))
            except (TypeError, ValueError):
                arousal = 0.5
            if emotion in {"boredom", "loneliness"} or (
                valence == "negative" and arousal <= 0.3
            ):
                score = max(score, 1.0)
        return score