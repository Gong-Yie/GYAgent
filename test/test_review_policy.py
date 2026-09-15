from self_cognition.core.dialogue import GroundingReview
from self_cognition.core.workspace import WorkspaceFixedContext, WorkspacePacket
from self_cognition.executive.dialogue.review_policy import ReviewPolicy


def _workspace(emotion=None):
    return WorkspacePacket(
        "user-1",
        0,
        (),
        fixed_context=WorkspaceFixedContext(
            emotion=() if emotion is None else (emotion,)
        ),
    )


def test_supported_review_is_accepted():
    decision = ReviewPolicy().decide(
        GroundingReview(True, "ok"),
        _workspace(),
    )

    assert decision.accepted
    assert not decision.emotion_modulated


def test_boredom_can_accept_social_response_without_unsupported_claims():
    decision = ReviewPolicy().decide(
        GroundingReview(
            False,
            "social response rejected by strict review",
            social_response=True,
        ),
        _workspace(
            {
                "field": "mood.current",
                "content": {
                    "emotion": "boredom",
                    "valence": "negative",
                    "arousal": 0.1,
                },
            }
        ),
    )

    assert decision.accepted
    assert decision.emotion_modulated


def test_boredom_cannot_accept_unsupported_factual_claims():
    decision = ReviewPolicy().decide(
        GroundingReview(
            False,
            "contains unsupported claim",
            social_response=True,
            unsupported_claims=("用户喜欢茶。",),
        ),
        _workspace(
            {
                "field": "mood.current",
                "content": {
                    "emotion": "boredom",
                    "valence": "negative",
                    "arousal": 0.1,
                },
            }
        ),
    )

    assert not decision.accepted
    assert not decision.emotion_modulated


def test_social_response_without_boredom_is_still_rejected():
    decision = ReviewPolicy().decide(
        GroundingReview(False, "rejected", social_response=True),
        _workspace(
            {
                "field": "mood.current",
                "content": {
                    "emotion": "neutral",
                    "valence": "neutral",
                    "arousal": 0.5,
                },
            }
        ),
    )

    assert not decision.accepted