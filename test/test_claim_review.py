from self_cognition.core.dialogue import GroundingReview, review_from_dict


def test_claim_level_review_round_trips():
    review = GroundingReview(
        supported=False,
        reason="contains unsupported assertions",
        social_response=False,
        unsupported_claims=("用户喜欢茶。",),
        uncertain_claims=("用户可能很忙。",),
    )

    assert GroundingReview.from_state_value(review.to_state_value()) == review
    assert review_from_dict(review.to_state_value()) == review


def test_old_review_shape_remains_readable():
    review = review_from_dict({"supported": True, "reason": "ok"})

    assert review.supported is True
    assert review.social_response is False
    assert review.unsupported_claims == ()
    assert review.uncertain_claims == ()