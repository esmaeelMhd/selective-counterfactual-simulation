from __future__ import annotations

from scs.experiments.cstr_replication import (
    FROZEN_CANDIDATE_JUDGES,
    FROZEN_PRIMARY_COVERAGES,
    validate_protocol_lock,
)


def test_protocol_lock_freezes_existing_calibrated_protocol() -> None:
    result = validate_protocol_lock("docs/calibrated_protocol_lock_v1.md")
    assert result["valid"] is True
    assert result["candidate_judges"] == FROZEN_CANDIDATE_JUDGES
    assert result["primary_coverages"] == FROZEN_PRIMARY_COVERAGES
    assert "oracle_error_rank" not in result["candidate_judges"]
    assert not [judge for judge in result["candidate_judges"] if judge.startswith("cstr_")]
