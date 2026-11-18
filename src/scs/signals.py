"""Which refusal signals a system can compute."""

from __future__ import annotations

FULL_STATE_SIGNALS = frozenset({"invariant_residual", "repair_amount"})

ALL_SIGNALS = (
    "support_distance",
    "uncertainty_score",
    "disagreement_score",
    "invariant_residual",
    "repair_amount",
)


def available_signals(system_id: str) -> frozenset[str]:
    """Signals that can be computed for ``system_id``.

    BSM1 does not expose a full-state invariant or a repair map, so those two
    signals are absent. Every other system keeps the full set.
    """
    if system_id == "bsm1":
        return frozenset(signal for signal in ALL_SIGNALS if signal not in FULL_STATE_SIGNALS)
    return frozenset(ALL_SIGNALS)


def signal_is_available(system_id: str, signal: str) -> bool:
    return signal in available_signals(system_id)
