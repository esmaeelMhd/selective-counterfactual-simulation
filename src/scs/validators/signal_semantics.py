from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SignalSemantics:
    signal_id: str
    description: str
    risk_orientation: str
    expected_higher_means: str
    requires_trajectory: bool
    requires_bounds: bool
    requires_repair_operator: bool
    requires_ensemble: bool
    system_applicability: dict[str, str]
    cstr_role: str
    twotank_role: str
    cstr_note: str
    universal_refusal_signal: bool
    failure_type_detected: list[str]
    failure_type_not_detected: list[str]
    known_blind_spots: list[str]
    is_universal_candidate: bool
    is_system_specific_candidate: bool


REQUIRED_SIGNAL_IDS = [
    "support_distance",
    "uncertainty_score",
    "disagreement_score",
    "invariant_residual",
    "repair_amount",
]


SIGNAL_SEMANTICS: dict[str, SignalSemantics] = {
    "support_distance": SignalSemantics(
        signal_id="support_distance",
        description="Distance between an action/disturbance trajectory and the model-training support.",
        risk_orientation="higher_is_riskier",
        expected_higher_means="trajectory is farther from the empirical training action/disturbance distribution",
        requires_trajectory=True,
        requires_bounds=False,
        requires_repair_operator=False,
        requires_ensemble=False,
        system_applicability={
            "two_tank": "applicable when action/disturbance support shift is a failure driver",
            "cstr": "applicable when feed/cooling disturbance support shift is a failure driver",
        },
        cstr_role="candidate_context_signal",
        twotank_role="candidate_context_signal",
        cstr_note="Useful for CSTR support shift, but not a universal refusal signal.",
        universal_refusal_signal=False,
        failure_type_detected=[
            "extrapolation outside observed action/disturbance support",
            "intervention magnitude shift",
        ],
        failure_type_not_detected=[
            "within-support dynamics model error",
            "wrong state trajectory when inputs remain in support",
            "state-bound violation by itself",
        ],
        known_blind_spots=[
            "does not inspect predicted state consistency",
            "can be low for hard in-distribution model errors",
        ],
        is_universal_candidate=False,
        is_system_specific_candidate=True,
    ),
    "uncertainty_score": SignalSemantics(
        signal_id="uncertainty_score",
        description="Mean predictive spread from rollout samples for a single simulator model.",
        risk_orientation="higher_is_riskier",
        expected_higher_means="model sample rollouts disagree with each other",
        requires_trajectory=True,
        requires_bounds=False,
        requires_repair_operator=False,
        requires_ensemble=True,
        system_applicability={
            "two_tank": "applicable if model sampling captures epistemic or rollout uncertainty",
            "cstr": "applicable if model sampling captures reaction/temperature uncertainty",
        },
        cstr_role="candidate_model_uncertainty_signal",
        twotank_role="candidate_model_uncertainty_signal",
        cstr_note="Not selected as the strongest current CSTR accepted-region separator.",
        universal_refusal_signal=False,
        failure_type_detected=[
            "model self-uncertainty",
            "unstable sampled rollouts",
        ],
        failure_type_not_detected=[
            "confident but biased dynamics",
            "deterministic model misspecification",
            "state-bound violation when all samples agree",
        ],
        known_blind_spots=[
            "deterministic or under-dispersed models can produce low uncertainty on wrong predictions",
            "sample noise is not a calibrated probability of RMSE failure",
        ],
        is_universal_candidate=False,
        is_system_specific_candidate=True,
    ),
    "disagreement_score": SignalSemantics(
        signal_id="disagreement_score",
        description="Mean trajectory disagreement across model predictions for the same scenario.",
        risk_orientation="higher_is_riskier",
        expected_higher_means="available simulator models predict meaningfully different trajectories",
        requires_trajectory=True,
        requires_bounds=False,
        requires_repair_operator=False,
        requires_ensemble=True,
        system_applicability={
            "two_tank": "applicable when model class diversity exposes intervention-shift failure",
            "cstr": "applicable when model class diversity exposes reaction or thermal dynamics failure",
        },
        cstr_role="candidate_model_disagreement_signal",
        twotank_role="candidate_model_disagreement_signal",
        cstr_note="Mixed CSTR separability in the CSTR weakness audit.",
        universal_refusal_signal=False,
        failure_type_detected=[
            "model-class disagreement",
            "ambiguous rollout behavior under the same intervention",
        ],
        failure_type_not_detected=[
            "shared bias across models",
            "wrong predictions when all models agree",
        ],
        known_blind_spots=[
            "ensembles with common bias can disagree little while all being wrong",
            "large disagreement is not specific to the physical cause of failure",
        ],
        is_universal_candidate=False,
        is_system_specific_candidate=True,
    ),
    "invariant_residual": SignalSemantics(
        signal_id="invariant_residual",
        description="Residual from system-specific physical accounting or dynamics consistency checks.",
        risk_orientation="higher_is_riskier",
        expected_higher_means="predicted trajectory is inconsistent with known system dynamics or accounting",
        requires_trajectory=True,
        requires_bounds=False,
        requires_repair_operator=False,
        requires_ensemble=False,
        system_applicability={
            "two_tank": "inventory accounting residual for external inflow/outflow consistency",
            "cstr": "one-step CSTR concentration/temperature dynamics residual under known equations",
        },
        cstr_role="informative_refusal_signal",
        twotank_role="informative_refusal_signal",
        cstr_note="Best accepted-region signal in CSTR weakness audit.",
        universal_refusal_signal=False,
        failure_type_detected=[
            "dynamics/accounting inconsistency",
            "trajectory that cannot be reproduced by the known system step",
        ],
        failure_type_not_detected=[
            "all possible trajectory errors",
            "errors that satisfy the checked invariant",
            "input support shift by itself",
        ],
        known_blind_spots=[
            "only detects violations of the implemented invariant",
            "does not prove the trajectory is causally correct",
        ],
        is_universal_candidate=False,
        is_system_specific_candidate=True,
    ),
    "repair_amount": SignalSemantics(
        signal_id="repair_amount",
        description="Magnitude of clipping/projection needed to bring a predicted state trajectory back inside system bounds.",
        risk_orientation="higher_is_riskier",
        expected_higher_means="predicted states violate configured physical bounds and require repair",
        requires_trajectory=True,
        requires_bounds=True,
        requires_repair_operator=True,
        requires_ensemble=False,
        system_applicability={
            "two_tank": "applicable to negative inventory and over-capacity predictions",
            "cstr": "applicable only to concentration/temperature predictions outside configured bounds",
        },
        cstr_role="diagnostic_only",
        twotank_role="diagnostic_constraint_signal",
        cstr_note="Correct as a bounds/projection signal but irrelevant for within-bound CSTR dynamic errors.",
        universal_refusal_signal=False,
        failure_type_detected=[
            "state-bound violation",
            "projection or clipping correction amount",
            "negative inventory or over-capacity state for bounded systems",
        ],
        failure_type_not_detected=[
            "within-bound dynamical error",
            "wrong reaction rate",
            "wrong temperature trajectory inside bounds",
            "wrong concentration trajectory inside bounds",
            "support shift without state-bound violation",
        ],
        known_blind_spots=[
            "within-bound dynamic errors can have high RMSE and zero repair",
            "bounded but physically inconsistent trajectories are not detected unless they leave bounds",
            "constant zero repair provides no ranking information for refusal",
        ],
        is_universal_candidate=False,
        is_system_specific_candidate=True,
    ),
}


def signal_semantics_registry() -> dict[str, dict[str, Any]]:
    return {signal_id: asdict(SIGNAL_SEMANTICS[signal_id]) for signal_id in REQUIRED_SIGNAL_IDS}


def render_signal_semantics_markdown(registry: dict[str, dict[str, Any]] | None = None) -> str:
    data = signal_semantics_registry() if registry is None else registry
    lines = [
        "# Signal Semantics Registry",
        "",
        "This registry describes what each refusal signal can and cannot detect. No signal is treated as universally reliable.",
        "",
    ]
    for signal_id in REQUIRED_SIGNAL_IDS:
        item = data[signal_id]
        lines.extend(
            [
                f"## {signal_id}",
                "",
                f"Description: {item['description']}",
                "",
                f"Risk orientation: {item['risk_orientation']}",
                "",
                f"Expected higher means: {item['expected_higher_means']}",
                "",
                "| field | value |",
                "| --- | --- |",
                f"| requires_trajectory | {item['requires_trajectory']} |",
                f"| requires_bounds | {item['requires_bounds']} |",
                f"| requires_repair_operator | {item['requires_repair_operator']} |",
                f"| requires_ensemble | {item['requires_ensemble']} |",
                f"| cstr_role | {item['cstr_role']} |",
                f"| twotank_role | {item['twotank_role']} |",
                f"| universal_refusal_signal | {item['universal_refusal_signal']} |",
                f"| is_universal_candidate | {item['is_universal_candidate']} |",
                f"| is_system_specific_candidate | {item['is_system_specific_candidate']} |",
                "",
                f"CSTR note: {item['cstr_note']}",
                "",
                "System applicability:",
            ]
        )
        for system_id, text in item["system_applicability"].items():
            lines.append(f"- {system_id}: {text}")
        lines.extend(["", "Failure types detected:"])
        lines.extend(f"- {value}" for value in item["failure_type_detected"])
        lines.extend(["", "Failure types not detected:"])
        lines.extend(f"- {value}" for value in item["failure_type_not_detected"])
        lines.extend(["", "Known blind spots:"])
        lines.extend(f"- {value}" for value in item["known_blind_spots"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_signal_semantics_artifacts(
    report_output: str | Path,
    results_output: str | Path = "results/repair_signal_semantics_audit/signal_semantics",
    docs_output: str | Path = "docs/signal_semantics_registry.md",
) -> dict[str, str]:
    registry = signal_semantics_registry()
    results_dir = Path(results_output)
    results_dir.mkdir(parents=True, exist_ok=True)
    registry_path = results_dir / "signal_semantics_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8")
    markdown = render_signal_semantics_markdown(registry)
    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown, encoding="utf-8")
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(markdown, encoding="utf-8")
    return {
        "registry_json": str(registry_path),
        "report": str(report_path),
        "docs": str(docs_path),
    }
