from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PLOT_JUDGES = [
    "combined_linear",
    "support_only",
    "uncertainty_only",
    "disagreement_only",
    "random_baseline",
    "oracle_error_rank",
]


def plot_risk_coverage(
    risk_coverage: pd.DataFrame,
    output_path: str | Path,
    judge_ids: list[str] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    judge_ids = judge_ids or DEFAULT_PLOT_JUDGES
    plot_df = (
        risk_coverage[risk_coverage["judge_id"].isin(judge_ids)]
        .groupby(["judge_id", "coverage"], as_index=False)["false_accept_rate"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for judge_id in judge_ids:
        judge_df = plot_df[plot_df["judge_id"] == judge_id].sort_values("coverage")
        if judge_df.empty:
            continue
        ax.plot(
            judge_df["coverage"],
            judge_df["false_accept_rate"],
            marker="o",
            linewidth=1.8,
            label=judge_id,
        )
    ax.set_xlabel("coverage")
    ax.set_ylabel("false_accept_rate")
    ax.set_title("Risk-Coverage Curve")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)

