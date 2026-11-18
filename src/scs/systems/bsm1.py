"""BSM1 plant adapter.

The dynamics come from ``bsm2-python`` (BSD 3-Clause, FAU-EVT). A 100-day
open-loop run with the published constant influent and the library's default
aeration reached the published reactor-5 steady state (SS 0.889, SO 0.491,
SNO 10.4, SNH 1.73, SND 0.688) to within 0.2 percent. One such run took 77
seconds. The packaged rain file does not parse, and the package has no storm
file, so those weather splits are not generated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scs.systems.base import TrajectoryBatch

MEASURED_NAMES = ("s_nh", "total_n", "cod", "tss", "bod5")
EFFLUENT_LIMITS = {
    "s_nh": 4.0,
    "total_n": 18.0,
    "cod": 100.0,
    "tss": 30.0,
    "bod5": 10.0,
}
NOMINAL_KLA = np.array([0.0, 0.0, 240.0, 240.0, 84.0], dtype=float)
PUBLISHED_CONSTANT_INFLUENT = np.array(
    [30, 69.5, 51.2, 202.32, 28.17, 0, 0, 0, 0, 31.56, 6.95, 10.59, 7, 211.2675, 18446, 15, 0, 0, 0, 0, 0],
    dtype=float,
)
REACTOR_WIDTH = 21
SETTLER_SOLIDS = 10
STATE_DIM = 5 * REACTOR_WIDTH + REACTOR_WIDTH + SETTLER_SOLIDS


def _package_data_dir() -> Path:
    try:
        import bsm2_python
    except ImportError as exc:
        raise ImportError("BSM1 needs the optional extra: pip install bsm2-python") from exc
    return Path(bsm2_python.__file__).parent / "data"


def influent_file_status() -> dict[str, str]:
    """Say whether each weather file can be used as a BSM1 disturbance."""
    data_dir = _package_data_dir()
    return {
        "dry": _status(data_dir / "dryinfluent.csv"),
        "rain": _status(data_dir / "raininfluent.csv"),
        "storm": _status(data_dir / "storminfluent.csv"),
    }


def _status(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        table = np.loadtxt(path, delimiter=",")
    except (ValueError, OSError):
        return "unreadable"
    if table.ndim != 2 or table.shape[1] < 16:
        return "bad shape"
    flow = table[:, 15]
    if not np.isfinite(flow).all() or float(np.median(flow)) < 1000.0:
        return "implausible flow"
    return "ok"


def _measured(performance: object, effluent: np.ndarray) -> np.ndarray:
    quantities = np.asarray(
        performance.advanced_quantities(effluent, ("totalN", "COD", "X_TSS", "BOD5e")),
        dtype=float,
    ).reshape(-1)
    return np.array([effluent[9], quantities[0], quantities[1], quantities[2], quantities[3]], dtype=float)


def _pack_state(plant: object) -> np.ndarray:
    parts = [getattr(plant, name) for name in ("y_out1", "y_out2", "y_out3", "y_out4", "y_out5", "ys_eff")]
    solids = getattr(plant, "ys_tss_internal", None)
    if solids is None:
        solids = np.zeros(SETTLER_SOLIDS, dtype=float)
    return np.concatenate(
        [*(np.asarray(part, dtype=float).reshape(-1) for part in parts), np.asarray(solids, dtype=float).reshape(-1)]
    )


def rollout_scenario(
    *,
    history: int = 4,
    horizon: int = 8,
    kla: np.ndarray | None = None,
    seed_label: str = "constant_influent",
) -> TrajectoryBatch:
    """Roll one short open-loop scenario and attach the measured history.

    The history is the effluent window just before the predicted horizon.
    Existing models can ignore it and still be called with the full state.
    """
    from bsm2_python.bsm1_ol import BSM1OL

    if history < 1 or horizon < 1:
        raise ValueError("history and horizon must be positive")
    aeration = NOMINAL_KLA if kla is None else np.asarray(kla, dtype=float)
    if aeration.shape != (5,):
        raise ValueError("kla must have shape (5,)")
    total = history + horizon
    times = np.arange(total + 2) / 1440.0
    influent = np.repeat(PUBLISHED_CONSTANT_INFLUENT[None, :], len(times), axis=0)
    data = np.column_stack([times, influent])
    plant = BSM1OL(data_in=data, timestep=1.0 / 1440.0)
    measured = np.empty((total, len(MEASURED_NAMES)), dtype=float)
    states = np.empty((total + 1, STATE_DIM), dtype=float)
    states[0] = _pack_state(plant)
    for index in range(total):
        plant.step(index, klas=aeration)
        states[index + 1] = _pack_state(plant)
        measured[index] = _measured(plant.performance, plant.ys_eff)
    predicted = states[history : history + horizon + 1]
    disturbances = np.repeat(PUBLISHED_CONSTANT_INFLUENT[None, :], horizon, axis=0)
    actions = np.repeat(aeration[None, :], horizon, axis=0)
    return TrajectoryBatch(
        states=predicted[None, :, :],
        actions=actions[None, :, :],
        disturbances=disturbances[None, :, :],
        scenario_type=[seed_label],
        split="model_train",
        system_id="bsm1",
        measured_history=measured[:history][None, :, :],
    )
