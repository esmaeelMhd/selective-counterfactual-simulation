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
# Working ranges, not frozen. Aerated reactors only; the two anoxic reactors stay at zero.
NOMINAL_SCALE = (0.9, 1.1)
CONTROL_SCALE = (1.5, 1.8)
HISTORY_MINUTES = 15
# Five reactor volumes sum to 5999 m3. At the constant influent flow of 18446 m3/d
# that is 7.8 hours. The working horizon is 8 hours, one hydraulic residence time.
HORIZON_MINUTES = 480
RESERVED_TEST_DAYS = 4
PRETEST_COUNTS = {
    "model_train": 160,
    "judge_calibration_id": 80,
    "judge_calibration_control": 80,
}
PUBLISHED_REACTOR5 = {1: 0.889, 7: 0.491, 8: 10.4, 9: 1.73, 10: 0.688}


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


# PAR1 entries used by bsm2_python PlantPerformance.advanced_quantities: F_P, I_XB, I_XP.
_FP = 0.08
_IXB = 0.08
_IXP = 0.06
EFFLUENT_OFFSET = 5 * REACTOR_WIDTH


def effluent_measurements(states: np.ndarray) -> np.ndarray:
    """Ammonia, total nitrogen, COD, suspended solids, and effluent BOD from a packed state.

    BOD is the library's BOD5e. The benchmark limit of 10 g/m3 applies to that series.
    """
    values = np.asarray(states, dtype=float)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != STATE_DIM:
        raise ValueError(f"states must have shape (time, {STATE_DIM})")
    effluent = values[:, EFFLUENT_OFFSET : EFFLUENT_OFFSET + REACTOR_WIDTH]
    s_nh = effluent[:, 9]
    total_n = (
        effluent[:, 9]
        + effluent[:, 10]
        + effluent[:, 11]
        + _IXB * (effluent[:, 4] + effluent[:, 5])
        + _IXP * (effluent[:, 6] + effluent[:, 2])
        + effluent[:, 8]
    )
    cod = effluent[:, [1, 0, 3, 2, 4, 5, 6]].sum(axis=1)
    tss = 0.75 * (effluent[:, 3] + effluent[:, 6] + effluent[:, 2] + effluent[:, 4] + effluent[:, 5])
    bod5 = 0.25 * (effluent[:, 1] + effluent[:, 3] + (1.0 - _FP) * (effluent[:, 4] + effluent[:, 5]))
    measured = np.column_stack([s_nh, total_n, cod, tss, bod5])
    return measured[0] if single else measured


def effluent_limit_flags(states: np.ndarray, thresholds: dict[str, float]) -> dict[str, bool]:
    """One bit per effluent limit. True when any timestep in the trajectory is above the limit."""
    measured = effluent_measurements(states)
    if measured.ndim == 1:
        measured = measured[None, :]
    return {
        f"{name}_above_limit": bool(np.any(measured[:, index] > float(thresholds[name])))
        for index, name in enumerate(MEASURED_NAMES)
    }


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


def assert_split_allowed(split: str) -> None:
    """Refuse weather files that are not usable and any confirmatory test split."""
    if split.startswith("judge_test"):
        raise RuntimeError("confirmatory BSM1 test split is not generated before the protocol freeze")
    tokens = set(split.split("_"))
    if "rain" in tokens or "storm" in tokens:
        raise RuntimeError(f"{split} is not generated; the packaged weather file is not usable")


def load_dry_influent() -> np.ndarray:
    status = influent_file_status()
    if status["dry"] != "ok":
        raise RuntimeError(f"dry influent is {status['dry']}")
    if status["rain"] == "ok" or status["storm"] == "ok":
        raise RuntimeError("weather status changed; refuse the old rain and storm policy before using those files")
    return np.loadtxt(_package_data_dir() / "dryinfluent.csv", delimiter=",")


def pretest_specs(
    *,
    n_model_train: int = PRETEST_COUNTS["model_train"],
    n_calibration_id: int = PRETEST_COUNTS["judge_calibration_id"],
    n_calibration_control: int = PRETEST_COUNTS["judge_calibration_control"],
    horizon: int = HORIZON_MINUTES,
    history: int = HISTORY_MINUTES,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Non-overlapping dry-weather windows. The dry file repeats. A tail stays unused."""
    for split in PRETEST_COUNTS:
        assert_split_allowed(split)
    if min(n_model_train, n_calibration_id, n_calibration_control, horizon, history) < 1:
        raise ValueError("counts, horizon, and history must be positive")
    span = history + horizon
    counts = [
        ("model_train", "dry_nominal", n_model_train, NOMINAL_SCALE),
        ("judge_calibration_id", "dry_nominal", n_calibration_id, NOMINAL_SCALE),
        ("judge_calibration_control", "dry_control_shift", n_calibration_control, CONTROL_SCALE),
    ]
    slots = list(range(0, sum(count for _, _, count, _ in counts) * span, span))
    rng = np.random.default_rng(seed)
    specs = []
    cursor = 0
    for split, label, count, scale in counts:
        for start in slots[cursor : cursor + count]:
            specs.append(
                {
                    "split": split,
                    "scenario_type": label,
                    "start_minute": start,
                    "kla": _scaled_kla(rng, scale),
                }
            )
        cursor += count
    return specs


def _scaled_kla(rng: np.random.Generator, scale: tuple[float, float]) -> np.ndarray:
    kla = NOMINAL_KLA.copy()
    kla[2:] *= float(rng.uniform(scale[0], scale[1]))
    return kla


_REACTOR_NAMES = ("reactor1", "reactor2", "reactor3", "reactor4", "reactor5")
_OUTPUT_NAMES = ("y_out1", "y_out2", "y_out3", "y_out4", "y_out5", "ys_eff", "ys_out", "y_out5_r", "ys_tss_internal")


def capture_plant(plant: object) -> dict[str, np.ndarray]:
    state = {
        name: np.asarray(getattr(getattr(plant, name), "y0"), dtype=float).copy() for name in _REACTOR_NAMES
    }
    state["settler"] = np.asarray(plant.settler.ys0, dtype=float).copy()
    for name in _OUTPUT_NAMES:
        state[name] = np.asarray(getattr(plant, name), dtype=float).copy()
    return state


def apply_plant(plant: object, state: dict[str, np.ndarray]) -> None:
    for name in _REACTOR_NAMES:
        getattr(plant, name).y0 = np.asarray(state[name], dtype=float).copy()
    plant.settler.ys0 = np.asarray(state["settler"], dtype=float).copy()
    for name in _OUTPUT_NAMES:
        setattr(plant, name, np.asarray(state[name], dtype=float).copy())


def open_loop_steady_state(days: int = 100) -> dict[str, np.ndarray]:
    """Constant-influent open loop. Raises unless reactor 5 matches the published point."""
    from bsm2_python.bsm1_ol import BSM1OL

    n_steps = round(days * 1440)
    plant = _new_plant(_constant_series(n_steps), NOMINAL_KLA)
    for index in range(n_steps):
        plant.step(index, klas=NOMINAL_KLA)
    reactor5 = np.asarray(plant.y_out5, dtype=float)
    for index, target in PUBLISHED_REACTOR5.items():
        if abs(float(reactor5[index]) - target) / target > 0.01:
            raise RuntimeError(f"reactor 5 index {index} is {reactor5[index]}, published target {target}")
    return capture_plant(plant)


def generate_bsm1_pretest_dataset(
    *,
    n_model_train: int = PRETEST_COUNTS["model_train"],
    n_calibration_id: int = PRETEST_COUNTS["judge_calibration_id"],
    n_calibration_control: int = PRETEST_COUNTS["judge_calibration_control"],
    horizon: int = HORIZON_MINUTES,
    history: int = HISTORY_MINUTES,
    seed: int = 0,
    steady_state: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, TrajectoryBatch], dict[str, object]]:
    """Train and calibration splits on dry weather. No rain, storm, or judge-test rows."""
    specs = pretest_specs(
        n_model_train=n_model_train,
        n_calibration_id=n_calibration_id,
        n_calibration_control=n_calibration_control,
        horizon=horizon,
        history=history,
        seed=seed,
    )
    dry = load_dry_influent()
    state = open_loop_steady_state() if steady_state is None else steady_state
    grouped: dict[str, list[TrajectoryBatch]] = {}
    for spec in specs:
        batch = _rollout_from_state(
            dry=dry,
            start_minute=int(spec["start_minute"]),
            history=history,
            horizon=horizon,
            kla=np.asarray(spec["kla"], dtype=float),
            steady_state=state,
            split=str(spec["split"]),
            scenario_type=str(spec["scenario_type"]),
        )
        grouped.setdefault(str(spec["split"]), []).append(batch)
    dataset = {split: _concat_batches(pieces) for split, pieces in grouped.items()}
    manifest = {
        "protocol_frozen": False,
        "confirmatory_test_generated": False,
        "weather": influent_file_status(),
        "splits": sorted(dataset),
        "horizon": horizon,
        "history": history,
        "seed": seed,
        "nominal_scale": list(NOMINAL_SCALE),
        "control_scale": list(CONTROL_SCALE),
        "reserved_test_days": RESERVED_TEST_DAYS,
        "reserved_start_minute": int(max(int(spec["start_minute"]) for spec in specs) + history + horizon),
        "start_minutes": {split: [int(spec["start_minute"]) for spec in specs if spec["split"] == split] for split in dataset},
    }
    return dataset, manifest


def _constant_series(n_steps: int) -> np.ndarray:
    times = np.arange(n_steps + 2) / 1440.0
    influent = np.repeat(PUBLISHED_CONSTANT_INFLUENT[None, :], len(times), axis=0)
    return np.column_stack([times, influent])


def _dry_series(dry: np.ndarray, start_minute: int, n_steps: int) -> np.ndarray:
    rows = []
    for minute in range(start_minute, start_minute + n_steps + 2):
        sample = (minute // 15) % len(dry)
        rows.append(dry[sample, 1:])
    times = np.arange(len(rows)) / 1440.0
    return np.column_stack([times, np.asarray(rows, dtype=float)])


def _new_plant(data: np.ndarray, kla: np.ndarray):
    from bsm2_python.bsm1_ol import BSM1OL

    plant = BSM1OL(data_in=data, timestep=1.0 / 1440.0)
    plant.klas = np.asarray(kla, dtype=float)
    return plant


def _rollout_from_state(
    *,
    dry: np.ndarray,
    start_minute: int,
    history: int,
    horizon: int,
    kla: np.ndarray,
    steady_state: dict[str, np.ndarray],
    split: str,
    scenario_type: str,
) -> TrajectoryBatch:
    assert_split_allowed(split)
    total = history + horizon
    plant = _new_plant(_dry_series(dry, start_minute, total), kla)
    apply_plant(plant, steady_state)
    measured = np.empty((total, len(MEASURED_NAMES)), dtype=float)
    states = np.empty((total + 1, STATE_DIM), dtype=float)
    states[0] = _pack_state(plant)
    for index in range(total):
        plant.step(index, klas=kla)
        states[index + 1] = _pack_state(plant)
        measured[index] = _measured(plant.performance, plant.ys_eff)
    disturbances = plant.y_in[history:total]
    return TrajectoryBatch(
        states=states[history : history + horizon + 1][None, :, :],
        actions=np.repeat(kla[None, :], horizon, axis=0)[None, :, :],
        disturbances=disturbances[None, :, :],
        scenario_type=[scenario_type],
        split=split,
        system_id="bsm1",
        measured_history=measured[:history][None, :, :],
    )


def _concat_batches(pieces: list[TrajectoryBatch]) -> TrajectoryBatch:
    return TrajectoryBatch(
        states=np.concatenate([piece.states for piece in pieces], axis=0),
        actions=np.concatenate([piece.actions for piece in pieces], axis=0),
        disturbances=np.concatenate([piece.disturbances for piece in pieces], axis=0),
        scenario_type=[label for piece in pieces for label in piece.scenario_type],
        split=pieces[0].split,
        system_id="bsm1",
        measured_history=np.concatenate([piece.measured_history for piece in pieces], axis=0),
    )
