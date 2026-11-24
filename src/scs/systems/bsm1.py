"""BSM1 plant adapter.

The dynamics come from ``bsm2-python`` (BSD 3-Clause, FAU-EVT). A 100-day
open-loop run with the published constant influent and the library's default
aeration reached the published reactor-5 steady state (SS 0.889, SO 0.491,
SNO 10.4, SNH 1.73, SND 0.688) to within 0.2 percent. One such run took 77
seconds.

Weather. The packaged dry file lists 14 days, but days 7 to 13 are an exact copy
of days 0 to 6, so it holds 7 unique days. The loader checks that and returns
those 7. The packaged rain file has three malformed flow cells and its flow is
in thousands of m3/d. ``load_rain_influent`` repairs both and says what it
changed. Its only rain event is on days 8 to 10, which belong to the confirmatory
test, so no rain split is generated before the freeze. The package has no storm
file, so there is no storm split.

Splits read disjoint blocks of the 7 unique dry days (``SPLIT_DAYS``), the last
1.5 days are reserved for the confirmatory test, and every window starts from
the plant state at its own minute of one continuous dry run.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np

from scs.parallel import configure_worker_threads, resolve_workers
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
# Open-loop hydraulics from asm1init_bsm1. Internal recycle is three times the
# nominal influent, return sludge equals that influent, and waste flow is 385 m3/d.
NOMINAL_INFLUENT_FLOW = 18446.0
INTERNAL_RECYCLE = 3.0 * NOMINAL_INFLUENT_FLOW
RETURN_SLUDGE = NOMINAL_INFLUENT_FLOW
WASTE_FLOW = 385.0
FLOW_INDEX = 14
STATE_DIM = 5 * REACTOR_WIDTH + REACTOR_WIDTH + SETTLER_SOLIDS
# Working ranges, not frozen. Aerated reactors only; the two anoxic reactors stay at zero.
NOMINAL_SCALE = (0.9, 1.1)
CONTROL_SCALE = (1.5, 1.8)
HISTORY_MINUTES = 15
# Five reactor volumes sum to 5999 m3. At the constant influent flow of 18446 m3/d
# that is 7.8 hours. The working horizon is 8 hours, one hydraulic residence time.
HORIZON_MINUTES = 480
# The dry file lists 14 days at 15-minute samples, and days 7 to 13 repeat days 0 to 6.
# That leaves 7 unique days. Each split owns its own block of them, so no two splits
# share an influent minute. The last 1.5 days are kept for the confirmatory test and are
# not read before the freeze. Windows in one split may overlap each other; windows in
# different splits never do. Windows inside one split share influent days, so the
# number of independent influent windows per split is far below the window count.
DRY_DAYS = 7
DRY_MINUTES = DRY_DAYS * 1440
SPLIT_DAYS = {
    "model_train": (0.0, 3.5),
    "judge_calibration_id": (3.5, 4.5),
    "judge_calibration_control": (4.5, 5.5),
}
RESERVED_DAYS = (5.5, float(DRY_DAYS))
RESERVED_TEST_DAYS = RESERVED_DAYS[1] - RESERVED_DAYS[0]
# Initial states come from one continuous dry run at nominal aeration: two full
# dry cycles of warm-up after the constant-influent steady state, then a pass in
# which the plant state is read at each window start.
WARMUP_MINUTES = 2 * DRY_MINUTES
# Per-window influent perturbation. Each window gets one flow factor and one load factor,
# held constant over the window and drawn within +-10 percent, which is about the day-to-day
# spread of daily mean flow in the dry file. Concentrations and total suspended solids
# scale with the load factor. Alkalinity and temperature are not loads and stay as they are.
# The same scheme applies to every split, including the confirmatory test.
INFLUENT_SCALE = (0.9, 1.1)
LOAD_COLUMNS = tuple(index for index in range(14) if index != 12)
# The protocol's rain event, days 8–10. Not read until the freeze allows a confirmatory split.
RAIN_EVENT_DAYS = (8.0, 11.0)
RAIN_FLOW_UNIT = 1000.0
RAIN_FLOW_COLUMN = 15
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
    """Say whether each weather file can be used as a BSM1 disturbance.

    ``repaired`` means the file needs the fixes in ``load_rain_influent``. ``ok``
    for rain or storm means the package changed, and the policy must be revisited.
    """
    data_dir = _package_data_dir()
    return {
        "dry": _status(data_dir / "dryinfluent.csv"),
        "rain": _rain_status(),
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


def _rain_status() -> str:
    path = _package_data_dir() / "raininfluent.csv"
    if not path.exists():
        return "missing"
    try:
        table, repairs = load_rain_influent()
    except (RuntimeError, ValueError):
        return "unreadable"
    if table.shape != (1344, 22):
        return "bad shape"
    return "repaired" if repairs else "ok"


def load_rain_influent() -> tuple[np.ndarray, list[dict[str, object]]]:
    """The packaged rain file with its malformed flow cells repaired and flow in m3/d.

    Three flow cells carry a second decimal point, such as ``30.044.50``. Removing it
    gives a value on the straight line between the readable cells around it, which is
    checked here. The flow column is in thousands of m3/d and is multiplied by 1000 so
    it matches the dry file. Any other malformed cell raises. This only reads influent
    inputs. No rain split is generated before the protocol freeze.
    """
    path = _package_data_dir() / "raininfluent.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    width = 22
    table = np.empty((len(rows), width), dtype=float)
    repairs: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        if len(row) != width:
            raise RuntimeError(f"rain row {row_index} has {len(row)} cells, expected {width}")
        for column, cell in enumerate(row):
            if cell.count(".") > 1:
                if column != RAIN_FLOW_COLUMN or cell.count(".") != 2:
                    raise RuntimeError(f"unexpected malformed rain cell {cell!r} at row {row_index}, column {column}")
                second = cell.index(".", cell.index(".") + 1)
                value = float(cell[:second] + cell[second + 1 :])
                repairs.append({"row": row_index, "column": column, "raw": cell, "value": value})
            else:
                value = float(cell)
            table[row_index, column] = value
    _check_rain_repairs(table, repairs)
    table[:, RAIN_FLOW_COLUMN] *= RAIN_FLOW_UNIT
    return table, repairs


def _check_rain_repairs(table: np.ndarray, repairs: list[dict[str, object]]) -> None:
    """Each repaired run of cells must follow the line between its readable neighbours."""
    rows = sorted(int(repair["row"]) for repair in repairs)
    flow = table[:, RAIN_FLOW_COLUMN]
    index = 0
    while index < len(rows):
        run = [rows[index]]
        while index + 1 < len(rows) and rows[index + 1] <= run[-1] + 1:
            index += 1
            run.append(rows[index])
        index += 1
        before, after = run[0] - 1, run[-1] + 1
        while before in rows:
            before -= 1
        while after in rows:
            after += 1
        if before < 0 or after >= len(flow):
            raise RuntimeError(f"repaired rain flow at row {run[0]} lies on a file boundary without readable neighbours")
        for row in run:
            line = flow[before] + (flow[after] - flow[before]) * (row - before) / (after - before)
            if abs(flow[row] - line) > 0.05 * max(abs(flow[before] - flow[after]), 1e-9) + 0.02:
                raise RuntimeError(f"repaired rain flow at row {row} is {flow[row]}, off the line value {line}")


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


def apply_hydraulic_flows(states: np.ndarray, disturbances: np.ndarray) -> np.ndarray:
    """Set reactor and effluent flow from the influent flow.

    Reactor flow is influent plus the fixed internal recycle and return sludge.
    Effluent flow is influent minus the fixed waste flow.
    """
    updated = np.array(states, dtype=float, copy=True)
    influent_flow = np.asarray(disturbances, dtype=float)[..., FLOW_INDEX]
    reactor_flow = influent_flow + INTERNAL_RECYCLE + RETURN_SLUDGE
    effluent_flow = influent_flow - WASTE_FLOW
    for block in range(5):
        updated[..., FLOW_INDEX + REACTOR_WIDTH * block] = reactor_flow
    updated[..., FLOW_INDEX + REACTOR_WIDTH * 5] = effluent_flow
    return updated


def hydraulic_projection(disturbances: np.ndarray):
    """The ``project`` hook that assigns the flow channels from these disturbances."""
    disturbances = np.asarray(disturbances, dtype=float)

    def project(step: int, states: np.ndarray) -> np.ndarray:
        return apply_hydraulic_flows(states, disturbances[:, step])

    return project


def pretest_overlap_gap(split: str = "model_train", horizon: int = HORIZON_MINUTES, history: int = HISTORY_MINUTES) -> int:
    """How many neighbouring windows of ``split`` overlap one window in time.

    The windows are spread evenly over the split's block of days, so a window
    overlaps every neighbour closer than one window length.
    """
    first, end = split_minutes(split)
    count = PRETEST_COUNTS[split]
    span = history + horizon
    stride = (end - first - span) / max(count - 1, 1)
    return int(np.ceil(span / stride)) if stride > 0 else count


def rollout_with_known_hydraulics(model, initial_states: np.ndarray, actions: np.ndarray, disturbances: np.ndarray) -> np.ndarray:
    """Roll out the model with the flow channels assigned from the influent at every step.

    The assignment runs inside the model's own rollout through ``project``, so a
    recurrent model keeps its hidden state from one step to the next.
    """
    return model.predict_rollouts(
        initial_states, actions, disturbances, project=hydraulic_projection(disturbances)
    )


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
        step_kla = NOMINAL_KLA if index < history else aeration
        plant.step(index, klas=step_kla)
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


class BSM1System:
    """Identity for the packaged plant. Rollouts come from the generator, not this stepper.

    Invariant and repair need the full state and are not available here. Calling them raises
    so a missing-signal judge cannot silently receive a stand-in score.
    """

    system_id = "bsm1"
    state_dim = STATE_DIM
    action_dim = 5
    disturbance_dim = REACTOR_WIDTH

    def reset(self, seed: int | None = None) -> np.ndarray:
        raise RuntimeError("BSM1 trajectories come from the packaged plant, not this stepper")

    def step(self, state: np.ndarray, action: np.ndarray, disturbance: np.ndarray, dt: float) -> np.ndarray:
        raise RuntimeError("BSM1 trajectories come from the packaged plant, not this stepper")

    def rollout(self, initial_state: np.ndarray, actions: np.ndarray, disturbances: np.ndarray, dt: float) -> np.ndarray:
        raise RuntimeError("BSM1 trajectories come from the packaged plant, not this stepper")

    def invariant_residual(self, states: np.ndarray, actions: np.ndarray, disturbances: np.ndarray, dt: float) -> np.ndarray:
        raise RuntimeError("invariant_residual is not available on BSM1")

    def repair_amount(self, raw_states: np.ndarray, repaired_states: np.ndarray) -> float:
        raise RuntimeError("repair_amount is not available on BSM1")


def confirmatory_window_capacity() -> dict[str, int]:
    """Non-overlapping 8-hour windows that fit in the weather the protocol already names."""
    span = HISTORY_MINUTES + HORIZON_MINUTES
    reserved = reserved_minutes()[1] - reserved_minutes()[0]
    rain_minutes = round((RAIN_EVENT_DAYS[1] - RAIN_EVENT_DAYS[0]) * 1440)
    return {
        "span_minutes": span,
        "dry_minutes": reserved,
        "dry_nonoverlapping": reserved // span,
        "rain_minutes": rain_minutes,
        "rain_nonoverlapping": rain_minutes // span,
    }


def assert_split_allowed(split: str) -> None:
    """Refuse the confirmatory test splits and weather that is not available before the freeze."""
    if split.startswith("judge_test"):
        raise RuntimeError("confirmatory BSM1 test split is not generated before the protocol freeze")
    tokens = set(split.split("_"))
    if "rain" in tokens:
        raise RuntimeError(f"{split} is not generated; the rain event is reserved for the confirmatory test")
    if "storm" in tokens:
        raise RuntimeError(f"{split} is not generated; the package has no storm file")


def load_dry_influent() -> np.ndarray:
    """The 7 unique dry days, as 672 rows of 15-minute samples."""
    status = influent_file_status()
    if status["dry"] != "ok":
        raise RuntimeError(f"dry influent is {status['dry']}")
    if status["rain"] == "ok" or status["storm"] == "ok":
        raise RuntimeError("weather status changed; revisit the rain and storm policy before using those files")
    table = np.loadtxt(_package_data_dir() / "dryinfluent.csv", delimiter=",")
    half = len(table) // 2
    if len(table) != 1344 or not np.array_equal(table[:half, 1:], table[half:, 1:]):
        raise RuntimeError("the dry file no longer repeats after 7 days; revisit the split design")
    return table[:half]


def pretest_specs(
    *,
    n_model_train: int = PRETEST_COUNTS["model_train"],
    n_calibration_id: int = PRETEST_COUNTS["judge_calibration_id"],
    n_calibration_control: int = PRETEST_COUNTS["judge_calibration_control"],
    horizon: int = HORIZON_MINUTES,
    history: int = HISTORY_MINUTES,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Dry-weather windows inside each split's own block of days. The reserved tail is not used."""
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
    rng = np.random.default_rng(seed)
    specs = []
    for split, label, count, scale in counts:
        first, end = split_minutes(split)
        last = end - span
        if last < first:
            raise ValueError(f"{split} block is shorter than one window of {span} minutes")
        for start in np.linspace(first, last, count).round().astype(int):
            specs.append(
                {
                    "split": split,
                    "scenario_type": label,
                    "start_minute": int(start),
                    "kla": _scaled_kla(rng, scale),
                    "flow_scale": float(rng.uniform(*INFLUENT_SCALE)),
                    "load_scale": float(rng.uniform(*INFLUENT_SCALE)),
                }
            )
    return specs


def split_minutes(split: str) -> tuple[int, int]:
    """Half-open range of dry-file minutes that a pretest split may read."""
    first, end = SPLIT_DAYS[split]
    return round(first * 1440), round(end * 1440)


def reserved_minutes() -> tuple[int, int]:
    """Half-open range of dry-file minutes kept for the confirmatory test."""
    return round(RESERVED_DAYS[0] * 1440), round(RESERVED_DAYS[1] * 1440)


def dry_start_states(
    dry: np.ndarray,
    start_minutes: list[int],
    steady_state: dict[str, np.ndarray],
    warmup_minutes: int = WARMUP_MINUTES,
) -> dict[int, dict[str, np.ndarray]]:
    """Plant state at each start minute of one continuous dry run at nominal aeration.

    The run starts from ``steady_state``, warms up for ``warmup_minutes``, and
    then reads the plant just before the step at each requested dry-file minute.
    """
    wanted = sorted({int(minute) for minute in start_minutes})
    if not wanted:
        return {}
    if wanted[0] < 0 or wanted[-1] >= DRY_MINUTES:
        raise ValueError("start minutes must lie inside one dry cycle")
    if warmup_minutes % DRY_MINUTES:
        raise ValueError("warm-up must be a whole number of dry cycles so minutes stay aligned")
    n_steps = warmup_minutes + wanted[-1]
    plant = _new_plant(_dry_series(dry, 0, n_steps), NOMINAL_KLA)
    apply_plant(plant, steady_state)
    captured: dict[int, dict[str, np.ndarray]] = {}
    targets = {warmup_minutes + minute: minute for minute in wanted}
    for index in range(n_steps + 1):
        if index in targets:
            captured[targets[index]] = capture_plant(plant)
        if index < n_steps:
            plant.step(index, klas=NOMINAL_KLA)
    return captured


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


_WORKER_TABLES: dict[str, np.ndarray] = {}
_WORKER_START_STATES: dict[str, dict[int, dict[str, np.ndarray]]] = {}


def _init_bsm1_worker(
    tables: dict[str, np.ndarray],
    start_states: dict[str, dict[int, dict[str, np.ndarray]]],
) -> None:
    global _WORKER_TABLES, _WORKER_START_STATES
    configure_worker_threads()
    _WORKER_TABLES = tables
    _WORKER_START_STATES = start_states


def _run_bsm1_window_task(
    task: tuple[str, int, int, int, np.ndarray, str, str, float, float, bool],
) -> TrajectoryBatch:
    weather, start_minute, history, horizon, kla, split, scenario_type, flow_scale, load_scale, check_split = task
    return _rollout_from_state(
        dry=_WORKER_TABLES[weather],
        start_minute=start_minute,
        history=history,
        horizon=horizon,
        kla=kla,
        initial_state=_WORKER_START_STATES[weather][start_minute],
        split=split,
        scenario_type=scenario_type,
        flow_scale=flow_scale,
        load_scale=load_scale,
        check_split=check_split,
    )


def _execute_bsm1_window_tasks(
    tasks: list[tuple[str, int, int, int, np.ndarray, str, str, float, float, bool]],
    tables: dict[str, np.ndarray],
    start_states: dict[str, dict[int, dict[str, np.ndarray]]],
    workers: int,
) -> list[TrajectoryBatch]:
    effective_workers = resolve_workers(workers)
    if effective_workers <= 1 or len(tasks) <= 1:
        return [
            _rollout_from_state(
                dry=tables[weather],
                start_minute=start_minute,
                history=history,
                horizon=horizon,
                kla=kla,
                initial_state=start_states[weather][start_minute],
                split=split,
                scenario_type=scenario_type,
                flow_scale=flow_scale,
                load_scale=load_scale,
                check_split=check_split,
            )
            for weather, start_minute, history, horizon, kla, split, scenario_type, flow_scale, load_scale, check_split in tasks
        ]
    n_procs = min(effective_workers, len(tasks))
    chunksize = max(1, len(tasks) // (n_procs * 4))
    with ProcessPoolExecutor(
        max_workers=n_procs,
        initializer=_init_bsm1_worker,
        initargs=(tables, start_states),
    ) as pool:
        return list(pool.map(_run_bsm1_window_task, tasks, chunksize=chunksize))


def generate_bsm1_pretest_dataset(
    *,
    n_model_train: int = PRETEST_COUNTS["model_train"],
    n_calibration_id: int = PRETEST_COUNTS["judge_calibration_id"],
    n_calibration_control: int = PRETEST_COUNTS["judge_calibration_control"],
    horizon: int = HORIZON_MINUTES,
    history: int = HISTORY_MINUTES,
    seed: int = 0,
    steady_state: dict[str, np.ndarray] | None = None,
    start_states: dict[int, dict[str, np.ndarray]] | None = None,
    workers: int = 1,
) -> tuple[dict[str, TrajectoryBatch], dict[str, object]]:
    """Train and calibration splits on dry weather. No rain, storm, or judge-test rows.

    Each window starts from the plant state at that dry-file minute of one
    continuous nominal run. ``start_states`` skips that run, for tests.
    """
    effective_workers = resolve_workers(workers)
    specs = pretest_specs(
        n_model_train=n_model_train,
        n_calibration_id=n_calibration_id,
        n_calibration_control=n_calibration_control,
        horizon=horizon,
        history=history,
        seed=seed,
    )
    dry = load_dry_influent()
    if len(dry) * 15 != DRY_MINUTES:
        raise RuntimeError(f"dry influent covers {len(dry) * 15} minutes, expected {DRY_MINUTES}")
    reserved_first, _reserved_end = reserved_minutes()
    for spec in specs:
        if int(spec["start_minute"]) + history + horizon > reserved_first:
            raise RuntimeError("a pretest window reaches the reserved confirmatory days")
    starts = [int(spec["start_minute"]) for spec in specs]
    if start_states is None:
        state = open_loop_steady_state() if steady_state is None else steady_state
        start_states = dry_start_states(dry, starts, state)
    missing = sorted(set(starts) - set(start_states))
    if missing:
        raise ValueError(f"no plant state for start minutes {missing[:5]}")
    tasks = [
        (
            "dry",
            int(spec["start_minute"]),
            history,
            horizon,
            np.asarray(spec["kla"], dtype=float),
            str(spec["split"]),
            str(spec["scenario_type"]),
            float(spec["flow_scale"]),
            float(spec["load_scale"]),
            True,
        )
        for spec in specs
    ]
    batches = _execute_bsm1_window_tasks(
        tasks,
        tables={"dry": dry},
        start_states={"dry": start_states},
        workers=effective_workers,
    )
    grouped: dict[str, list[TrajectoryBatch]] = {}
    for spec, batch in zip(specs, batches):
        grouped.setdefault(str(spec["split"]), []).append(batch)
    dataset = {split: _concat_batches(pieces) for split, pieces in grouped.items()}
    try:
        import bsm2_python

        bsm2_version = str(getattr(bsm2_python, "__version__", "unknown"))
    except ImportError:
        bsm2_version = "missing"
    manifest = {
        "protocol_frozen": False,
        "confirmatory_test_generated": False,
        "bsm2_python_version": bsm2_version,
        "weather": influent_file_status(),
        "splits": sorted(dataset),
        "horizon": horizon,
        "history": history,
        "seed": seed,
        "nominal_scale": list(NOMINAL_SCALE),
        "control_scale": list(CONTROL_SCALE),
        "unique_dry_days": DRY_DAYS,
        "influent_scale": list(INFLUENT_SCALE),
        "influent_scales": {
            split: [[round(float(spec["flow_scale"]), 4), round(float(spec["load_scale"]), 4)] for spec in specs if spec["split"] == split]
            for split in dataset
        },
        "independent_windows": {
            split: int((split_minutes(split)[1] - split_minutes(split)[0]) // (history + horizon)) for split in dataset
        },
        "rain_repairs": [
            {key: repair[key] for key in ("row", "raw", "value")} for repair in load_rain_influent()[1]
        ],
        "dry_minutes": DRY_MINUTES,
        "split_minutes": {split: list(split_minutes(split)) for split in dataset},
        "reserved_test_days": RESERVED_TEST_DAYS,
        "reserved_minutes": list(reserved_minutes()),
        "initial_state": f"continuous nominal dry run after {WARMUP_MINUTES} minutes of warm-up",
        "start_minutes": {split: [int(spec["start_minute"]) for spec in specs if spec["split"] == split] for split in dataset},
    }
    return dataset, manifest


_STEADY_STATE: dict[str, np.ndarray] | None = None
_START_STATES: dict[tuple[int, ...], dict[int, dict[str, np.ndarray]]] = {}


def _cached_start_states(dry: np.ndarray, starts: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Plant states at the given dry-file minutes. One continuous run per set of minutes and process.

    The states do not depend on the seed, so a stage that scores several seeds pays for the
    steady-state run and the warm-up once.
    """
    global _STEADY_STATE
    key = tuple(sorted(set(int(start) for start in starts)))
    if key not in _START_STATES:
        if _STEADY_STATE is None:
            _STEADY_STATE = open_loop_steady_state()
        _START_STATES[key] = dry_start_states(dry, list(key), _STEADY_STATE)
    return _START_STATES[key]


def _load_or_build_splits(
    cache_dir: Path | None,
    build,
    expected_meta: dict[str, object],
) -> dict[str, TrajectoryBatch]:
    if cache_dir is not None and cache_dir.is_dir():
        meta_path = cache_dir / "cache_meta.json"
        if meta_path.is_file():
            try:
                saved = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                saved = None
            splits = saved.get("splits") if isinstance(saved, dict) else None
            if (
                isinstance(saved, dict)
                and saved.get("params") == expected_meta
                and isinstance(splits, list)
                and splits
                and all((cache_dir / f"{split}.npz").is_file() for split in splits)
            ):
                from scs.data.schemas import load_dataset

                return load_dataset(cache_dir)
    dataset = build()
    if cache_dir is not None:
        from scs.data.schemas import save_dataset

        cache_dir.mkdir(parents=True, exist_ok=True)
        for stale in cache_dir.glob("*.npz"):
            stale.unlink()
        meta_path = cache_dir / "cache_meta.json"
        if meta_path.exists():
            meta_path.unlink()
        save_dataset(dataset, cache_dir)
        payload = {"params": expected_meta, "splits": sorted(dataset)}
        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


def generate_bsm1_pretest_stage_dataset(
    config: dict[str, object], seed: int, workers: int = 1
) -> dict[str, TrajectoryBatch]:
    """Train and calibration BSM1 splits for one seed, without touching confirmatory days."""
    effective_workers = resolve_workers(workers)
    data = config["data"]
    counts = dict(
        n_model_train=int(data["n_model_train"]),
        n_calibration_id=int(data["n_calibration_id"]),
        n_calibration_control=int(data["n_calibration_ood"]),
        horizon=int(data["horizon"]),
        history=int(data.get("history_minutes", HISTORY_MINUTES)),
    )
    cache_root = data.get("cache_dir")
    cache_dir = Path(str(cache_root)) / f"seed_{int(seed)}" / "pretest" if cache_root else None
    expected_meta: dict[str, object] = {
        "kind": "pretest",
        "seed": int(seed),
        **counts,
        "nominal_scale": list(NOMINAL_SCALE),
        "control_scale": list(CONTROL_SCALE),
        "influent_scale": list(INFLUENT_SCALE),
    }

    def _build() -> dict[str, TrajectoryBatch]:
        dry = load_dry_influent()
        starts = [int(spec["start_minute"]) for spec in pretest_specs(seed=seed, **counts)]
        dataset, _manifest = generate_bsm1_pretest_dataset(
            seed=seed,
            start_states=_cached_start_states(dry, starts),
            workers=effective_workers,
            **counts,
        )
        return dataset

    return _load_or_build_splits(cache_dir, _build, expected_meta)


def generate_bsm1_test_stage_dataset(
    config: dict[str, object], seed: int, workers: int = 1
) -> dict[str, TrajectoryBatch]:
    """Confirmatory or rehearsal BSM1 test splits for one seed."""
    effective_workers = resolve_workers(workers)
    data = config["data"]
    stage = str(config.get("stage", "A"))
    if stage not in ("B", "B-rehearsal"):
        return {}
    rehearsal = stage == "B-rehearsal"
    if not rehearsal:
        from scs.experiments.v3 import protocol_is_frozen

        if not protocol_is_frozen():
            raise RuntimeError(
                "the confirmatory dry and rain windows are generated only after the freeze, "
                "and docs/v3/protocol.md is still a draft"
            )
    scale = [float(value) for value in (data.get("test_control_scale") or CONTROL_SCALE)]
    cache_root = data.get("cache_dir")
    cache_dir = Path(str(cache_root)) / f"seed_{int(seed)}" / "test" if cache_root else None
    expected_meta: dict[str, object] = {
        "kind": "test",
        "stage": stage,
        "rehearsal": rehearsal,
        "seed": int(seed),
        "n_per_split": int(data["n_test_per_split"]),
        "horizon": int(data["horizon"]),
        "history": int(data.get("history_minutes", HISTORY_MINUTES)),
        "control_scale": scale,
        "nominal_scale": list(NOMINAL_SCALE),
        "influent_scale": list(INFLUENT_SCALE),
    }

    def _build() -> dict[str, TrajectoryBatch]:
        return generate_bsm1_confirmatory_dataset(
            n_per_split=int(data["n_test_per_split"]),
            horizon=int(data["horizon"]),
            history=int(data.get("history_minutes", HISTORY_MINUTES)),
            seed=seed,
            control_scale=scale,
            rehearsal=rehearsal,
            workers=effective_workers,
        )

    return _load_or_build_splits(cache_dir, _build, expected_meta)


def generate_bsm1_stage_dataset(
    config: dict[str, object], seed: int, workers: int = 1
) -> dict[str, TrajectoryBatch]:
    """The splits a v3 stage scores for BSM1, from a stage config.

    ``model_train``, ``judge_calibration_id`` and ``judge_calibration_control`` come from the
    pretest generator. Stage ``B`` adds the four confirmatory test splits, which the generator
    refuses to read from the reserved days until the protocol is frozen. Stage ``B-rehearsal``
    adds the same four splits cut from days that are not reserved, so every scoring step runs
    before the freeze. Those rows are not evidence.
    """
    dataset = generate_bsm1_pretest_stage_dataset(config, seed, workers=workers)
    dataset.update(generate_bsm1_test_stage_dataset(config, seed, workers=workers))
    return dataset


# The four confirmatory test splits. Control windows reuse the influent days and the flow and
# load factors of the nominal window with the same index, so the pair differs in aeration only.
CONFIRMATORY_SPLITS = {
    "judge_test_dry_holdout": ("dry", False),
    "judge_test_dry_control": ("dry", True),
    "judge_test_rain": ("rain", False),
    "judge_test_rain_control": ("rain", True),
}
_RAIN_START_STATES: dict[tuple[int, ...], dict[int, dict[str, np.ndarray]]] = {}


def _rain_event_minutes() -> tuple[int, int]:
    return round(RAIN_EVENT_DAYS[0] * 1440), round(RAIN_EVENT_DAYS[1] * 1440)


def confirmatory_blocks(rehearsal: bool = False) -> dict[str, tuple[int, int]]:
    """Half-open minute range each weather's confirmatory windows are cut from.

    The real blocks are the reserved dry days and the rain event. The rehearsal blocks are
    the calibration days of the dry file and the same days of the rain file, which hold no
    rain event, so nothing reserved is read.
    """
    if rehearsal:
        block = (split_minutes("judge_calibration_id")[0], split_minutes("judge_calibration_control")[1])
        return {"dry": block, "rain": block}
    return {"dry": reserved_minutes(), "rain": _rain_event_minutes()}


def _reads_reserved(weather: str, first: int, end: int) -> bool:
    reserved = reserved_minutes() if weather == "dry" else _rain_event_minutes()
    return first < reserved[1] and end > reserved[0]


def rain_start_states(
    dry: np.ndarray,
    rain: np.ndarray,
    start_minutes: list[int],
    steady_state: dict[str, np.ndarray],
    warmup_minutes: int = WARMUP_MINUTES,
) -> dict[int, dict[str, np.ndarray]]:
    """Plant state at each start minute of the rain file, after the same dry warm-up.

    The run warms up on the dry file for whole cycles, then follows the rain file from its
    first row. The rain file's first week carries the dry flows, so the hand-over is smooth.
    """
    wanted = sorted({int(minute) for minute in start_minutes})
    if not wanted:
        return {}
    if wanted[0] < 0 or wanted[-1] + 2 >= len(rain) * 15:
        raise ValueError("start minutes must lie inside the rain file")
    if warmup_minutes % DRY_MINUTES:
        raise ValueError("warm-up must be a whole number of dry cycles so minutes stay aligned")
    n_steps = warmup_minutes + wanted[-1]
    rows = [
        dry[(minute // 15) % len(dry), 1:] if minute < warmup_minutes else rain[(minute - warmup_minutes) // 15, 1:]
        for minute in range(n_steps + 2)
    ]
    data = np.column_stack([np.arange(len(rows)) / 1440.0, np.asarray(rows, dtype=float)])
    plant = _new_plant(data, NOMINAL_KLA)
    apply_plant(plant, steady_state)
    captured: dict[int, dict[str, np.ndarray]] = {}
    targets = {warmup_minutes + minute: minute for minute in wanted}
    for index in range(n_steps + 1):
        if index in targets:
            captured[targets[index]] = capture_plant(plant)
        if index < n_steps:
            plant.step(index, klas=NOMINAL_KLA)
    return captured


def _cached_rain_start_states(dry: np.ndarray, rain: np.ndarray, starts: list[int]) -> dict[int, dict[str, np.ndarray]]:
    global _STEADY_STATE
    key = tuple(sorted(set(int(start) for start in starts)))
    if key not in _RAIN_START_STATES:
        if _STEADY_STATE is None:
            _STEADY_STATE = open_loop_steady_state()
        _RAIN_START_STATES[key] = rain_start_states(dry, rain, list(key), _STEADY_STATE)
    return _RAIN_START_STATES[key]


def generate_bsm1_confirmatory_dataset(
    *,
    n_per_split: int,
    horizon: int = HORIZON_MINUTES,
    history: int = HISTORY_MINUTES,
    seed: int = 0,
    control_scale: tuple[float, float] | list[float] | None = None,
    rehearsal: bool = False,
    workers: int = 1,
) -> dict[str, TrajectoryBatch]:
    """The four confirmatory test splits: dry hold-out and rain, each at nominal and control-shifted aeration.

    Reading the reserved dry days or the rain event needs a frozen protocol, and this raises
    otherwise. ``rehearsal`` cuts the same splits from days that are not reserved. Windows are
    spread evenly over the block, so they overlap each other, and the protocol says so.
    """
    effective_workers = resolve_workers(workers)
    if n_per_split < 1:
        raise ValueError("n_per_split must be positive")
    scale = tuple(float(value) for value in (control_scale if control_scale is not None else CONTROL_SCALE))
    span = history + horizon
    blocks = confirmatory_blocks(rehearsal)
    for weather, (first, end) in blocks.items():
        if end - first < span:
            raise ValueError(f"the {weather} block is shorter than one window of {span} minutes")
    reserved = [weather for weather, (first, end) in blocks.items() if _reads_reserved(weather, first, end)]
    if reserved:
        from scs.experiments.v3 import protocol_is_frozen

        if not protocol_is_frozen():
            raise RuntimeError(
                f"the confirmatory {' and '.join(reserved)} windows are generated only after the freeze, "
                "and docs/v3/protocol.md is still a draft"
            )
    dry = load_dry_influent()
    rain = load_rain_influent()[0]
    tables = {"dry": dry, "rain": rain}
    rng = np.random.default_rng(20_000 + int(seed))
    windows: dict[str, list[dict[str, object]]] = {}
    for weather, (first, end) in blocks.items():
        starts = np.linspace(first, end - span, n_per_split).round().astype(int)
        factors = [(float(rng.uniform(*INFLUENT_SCALE)), float(rng.uniform(*INFLUENT_SCALE))) for _ in starts]
        windows[weather] = [
            {"start": int(start), "flow_scale": flow, "load_scale": load}
            for start, (flow, load) in zip(starts, factors)
        ]
    states = {
        "dry": _cached_start_states(dry, [int(window["start"]) for window in windows["dry"]]),
        "rain": _cached_rain_start_states(dry, rain, [int(window["start"]) for window in windows["rain"]]),
    }
    tasks: list[tuple[str, int, int, int, np.ndarray, str, str, float, float, bool]] = []
    task_splits: list[str] = []
    for split, (weather, shifted) in CONFIRMATORY_SPLITS.items():
        for window_index, window in enumerate(windows[weather]):
            kla_rng = np.random.default_rng(
                [int(seed), int(window_index), int(window["start"]), int(shifted), 1 if weather == "rain" else 0]
            )
            tasks.append(
                (
                    weather,
                    int(window["start"]),
                    history,
                    horizon,
                    _scaled_kla(kla_rng, scale if shifted else NOMINAL_SCALE),
                    split,
                    f"{weather}_{'control_shift' if shifted else 'nominal'}",
                    float(window["flow_scale"]),
                    float(window["load_scale"]),
                    False,
                )
            )
            task_splits.append(split)
    batches = _execute_bsm1_window_tasks(tasks, tables=tables, start_states=states, workers=effective_workers)
    grouped: dict[str, list[TrajectoryBatch]] = {split: [] for split in CONFIRMATORY_SPLITS}
    for split, batch in zip(task_splits, batches):
        grouped[split].append(batch)
    return {split: _concat_batches(pieces) for split, pieces in grouped.items()}


def _constant_series(n_steps: int) -> np.ndarray:
    times = np.arange(n_steps + 2) / 1440.0
    influent = np.repeat(PUBLISHED_CONSTANT_INFLUENT[None, :], len(times), axis=0)
    return np.column_stack([times, influent])


def history_start_sample(start_minute: int, history: int, n_rows: int) -> int:
    """Row of the dry file that holds the first predicted minute of a window."""
    return ((start_minute + history) // 15) % n_rows


def _dry_series(
    dry: np.ndarray, start_minute: int, n_steps: int, flow_scale: float = 1.0, load_scale: float = 1.0
) -> np.ndarray:
    """Dry influent from ``start_minute``, with the window's flow and load factors applied."""
    rows = []
    for minute in range(start_minute, start_minute + n_steps + 2):
        sample = (minute // 15) % len(dry)
        rows.append(dry[sample, 1:])
    influent = np.asarray(rows, dtype=float)
    influent[:, FLOW_INDEX] *= flow_scale
    influent[:, list(LOAD_COLUMNS)] *= load_scale
    times = np.arange(len(rows)) / 1440.0
    return np.column_stack([times, influent])


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
    initial_state: dict[str, np.ndarray],
    split: str,
    scenario_type: str,
    flow_scale: float = 1.0,
    load_scale: float = 1.0,
    check_split: bool = True,
) -> TrajectoryBatch:
    if check_split:
        assert_split_allowed(split)
    total = history + horizon
    plant = _new_plant(_dry_series(dry, start_minute, total, flow_scale, load_scale), NOMINAL_KLA)
    apply_plant(plant, initial_state)
    measured = np.empty((total, len(MEASURED_NAMES)), dtype=float)
    states = np.empty((total + 1, STATE_DIM), dtype=float)
    states[0] = _pack_state(plant)
    for index in range(total):
        # The measured-history window precedes the counterfactual intervention at t=0.
        step_kla = NOMINAL_KLA if index < history else kla
        plant.step(index, klas=step_kla)
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
