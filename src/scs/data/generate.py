from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scs.data.schemas import save_dataset
from scs.systems.base import TrajectoryBatch
from scs.systems.cstr import CSTRSystem
from scs.systems.heat_exchanger import HeatExchangerSystem
from scs.systems.two_tank import TwoTankSystem

INTERVENTION_TYPES = {
    "normal_policy",
    "held_out_action_magnitude",
    "action_step_change",
    "inflow_spike",
    "valve_or_pump_degradation",
    "combined_intervention",
    "feed_concentration_spike",
    "feed_temperature_spike",
    "cooling_step_change",
    "reaction_rate_shift",
    "combined_feed_and_cooling_shift",
    "unsafe_temperature_event",
    "inlet_temperature_spike",
    "flow_rate_shift",
    "cooling_or_heating_change",
    "heat_transfer_coefficient_shift",
    "combined_disturbance_shift",
    "unsafe_outlet_temperature_event",
}

TWO_TANK_SEVERITY_SETTINGS = {
    "low": {
        "action_center": 1.12,
        "action_noise": 0.07,
        "action_wave": 0.36,
        "action_step": 0.35,
        "inflow_spike": 0.35,
        "combined_action_center": 1.15,
        "combined_inflow_spike": 0.30,
        "combined_demand_drop": 0.08,
        "pump_factor": 0.75,
    },
    "medium": {
        "action_center": 1.55,
        "action_noise": 0.12,
        "action_wave": 0.22,
        "action_step": 0.85,
        "inflow_spike": 1.15,
        "combined_action_center": 1.65,
        "combined_inflow_spike": 0.95,
        "combined_demand_drop": 0.20,
        "pump_factor": 0.45,
    },
    "high": {
        "action_center": 2.05,
        "action_noise": 0.14,
        "action_wave": 0.30,
        "action_step": 1.20,
        "inflow_spike": 1.85,
        "combined_action_center": 2.15,
        "combined_inflow_spike": 1.65,
        "combined_demand_drop": 0.32,
        "pump_factor": 0.25,
    },
    "extreme": {
        "action_center": 2.42,
        "action_noise": 0.16,
        "action_wave": 0.36,
        "action_step": 1.55,
        "inflow_spike": 2.65,
        "combined_action_center": 2.45,
        "combined_inflow_spike": 2.35,
        "combined_demand_drop": 0.40,
        "pump_factor": 0.12,
    },
}


def two_tank_severity_settings(severity: str) -> dict[str, float]:
    if severity not in TWO_TANK_SEVERITY_SETTINGS:
        raise ValueError(f"unknown TwoTank severity: {severity}")
    return TWO_TANK_SEVERITY_SETTINGS[severity]


def _normal_action(state: np.ndarray, rng: np.random.Generator) -> float:
    feedback = 0.08 * (float(state[0] - state[1]) - 1.0)
    return float(np.clip(0.78 + feedback + rng.normal(0.0, 0.045), 0.42, 1.05))


def _normal_disturbance(t: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    phase = 2.0 * np.pi * t / max(horizon, 1)
    d_in = 0.50 + 0.035 * np.sin(phase) + rng.normal(0.0, 0.025)
    d_out = 0.45 + 0.030 * np.cos(phase * 0.7) + rng.normal(0.0, 0.025)
    return np.array([max(d_in, 0.0), max(d_out, 0.0)], dtype=float)


def _intervention_action(
    scenario_type: str,
    base_action: float,
    t: int,
    horizon: int,
    rng: np.random.Generator,
    severity: str,
) -> float:
    settings = two_tank_severity_settings(severity)
    if scenario_type == "held_out_action_magnitude":
        phase = 2.0 * np.pi * t / max(horizon, 1)
        return float(
            np.clip(
                settings["action_center"]
                + settings["action_wave"] * np.sin(phase)
                + rng.normal(0.0, settings["action_noise"]),
                0.0,
                2.7,
            )
        )
    if scenario_type == "action_step_change" and t >= horizon // 2:
        return float(np.clip(base_action + settings["action_step"], 0.0, 2.7))
    if scenario_type == "valve_or_pump_degradation" and t >= horizon // 2:
        return float(np.clip(base_action * settings["pump_factor"], 0.0, 2.7))
    if scenario_type == "combined_intervention":
        if t >= horizon // 3:
            return float(
                np.clip(
                    settings["combined_action_center"] + rng.normal(0.0, settings["action_noise"]),
                    0.0,
                    2.7,
                )
            )
        return base_action
    return base_action


def _intervention_disturbance(
    scenario_type: str,
    base_disturbance: np.ndarray,
    t: int,
    horizon: int,
    severity: str,
) -> np.ndarray:
    settings = two_tank_severity_settings(severity)
    disturbance = np.array(base_disturbance, dtype=float)
    spike_start = horizon // 3
    spike_end = min(horizon, spike_start + max(4, horizon // 8))
    if scenario_type == "inflow_spike" and spike_start <= t < spike_end:
        disturbance[0] += settings["inflow_spike"]
    if scenario_type == "combined_intervention":
        if spike_start <= t < spike_end:
            disturbance[0] += settings["combined_inflow_spike"]
        if t >= horizon // 2:
            disturbance[1] = max(0.05, disturbance[1] - settings["combined_demand_drop"])
    return disturbance


def _simulate_two_tank_one(
    system: TwoTankSystem,
    scenario_type: str,
    horizon: int,
    dt: float,
    rng: np.random.Generator,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scenario_type not in INTERVENTION_TYPES:
        raise ValueError(f"unknown scenario_type: {scenario_type}")

    states = np.empty((horizon + 1, system.state_dim), dtype=float)
    actions = np.empty((horizon, system.action_dim), dtype=float)
    disturbances = np.empty((horizon, system.disturbance_dim), dtype=float)
    states[0] = system.reset(seed=int(rng.integers(0, 2**31 - 1)))

    for t in range(horizon):
        base_action = _normal_action(states[t], rng)
        action = _intervention_action(scenario_type, base_action, t, horizon, rng, severity)
        disturbance = _intervention_disturbance(
            scenario_type,
            _normal_disturbance(t, horizon, rng),
            t,
            horizon,
            severity,
        )
        actions[t] = np.array([action], dtype=float)
        disturbances[t] = disturbance
        states[t + 1] = system.step(states[t], actions[t], disturbances[t], dt)

    return states, actions, disturbances


def _simulate_two_tank_batch(
    split: str,
    scenario_type: str,
    n_trajectories: int,
    horizon: int,
    dt: float,
    seed: int,
    severity: str = "medium",
) -> TrajectoryBatch:
    system = TwoTankSystem()
    rng = np.random.default_rng(seed)
    states = np.empty((n_trajectories, horizon + 1, system.state_dim), dtype=float)
    actions = np.empty((n_trajectories, horizon, system.action_dim), dtype=float)
    disturbances = np.empty((n_trajectories, horizon, system.disturbance_dim), dtype=float)
    scenario_type_list: list[str] = []

    for i in range(n_trajectories):
        states[i], actions[i], disturbances[i] = _simulate_two_tank_one(
            system=system,
            scenario_type=scenario_type,
            horizon=horizon,
            dt=dt,
            rng=rng,
            severity=severity,
        )
        scenario_type_list.append(scenario_type)

    return TrajectoryBatch(
        states=states,
        actions=actions,
        disturbances=disturbances,
        scenario_type=scenario_type_list,
        split=split,
        system_id=system.system_id,
    )


def _normal_cstr_action(state: np.ndarray, rng: np.random.Generator) -> float:
    temperature = float(state[1])
    feedback = 0.055 * (temperature - 345.0)
    return float(np.clip(9.5 + feedback + rng.normal(0.0, 0.25), 8.0, 11.2))


def _normal_cstr_disturbance(t: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    phase = 2.0 * np.pi * t / max(horizon, 1)
    feed_concentration = 1.0 + 0.035 * np.sin(phase * 0.8) + rng.normal(0.0, 0.025)
    feed_temperature = 340.0 + 2.5 * np.sin(phase * 0.6) + rng.normal(0.0, 1.2)
    flow_rate = 0.60 + 0.025 * np.cos(phase * 0.7) + rng.normal(0.0, 0.015)
    return np.array(
        [
            np.clip(feed_concentration, 0.88, 1.12),
            np.clip(feed_temperature, 334.0, 346.0),
            np.clip(flow_rate, 0.52, 0.68),
        ],
        dtype=float,
    )


def _intervention_cstr_action(
    scenario_type: str,
    base_action: float,
    t: int,
    horizon: int,
    rng: np.random.Generator,
) -> float:
    if scenario_type in {"held_out_action_magnitude", "cooling_step_change"}:
        phase = 2.0 * np.pi * t / max(horizon, 1)
        return float(np.clip(14.5 + 1.8 * np.sin(phase) + rng.normal(0.0, 0.35), 11.8, 17.2))
    if scenario_type == "action_step_change" and t >= horizon // 2:
        return float(np.clip(base_action + 4.0, 6.0, 16.0))
    if scenario_type == "valve_or_pump_degradation" and t >= horizon // 2:
        return float(np.clip(base_action * 0.55, 4.0, 16.0))
    if scenario_type in {"combined_intervention", "combined_feed_and_cooling_shift"}:
        if t >= horizon // 3:
            return float(np.clip(5.6 + rng.normal(0.0, 0.25), 4.6, 6.6))
        return base_action
    if scenario_type == "unsafe_temperature_event" and t >= horizon // 4:
        return float(np.clip(base_action * 0.35, 3.0, 12.0))
    return base_action


def _intervention_cstr_disturbance(
    scenario_type: str,
    base_disturbance: np.ndarray,
    t: int,
    horizon: int,
) -> np.ndarray:
    disturbance = np.array(base_disturbance, dtype=float)
    spike_start = horizon // 3
    spike_end = min(horizon, spike_start + max(4, horizon // 8))
    if scenario_type in {"inflow_spike", "feed_concentration_spike"} and spike_start <= t < spike_end:
        disturbance[0] += 0.58
    if scenario_type == "feed_temperature_spike" and spike_start <= t < spike_end:
        disturbance[1] += 34.0
    if scenario_type == "reaction_rate_shift" and t >= horizon // 2:
        disturbance[2] = np.clip(disturbance[2] * 1.75, 0.4, 1.25)
    if scenario_type in {"combined_intervention", "combined_feed_and_cooling_shift"}:
        if spike_start <= t < spike_end:
            disturbance[0] += 0.44
            disturbance[1] += 22.0
        if t >= horizon // 2:
            disturbance[2] = np.clip(disturbance[2] * 1.45, 0.4, 1.25)
    if scenario_type == "unsafe_temperature_event":
        if t >= horizon // 4:
            disturbance[1] += 46.0
            disturbance[2] = np.clip(disturbance[2] * 1.55, 0.4, 1.25)
    return disturbance


def _simulate_cstr_one(
    system: CSTRSystem,
    scenario_type: str,
    horizon: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scenario_type not in INTERVENTION_TYPES:
        raise ValueError(f"unknown scenario_type: {scenario_type}")

    states = np.empty((horizon + 1, system.state_dim), dtype=float)
    actions = np.empty((horizon, system.action_dim), dtype=float)
    disturbances = np.empty((horizon, system.disturbance_dim), dtype=float)
    states[0] = system.reset(seed=int(rng.integers(0, 2**31 - 1)))

    for t in range(horizon):
        base_action = _normal_cstr_action(states[t], rng)
        action = _intervention_cstr_action(scenario_type, base_action, t, horizon, rng)
        disturbance = _intervention_cstr_disturbance(
            scenario_type,
            _normal_cstr_disturbance(t, horizon, rng),
            t,
            horizon,
        )
        actions[t] = np.array([action], dtype=float)
        disturbances[t] = disturbance
        states[t + 1] = system.step(states[t], actions[t], disturbances[t], dt)

    return states, actions, disturbances


def _simulate_cstr_batch(
    split: str,
    scenario_type: str,
    n_trajectories: int,
    horizon: int,
    dt: float,
    seed: int,
) -> TrajectoryBatch:
    system = CSTRSystem()
    rng = np.random.default_rng(seed)
    states = np.empty((n_trajectories, horizon + 1, system.state_dim), dtype=float)
    actions = np.empty((n_trajectories, horizon, system.action_dim), dtype=float)
    disturbances = np.empty((n_trajectories, horizon, system.disturbance_dim), dtype=float)
    scenario_type_list: list[str] = []

    for i in range(n_trajectories):
        states[i], actions[i], disturbances[i] = _simulate_cstr_one(
            system=system,
            scenario_type=scenario_type,
            horizon=horizon,
            dt=dt,
            rng=rng,
        )
        scenario_type_list.append(scenario_type)

    return TrajectoryBatch(
        states=states,
        actions=actions,
        disturbances=disturbances,
        scenario_type=scenario_type_list,
        split=split,
        system_id=system.system_id,
    )


def _generate_two_tank_dataset(
    n_train: int,
    n_id_test: int,
    n_ood_test: int,
    horizon: int,
    dt: float,
    seed: int,
    severity: str = "medium",
    include_pump_degradation: bool = False,
) -> dict[str, TrajectoryBatch]:
    seeds = {
        "train": seed + 101,
        "id_test": seed + 202,
        "ood_action_magnitude": seed + 303,
        "ood_inflow_spike": seed + 404,
        "ood_combined": seed + 505,
    }
    dataset = {
        "train": _simulate_two_tank_batch("train", "normal_policy", n_train, horizon, dt, seeds["train"]),
        "id_test": _simulate_two_tank_batch("id_test", "normal_policy", n_id_test, horizon, dt, seeds["id_test"]),
        "ood_action_magnitude": _simulate_two_tank_batch(
            "ood_action_magnitude",
            "held_out_action_magnitude",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_action_magnitude"],
            severity=severity,
        ),
        "ood_inflow_spike": _simulate_two_tank_batch(
            "ood_inflow_spike",
            "inflow_spike",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_inflow_spike"],
            severity=severity,
        ),
        "ood_combined": _simulate_two_tank_batch(
            "ood_combined",
            "combined_intervention",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_combined"],
            severity=severity,
        ),
    }
    if include_pump_degradation:
        dataset["pump_degradation"] = _simulate_two_tank_batch(
            "pump_degradation",
            "valve_or_pump_degradation",
            n_ood_test,
            horizon,
            dt,
            seed + 606,
            severity=severity,
        )
    return dataset


def _generate_cstr_dataset(
    n_train: int,
    n_id_test: int,
    n_ood_test: int,
    horizon: int,
    dt: float,
    seed: int,
) -> dict[str, TrajectoryBatch]:
    seeds = {
        "train": seed + 1101,
        "id_test": seed + 1202,
        "ood_action_magnitude": seed + 1303,
        "ood_inflow_spike": seed + 1404,
        "ood_combined": seed + 1505,
        "ood_feed_temperature_spike": seed + 1606,
        "ood_reaction_rate_shift": seed + 1707,
        "ood_unsafe_temperature_event": seed + 1808,
    }
    return {
        "train": _simulate_cstr_batch("train", "normal_policy", n_train, horizon, dt, seeds["train"]),
        "id_test": _simulate_cstr_batch("id_test", "normal_policy", n_id_test, horizon, dt, seeds["id_test"]),
        "ood_action_magnitude": _simulate_cstr_batch(
            "ood_action_magnitude",
            "cooling_step_change",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_action_magnitude"],
        ),
        "ood_inflow_spike": _simulate_cstr_batch(
            "ood_inflow_spike",
            "feed_concentration_spike",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_inflow_spike"],
        ),
        "ood_combined": _simulate_cstr_batch(
            "ood_combined",
            "combined_feed_and_cooling_shift",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_combined"],
        ),
        "ood_feed_temperature_spike": _simulate_cstr_batch(
            "ood_feed_temperature_spike",
            "feed_temperature_spike",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_feed_temperature_spike"],
        ),
        "ood_reaction_rate_shift": _simulate_cstr_batch(
            "ood_reaction_rate_shift",
            "reaction_rate_shift",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_reaction_rate_shift"],
        ),
        "ood_unsafe_temperature_event": _simulate_cstr_batch(
            "ood_unsafe_temperature_event",
            "unsafe_temperature_event",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_unsafe_temperature_event"],
        ),
    }


def _normal_heat_exchanger_action(state: np.ndarray, rng: np.random.Generator) -> float:
    hot_out, cold_out = np.asarray(state, dtype=float)
    target_hot = 88.0
    feedback = 0.035 * (hot_out - target_hot) - 0.010 * (cold_out - 28.0)
    return float(np.clip(1.25 + feedback + rng.normal(0.0, 0.06), 0.85, 1.65))


def _normal_heat_exchanger_disturbance(t: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    phase = 2.0 * np.pi * t / max(horizon, 1)
    hot_in = 118.0 + 2.4 * np.sin(phase) + rng.normal(0.0, 0.75)
    cold_in = 18.0 + 1.0 * np.cos(phase * 0.6) + rng.normal(0.0, 0.35)
    return np.array([hot_in, cold_in], dtype=float)


def _intervention_heat_exchanger_action(
    scenario_type: str,
    base_action: float,
    t: int,
    horizon: int,
    rng: np.random.Generator,
) -> float:
    if scenario_type in {"held_out_action_magnitude", "flow_rate_shift"}:
        phase = 2.0 * np.pi * t / max(horizon, 1)
        return float(np.clip(2.65 + 0.55 * np.sin(phase) + rng.normal(0.0, 0.10), 1.85, 3.35))
    if scenario_type in {"action_step_change", "cooling_or_heating_change"} and t >= horizon // 2:
        return float(np.clip(base_action + 1.15, 0.2, 3.4))
    if scenario_type in {"valve_or_pump_degradation", "heat_transfer_coefficient_shift"} and t >= horizon // 2:
        return float(np.clip(base_action * 0.42, 0.2, 3.4))
    if scenario_type in {"combined_intervention", "combined_disturbance_shift"}:
        if t >= horizon // 3:
            return float(np.clip(0.55 + rng.normal(0.0, 0.06), 0.25, 0.85))
        return base_action
    if scenario_type == "unsafe_outlet_temperature_event" and t >= horizon // 4:
        return float(np.clip(base_action * 0.30, 0.15, 0.70))
    return base_action


def _intervention_heat_exchanger_disturbance(
    scenario_type: str,
    base_disturbance: np.ndarray,
    t: int,
    horizon: int,
) -> np.ndarray:
    disturbance = np.array(base_disturbance, dtype=float)
    spike_start = horizon // 3
    spike_end = min(horizon, spike_start + max(4, horizon // 8))
    if scenario_type in {"inflow_spike", "inlet_temperature_spike"} and spike_start <= t < spike_end:
        disturbance[0] += 42.0
    if scenario_type == "flow_rate_shift" and t >= horizon // 2:
        disturbance[1] -= 4.0
    if scenario_type == "heat_transfer_coefficient_shift" and t >= horizon // 2:
        disturbance[0] += 18.0
    if scenario_type in {"combined_intervention", "combined_disturbance_shift"}:
        if spike_start <= t < spike_end:
            disturbance[0] += 32.0
        if t >= horizon // 2:
            disturbance[1] += 7.0
    if scenario_type == "unsafe_outlet_temperature_event" and t >= horizon // 4:
        disturbance[0] += 58.0
        disturbance[1] += 9.0
    return disturbance


def _simulate_heat_exchanger_one(
    system: HeatExchangerSystem,
    scenario_type: str,
    horizon: int,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scenario_type not in INTERVENTION_TYPES:
        raise ValueError(f"unknown scenario_type: {scenario_type}")

    states = np.empty((horizon + 1, system.state_dim), dtype=float)
    actions = np.empty((horizon, system.action_dim), dtype=float)
    disturbances = np.empty((horizon, system.disturbance_dim), dtype=float)
    states[0] = system.reset(seed=int(rng.integers(0, 2**31 - 1)))

    for t in range(horizon):
        base_action = _normal_heat_exchanger_action(states[t], rng)
        action = _intervention_heat_exchanger_action(scenario_type, base_action, t, horizon, rng)
        disturbance = _intervention_heat_exchanger_disturbance(
            scenario_type,
            _normal_heat_exchanger_disturbance(t, horizon, rng),
            t,
            horizon,
        )
        actions[t] = np.array([action], dtype=float)
        disturbances[t] = disturbance
        states[t + 1] = system.step(states[t], actions[t], disturbances[t], dt)

    return states, actions, disturbances


def _simulate_heat_exchanger_batch(
    split: str,
    scenario_type: str,
    n_trajectories: int,
    horizon: int,
    dt: float,
    seed: int,
) -> TrajectoryBatch:
    system = HeatExchangerSystem()
    rng = np.random.default_rng(seed)
    states = np.empty((n_trajectories, horizon + 1, system.state_dim), dtype=float)
    actions = np.empty((n_trajectories, horizon, system.action_dim), dtype=float)
    disturbances = np.empty((n_trajectories, horizon, system.disturbance_dim), dtype=float)
    scenario_type_list: list[str] = []

    for i in range(n_trajectories):
        states[i], actions[i], disturbances[i] = _simulate_heat_exchanger_one(
            system=system,
            scenario_type=scenario_type,
            horizon=horizon,
            dt=dt,
            rng=rng,
        )
        scenario_type_list.append(scenario_type)

    return TrajectoryBatch(
        states=states,
        actions=actions,
        disturbances=disturbances,
        scenario_type=scenario_type_list,
        split=split,
        system_id=system.system_id,
    )


def _generate_heat_exchanger_dataset(
    n_train: int,
    n_id_test: int,
    n_ood_test: int,
    horizon: int,
    dt: float,
    seed: int,
) -> dict[str, TrajectoryBatch]:
    seeds = {
        "train": seed + 2101,
        "id_test": seed + 2202,
        "ood_action_magnitude": seed + 2303,
        "ood_inflow_spike": seed + 2404,
        "ood_combined": seed + 2505,
    }
    return {
        "train": _simulate_heat_exchanger_batch("train", "normal_policy", n_train, horizon, dt, seeds["train"]),
        "id_test": _simulate_heat_exchanger_batch("id_test", "normal_policy", n_id_test, horizon, dt, seeds["id_test"]),
        "ood_action_magnitude": _simulate_heat_exchanger_batch(
            "ood_action_magnitude",
            "held_out_action_magnitude",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_action_magnitude"],
        ),
        "ood_inflow_spike": _simulate_heat_exchanger_batch(
            "ood_inflow_spike",
            "inflow_spike",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_inflow_spike"],
        ),
        "ood_combined": _simulate_heat_exchanger_batch(
            "ood_combined",
            "combined_intervention",
            n_ood_test,
            horizon,
            dt,
            seeds["ood_combined"],
        ),
    }


def generate_calibrated_heat_exchanger_dataset(
    n_model_train: int,
    n_calibration_id: int,
    n_calibration_ood: int,
    n_test_id: int,
    n_test_ood: int,
    horizon: int,
    dt: float,
    seed: int,
) -> dict[str, TrajectoryBatch]:
    """Generate separated HeatExchanger roles for v2 frozen evaluation."""
    split_specs = [
        ("model_train", "normal_policy", "id", n_model_train, seed + 1001),
        ("judge_calibration_id", "normal_policy", "id", n_calibration_id, seed + 2001),
        (
            "judge_calibration_inlet_temperature_spike",
            "inlet_temperature_spike",
            "inlet_temperature_spike",
            n_calibration_ood,
            seed + 2101,
        ),
        (
            "judge_calibration_flow_rate_shift",
            "flow_rate_shift",
            "flow_rate_shift",
            n_calibration_ood,
            seed + 2201,
        ),
        (
            "judge_calibration_cooling_or_heating_change",
            "cooling_or_heating_change",
            "cooling_or_heating_change",
            n_calibration_ood,
            seed + 2301,
        ),
        (
            "judge_calibration_heat_transfer_coefficient_shift",
            "heat_transfer_coefficient_shift",
            "heat_transfer_coefficient_shift",
            n_calibration_ood,
            seed + 2401,
        ),
        (
            "judge_calibration_combined_disturbance_shift",
            "combined_disturbance_shift",
            "combined_disturbance_shift",
            n_calibration_ood,
            seed + 2501,
        ),
        (
            "judge_calibration_unsafe_outlet_temperature_event",
            "unsafe_outlet_temperature_event",
            "unsafe_outlet_temperature_event",
            n_calibration_ood,
            seed + 2601,
        ),
        ("judge_test_id", "normal_policy", "id", n_test_id, seed + 3001),
        (
            "judge_test_inlet_temperature_spike",
            "inlet_temperature_spike",
            "inlet_temperature_spike",
            n_test_ood,
            seed + 3101,
        ),
        (
            "judge_test_flow_rate_shift",
            "flow_rate_shift",
            "flow_rate_shift",
            n_test_ood,
            seed + 3201,
        ),
        (
            "judge_test_cooling_or_heating_change",
            "cooling_or_heating_change",
            "cooling_or_heating_change",
            n_test_ood,
            seed + 3301,
        ),
        (
            "judge_test_heat_transfer_coefficient_shift",
            "heat_transfer_coefficient_shift",
            "heat_transfer_coefficient_shift",
            n_test_ood,
            seed + 3401,
        ),
        (
            "judge_test_combined_disturbance_shift",
            "combined_disturbance_shift",
            "combined_disturbance_shift",
            n_test_ood,
            seed + 3501,
        ),
        (
            "judge_test_unsafe_outlet_temperature_event",
            "unsafe_outlet_temperature_event",
            "unsafe_outlet_temperature_event",
            n_test_ood,
            seed + 3601,
        ),
    ]
    return {
        split: TrajectoryBatch(
            states=batch.states,
            actions=batch.actions,
            disturbances=batch.disturbances,
            scenario_type=[reported_scenario_type] * batch.n_trajectories,
            split=split,
            system_id=batch.system_id,
        )
        for split, simulator_scenario_type, reported_scenario_type, n_trajectories, split_seed in split_specs
        for batch in [
            _simulate_heat_exchanger_batch(
                split=split,
                scenario_type=simulator_scenario_type,
                n_trajectories=n_trajectories,
                horizon=horizon,
                dt=dt,
                seed=split_seed,
            )
        ]
    }


def generate_dataset(
    system_id: str,
    n_train: int,
    n_id_test: int,
    n_ood_test: int,
    horizon: int,
    dt: float,
    seed: int,
    severity: str = "medium",
    include_pump_degradation: bool = False,
) -> dict[str, TrajectoryBatch]:
    if system_id == "two_tank":
        return _generate_two_tank_dataset(
            n_train,
            n_id_test,
            n_ood_test,
            horizon,
            dt,
            seed,
            severity=severity,
            include_pump_degradation=include_pump_degradation,
        )
    if system_id == "cstr":
        return _generate_cstr_dataset(n_train, n_id_test, n_ood_test, horizon, dt, seed)
    if system_id == "heat_exchanger":
        return _generate_heat_exchanger_dataset(n_train, n_id_test, n_ood_test, horizon, dt, seed)
    raise ValueError("dataset generation supports system_id='two_tank', system_id='cstr', or system_id='heat_exchanger'")


def generate_calibrated_two_tank_dataset(
    n_model_train: int,
    n_calibration_id: int,
    n_calibration_ood: int,
    n_test_id: int,
    n_test_ood: int,
    horizon: int,
    dt: float,
    seed: int,
    severity: str = "medium",
) -> dict[str, TrajectoryBatch]:
    """Generate separated TwoTank data roles for calibrated judge evaluation.

    Split roles:
    - ``model_train`` fits simulator models only.
    - ``judge_calibration_*`` fits/selects refusal judges only.
    - ``judge_test_*`` evaluates fixed refusal judges only.
    """
    split_specs = [
        ("model_train", "normal_policy", n_model_train, seed + 1001),
        ("judge_calibration_id", "normal_policy", n_calibration_id, seed + 2001),
        ("judge_calibration_ood_action_magnitude", "held_out_action_magnitude", n_calibration_ood, seed + 2101),
        ("judge_calibration_ood_inflow_spike", "inflow_spike", n_calibration_ood, seed + 2201),
        ("judge_calibration_ood_combined", "combined_intervention", n_calibration_ood, seed + 2301),
        ("judge_calibration_pump_degradation", "valve_or_pump_degradation", n_calibration_ood, seed + 2401),
        ("judge_test_id", "normal_policy", n_test_id, seed + 3001),
        ("judge_test_ood_action_magnitude", "held_out_action_magnitude", n_test_ood, seed + 3101),
        ("judge_test_ood_inflow_spike", "inflow_spike", n_test_ood, seed + 3201),
        ("judge_test_ood_combined", "combined_intervention", n_test_ood, seed + 3301),
        ("judge_test_pump_degradation", "valve_or_pump_degradation", n_test_ood, seed + 3401),
    ]
    return {
        split: _simulate_two_tank_batch(
            split=split,
            scenario_type=scenario_type,
            n_trajectories=n_trajectories,
            horizon=horizon,
            dt=dt,
            seed=split_seed,
            severity=severity,
        )
        for split, scenario_type, n_trajectories, split_seed in split_specs
    }


def _simulate_cstr_calibrated_batch(
    split: str,
    simulator_scenario_type: str,
    reported_scenario_type: str,
    n_trajectories: int,
    horizon: int,
    dt: float,
    seed: int,
) -> TrajectoryBatch:
    batch = _simulate_cstr_batch(
        split=split,
        scenario_type=simulator_scenario_type,
        n_trajectories=n_trajectories,
        horizon=horizon,
        dt=dt,
        seed=seed,
    )
    return TrajectoryBatch(
        states=batch.states,
        actions=batch.actions,
        disturbances=batch.disturbances,
        scenario_type=[reported_scenario_type] * batch.n_trajectories,
        split=split,
        system_id=batch.system_id,
    )


def generate_calibrated_cstr_dataset(
    n_model_train: int,
    n_calibration_id: int,
    n_calibration_ood: int,
    n_test_id: int,
    n_test_ood: int,
    horizon: int,
    dt: float,
    seed: int,
) -> dict[str, TrajectoryBatch]:
    """Generate separated CSTR data roles for frozen calibrated replication."""
    split_specs = [
        ("model_train", "normal_policy", "id", n_model_train, seed + 1001),
        ("judge_calibration_id", "normal_policy", "id", n_calibration_id, seed + 2001),
        (
            "judge_calibration_feed_concentration_spike",
            "feed_concentration_spike",
            "feed_concentration_spike",
            n_calibration_ood,
            seed + 2101,
        ),
        (
            "judge_calibration_feed_temperature_spike",
            "feed_temperature_spike",
            "feed_temperature_spike",
            n_calibration_ood,
            seed + 2201,
        ),
        (
            "judge_calibration_cooling_step_change",
            "cooling_step_change",
            "cooling_step_change",
            n_calibration_ood,
            seed + 2301,
        ),
        (
            "judge_calibration_reaction_rate_shift",
            "reaction_rate_shift",
            "reaction_rate_shift",
            n_calibration_ood,
            seed + 2401,
        ),
        (
            "judge_calibration_combined_feed_and_cooling_shift",
            "combined_feed_and_cooling_shift",
            "combined_feed_and_cooling_shift",
            n_calibration_ood,
            seed + 2501,
        ),
        (
            "judge_calibration_unsafe_temperature_event",
            "unsafe_temperature_event",
            "unsafe_temperature_event",
            n_calibration_ood,
            seed + 2601,
        ),
        ("judge_test_id", "normal_policy", "id", n_test_id, seed + 3001),
        (
            "judge_test_feed_concentration_spike",
            "feed_concentration_spike",
            "feed_concentration_spike",
            n_test_ood,
            seed + 3101,
        ),
        (
            "judge_test_feed_temperature_spike",
            "feed_temperature_spike",
            "feed_temperature_spike",
            n_test_ood,
            seed + 3201,
        ),
        (
            "judge_test_cooling_step_change",
            "cooling_step_change",
            "cooling_step_change",
            n_test_ood,
            seed + 3301,
        ),
        (
            "judge_test_reaction_rate_shift",
            "reaction_rate_shift",
            "reaction_rate_shift",
            n_test_ood,
            seed + 3401,
        ),
        (
            "judge_test_combined_feed_and_cooling_shift",
            "combined_feed_and_cooling_shift",
            "combined_feed_and_cooling_shift",
            n_test_ood,
            seed + 3501,
        ),
        (
            "judge_test_unsafe_temperature_event",
            "unsafe_temperature_event",
            "unsafe_temperature_event",
            n_test_ood,
            seed + 3601,
        ),
    ]
    return {
        split: _simulate_cstr_calibrated_batch(
            split=split,
            simulator_scenario_type=simulator_scenario_type,
            reported_scenario_type=reported_scenario_type,
            n_trajectories=n_trajectories,
            horizon=horizon,
            dt=dt,
            seed=split_seed,
        )
        for split, simulator_scenario_type, reported_scenario_type, n_trajectories, split_seed in split_specs
    }


def summarize_dataset(dataset: dict[str, TrajectoryBatch]) -> dict[str, dict[str, float | int | str]]:
    summary: dict[str, dict[str, float | int | str]] = {}
    for split, batch in dataset.items():
        summary[split] = {
            "system_id": batch.system_id,
            "n_trajectories": batch.n_trajectories,
            "horizon": batch.horizon,
            "state_dim": int(batch.states.shape[-1]),
            "action_dim": int(batch.actions.shape[-1]),
            "disturbance_dim": int(batch.disturbances.shape[-1]),
            "action_min": float(np.min(batch.actions)),
            "action_max": float(np.max(batch.actions)),
            "disturbance_0_max": float(np.max(batch.disturbances[..., 0])),
            "scenario_type": ",".join(sorted(set(batch.scenario_type))),
        }
        if batch.disturbances.shape[-1] > 1:
            summary[split]["disturbance_1_max"] = float(np.max(batch.disturbances[..., 1]))
    return summary


def generate_and_save_dataset(config: dict, output_dir: str | Path) -> dict[str, TrajectoryBatch]:
    dataset = generate_dataset(
        system_id=str(config["system_id"]),
        n_train=int(config["n_train"]),
        n_id_test=int(config["n_id_test"]),
        n_ood_test=int(config["n_ood_test"]),
        horizon=int(config["horizon"]),
        dt=float(config["dt"]),
        seed=int(config["seed"]),
        severity=str(config.get("severity", "medium")),
        include_pump_degradation=bool(config.get("include_pump_degradation", False)),
    )
    data_dir = Path(output_dir) / "data"
    save_dataset(dataset, data_dir)
    with (Path(output_dir) / "data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summarize_dataset(dataset), handle, indent=2, sort_keys=True)
    return dataset
