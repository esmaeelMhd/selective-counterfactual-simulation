from __future__ import annotations

import numpy as np

from scs.data.generate import generate_dataset
from scs.data.schemas import load_dataset, save_dataset
from scs.data.splits import action_range, assert_not_identical, max_inflow


def test_dataset_splits_and_shift(tmp_path) -> None:
    dataset = generate_dataset(
        system_id="two_tank",
        n_train=40,
        n_id_test=12,
        n_ood_test=12,
        horizon=20,
        dt=0.1,
        seed=7,
    )
    assert {"train", "id_test", "ood_action_magnitude", "ood_inflow_spike", "ood_combined"} <= set(dataset)
    assert_not_identical(dataset["train"], dataset["id_test"])
    assert action_range(dataset["ood_action_magnitude"]) > action_range(dataset["train"]) * 1.25
    assert max_inflow(dataset["ood_inflow_spike"]) > max_inflow(dataset["train"]) * 1.25
    assert dataset["train"].scenario_type
    assert set(dataset["ood_action_magnitude"].scenario_type) == {"held_out_action_magnitude"}

    save_dataset(dataset, tmp_path)
    loaded = load_dataset(tmp_path)
    assert loaded["train"].states.shape == dataset["train"].states.shape
    assert np.allclose(loaded["id_test"].actions, dataset["id_test"].actions)


def test_cstr_dataset_splits_and_shift() -> None:
    dataset = generate_dataset(
        system_id="cstr",
        n_train=40,
        n_id_test=12,
        n_ood_test=12,
        horizon=20,
        dt=0.1,
        seed=13,
    )
    assert {"train", "id_test", "ood_action_magnitude", "ood_inflow_spike", "ood_combined"} <= set(dataset)
    assert_not_identical(dataset["train"], dataset["id_test"])
    assert action_range(dataset["ood_action_magnitude"]) > action_range(dataset["train"]) * 1.25
    assert max_inflow(dataset["ood_inflow_spike"]) > max_inflow(dataset["train"]) * 1.25
    assert set(dataset["ood_inflow_spike"].scenario_type) == {"inflow_spike"}


def test_heat_exchanger_dataset_splits_and_shift() -> None:
    dataset = generate_dataset(
        system_id="heat_exchanger",
        n_train=40,
        n_id_test=12,
        n_ood_test=12,
        horizon=20,
        dt=0.1,
        seed=17,
    )
    assert {"train", "id_test", "ood_action_magnitude", "ood_inflow_spike", "ood_combined"} <= set(dataset)
    assert_not_identical(dataset["train"], dataset["id_test"])
    assert action_range(dataset["ood_action_magnitude"]) > action_range(dataset["train"]) * 1.25
    assert max_inflow(dataset["ood_inflow_spike"]) > max_inflow(dataset["train"]) * 1.15
    assert set(dataset["ood_combined"].scenario_type) == {"combined_intervention"}

