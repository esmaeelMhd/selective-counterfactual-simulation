from __future__ import annotations

from pathlib import Path

import numpy as np

from scs.systems.base import TrajectoryBatch


def save_batch(batch: TrajectoryBatch, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        states=batch.states,
        actions=batch.actions,
        disturbances=batch.disturbances,
        scenario_type=np.asarray(batch.scenario_type, dtype=str),
        split=np.asarray(batch.split),
        system_id=np.asarray(batch.system_id),
    )


def load_batch(path: str | Path) -> TrajectoryBatch:
    data = np.load(Path(path), allow_pickle=False)
    return TrajectoryBatch(
        states=np.asarray(data["states"], dtype=float),
        actions=np.asarray(data["actions"], dtype=float),
        disturbances=np.asarray(data["disturbances"], dtype=float),
        scenario_type=[str(value) for value in data["scenario_type"].tolist()],
        split=str(data["split"].tolist()),
        system_id=str(data["system_id"].tolist()),
    )


def save_dataset(dataset: dict[str, TrajectoryBatch], directory: str | Path) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for split, batch in dataset.items():
        save_batch(batch, target / f"{split}.npz")


def load_dataset(directory: str | Path) -> dict[str, TrajectoryBatch]:
    source = Path(directory)
    return {path.stem: load_batch(path) for path in sorted(source.glob("*.npz"))}

