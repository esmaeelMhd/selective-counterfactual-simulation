from __future__ import annotations

from scs.models.hold_last import HoldLastModel
from scs.models.linear_narx import LinearNARXModel
from scs.models.mlp_state_space import MLPStateSpaceModel
from scs.systems.cstr import CSTRSystem
from scs.systems.heat_exchanger import HeatExchangerSystem
from scs.systems.two_tank import TwoTankSystem


def make_system(system_id: str):
    if system_id == "two_tank":
        return TwoTankSystem()
    if system_id == "cstr":
        return CSTRSystem()
    if system_id == "heat_exchanger":
        return HeatExchangerSystem()
    raise ValueError(f"unknown system_id: {system_id}")


def make_model(model_id: str, seed: int = 0):
    if model_id == "hold_last":
        return HoldLastModel()
    if model_id == "linear_narx":
        return LinearNARXModel(random_state=seed)
    if model_id == "mlp_state_space":
        return MLPStateSpaceModel(random_state=seed)
    raise ValueError(f"unknown model_id: {model_id}")
