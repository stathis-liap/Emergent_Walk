"""
Backward-compatibility shim.

This module was split into focused files:
    config.py           — Config + EW_CFG_* overrides + FOOT_NAMES
    scene_builder.py    — scene XML generation, model/sensor/actuator setup
    mppi_controller.py  — MPPIController + generate_trot_seed
    simulate.py         — runnable live-sim / video / metrics harness

Import from those modules directly. The names below are re-exported so older
`from mppi_core import ...` code keeps working. To run the simulation use:
    python simulate.py
"""

from config import Config, FOOT_NAMES
from scene_builder import (
    make_scene_with_sensors,
    prepare_model,
    actuator_joint_perm,
    foot_sensor_columns,
)
from mppi_controller import MPPIController, generate_trot_seed

__all__ = [
    "Config",
    "FOOT_NAMES",
    "make_scene_with_sensors",
    "prepare_model",
    "actuator_joint_perm",
    "foot_sensor_columns",
    "MPPIController",
    "generate_trot_seed",
]


if __name__ == "__main__":
    import runpy
    runpy.run_path("simulate.py", run_name="__main__")
