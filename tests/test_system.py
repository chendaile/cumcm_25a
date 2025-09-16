import json
from pathlib import Path

import numpy as np

from cumcm.analysis import run_simulation
from cumcm.plan import DronePlan, JammerPlan, Plan
from cumcm.system import GlobalSystem
from cumcm.workflows import FORWARD_VECTOR_PATH, POSITIONS_PATH


def test_global_system_without_jammers_has_zero_coverage():
    system = GlobalSystem.from_json(POSITIONS_PATH, FORWARD_VECTOR_PATH)
    durations = system.cover_durations(list(system.missiles.keys()))
    assert all(value == 0 for value in durations.values())


def test_run_simulation_with_initial_plan_produces_positive_duration():
    payload = json.loads(Path("data-bin/ga_initial_params.json").read_text(encoding="utf-8"))
    fy1 = payload["FY1"]
    plan = Plan({
        "FY1": DronePlan(
            velocity=np.array([fy1["velocity"]["velocity_x"], fy1["velocity"]["velocity_y"], 0.0]),
            jammers=[
                JammerPlan(item["father_t"], item["smoke_delay"]) for item in fy1["jammers"]
            ],
        )
    })
    plan.clamp()
    result = run_simulation(POSITIONS_PATH, FORWARD_VECTOR_PATH, plan, ["M1"])
    assert result.total_duration >= 0
    assert "M1" in result.durations
