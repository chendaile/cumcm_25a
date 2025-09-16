"""High level workflows combining optimisation, analysis and visualisation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .analysis import SimulationResult, export_to_excel, run_simulation, save_plan_json
from .ga import GeneticOptimizer
from .plan import Plan
from .system import SystemConfig
from .visualization import export_frames

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data-bin"
OUTPUT_DIR = PROJECT_ROOT / "output"

POSITIONS_PATH = DATA_DIR / "initial_positions.json"
FORWARD_VECTOR_PATH = DATA_DIR / "initial_drones_forward_vector.json"


@dataclass
class WorkflowResult:
    optimisation_result: SimulationResult
    fitness_history: list[float]


def run_genetic_workflow(drone_ids: Sequence[str], n_jammers: int, population_size: int,
                         generations: int, targeted_missiles: Sequence[str], *,
                         random_seed: int | None = None,
                         save_json: bool = True,
                         export_excel_path: Path | None = None,
                         save_json_path: Path | None = None,
                         video: bool = False) -> WorkflowResult:
    positions = _load_json(POSITIONS_PATH)
    vectors = _load_json(FORWARD_VECTOR_PATH)

    optimizer = GeneticOptimizer(
        initial_positions=positions,
        initial_vectors=vectors,
        drone_ids=list(drone_ids),
        n_jammers=n_jammers,
        population_size=population_size,
        generations=generations,
        targeted_missile_ids=list(targeted_missiles),
        system_config=SystemConfig(),
        random_seed=random_seed,
    )

    optimisation_output = optimizer.optimize()
    simulation = run_simulation(POSITIONS_PATH, FORWARD_VECTOR_PATH,
                                optimisation_output.best_plan, targeted_missiles)

    if save_json:
        if save_json_path is None:
            save_json_path = OUTPUT_DIR / "log" / f"optimisation_{'_'.join(drone_ids)}.json"
        save_plan_json(optimisation_output.best_plan, save_json_path,
                       metadata={"fitness": optimisation_output.best_fitness,
                                 "targeted_missiles": list(targeted_missiles)})

    if export_excel_path is None:
        export_excel_path = OUTPUT_DIR / "excel" / f"plan_{'_'.join(drone_ids)}.xlsx"
    export_to_excel(simulation, targeted_missiles, export_excel_path)

    if video:
        frames_dir = OUTPUT_DIR / "photos" / f"frames_{'_'.join(drone_ids)}"
        time_points = np.arange(0.0, 25.0, 0.5)
        export_frames(simulation.system, time_points, frames_dir)

    return WorkflowResult(optimisation_result=simulation,
                          fitness_history=optimisation_output.history)


def verify_plan(plan: Plan, targeted_missiles: Sequence[str]) -> SimulationResult:
    return run_simulation(POSITIONS_PATH, FORWARD_VECTOR_PATH, plan, targeted_missiles)


def load_plan(path: str | Path) -> Plan:
    payload = _load_json(Path(path))
    plan_data = payload.get("plan", payload)
    plan = Plan.from_dict(plan_data)
    plan.clamp()
    return plan


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
