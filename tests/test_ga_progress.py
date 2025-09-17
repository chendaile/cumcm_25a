from __future__ import annotations

import json
from pathlib import Path

from cumcm.ga import GeneticOptimizer
from cumcm.workflows import FORWARD_VECTOR_PATH, POSITIONS_PATH


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_optimize_invokes_progress_callback():
    positions = _load_json(Path(POSITIONS_PATH))
    vectors = _load_json(Path(FORWARD_VECTOR_PATH))
    optimizer = GeneticOptimizer(
        initial_positions=positions,
        initial_vectors=vectors,
        drone_ids=["FY1"],
        n_jammers=1,
        population_size=4,
        generations=2,
        targeted_missile_ids=["M1"],
        random_seed=123,
    )

    progress_events = []

    def _progress(generation: int, generation_best: float, best_fitness: float, history: list[float]) -> None:
        progress_events.append((generation, generation_best, best_fitness, list(history)))

    result = optimizer.optimize(progress_callback=_progress)

    assert len(progress_events) == optimizer.generations
    assert progress_events[0][0] == 0
    assert progress_events[-1][0] == optimizer.generations - 1
    assert result.history == progress_events[-1][3]
