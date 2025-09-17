"""Simple yet well-documented genetic algorithm for smoke jamming."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from .plan import DronePlan, JammerPlan, Plan
from .system import GlobalSystem, SystemConfig, apply_plan


@dataclass
class GAResult:
    best_plan: Plan
    best_fitness: float
    history: List[float]


@dataclass
class GeneticOptimizer:
    initial_positions: Dict
    initial_vectors: Dict
    drone_ids: Sequence[str]
    n_jammers: int
    population_size: int
    generations: int
    targeted_missile_ids: Sequence[str]
    system_config: SystemConfig = field(default_factory=SystemConfig)
    seed_param_path: Path | None = None
    random_seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)
        self.seed_params = self._load_seed_params()
        self.fitness_cache: Dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(self, progress_callback: Callable[[int, float, float, List[float]], None] | None = None) -> GAResult:
        population = [self._create_individual() for _ in range(self.population_size)]
        best_plan = population[0]
        best_fitness = float("-inf")
        history: List[float] = []

        for generation in range(self.generations):
            fitnesses = np.array([self._evaluate_with_cache(plan) for plan in population])
            history.append(float(fitnesses.max()))

            idx_best = int(np.argmax(fitnesses))
            if fitnesses[idx_best] > best_fitness:
                best_fitness = float(fitnesses[idx_best])
                best_plan = population[idx_best]

            if progress_callback is not None:
                progress_callback(
                    generation,
                    float(fitnesses[idx_best]),
                    best_fitness,
                    history.copy(),
                )

            population = self._evolve(population, fitnesses)

        return GAResult(best_plan=best_plan, best_fitness=best_fitness, history=history)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _load_seed_params(self) -> Dict[str, Dict]:
        if self.seed_param_path is None:
            path = Path("data-bin/ga_initial_params.json")
        else:
            path = Path(self.seed_param_path)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _create_individual(self) -> Plan:
        plans: dict[str, DronePlan] = {}
        for drone_id in self.drone_ids:
            seed = self.seed_params.get(drone_id, self.seed_params.get("FY1", {}))
            vx, vy = self._initial_velocity(seed)
            velocity = np.array([vx, vy, 0.0], dtype=float)

            jammer_seed = seed.get("jammers", []) if seed else []
            jammers: List[JammerPlan] = []
            for idx in range(self.n_jammers):
                if idx < len(jammer_seed):
                    release = jammer_seed[idx].get("father_t", 0.0)
                    delay = jammer_seed[idx].get("smoke_delay", 3.0)
                else:
                    release = float(self.rng.uniform(0.0, 20.0))
                    delay = float(self.rng.uniform(0.0, 10.0))
                release += float(self.rng.normal(0.0, 0.6))
                delay += float(self.rng.normal(0.0, 0.4))
                jammers.append(JammerPlan(release, delay))

            plans[drone_id] = DronePlan(velocity=velocity, jammers=jammers)
            plans[drone_id].clamp()
        return Plan(plans)

    def _initial_velocity(self, seed: Dict) -> tuple[float, float]:
        vx = seed.get("velocity", {}).get("velocity_x", float(self.rng.uniform(-140, 140)))
        vy = seed.get("velocity", {}).get("velocity_y", float(self.rng.uniform(-140, 140)))
        vx += float(self.rng.normal(0.0, 5.0))
        vy += float(self.rng.normal(0.0, 5.0))
        return vx, vy

    # ------------------------------------------------------------------
    # Evolution operators
    # ------------------------------------------------------------------
    def _evolve(self, population: List[Plan], fitnesses: np.ndarray) -> List[Plan]:
        elite_size = max(2, self.population_size // 10)
        indices_sorted = np.argsort(fitnesses)[::-1]
        new_population = [population[idx] for idx in indices_sorted[:elite_size]]

        while len(new_population) < self.population_size:
            parent_indices = self._tournament_selection(fitnesses, k=2)
            parent_a, parent_b = population[parent_indices[0]], population[parent_indices[1]]
            child = self._crossover(parent_a, parent_b)
            self._mutate(child)
            child.clamp()
            new_population.append(child)

        return new_population[:self.population_size]

    def _tournament_selection(self, fitnesses: np.ndarray, k: int = 2, size: int = 3) -> List[int]:
        selected: List[int] = []
        for _ in range(k):
            candidates = self.rng.integers(0, len(fitnesses), size=size)
            best_idx = candidates[np.argmax(fitnesses[candidates])]
            selected.append(int(best_idx))
        return selected

    def _crossover(self, parent_a: Plan, parent_b: Plan) -> Plan:
        child_plans: dict[str, DronePlan] = {}
        for drone_id in self.drone_ids:
            plan_a = parent_a.drones[drone_id]
            plan_b = parent_b.drones[drone_id]
            alpha = float(self.rng.uniform(0.3, 0.7))
            velocity = alpha * plan_a.velocity + (1 - alpha) * plan_b.velocity

            jammers: List[JammerPlan] = []
            for idx in range(self.n_jammers):
                jammer_a = plan_a.jammers[idx % len(plan_a.jammers)]
                jammer_b = plan_b.jammers[idx % len(plan_b.jammers)]
                beta = float(self.rng.uniform(0.2, 0.8))
                release = beta * jammer_a.release_time + (1 - beta) * jammer_b.release_time
                delay = beta * jammer_a.smoke_delay + (1 - beta) * jammer_b.smoke_delay
                jammers.append(JammerPlan(release, delay))

            child_plans[drone_id] = DronePlan(velocity=velocity.copy(), jammers=jammers)
        return Plan(child_plans)

    def _mutate(self, plan: Plan) -> None:
        for drone_id, drone_plan in plan.drones.items():
            if self.rng.random() < 0.5:
                noise = self.rng.normal(0.0, 6.0, size=2)
                drone_plan.velocity[:2] += noise
            for jammer in drone_plan.jammers:
                if self.rng.random() < 0.4:
                    jammer.release_time += float(self.rng.normal(0.0, 0.8))
                if self.rng.random() < 0.4:
                    jammer.smoke_delay += float(self.rng.normal(0.0, 0.6))

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------
    def _evaluate_with_cache(self, plan: Plan) -> float:
        signature = self._plan_signature(plan)
        if signature not in self.fitness_cache:
            self.fitness_cache[signature] = self._evaluate(plan)
        return self.fitness_cache[signature]

    def _evaluate(self, plan: Plan) -> float:
        system = GlobalSystem(self.initial_positions, self.initial_vectors,
                              config=self.system_config)
        apply_plan(system, plan)
        return system.total_cover_duration(self.targeted_missile_ids)

    def _plan_signature(self, plan: Plan) -> tuple:
        signature = []
        for drone_id in self.drone_ids:
            drone_plan = plan.drones[drone_id]
            velocity = tuple(np.round(drone_plan.velocity[:2], 2))
            jammers = tuple(
                (round(j.release_time, 2), round(j.smoke_delay, 2))
                for j in drone_plan.jammers
            )
            signature.append((drone_id, velocity, jammers))
        return tuple(signature)
