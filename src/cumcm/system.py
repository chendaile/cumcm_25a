"""High level orchestration of drones, missiles and jammers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .entities import Drone, Jammer, Missile, TrueGoal
from .geometry import check_occlusion
from .plan import Plan


@dataclass
class SystemConfig:
    """Immutable configuration for the :class:`GlobalSystem`."""

    time_horizon: float = 30.0
    time_step: float = 0.02

    @property
    def time_grid(self) -> np.ndarray:
        return np.arange(0.0, self.time_horizon + self.time_step, self.time_step)


class GlobalSystem:
    """Simulation environment used for evaluation and optimisation."""

    def __init__(
        self,
        initial_positions: Mapping[str, Mapping[str, Sequence[float]]],
        drones_forward_vector: Mapping[str, Sequence[float]],
        config: SystemConfig | None = None,
    ) -> None:
        self.config = config or SystemConfig()

        drones_positions = initial_positions.get("drones", {})
        missiles_positions = initial_positions.get("missiles", {})
        target_positions = initial_positions.get("target", {})

        self.drones: Dict[str, Drone] = {}
        self.jammers: Dict[str, List[Jammer]] = {}
        for drone_id, pos in drones_positions.items():
            if drone_id not in drones_forward_vector:
                continue
            drone = Drone(np.array(pos, dtype=float),
                          np.array(drones_forward_vector[drone_id], dtype=float))
            self.drones[drone_id] = drone
            self.jammers[drone_id] = []

        self.missiles: Dict[str, Missile] = {
            missile_id: Missile(np.array(position, dtype=float))
            for missile_id, position in missiles_positions.items()
        }

        true_target = target_positions.get("true_target")
        if true_target is None:
            raise ValueError("initial_positions must contain target.true_target")
        self.true_goal = TrueGoal(np.array(true_target, dtype=float))
        self._occlusion_points = list(self.true_goal.sample_occlusion_points())

    @classmethod
    def from_json(
        cls,
        position_path: str | Path,
        forward_vector_path: str | Path,
        *,
        config: SystemConfig | None = None,
    ) -> "GlobalSystem":
        import json

        with Path(position_path).open("r", encoding="utf-8") as f:
            initial_positions = json.load(f)
        with Path(forward_vector_path).open("r", encoding="utf-8") as f:
            forward_vectors = json.load(f)
        return cls(initial_positions, forward_vectors, config=config)

    # ------------------------------------------------------------------
    # State management helpers
    # ------------------------------------------------------------------
    def reset_jammers(self, drone_id: str) -> None:
        if drone_id in self.jammers:
            self.jammers[drone_id] = []

    def add_jammer(self, drone_id: str, release_delay: float,
                   smoke_release_delay: float) -> Jammer:
        jammer = self.drones[drone_id].create_jammer(release_delay, smoke_release_delay)
        self.jammers[drone_id].append(jammer)
        return jammer

    def add_jammers(self, drone_id: str, jammers: Iterable[tuple[float, float]]) -> None:
        self.reset_jammers(drone_id)
        for release_delay, smoke_delay in jammers:
            self.add_jammer(drone_id, release_delay, smoke_delay)

    def update_drone_velocity(self, drone_id: str, velocity_vector: Sequence[float]) -> None:
        drone = self.drones[drone_id]
        new_vector = np.array(velocity_vector, dtype=float)
        if new_vector.shape == (2,):
            new_vector = np.array([new_vector[0], new_vector[1], 0.0], dtype=float)
        drone.forward_vector = new_vector
        drone._validate_speed()

    # ------------------------------------------------------------------
    # Occlusion calculations
    # ------------------------------------------------------------------
    def detect_occlusion_single_jammer(self, global_t: float, missile: Missile,
                                       jammer: Jammer) -> bool:
        if not jammer.smoke.is_active(global_t):
            return False
        try:
            missile_pos = missile.position_at(global_t)
        except ValueError:
            return False
        smoke_pos = jammer.smoke.position_at(global_t)

        for target_point in self._occlusion_points:
            if not check_occlusion(missile_pos, target_point, smoke_pos,
                                   jammer.smoke.radius):
                return False
        return True

    def detect_occlusion_all_jammers(self, global_t: float, missile: Missile,
                                     jammers: Iterable[Jammer]) -> bool:
        return any(self.detect_occlusion_single_jammer(global_t, missile, jammer)
                   for jammer in jammers)

    # ------------------------------------------------------------------
    # Coverage statistics
    # ------------------------------------------------------------------
    def cover_intervals(self, missile_ids: Sequence[str]) -> Dict[str, List[tuple[float, float]]]:
        intervals: Dict[str, List[tuple[float, float]]] = {}
        time_grid = self.config.time_grid
        all_jammers = [jammer for jammers in self.jammers.values() for jammer in jammers]

        for missile_id in missile_ids:
            missile = self.missiles[missile_id]
            covered_times: List[float] = []
            for global_t in time_grid:
                if self.detect_occlusion_all_jammers(global_t, missile, all_jammers):
                    covered_times.append(float(global_t))

            intervals[missile_id] = self._times_to_intervals(covered_times)
        return intervals

    def cover_durations(self, missile_ids: Sequence[str]) -> Dict[str, float]:
        intervals = self.cover_intervals(missile_ids)
        return {
            missile_id: sum(end - start for start, end in ranges)
            for missile_id, ranges in intervals.items()
        }

    def merged_coverage(self, missile_ids: Sequence[str]) -> List[tuple[float, float]]:
        if not missile_ids:
            return []
        intervals = self.cover_intervals(missile_ids)
        result = intervals[missile_ids[0]]
        for missile_id in missile_ids[1:]:
            result = self._intersect_intervals(result, intervals[missile_id])
            if not result:
                break
        return result

    def total_cover_duration(self, missile_ids: Sequence[str]) -> float:
        return sum(end - start for start, end in self.merged_coverage(missile_ids))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _times_to_intervals(times: List[float], gap_threshold: float = 0.1) -> List[tuple[float, float]]:
        if not times:
            return []
        intervals: List[tuple[float, float]] = []
        start = previous = times[0]
        for current in times[1:]:
            if current - previous > gap_threshold:
                intervals.append((start, previous))
                start = current
            previous = current
        intervals.append((start, previous))
        return intervals

    @staticmethod
    def _intersect_intervals(intervals_a: List[tuple[float, float]],
                             intervals_b: List[tuple[float, float]]) -> List[tuple[float, float]]:
        result: List[tuple[float, float]] = []
        idx_a = idx_b = 0
        while idx_a < len(intervals_a) and idx_b < len(intervals_b):
            start_a, end_a = intervals_a[idx_a]
            start_b, end_b = intervals_b[idx_b]
            start = max(start_a, start_b)
            end = min(end_a, end_b)
            if start < end:
                result.append((start, end))
            if end_a <= end_b:
                idx_a += 1
            else:
                idx_b += 1
        return result


def apply_plan(system: GlobalSystem, plan: Plan) -> None:
    """Apply a plan of velocities and jammers to the system."""

    for drone_id, drone_plan in plan.drones.items():
        system.update_drone_velocity(drone_id, drone_plan.velocity)
        system.add_jammers(drone_id,
                           [jammer.as_tuple() for jammer in drone_plan.jammers])
