"""Data models describing drone and jammer schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Sequence

import numpy as np


@dataclass
class JammerPlan:
    release_time: float
    smoke_delay: float

    def clamp(self, *, time_range: tuple[float, float] = (0.0, 25.0)) -> None:
        start, end = time_range
        self.release_time = float(np.clip(self.release_time, start, end))
        self.smoke_delay = float(np.clip(self.smoke_delay, 0.0, end))

    def as_tuple(self) -> tuple[float, float]:
        return (self.release_time, self.smoke_delay)


@dataclass
class DronePlan:
    velocity: np.ndarray
    jammers: List[JammerPlan] = field(default_factory=list)

    def clamp(self, *, speed_limits: tuple[float, float] = (70.0, 140.0)) -> None:
        min_speed, max_speed = speed_limits
        velocity = np.array(self.velocity, dtype=float, copy=True)
        if velocity.size < 2:
            raise ValueError("Velocity must have at least two components")
        if velocity.size == 2:
            velocity = np.append(velocity, 0.0)
        vector = velocity[:3]
        magnitude = float(np.linalg.norm(vector))
        if magnitude == 0.0:
            vector = np.array([-min_speed, 0.0, 0.0], dtype=float)
        else:
            if magnitude < min_speed:
                vector = vector * (min_speed / magnitude)
            elif magnitude > max_speed:
                vector = vector * (max_speed / magnitude)
        self.velocity = vector
        for jammer in self.jammers:
            jammer.clamp()
        self.jammers.sort(key=lambda item: item.release_time)
        for idx in range(1, len(self.jammers)):
            prev = self.jammers[idx - 1].release_time
            self.jammers[idx].release_time = max(
                self.jammers[idx].release_time, prev + 1.0)

    def as_tuple(self) -> tuple[np.ndarray, List[tuple[float, float]]]:
        return (self.velocity, [jammer.as_tuple() for jammer in self.jammers])


@dataclass
class Plan:
    drones: dict[str, DronePlan]

    def clamp(self) -> None:
        for drone_plan in self.drones.values():
            drone_plan.clamp()

    def as_mapping(self) -> dict[str, tuple[np.ndarray, List[tuple[float, float]]]]:
        return {drone_id: plan.as_tuple() for drone_id, plan in self.drones.items()}

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Sequence]) -> "Plan":
        drones: dict[str, DronePlan] = {}
        for drone_id, value in mapping.items():
            if isinstance(value, Mapping) and "velocity" in value:
                velocity = np.asarray(value["velocity"], dtype=float)
                jammers_raw = value.get("jammers", [])
                jammers = [
                    JammerPlan(j.get("release_time", 0.0), j.get("smoke_delay", 0.0))
                    for j in jammers_raw
                ]
            else:
                velocity = np.asarray(value[0], dtype=float)
                jammers = [JammerPlan(*jam) for jam in value[1]]
            drones[drone_id] = DronePlan(velocity=velocity, jammers=jammers)
        return cls(drones)
