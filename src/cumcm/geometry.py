"""Geometric helpers used across the project."""

from __future__ import annotations

import numpy as np


def check_occlusion(missile_pos: np.ndarray, target_pos: np.ndarray,
                    smoke_pos: np.ndarray, smoke_radius: float = 10.0) -> bool:
    """Return ``True`` if the smoke sphere occludes the line of sight."""

    missile_to_target = target_pos - missile_pos
    missile_to_smoke = smoke_pos - missile_pos

    if np.dot(missile_to_smoke, missile_to_target) <= 0:
        return False

    target_norm = np.linalg.norm(missile_to_target)
    if target_norm == 0:
        return True

    proj_length = np.dot(missile_to_smoke, missile_to_target) / target_norm
    proj_point = missile_pos + (proj_length / target_norm) * missile_to_target
    distance = np.linalg.norm(smoke_pos - proj_point)
    return distance <= smoke_radius
