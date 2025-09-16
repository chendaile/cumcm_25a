"""Matplotlib based visualisation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import compute_interference_stats
from .system import GlobalSystem


def plot_scene(system: GlobalSystem, global_t: float, *, save_path: str | Path | None = None,
               show: bool = False) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for missile_id, missile in system.missiles.items():
        try:
            missile_pos = missile.position_at(global_t)
        except ValueError:
            continue
        ax.scatter(*missile_pos, color="tab:blue", s=60, label=f"{missile_id} Missile")
        ax.text(*missile_pos, f" {missile_id}", color="tab:blue")

    target_pos = system.true_goal.bottom_center_pos
    ax.scatter(*target_pos, color="tab:green", s=50, label="True Target")

    for drone_id, drone in system.drones.items():
        drone_pos = drone.position_at(global_t)
        ax.scatter(*drone_pos, color="tab:red", s=60, label=drone_id)
        ax.text(*drone_pos, f" {drone_id}", color="tab:red")

    colors = plt.cm.get_cmap("viridis", 8)
    all_jammers = [jammer for jammers in system.jammers.values() for jammer in jammers]
    for idx, jammer in enumerate(all_jammers):
        if not jammer.smoke.is_active(global_t):
            continue
        smoke_pos = jammer.smoke.position_at(global_t)
        color = colors(idx)

        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        x = jammer.smoke.radius * np.outer(np.cos(u), np.sin(v)) + smoke_pos[0]
        y = jammer.smoke.radius * np.outer(np.sin(u), np.sin(v)) + smoke_pos[1]
        z = jammer.smoke.radius * np.outer(np.ones_like(u), np.cos(v)) + smoke_pos[2]
        ax.plot_surface(x, y, z, alpha=0.3, color=color)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Scene at t={global_t:.2f}s")
    ax.set_xlim([0, 22000])
    ax.set_ylim([-3500, 1500])
    ax.set_zlim([0, 2000])

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def export_frames(system: GlobalSystem, time_points: Sequence[float], output_dir: str | Path) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for idx, global_t in enumerate(time_points):
        path = Path(output_dir) / f"frame_{idx:04d}_t_{global_t:.2f}s.png"
        plot_scene(system, float(global_t), save_path=path, show=False)


def jam_interference_summary(system: GlobalSystem, targeted_missiles: Sequence[str]) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    for drone_id, jammers in system.jammers.items():
        for idx, jammer in enumerate(jammers, start=1):
            duration, missile = compute_interference_stats(system, jammer, targeted_missiles)
            summary.append({
                "drone": drone_id,
                "jammer": idx,
                "duration": duration,
                "missile": missile,
            })
    return summary
