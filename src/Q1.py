"""Utilities to reproduce the analysis of question 1."""

from __future__ import annotations

from pathlib import Path

from cumcm import GlobalSystem, plot_scene

DATA_DIR = Path(__file__).resolve().parents[1] / "data-bin"


def _load_system() -> GlobalSystem:
    return GlobalSystem.from_json(
        DATA_DIR / "initial_positions.json",
        DATA_DIR / "initial_drones_forward_vector-Q1.json",
    )


def find_cover_seconds_Q1() -> None:
    system = _load_system()
    system.add_jammer("FY1", 1.5, 3.6)
    durations = system.cover_durations(list(system.missiles.keys()))
    total = sum(durations.values())
    print(f"Total coverage duration: {total:.2f} seconds")
    for missile_id, duration in durations.items():
        print(f"  {missile_id}: {duration:.2f}s")


def test_Q1(tmp_time: float = 7.9) -> None:
    system = _load_system()
    jammer = system.add_jammer("FY1", 1.5, 3.6)
    missile = system.missiles["M1"]
    result = system.detect_occlusion_single_jammer(tmp_time, missile, jammer)
    print(f"t={tmp_time:.2f}s: occlusion detected = {result}")
    plot_scene(system, tmp_time, show=True)


if __name__ == "__main__":
    test_Q1()
