"""Command helpers to run the genetic optimisation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from cumcm.analysis import export_to_excel
from cumcm.plan import Plan
from cumcm.visualization import export_frames
from cumcm.workflows import OUTPUT_DIR, WorkflowResult, load_plan, run_genetic_workflow, verify_plan


def Lets_optimize(drone_ids: Sequence[str], n_jammers: int, population_size: int,
                  generations: int, Qname: str,
                  targeted_missile_ids: Sequence[str], random_seed: int | None = None,
                  video: bool = False) -> WorkflowResult:
    save_json = OUTPUT_DIR / "log" / f"{Qname}.json"
    excel_path = OUTPUT_DIR / "excel" / f"{Qname}.xlsx"

    workflow = run_genetic_workflow(drone_ids, n_jammers, population_size,
                                    generations, targeted_missile_ids,
                                    random_seed=random_seed,
                                    save_json=True,
                                    save_json_path=save_json,
                                    export_excel_path=excel_path,
                                    video=video)

    _print_summary(workflow)
    return workflow


def test(plan_source: str | Path | Plan, targeted_missiles: Sequence[str],
         video: bool = False) -> None:
    if isinstance(plan_source, (str, Path)):
        plan = load_plan(plan_source)
    else:
        plan = plan_source
    result = verify_plan(plan, targeted_missiles)
    _print_result(result)
    if video:
        frames_dir = OUTPUT_DIR / "photos" / "verification_frames"
        time_points = result.system.config.time_grid[::10]
        export_frames(result.system, time_points, frames_dir)
        print(f"Frames exported to {frames_dir}")


def export_physical_parameters_to_excel(plan_source: str | Path | Plan,
                                        targeted_missiles: Sequence[str],
                                        output_path: str | Path) -> None:
    if isinstance(plan_source, (str, Path)):
        plan = load_plan(plan_source)
    else:
        plan = plan_source
    result = verify_plan(plan, targeted_missiles)
    export_to_excel(result, targeted_missiles, output_path)
    print(f"物理参数已保存到 {output_path}")


def _print_summary(workflow: WorkflowResult) -> None:
    result = workflow.optimisation_result
    print("Optimisation finished.")
    _print_result(result)
    print(f"Best coverage: {result.total_duration:.2f}s")
    print(f"Generations evaluated: {len(workflow.fitness_history)}")


def _print_result(result) -> None:
    print("Individual missile coverage:")
    for missile_id, duration in result.durations.items():
        print(f"  {missile_id}: {duration:.2f}s")
        intervals = result.intervals.get(missile_id, [])
        if intervals:
            for idx, (start, end) in enumerate(intervals, start=1):
                print(f"    Interval {idx}: {start:.2f}s - {end:.2f}s")
        else:
            print("    No coverage intervals")


if __name__ == "__main__":
    workflow = Lets_optimize([
        "FY1", "FY2", "FY3", "FY4", "FY5"
    ], n_jammers=3, population_size=50, generations=10,
        Qname="demo", targeted_missile_ids=["M1", "M2", "M3"], random_seed=42, video=False)
    _print_summary(workflow)
