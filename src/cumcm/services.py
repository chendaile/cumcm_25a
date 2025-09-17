"""Shared service-layer helpers for running workflows and formatting results."""

from __future__ import annotations

import base64
import json
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence

from .analysis import (
    compute_interference_stats,
    run_simulation,
    save_plan_json,
    serialize_plan,
)
from .plan import Plan
from .questions import QUESTION_PRESETS
from .system import GlobalSystem
from .visualization import jam_interference_summary, plot_scene
from .workflows import (
    FORWARD_VECTOR_PATH,
    OUTPUT_DIR,
    POSITIONS_PATH,
    run_genetic_workflow,
    verify_plan,
)

ProgressCallback = Callable[[Dict[str, Any]], None]


def default_missiles() -> list[str]:
    """Return the default missile identifiers used throughout the project."""
    data = json.loads(Path(POSITIONS_PATH).read_text(encoding="utf-8"))
    return list(data.get("missiles", {}).keys())


def available_plans() -> list[dict[str, Any]]:
    """Enumerate plan JSON files stored in the output directory."""
    plans: list[dict[str, Any]] = []
    log_dir = OUTPUT_DIR / "log"
    if not log_dir.exists():
        return plans
    for path in sorted(log_dir.glob("*.json"), reverse=True):
        size_kb = max(path.stat().st_size // 1024, 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        targeted = metadata.get("targeted_missiles", payload.get("targeted_missiles"))
        plans.append(
            {
                "name": path.name,
                "size_kb": size_kb,
                "targeted_missiles": targeted,
                "metadata": metadata,
                "downloads": {
                    "json": f"/download/log/{path.name}",
                    "excel": f"/download/excel/{path.stem}.xlsx",
                },
            }
        )
    return plans


def load_plan(path: Path) -> tuple[Plan, Dict[str, Any]]:
    """Load a plan JSON file and return both the plan and raw payload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan_data = payload.get("plan", payload)
    plan = Plan.from_dict(plan_data)
    plan.clamp()
    return plan, payload


def simulation_payload(simulation, targeted: Iterable[str]) -> Dict[str, Any]:
    """Convert a simulation result into the JSON payload used by front-ends."""
    targeted_list = list(targeted)
    return {
        "durations": simulation.durations,
        "intervals": {
            key: [[float(start), float(end)] for start, end in values]
            for key, values in simulation.intervals.items()
        },
        "total_duration": simulation.total_duration,
        "plan": serialize_plan(simulation.plan),
        "targeted_missiles": targeted_list,
        "drone_count": len(simulation.plan.drones),
        "jammers": jam_interference_summary(simulation.system, targeted_list),
    }


def build_dashboard_payload(selected: Optional[str]) -> Dict[str, Any]:
    """Construct the payload powering the web dashboard."""
    plans = available_plans()
    plan_names = [plan["name"] for plan in plans]
    selected_plan = selected or (plan_names[0] if plan_names else None)

    summary: Optional[Dict[str, Any]] = None
    if selected_plan and selected_plan in plan_names:
        plan_path = OUTPUT_DIR / "log" / selected_plan
        plan, payload = load_plan(plan_path)
        targeted = payload.get("targeted_missiles") or default_missiles()
        simulation = run_simulation(POSITIONS_PATH, FORWARD_VECTOR_PATH, plan, targeted)
        summary = simulation_payload(simulation, targeted)
        summary["metadata"] = {
            "fitness": payload.get("fitness"),
            "generated_at": payload.get("generated_at"),
            "question": payload.get("question"),
            "parameters": payload.get("parameters"),
        }
        summary["plan_name"] = selected_plan
        summary["downloads"] = {
            "json": f"/download/log/{selected_plan}",
            "excel": f"/download/excel/{Path(selected_plan).stem}.xlsx",
        }
    return {
        "plans": plans,
        "selected_plan": selected_plan,
        "summary": summary,
    }


def _save_plan_with_metadata(simulation, params: Dict[str, Any], json_path: Path) -> None:
    targeted = params.get("targeted_missile_ids", [])
    history = params.get("history", [])
    best_fitness = max(history) if history else None
    metadata = {
        "fitness": best_fitness,
        "targeted_missiles": list(targeted),
        "generated_at": datetime.utcnow().isoformat(),
        "question": params.get("Qname"),
        "parameters": {
            "drone_ids": params.get("drone_ids"),
            "n_jammers": params.get("n_jammers"),
            "population_size": params.get("population_size"),
            "generations": params.get("generations"),
            "random_seed": params.get("random_seed"),
        },
    }
    save_plan_json(simulation.plan, json_path, metadata=metadata)


def _run_q1_analysis(overrides: Dict[str, Any]) -> Dict[str, Any]:
    params = deepcopy(QUESTION_PRESETS["Q1"]["params"])
    params.update(overrides)

    drone_id = params.get("drone_id", "FY1")
    release_time = float(params.get("release_time", 1.5))
    smoke_delay = float(params.get("smoke_delay", 3.6))
    time_point = float(params.get("time", 7.9))

    q1_vector_path = FORWARD_VECTOR_PATH.parent / "initial_drones_forward_vector-Q1.json"
    vector_path = q1_vector_path if q1_vector_path.exists() else FORWARD_VECTOR_PATH

    system = GlobalSystem.from_json(POSITIONS_PATH, vector_path)
    jammer = system.add_jammer(drone_id, release_time, smoke_delay)

    targeted = list(system.missiles.keys())
    durations = system.cover_durations(targeted)
    intervals = system.cover_intervals(targeted)
    total_duration = sum(durations.values())

    occlusion = False
    if "M1" in system.missiles:
        occlusion = system.detect_occlusion_single_jammer(
            time_point, system.missiles["M1"], jammer
        )

    image_dir = OUTPUT_DIR / "photos" / "webapp"
    image_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    image_path = image_dir / f"Q1_scene_{timestamp}.png"
    plot_scene(system, time_point, save_path=image_path, show=False)
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")

    duration, missile_id = compute_interference_stats(system, jammer, targeted)

    payload = {
        "durations": durations,
        "intervals": {
            key: [[float(start), float(end)] for start, end in values]
            for key, values in intervals.items()
        },
        "total_duration": float(total_duration),
        "targeted_missiles": targeted,
        "drone_count": len(system.drones),
        "jammers": [
            {
                "drone": drone_id,
                "jammer": 1,
                "duration": duration,
                "missile": missile_id,
            }
        ],
        "context": {
            "occlusion": occlusion,
            "time": time_point,
            "release_time": release_time,
            "smoke_delay": smoke_delay,
            "image": f"data:image/png;base64,{encoded_image}",
            "download": f"/download/photos/{image_path.relative_to(OUTPUT_DIR / 'photos')}"
            if image_path.exists()
            else None,
        },
        "downloads": {
            "image": f"/download/photos/{image_path.relative_to(OUTPUT_DIR / 'photos')}"
            if image_path.exists()
            else None,
        },
    }
    return payload


def _optimisation_params(question_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    preset = QUESTION_PRESETS[question_id]
    params = deepcopy(preset.get("params", {}))
    params.update(overrides)
    return params


def run_optimisation_question(
    question_id: str,
    overrides: Dict[str, Any],
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    params = _optimisation_params(question_id, overrides)

    targeted = params.get("targeted_missile_ids", [])
    qname = params.get("Qname", question_id)
    json_path = OUTPUT_DIR / "log" / f"{qname}.json"
    excel_path = OUTPUT_DIR / "excel" / f"{qname}.xlsx"

    def _on_progress(generation: int, generation_best: float, best_fitness: float, history: Sequence[float]) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "event": "progress",
                "generation": generation,
                "generation_best": generation_best,
                "best_fitness": best_fitness,
                "history": list(history),
            }
        )

    workflow = run_genetic_workflow(
        params.get("drone_ids", []),
        params.get("n_jammers", 1),
        params.get("population_size", 100),
        params.get("generations", 50),
        targeted,
        random_seed=params.get("random_seed"),
        save_json=False,
        export_excel_path=excel_path,
        progress_callback=_on_progress,
    )

    history = workflow.fitness_history
    simulation = workflow.optimisation_result

    params_with_history = dict(params)
    params_with_history["history"] = history
    params_with_history["Qname"] = qname
    _save_plan_with_metadata(simulation, params_with_history, json_path)

    summary = simulation_payload(simulation, targeted)
    summary["history"] = history
    summary["plan_name"] = json_path.name
    summary["downloads"] = {
        "json": f"/download/log/{json_path.name}",
        "excel": f"/download/excel/{qname}.xlsx",
    }
    summary["metadata"] = {
        "fitness": max(history) if history else None,
        "generated_at": datetime.utcnow().isoformat(),
        "question": qname,
        "parameters": {
            "drone_ids": params.get("drone_ids"),
            "n_jammers": params.get("n_jammers"),
            "population_size": params.get("population_size"),
            "generations": params.get("generations"),
            "random_seed": params.get("random_seed"),
        },
    }
    return summary


def run_question(
    question_id: str,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    if question_id not in QUESTION_PRESETS:
        raise KeyError(f"未知的题目编号: {question_id}")
    overrides = overrides or {}
    preset = QUESTION_PRESETS[question_id]
    if preset.get("type") == "analysis":
        return _run_q1_analysis(overrides)
    return run_optimisation_question(question_id, overrides, progress_callback=progress_callback)


def summarise_plan(plan_name: str, missiles: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Return a summary payload for a stored plan."""
    plan_path = OUTPUT_DIR / "log" / plan_name
    if not plan_path.exists():
        raise FileNotFoundError(f"方案不存在: {plan_name}")
    plan, payload = load_plan(plan_path)
    targeted = list(missiles or payload.get("targeted_missiles") or default_missiles())
    simulation = verify_plan(plan, targeted)
    summary = simulation_payload(simulation, targeted)
    summary["plan_name"] = plan_name
    summary["downloads"] = {
        "json": f"/download/log/{plan_name}",
        "excel": f"/download/excel/{Path(plan_name).stem}.xlsx",
    }
    summary["metadata"] = {
        "fitness": payload.get("fitness"),
        "generated_at": payload.get("generated_at"),
        "question": payload.get("question"),
        "parameters": payload.get("parameters"),
    }
    return summary


def stream_question(
    question_id: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield incremental optimisation updates followed by the final summary."""
    overrides = overrides or {}
    queue: SimpleQueue[Optional[Dict[str, Any]]] = SimpleQueue()

    def _progress(payload: Dict[str, Any]) -> None:
        queue.put(payload)

    def _worker() -> None:
        try:
            summary = run_question(question_id, overrides, progress_callback=_progress)
            queue.put({"event": "complete", "summary": summary})
        except Exception as exc:  # pragma: no cover - surfaced to caller
            queue.put({"event": "error", "error": str(exc)})
        finally:
            queue.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        item = queue.get()
        if item is None:
            break
        yield item


__all__ = [
    "available_plans",
    "build_dashboard_payload",
    "default_missiles",
    "load_plan",
    "run_question",
    "run_optimisation_question",
    "simulation_payload",
    "stream_question",
    "summarise_plan",
]
