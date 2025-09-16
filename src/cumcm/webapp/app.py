"""Flask web application for browsing optimisation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template_string, request

from ..analysis import run_simulation, serialize_plan
from ..plan import Plan
from ..visualization import jam_interference_summary
from ..workflows import FORWARD_VECTOR_PATH, OUTPUT_DIR, POSITIONS_PATH

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Smoke Jamming Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; margin: 0; background: #f5f7fb; color: #222; }
        header { background: #202a44; color: white; padding: 1.5rem 2rem; }
        main { padding: 2rem; max-width: 1100px; margin: 0 auto; }
        .card { background: white; border-radius: 12px; box-shadow: 0 10px 20px rgba(0,0,0,0.08); padding: 1.5rem; margin-bottom: 1.5rem; }
        select { padding: 0.5rem 0.75rem; border-radius: 8px; border: 1px solid #ccd3e0; font-size: 1rem; }
        h1 { margin: 0; font-weight: 600; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { padding: 0.6rem; border-bottom: 1px solid #edf1f9; text-align: left; }
        th { background: #f0f4ff; }
        .metrics { display: flex; gap: 1.5rem; flex-wrap: wrap; }
        .metric { flex: 1 0 220px; background: linear-gradient(135deg, #e0ecff, #f8faff); border-radius: 10px; padding: 1rem; }
        .metric span { display: block; font-size: 0.9rem; color: #475168; margin-bottom: 0.3rem; }
        .metric strong { font-size: 1.4rem; color: #14213d; }
    </style>
</head>
<body>
    <header>
        <h1>Smoke Jamming Mission Dashboard</h1>
    </header>
    <main>
        <div class="card">
            <label for="plan-select">选择方案：</label>
            <select id="plan-select" onchange="onPlanChange(this.value)">
                {% for plan in plans %}
                <option value="{{ plan }}" {% if plan == selected_plan %}selected{% endif %}>{{ plan }}</option>
                {% endfor %}
            </select>
        </div>
        {% if summary %}
        <div class="card">
            <div class="metrics">
                <div class="metric">
                    <span>总遮挡时间</span>
                    <strong>{{ summary.total_duration | round(2) }} s</strong>
                </div>
                <div class="metric">
                    <span>目标导弹</span>
                    <strong>{{ ', '.join(summary.targeted_missiles) }}</strong>
                </div>
                <div class="metric">
                    <span>无人机数量</span>
                    <strong>{{ summary.drone_count }}</strong>
                </div>
            </div>
        </div>
        <div class="card">
            <h2>遮挡时长分布</h2>
            <canvas id="duration-chart" height="120"></canvas>
        </div>
        <div class="card">
            <h2>干扰弹详情</h2>
            <table>
                <thead>
                    <tr><th>无人机</th><th>干扰弹</th><th>主要干扰导弹</th><th>有效时长 (s)</th></tr>
                </thead>
                <tbody>
                {% for row in summary.jammers %}
                    <tr>
                        <td>{{ row.drone }}</td>
                        <td>#{{ row.jammer }}</td>
                        <td>{{ row.missile or '—' }}</td>
                        <td>{{ row.duration | round(2) }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        {% else %}
        <div class="card">
            <p>当前尚无优化结果，请先运行优化流程生成方案。</p>
        </div>
        {% endif %}
    </main>
    <script>
        const planData = {{ summary | tojson | safe }};
        function onPlanChange(value) {
            if (!value) return;
            const url = new URL(window.location.href);
            url.searchParams.set('plan', value);
            window.location.href = url.toString();
        }
        if (planData) {
            const ctx = document.getElementById('duration-chart');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: Object.keys(planData.durations),
                    datasets: [{
                        label: '遮挡时间 (s)',
                        data: Object.values(planData.durations),
                        backgroundColor: '#4c6ef5'
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: true } }
                }
            });
        }
    </script>
</body>
</html>
"""


def _available_plans() -> list[str]:
    log_dir = OUTPUT_DIR / "log"
    if not log_dir.exists():
        return []
    return sorted([path.name for path in log_dir.glob("*.json")])


def _load_plan(path: Path) -> tuple[Plan, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan = Plan.from_dict(payload.get("plan", {}))
    plan.clamp()
    return plan, payload


def _default_missiles() -> list[str]:
    data = json.loads(Path(POSITIONS_PATH).read_text(encoding="utf-8"))
    return list(data.get("missiles", {}).keys())


@app.route("/")
def index() -> str:
    plans = _available_plans()
    selected = request.args.get("plan")
    if plans and selected is None:
        selected = plans[0]

    summary: Optional[Dict[str, Any]] = None
    if selected:
        plan_path = OUTPUT_DIR / "log" / selected
        plan, payload = _load_plan(plan_path)
        targeted = payload.get("targeted_missiles") or _default_missiles()
        simulation = run_simulation(POSITIONS_PATH, FORWARD_VECTOR_PATH, plan, targeted)
        summary = {
            "durations": simulation.durations,
            "intervals": simulation.intervals,
            "total_duration": simulation.total_duration,
            "plan": serialize_plan(plan),
            "targeted_missiles": targeted,
            "drone_count": len(plan.drones),
            "jammers": jam_interference_summary(simulation.system, targeted),
        }

    return render_template_string(TEMPLATE, plans=plans, selected_plan=selected,
                                  summary=summary)


@app.route("/api/plan/<plan_name>")
def plan_api(plan_name: str):
    plan_path = OUTPUT_DIR / "log" / plan_name
    if not plan_path.exists():
        return jsonify({"error": "Plan not found"}), 404
    plan, payload = _load_plan(plan_path)
    targeted = payload.get("targeted_missiles") or _default_missiles()
    simulation = run_simulation(POSITIONS_PATH, FORWARD_VECTOR_PATH, plan, targeted)
    return jsonify({
        "durations": simulation.durations,
        "intervals": simulation.intervals,
        "total_duration": simulation.total_duration,
        "plan": serialize_plan(plan),
        "jammers": jam_interference_summary(simulation.system, targeted),
        "targeted_missiles": targeted,
    })


if __name__ == "__main__":
    app.run(debug=True)
