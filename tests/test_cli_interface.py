from __future__ import annotations

import json
from pathlib import Path

import pytest

from cumcm import cli
from cumcm.workflows import OUTPUT_DIR


@pytest.fixture(autouse=True)
def _prepare_output_dirs():
    log_dir = OUTPUT_DIR / "log"
    excel_dir = OUTPUT_DIR / "excel"
    log_dir.mkdir(parents=True, exist_ok=True)
    excel_dir.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup created files to avoid interfering with other tests
    for directory in (log_dir, excel_dir):
        for path in directory.glob("test_cli_*"):
            path.unlink()


def test_cli_solve_uses_overrides(monkeypatch, capsys):
    overrides_captured = {}

    def fake_run_question(question_id, overrides, progress_callback=None):
        overrides_captured.update(overrides)
        if progress_callback:
            progress_callback(
                {
                    "event": "progress",
                    "generation": 0,
                    "generation_best": 12.5,
                    "best_fitness": 12.5,
                    "history": [12.5],
                }
            )
        return {
            "plan_name": "test_cli_plan.json",
            "targeted_missiles": ["M1"],
            "total_duration": 15.0,
            "metadata": {"fitness": 18.2},
            "downloads": {},
        }

    monkeypatch.setattr(cli, "run_question", fake_run_question)

    exit_code = cli.main(["solve", "Q5", "--population-size", "10", "--generations", "3"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "优化完成" in out
    assert overrides_captured["population_size"] == 10
    assert overrides_captured["generations"] == 3


def test_cli_show_lists_plans(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "available_plans",
        lambda: [{"name": "demo.json", "size_kb": 8, "targeted_missiles": ["M1"]}],
    )

    exit_code = cli.main(["show"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "demo.json" in out


def test_cli_show_plan_exports(monkeypatch, capsys, tmp_path):
    export_dir = tmp_path / "export"
    plan_name = "test_cli_plan.json"
    (OUTPUT_DIR / "log" / plan_name).write_text("{}", encoding="utf-8")
    excel_path = OUTPUT_DIR / "excel" / f"{Path(plan_name).stem}.xlsx"
    excel_path.write_text("excel", encoding="utf-8")

    def fake_summary(name, missiles=None):
        return {
            "plan_name": name,
            "targeted_missiles": missiles or ["M1"],
            "total_duration": 11.0,
            "metadata": {"fitness": 13.3, "generated_at": "2024-01-01T00:00:00"},
        }

    monkeypatch.setattr(cli, "summarise_plan", fake_summary)

    exit_code = cli.main([
        "show",
        "--plan",
        plan_name,
        "--missiles",
        "M1",
        "--export-dir",
        str(export_dir),
        "--as-json",
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "已导出 JSON" in out
    exported_json = export_dir / plan_name
    exported_excel = export_dir / excel_path.name
    assert exported_json.exists()
    assert exported_excel.exists()
    assert json.loads((OUTPUT_DIR / "log" / plan_name).read_text(encoding="utf-8")) == {}
