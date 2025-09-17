from __future__ import annotations

from cumcm import services


def test_stream_question_yields_progress_and_completion(monkeypatch):
    events = []

    def fake_run_question(question_id, overrides, progress_callback=None):
        if progress_callback:
            progress_callback({
                "event": "progress",
                "generation": 0,
                "generation_best": 10.0,
                "best_fitness": 10.0,
                "history": [10.0],
            })
        return {"metadata": {"fitness": 10.0}, "targeted_missiles": [], "total_duration": 0.0}

    monkeypatch.setattr(services, "run_question", fake_run_question)

    stream = services.stream_question("Q5")
    for item in stream:
        events.append(item)

    assert any(event.get("event") == "progress" for event in events)
    assert events[-1].get("event") == "complete"
