"""Question preset definitions and helpers shared between interfaces."""

from __future__ import annotations

from typing import Any, Dict, Iterable

QUESTION_PRESETS: Dict[str, Dict[str, Any]] = {
    "Q1": {
        "label": "Q1 单机遮挡演示",
        "category": "分析演示",
        "description": "验证 FY1 在固定投放策略下对 M1 的遮挡情况，并生成可视化截图。",
        "type": "analysis",
        "params": {
            "drone_id": "FY1",
            "release_time": 1.5,
            "smoke_delay": 3.6,
            "time": 7.9,
        },
        "info": {
            "无人机": "FY1",
            "投放延迟": "1.5 s",
            "起爆延迟": "3.6 s",
            "默认观测时间": "7.9 s",
        },
        "form": {
            "time": {
                "label": "观测时刻 (s)",
                "input_type": "number",
                "value_type": "float",
                "min": 0,
                "step": 0.1,
            }
        },
    },
    "Q2": {
        "label": "Q2 单机单弹优化",
        "category": "官方题目",
        "description": "单架无人机携带 1 枚干扰弹，对导弹 M1 进行遮挡优化。",
        "type": "optimisation",
        "params": {
            "drone_ids": ["FY1"],
            "n_jammers": 1,
            "population_size": 200,
            "generations": 80,
            "Qname": "Q2",
            "targeted_missile_ids": ["M1"],
            "random_seed": 123,
        },
        "info": {
            "无人机": "FY1",
            "干扰弹数量": 1,
            "目标导弹": "M1",
        },
        "form": {
            "population_size": {
                "label": "种群规模",
                "input_type": "number",
                "value_type": "int",
                "min": 50,
                "step": 10,
            },
            "generations": {
                "label": "迭代代数",
                "input_type": "number",
                "value_type": "int",
                "min": 10,
                "step": 10,
            },
            "random_seed": {
                "label": "随机种子",
                "input_type": "number",
                "value_type": "int",
                "placeholder": "可选",
            },
        },
    },
    "Q3": {
        "label": "Q3 单机三弹优化",
        "category": "官方题目",
        "description": "单架无人机携带 3 枚干扰弹，对导弹 M1 进行遮挡优化。",
        "type": "optimisation",
        "params": {
            "drone_ids": ["FY1"],
            "n_jammers": 3,
            "population_size": 150,
            "generations": 200,
            "Qname": "Q3",
            "targeted_missile_ids": ["M1"],
            "random_seed": None,
        },
        "info": {
            "无人机": "FY1",
            "干扰弹数量": 3,
            "目标导弹": "M1",
        },
        "form": {
            "population_size": {
                "label": "种群规模",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "generations": {
                "label": "迭代代数",
                "input_type": "number",
                "value_type": "int",
                "min": 50,
                "step": 10,
            },
            "random_seed": {
                "label": "随机种子",
                "input_type": "number",
                "value_type": "int",
                "placeholder": "可选",
            },
        },
    },
    "Q4": {
        "label": "Q4 三机协同优化",
        "category": "官方题目",
        "description": "三架无人机协同投放 1 枚干扰弹，覆盖导弹 M1。",
        "type": "optimisation",
        "params": {
            "drone_ids": ["FY1", "FY2", "FY3"],
            "n_jammers": 1,
            "population_size": 500,
            "generations": 120,
            "Qname": "Q4",
            "targeted_missile_ids": ["M1"],
            "random_seed": 1234,
        },
        "info": {
            "无人机": "FY1, FY2, FY3",
            "干扰弹数量": 1,
            "目标导弹": "M1",
        },
        "form": {
            "population_size": {
                "label": "种群规模",
                "input_type": "number",
                "value_type": "int",
                "min": 100,
                "step": 20,
            },
            "generations": {
                "label": "迭代代数",
                "input_type": "number",
                "value_type": "int",
                "min": 50,
                "step": 10,
            },
            "random_seed": {
                "label": "随机种子",
                "input_type": "number",
                "value_type": "int",
                "placeholder": "可选",
            },
        },
    },
    "Q5": {
        "label": "Q5 五机三弹优化",
        "category": "官方题目",
        "description": "五架无人机协同投放 3 枚干扰弹，覆盖多枚导弹。",
        "type": "optimisation",
        "params": {
            "drone_ids": ["FY1", "FY2", "FY3", "FY4", "FY5"],
            "n_jammers": 3,
            "population_size": 300,
            "generations": 120,
            "Qname": "Q5",
            "targeted_missile_ids": ["M1", "M2", "M3"],
            "random_seed": 2025,
        },
        "info": {
            "无人机": "FY1, FY2, FY3, FY4, FY5",
            "干扰弹数量": 3,
            "目标导弹": "M1, M2, M3",
        },
        "form": {
            "population_size": {
                "label": "种群规模",
                "input_type": "number",
                "value_type": "int",
                "min": 120,
                "step": 20,
            },
            "generations": {
                "label": "迭代代数",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "random_seed": {
                "label": "随机种子",
                "input_type": "number",
                "value_type": "int",
                "placeholder": "可选",
            },
        },
    },
    "Q5_FY1": {
        "label": "Q5 单机拆分 - FY1",
        "category": "辅助题目",
        "description": "拆分 Q5 任务，单独优化 FY1 的 3 枚干扰弹。",
        "type": "optimisation",
        "params": {
            "drone_ids": ["FY1"],
            "n_jammers": 3,
            "population_size": 120,
            "generations": 150,
            "Qname": "Q5_FY1",
            "targeted_missile_ids": ["M1", "M2", "M3"],
            "random_seed": None,
        },
        "info": {
            "无人机": "FY1",
            "干扰弹数量": 3,
            "目标导弹": "M1, M2, M3",
        },
        "form": {
            "population_size": {
                "label": "种群规模",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "generations": {
                "label": "迭代代数",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "random_seed": {
                "label": "随机种子",
                "input_type": "number",
                "value_type": "int",
                "placeholder": "可选",
            },
        },
    },
    "Q5_FY2": {
        "label": "Q5 单机拆分 - FY2",
        "category": "辅助题目",
        "description": "拆分 Q5 任务，单独优化 FY2 的 3 枚干扰弹。",
        "type": "optimisation",
        "params": {
            "drone_ids": ["FY2"],
            "n_jammers": 3,
            "population_size": 150,
            "generations": 150,
            "Qname": "Q5_FY2",
            "targeted_missile_ids": ["M1", "M2", "M3"],
            "random_seed": None,
        },
        "info": {
            "无人机": "FY2",
            "干扰弹数量": 3,
            "目标导弹": "M1, M2, M3",
        },
        "form": {
            "population_size": {
                "label": "种群规模",
                "input_type": "number",
                "value_type": "int",
                "min": 100,
                "step": 10,
            },
            "generations": {
                "label": "迭代代数",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "random_seed": {
                "label": "随机种子",
                "input_type": "number",
                "value_type": "int",
                "placeholder": "可选",
            },
        },
    },
    "Q5_FY3": {
        "label": "Q5 单机拆分 - FY3",
        "category": "辅助题目",
        "description": "拆分 Q5 任务，单独优化 FY3 的 3 枚干扰弹。",
        "type": "optimisation",
        "params": {
            "drone_ids": ["FY3"],
            "n_jammers": 3,
            "population_size": 150,
            "generations": 150,
            "Qname": "Q5_FY3",
            "targeted_missile_ids": ["M1", "M2", "M3"],
            "random_seed": None,
        },
        "info": {
            "无人机": "FY3",
            "干扰弹数量": 3,
            "目标导弹": "M1, M2, M3",
        },
        "form": {
            "population_size": {
                "label": "种群规模",
                "input_type": "number",
                "value_type": "int",
                "min": 100,
                "step": 10,
            },
            "generations": {
                "label": "迭代代数",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "random_seed": {
                "label": "随机种子",
                "input_type": "number",
                "value_type": "int",
                "placeholder": "可选",
            },
        },
    },
    "Q5_FY4": {
        "label": "Q5 单机拆分 - FY4",
        "category": "辅助题目",
        "description": "拆分 Q5 任务，单独优化 FY4 的 1 枚干扰弹。",
        "type": "optimisation",
        "params": {
            "drone_ids": ["FY4"],
            "n_jammers": 1,
            "population_size": 120,
            "generations": 150,
            "Qname": "Q5_FY4",
            "targeted_missile_ids": ["M1"],
            "random_seed": None,
        },
        "info": {
            "无人机": "FY4",
            "干扰弹数量": 1,
            "目标导弹": "M1",
        },
        "form": {
            "population_size": {
                "label": "种群规模",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "generations": {
                "label": "迭代代数",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "random_seed": {
                "label": "随机种子",
                "input_type": "number",
                "value_type": "int",
                "placeholder": "可选",
            },
        },
    },
    "Q5_FY5": {
        "label": "Q5 单机拆分 - FY5",
        "category": "辅助题目",
        "description": "拆分 Q5 任务，单独优化 FY5 的 1 枚干扰弹。",
        "type": "optimisation",
        "params": {
            "drone_ids": ["FY5"],
            "n_jammers": 1,
            "population_size": 120,
            "generations": 150,
            "Qname": "Q5_FY5",
            "targeted_missile_ids": ["M1", "M2", "M3"],
            "random_seed": None,
        },
        "info": {
            "无人机": "FY5",
            "干扰弹数量": 1,
            "目标导弹": "M1, M2, M3",
        },
        "form": {
            "population_size": {
                "label": "种群规模",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "generations": {
                "label": "迭代代数",
                "input_type": "number",
                "value_type": "int",
                "min": 80,
                "step": 10,
            },
            "random_seed": {
                "label": "随机种子",
                "input_type": "number",
                "value_type": "int",
                "placeholder": "可选",
            },
        },
    },
}


def question_metadata() -> list[dict[str, Any]]:
    """Return lightweight metadata for populating UI selections."""
    data = []
    for key, value in QUESTION_PRESETS.items():
        entry = {
            "id": key,
            "label": value.get("label", key),
            "category": value.get("category", ""),
            "description": value.get("description"),
            "info": value.get("info", {}),
        }
        form = value.get("form")
        if form:
            fields = []
            for field_key, field in form.items():
                field_entry = dict(field)
                field_entry["name"] = field_key
                fields.append(field_entry)
            entry["form"] = fields
        data.append(entry)
    return data


def parse_overrides(payload: Dict[str, Any], question_id: str) -> Dict[str, Any]:
    """Normalise override values according to the preset form definition."""
    overrides: Dict[str, Any] = {}
    preset = QUESTION_PRESETS.get(question_id, {})
    form = preset.get("form", {})
    for key, config in form.items():
        if key not in payload:
            continue
        value = payload[key]
        if value in (None, ""):
            continue
        value_type = config.get("value_type")
        if value_type == "int":
            overrides[key] = int(value)
        elif value_type == "float":
            overrides[key] = float(value)
        else:
            overrides[key] = value
    return overrides


def default_cli_fields(question_id: str) -> Iterable[str]:
    """Return CLI override field names for the given question."""
    preset = QUESTION_PRESETS.get(question_id, {})
    return preset.get("form", {}).keys()


__all__ = ["QUESTION_PRESETS", "question_metadata", "parse_overrides", "default_cli_fields"]
