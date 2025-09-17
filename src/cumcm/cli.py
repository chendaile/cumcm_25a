"""Command line interface for solving and inspecting optimisation tasks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .questions import QUESTION_PRESETS
from .services import available_plans, run_question, summarise_plan
from .workflows import OUTPUT_DIR


def _register_override_arguments(parser: argparse.ArgumentParser) -> None:
    """Add dynamic override arguments derived from question presets."""
    seen: set[str] = set()
    for preset in QUESTION_PRESETS.values():
        for field_name, config in preset.get("form", {}).items():
            if field_name in seen:
                continue
            option = f"--{field_name.replace('_', '-')}"
            kwargs: Dict[str, Any] = {
                "help": config.get("label") or field_name.replace("_", " "),
                "default": None,
            }
            value_type = config.get("value_type")
            if value_type == "int":
                kwargs["type"] = int
            elif value_type == "float":
                kwargs["type"] = float
            else:
                kwargs["type"] = str
            parser.add_argument(option, **kwargs)
            seen.add(field_name)


def _collect_overrides(args: argparse.Namespace, question_id: str) -> Dict[str, Any]:
    preset = QUESTION_PRESETS.get(question_id, {})
    overrides: Dict[str, Any] = {}
    for field_name in preset.get("form", {}):
        value = getattr(args, field_name, None)
        if value is not None:
            overrides[field_name] = value
    return overrides


def _cmd_solve(args: argparse.Namespace) -> int:
    question_id = args.question
    if question_id not in QUESTION_PRESETS:
        raise SystemExit(f"未知的题目编号: {question_id}")

    overrides = _collect_overrides(args, question_id)
    if args.name:
        overrides["Qname"] = args.name
    if args.missiles:
        overrides["targeted_missile_ids"] = args.missiles

    print(f"[cumcm] 正在运行 {QUESTION_PRESETS[question_id]['label']}...")

    def _progress(update: Dict[str, Any]) -> None:
        if update.get("event") != "progress":
            return
        generation = update.get("generation", 0) + 1
        best = update.get("best_fitness")
        generation_best = update.get("generation_best")
        if best is None or generation_best is None:
            return
        print(f"  · 第 {generation} 代: 当前最优 {generation_best:.2f} / 历史最优 {best:.2f}")

    summary = run_question(question_id, overrides, progress_callback=_progress)

    metadata = summary.get("metadata", {})
    plan_name = summary.get("plan_name")
    print("[cumcm] 优化完成。")
    if plan_name:
        print(f"  · 方案文件: {(OUTPUT_DIR / 'log' / plan_name).resolve()}")
    excel_path = OUTPUT_DIR / "excel" / f"{Path(plan_name).stem}.xlsx" if plan_name else None
    if excel_path and excel_path.exists():
        print(f"  · Excel 导出: {excel_path.resolve()}")
    print(f"  · 覆盖导弹: {', '.join(summary.get('targeted_missiles', []))}")
    print(f"  · 总遮挡时间: {summary.get('total_duration', 0):.2f} s")
    if metadata.get("fitness") is not None:
        print(f"  · 最佳适应度: {metadata['fitness']:.2f}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    if not args.plan:
        plans = available_plans()
        if not plans:
            print("[cumcm] 暂无已保存的方案。")
            return 0
        print("[cumcm] 可用方案：")
        for plan in plans:
            missiles = plan.get("targeted_missiles") or []
            print(f"  · {plan['name']:30} {plan['size_kb']:>4} KB 目标: {', '.join(missiles)}")
        return 0

    missiles = args.missiles or None
    summary = summarise_plan(args.plan, missiles=missiles)
    metadata = summary.get("metadata", {})
    print(f"[cumcm] 方案 {summary['plan_name']} 摘要：")
    print(f"  · 目标导弹: {', '.join(summary.get('targeted_missiles', []))}")
    print(f"  · 总遮挡时间: {summary.get('total_duration', 0):.2f} s")
    if metadata.get("fitness") is not None:
        print(f"  · 最佳适应度: {metadata['fitness']:.2f}")
    if metadata.get("generated_at"):
        print(f"  · 生成时间: {metadata['generated_at']}")

    if args.export_dir:
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        json_src = OUTPUT_DIR / "log" / summary["plan_name"]
        excel_src = OUTPUT_DIR / "excel" / f"{Path(summary['plan_name']).stem}.xlsx"
        json_dst = export_dir / json_src.name
        shutil.copy2(json_src, json_dst)
        print(f"  · 已导出 JSON: {json_dst}")
        if excel_src.exists():
            excel_dst = export_dir / excel_src.name
            shutil.copy2(excel_src, excel_dst)
            print(f"  · 已导出 Excel: {excel_dst}")
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cumcm", description="遮挡优化命令行工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="运行优化题目并生成方案")
    solve_parser.add_argument("question", help="题目编号，例如 Q5")
    solve_parser.add_argument("--name", help="保存方案时使用的名称", default=None)
    solve_parser.add_argument(
        "--missiles",
        nargs="*",
        help="覆盖的导弹编号（默认使用题目配置）",
        default=None,
    )
    _register_override_arguments(solve_parser)
    solve_parser.set_defaults(func=_cmd_solve)

    show_parser = subparsers.add_parser("show", help="浏览或导出已生成的方案")
    show_parser.add_argument("--plan", help="要查看的方案文件名", default=None)
    show_parser.add_argument(
        "--missiles",
        nargs="*",
        help="重新计算时使用的导弹编号",
        default=None,
    )
    show_parser.add_argument(
        "--export-dir",
        help="将方案及其导出文件复制到该目录",
        default=None,
    )
    show_parser.add_argument(
        "--as-json",
        action="store_true",
        help="以 JSON 形式输出详细摘要",
    )
    show_parser.set_defaults(func=_cmd_show)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
