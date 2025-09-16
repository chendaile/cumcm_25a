# CUMCM 2025A 项目

全新的代码架构将原有的脚本式实现拆分为清晰的模块：核心物理模型、遗传算法、分析工具以及 Web 可视化。`src/cumcm` 目录提供了正式的 Python 包，便于在命令行、测试以及网站中重复使用。

## 目录结构

```
├── src/
│   ├── cumcm/
│   │   ├── __init__.py              # 对外暴露的统一接口
│   │   ├── analysis.py              # 仿真统计、Excel 导出、JSON 序列化
│   │   ├── entities.py              # 无人机、导弹、干扰弹等核心对象
│   │   ├── ga.py                    # 遗传算法求解器
│   │   ├── geometry.py              # 遮挡检测相关的几何计算
│   │   ├── plan.py                  # 方案（速度 + 干扰弹）的数据模型
│   │   ├── system.py                # 全局系统仿真（遮挡计算等）
│   │   ├── visualization.py         # 基于 Matplotlib 的静态可视化
│   │   ├── workflows.py             # 高层工作流封装（优化 / 校验）
│   │   └── webapp/
│   │       └── app.py               # Flask 小型网站，展示优化结果
│   ├── Lets_Optimize.py             # 命令式入口，封装常用流程
│   ├── Q1.py / Q2345.py             # 比赛题目脚本，调用新架构
├── tests/                           # Pytest 单元测试
│   ├── test_plan.py
│   └── test_system.py
├── data-bin/                        # 输入数据（位置、初始参数等）
└── output/                          # 结果输出（Excel、日志、截图等）
```

## 运行遗传算法示例

```bash
cd cumcm_25a
export PYTHONPATH="src"
python src/Lets_Optimize.py  # 内置了一个 demo，运行少量世代以示范流程
```

在实际任务中，可通过 `Lets_Optimize.Lets_optimize` 函数灵活控制参数，例如：

```python
from Lets_Optimize import Lets_optimize

workflow = Lets_optimize(
    drone_ids=["FY1", "FY2"],
    n_jammers=2,
    population_size=300,
    generations=120,
    Qname="Q4",
    targeted_missile_ids=["M1"],
    random_seed=42,
)
```

优化完成后会自动：

1. 在 `output/log/` 写入 JSON 方案及统计信息。
2. 在 `output/excel/` 输出物理参数表。
3. （可选）导出可视化帧到 `output/photos/`。

## 复现 / 校验既有方案

```python
from Lets_Optimize import test

test("output/log/Q5.json", targeted_missiles=["M1", "M2", "M3"], video=False)
```

若需要单独导出 Excel：

```python
from Lets_Optimize import export_physical_parameters_to_excel

export_physical_parameters_to_excel(
    "output/log/Q5.json",
    targeted_missiles=["M1", "M2", "M3"],
    output_path="output/excel/Q5-review.xlsx",
)
```

## 启动可视化网站

确保已经生成过至少一个 JSON 方案（位于 `output/log/`），然后执行：

```bash
cd cumcm_25a
export PYTHONPATH="src"
python -m cumcm.webapp.app
```

浏览器访问 `http://127.0.0.1:5000/` 即可查看：

- 方案列表与基础指标
- 各导弹遮挡时长柱状图（Chart.js）
- 每枚干扰弹的主要干扰目标与实际持续时间

## 运行测试

```bash
cd cumcm_25a
export PYTHONPATH="src"
pytest
```

测试覆盖了方案约束、仿真计算等核心逻辑，保证重构后的稳定性。

---

如需扩展，可参考 `src/cumcm/workflows.py` 内的 API —— 所有模块均已解耦，便于接入新的优化器或前端界面。
