"""Interactive Flask web application for optimisation and verification."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from flask import (
    Flask,
    abort,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
)

from ..analysis import compute_interference_stats, run_simulation, save_plan_json, serialize_plan
from ..plan import Plan
from ..system import GlobalSystem
from ..visualization import jam_interference_summary, plot_scene
from ..workflows import (
    FORWARD_VECTOR_PATH,
    OUTPUT_DIR,
    POSITIONS_PATH,
    run_genetic_workflow,
    verify_plan,
)

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["JSON_SORT_KEYS"] = False

TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-cn">
<head>
    <meta charset="utf-8" />
    <title>遮挡优化控制台</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f3f6fb; color: #1f2933; }
        header { background: linear-gradient(135deg, #14213d, #274060); color: white; padding: 1.6rem 2.4rem; }
        header h1 { margin: 0; font-weight: 600; font-size: 1.8rem; }
        main { padding: 2.4rem; max-width: 1200px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.6rem; }
        .card { background: white; border-radius: 16px; box-shadow: 0 12px 30px rgba(15, 35, 95, 0.08); padding: 1.8rem; display: flex; flex-direction: column; gap: 1.2rem; }
        h2 { margin: 0; font-weight: 600; color: #1b2a4b; }
        label { font-weight: 600; color: #37425b; display: block; margin-bottom: 0.4rem; }
        select, input { width: 100%; padding: 0.6rem 0.75rem; border: 1px solid #d4d9e6; border-radius: 10px; font-size: 0.95rem; }
        button { border: none; border-radius: 10px; padding: 0.65rem 1.2rem; background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; font-weight: 600; cursor: pointer; transition: transform 0.2s ease, box-shadow 0.2s ease; }
        button:hover { transform: translateY(-1px); box-shadow: 0 12px 22px rgba(37, 99, 235, 0.2); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 0.55rem 0.6rem; border-bottom: 1px solid #e5e9f4; text-align: left; }
        th { background: #eef2ff; font-weight: 600; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }
        .metric { padding: 0.9rem 1rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.14), rgba(37, 99, 235, 0.2)); border-radius: 12px; }
        .metric span { display: block; font-size: 0.8rem; color: #47516b; margin-bottom: 0.3rem; }
        .metric strong { font-size: 1.25rem; color: #12284a; }
        .muted { color: #6c7a96; font-size: 0.9rem; }
        .downloads { display: flex; flex-wrap: wrap; gap: 0.8rem; }
        .downloads a { background: rgba(59, 130, 246, 0.08); padding: 0.45rem 0.75rem; border-radius: 8px; color: #1d4ed8; text-decoration: none; font-weight: 600; }
        .downloads a:hover { background: rgba(37, 99, 235, 0.12); }
        .section-title { font-size: 1.05rem; font-weight: 600; color: #1c2a4d; }
        .context-block { background: #f4f7ff; border: 1px solid #d9e2ff; border-radius: 12px; padding: 0.9rem 1.1rem; }
        .context-block img { max-width: 100%; border-radius: 12px; margin-top: 0.6rem; }
        .question-info { background: #f5f7fa; border-radius: 12px; padding: 0.8rem 1rem; border: 1px solid #e1e7f5; }
        .question-info ul { padding-left: 1.2rem; margin: 0.4rem 0 0; }
        .question-info li { margin-bottom: 0.2rem; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem; }
        .status-log { background: #0f172a; color: #f1f5f9; padding: 1rem; border-radius: 12px; font-size: 0.9rem; max-height: 220px; overflow-y: auto; }
        .status-log strong { color: #38bdf8; }
        @media (max-width: 720px) {
            main { padding: 1.6rem; }
            header { padding: 1.4rem 1.6rem; }
        }
    </style>
</head>
<body>
    <header>
        <h1>烟幕干扰全流程控制台</h1>
        <p class="muted">一站式管理：题目运行、方案验证、可视化展示与结果下载</p>
    </header>
    <main>
        <div class="grid">
            <section class="card" style="grid-column: 1 / -1;">
                <h2>方案浏览</h2>
                <label for="plan-select">选择已有方案</label>
                <select id="plan-select"></select>
                <div class="downloads" id="plan-downloads"></div>
                <div id="plan-summary" class="summary-block">
                    <div class="metrics" id="plan-metrics"></div>
                    <canvas id="plan-chart" height="110"></canvas>
                    <div id="plan-history-wrapper" style="display:none;">
                        <p class="section-title">优化历史</p>
                        <canvas id="plan-history" height="110"></canvas>
                    </div>
                    <div>
                        <p class="section-title">干扰弹详情</p>
                        <div id="plan-jammers"></div>
                    </div>
                    <div>
                        <p class="section-title">遮挡区间</p>
                        <div id="plan-intervals"></div>
                    </div>
                    <div id="plan-context"></div>
                </div>
            </section>

            <section class="card">
                <h2>题目运行</h2>
                <label for="question-select">选择题目</label>
                <select id="question-select"></select>
                <div class="question-info" id="question-info"></div>
                <div class="form-grid" id="question-parameters"></div>
                <button id="run-question">运行题目</button>
                <div class="downloads" id="question-downloads"></div>
                <div id="question-result" class="summary-block">
                    <div class="metrics" id="question-metrics"></div>
                    <canvas id="question-chart" height="110"></canvas>
                    <div id="question-history-wrapper" style="display:none;">
                        <p class="section-title">优化历史</p>
                        <canvas id="question-history" height="110"></canvas>
                    </div>
                    <div id="question-jammers"></div>
                    <div id="question-intervals"></div>
                    <div id="question-context"></div>
                </div>
            </section>

            <section class="card">
                <h2>方案验证</h2>
                <label for="verify-plan">待验证方案</label>
                <select id="verify-plan"></select>
                <label for="verify-missiles">目标导弹（可多选）</label>
                <select id="verify-missiles" multiple size="6"></select>
                <button id="verify-button">重新计算遮挡</button>
                <div class="downloads" id="verify-downloads"></div>
                <div id="verify-result" class="summary-block">
                    <div class="metrics" id="verify-metrics"></div>
                    <canvas id="verify-chart" height="110"></canvas>
                    <div id="verify-history-wrapper" style="display:none;"><p class="section-title">优化历史</p><canvas id="verify-history" height="110"></canvas></div>
                    <div id="verify-jammers"></div>
                    <div id="verify-intervals"></div>
                    <div id="verify-context"></div>
                </div>
            </section>

            <section class="card" style="grid-column: 1 / -1;">
                <h2>系统日志</h2>
                <div class="status-log" id="status-log"></div>
            </section>
        </div>
    </main>

    <script>
        const initialData = {{ initial_payload | tojson }};
        const questionPresets = {{ question_metadata | tojson }};
        const missileOptions = {{ missile_options | tojson }};

        const charts = {};

        function logStatus(message, type = 'info') {
            const log = document.getElementById('status-log');
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.innerHTML = `<strong>[${time}]</strong> ${message}`;
            if (type === 'error') {
                entry.style.color = '#f87171';
            } else if (type === 'success') {
                entry.style.color = '#34d399';
            }
            log.prepend(entry);
        }

        function createMetric(label, value) {
            return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
        }

        function formatNumber(value, digits = 2) {
            const num = Number(value);
            if (Number.isNaN(num)) return '—';
            return num.toFixed(digits);
        }

        function ensureChart(id, type) {
            const canvas = document.getElementById(id);
            if (!canvas) return null;
            if (charts[id]) {
                return charts[id];
            }
            charts[id] = new Chart(canvas, {
                type: type,
                data: { labels: [], datasets: [] },
                options: { responsive: true, plugins: { legend: { display: true } } }
            });
            return charts[id];
        }

        function updateBarChart(chartId, datasetLabel, dataObj) {
            const chart = ensureChart(chartId, 'bar');
            if (!chart) return;
            const labels = Object.keys(dataObj || {});
            const values = labels.map(key => dataObj[key]);
            chart.data.labels = labels;
            chart.data.datasets = [{
                label: datasetLabel,
                data: values,
                backgroundColor: '#3b82f6'
            }];
            chart.update();
        }

        function updateLineChart(chartId, data) {
            const chart = ensureChart(chartId, 'line');
            if (!chart) return;
            chart.data.labels = data.map((_, idx) => idx + 1);
            chart.data.datasets = [{
                label: '适应度',
                data: data,
                tension: 0.3,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.2)',
                fill: true,
            }];
            chart.update();
        }

        function renderJammerTable(containerId, jammers) {
            const container = document.getElementById(containerId);
            if (!container) return;
            if (!jammers || jammers.length === 0) {
                container.innerHTML = '<p class="muted">暂无干扰弹数据。</p>';
                return;
            }
            const rows = jammers.map(j => `<tr><td>${j.drone}</td><td>${j.jammer}</td><td>${j.missile || '—'}</td><td>${formatNumber(j.duration)}</td></tr>`).join('');
            container.innerHTML = `<table><thead><tr><th>无人机</th><th>干扰弹</th><th>主要干扰导弹</th><th>有效时长 (s)</th></tr></thead><tbody>${rows}</tbody></table>`;
        }

        function renderIntervals(containerId, intervals) {
            const container = document.getElementById(containerId);
            if (!container) return;
            if (!intervals || Object.keys(intervals).length === 0) {
                container.innerHTML = '<p class="muted">暂无遮挡区间。</p>';
                return;
            }
            const html = Object.entries(intervals).map(([missile, ranges]) => {
                if (!ranges || ranges.length === 0) {
                    return `<div><strong>${missile}</strong>: —</div>`;
                }
                const parts = ranges.map(item => {
                    const [start, end] = item;
                    return `${formatNumber(start)}s - ${formatNumber(end)}s`;
                }).join(', ');
                return `<div><strong>${missile}</strong>: ${parts}</div>`;
            }).join('');
            container.innerHTML = html;
        }

        function renderContext(containerId, context) {
            const container = document.getElementById(containerId);
            if (!container) return;
            if (!context) {
                container.innerHTML = '';
                return;
            }
            let html = '<div class="context-block">';
            if (context.message) {
                html += `<p>${context.message}</p>`;
            }
            if (Object.prototype.hasOwnProperty.call(context, 'occlusion')) {
                html += `<p>观测时刻 ${formatNumber(context.time)}s ，M1 遮挡结果：<strong>${context.occlusion ? '是' : '否'}</strong></p>`;
            }
            if (context.release_time !== undefined && context.smoke_delay !== undefined) {
                html += `<p>投放延迟：${formatNumber(context.release_time)}s，起爆延迟：${formatNumber(context.smoke_delay)}s</p>`;
            }
            if (context.image) {
                html += `<img src="${context.image}" alt="场景截图" />`;
            }
            if (context.download) {
                html += `<p style="margin-top:0.6rem;"><a href="${context.download}" target="_blank">下载图像</a></p>`;
            }
            html += '</div>';
            container.innerHTML = html;
        }

        function updateDownloads(containerId, downloads) {
            const container = document.getElementById(containerId);
            if (!container) return;
            if (!downloads) {
                container.innerHTML = '';
                return;
            }
            const links = [];
            if (downloads.json) {
                links.push(`<a href="${downloads.json}" target="_blank">下载方案 JSON</a>`);
            }
            if (downloads.excel) {
                links.push(`<a href="${downloads.excel}" target="_blank">下载 Excel</a>`);
            }
            if (downloads.image) {
                links.push(`<a href="${downloads.image}" target="_blank">下载图像</a>`);
            }
            container.innerHTML = links.join('');
        }

        function updateSummary(prefix, summary, downloads) {
            updateDownloads(`${prefix}-downloads`, downloads);
            const metricContainer = document.getElementById(`${prefix}-metrics`);
            if (!summary) {
                if (metricContainer) {
                    metricContainer.innerHTML = '<p class="muted">暂无数据，请先运行任务。</p>';
                }
                renderJammerTable(`${prefix}-jammers`, []);
                renderIntervals(`${prefix}-intervals`, {});
                renderContext(`${prefix}-context`, null);
                const historyWrapper = document.getElementById(`${prefix}-history-wrapper`);
                if (historyWrapper) historyWrapper.style.display = 'none';
                const chart = charts[`${prefix}-chart`];
                if (chart) {
                    chart.data.labels = [];
                    chart.data.datasets = [];
                    chart.update();
                }
                return;
            }
            const metrics = [];
            metrics.push(createMetric('总遮挡时间', `${formatNumber(summary.total_duration)} s`));
            metrics.push(createMetric('目标导弹', summary.targeted_missiles && summary.targeted_missiles.length ? summary.targeted_missiles.join(', ') : '—'));
            metrics.push(createMetric('无人机数量', summary.drone_count ?? '—'));
            if (summary.metadata && summary.metadata.fitness !== undefined && summary.metadata.fitness !== null) {
                metrics.push(createMetric('最佳适应度', formatNumber(summary.metadata.fitness)));
            }
            if (summary.metadata && summary.metadata.generated_at) {
                metrics.push(createMetric('生成时间', new Date(summary.metadata.generated_at).toLocaleString()));
            }
            metricContainer.innerHTML = metrics.join('');
            updateBarChart(`${prefix}-chart`, '遮挡时间 (s)', summary.durations || {});
            renderJammerTable(`${prefix}-jammers`, summary.jammers || []);
            renderIntervals(`${prefix}-intervals`, summary.intervals || {});
            renderContext(`${prefix}-context`, summary.context);
            const historyWrapper = document.getElementById(`${prefix}-history-wrapper`);
            if (summary.history && summary.history.length) {
                if (historyWrapper) historyWrapper.style.display = 'block';
                updateLineChart(`${prefix}-history`, summary.history);
            } else if (historyWrapper) {
                historyWrapper.style.display = 'none';
            }
        }

        function populatePlanSelectors(plans) {
            const planSelect = document.getElementById('plan-select');
            const verifySelect = document.getElementById('verify-plan');
            planSelect.innerHTML = '';
            verifySelect.innerHTML = '';
            if (!plans || plans.length === 0) {
                planSelect.innerHTML = '<option value="">暂无方案</option>';
                verifySelect.innerHTML = '<option value="">暂无方案</option>';
                return;
            }
            plans.forEach(plan => {
                const option = document.createElement('option');
                option.value = plan.name;
                option.textContent = `${plan.name} (${plan.size_kb} KB)`;
                planSelect.appendChild(option);
                const verifyOption = option.cloneNode(true);
                verifySelect.appendChild(verifyOption);
            });
        }

        function populateMissileOptions(options) {
            const select = document.getElementById('verify-missiles');
            select.innerHTML = '';
            options.forEach(missile => {
                const opt = document.createElement('option');
                opt.value = missile;
                opt.textContent = missile;
                select.appendChild(opt);
            });
        }

        function setVerifySelection(missiles) {
            const select = document.getElementById('verify-missiles');
            const values = new Set(missiles || []);
            Array.from(select.options).forEach(opt => {
                opt.selected = values.has(opt.value);
            });
        }

        function renderQuestionInfo(preset) {
            const container = document.getElementById('question-info');
            if (!preset) {
                container.innerHTML = '<p class="muted">请选择题目。</p>';
                return;
            }
            const infoEntries = Object.entries(preset.info || {}).map(([key, value]) => `<li><strong>${key}</strong>: ${Array.isArray(value) ? value.join(', ') : value}</li>`).join('');
            container.innerHTML = `<p>${preset.description || ''}</p><ul>${infoEntries}</ul>`;
        }

        function renderQuestionForm(preset) {
            const container = document.getElementById('question-parameters');
            container.innerHTML = '';
            if (!preset || !preset.form) return;
            preset.form.forEach(field => {
                const wrapper = document.createElement('div');
                wrapper.innerHTML = `<label>${field.label}</label>`;
                const input = document.createElement('input');
                input.type = field.input_type || 'number';
                input.value = field.default ?? '';
                input.dataset.field = field.name;
                if (field.min !== undefined) input.min = field.min;
                if (field.max !== undefined) input.max = field.max;
                if (field.step !== undefined) input.step = field.step;
                if (field.placeholder) input.placeholder = field.placeholder;
                wrapper.appendChild(input);
                container.appendChild(wrapper);
            });
        }

        function initQuestions() {
            const select = document.getElementById('question-select');
            select.innerHTML = '';
            questionPresets.forEach(preset => {
                const option = document.createElement('option');
                option.value = preset.id;
                option.textContent = `${preset.category} · ${preset.label}`;
                select.appendChild(option);
            });
            const defaultPreset = questionPresets[0];
            renderQuestionInfo(defaultPreset);
            renderQuestionForm(defaultPreset);
            select.value = defaultPreset ? defaultPreset.id : '';
            select.addEventListener('change', () => {
                const preset = questionPresets.find(item => item.id === select.value);
                renderQuestionInfo(preset);
                renderQuestionForm(preset);
                updateSummary('question', null, null);
            });
        }

        async function refreshDashboard(selectedPlan = null, silent = false) {
            let url = '/api/dashboard';
            if (selectedPlan) {
                const params = new URLSearchParams({ plan: selectedPlan });
                url += `?${params.toString()}`;
            }
            const response = await fetch(url);
            if (!response.ok) {
                logStatus('刷新方案数据失败。', 'error');
                return;
            }
            const data = await response.json();
            populatePlanSelectors(data.plans);
            if (data.selected_plan) {
                document.getElementById('plan-select').value = data.selected_plan;
                document.getElementById('verify-plan').value = data.selected_plan;
            }
            if (data.summary) {
                updateSummary('plan', data.summary, data.summary.downloads);
                setVerifySelection(data.summary.targeted_missiles);
            } else {
                updateSummary('plan', null, null);
            }
            if (!silent) {
                logStatus('方案列表已更新。', 'success');
            }
            return data;
        }

        function gatherQuestionOverrides(preset) {
            const overrides = {};
            if (!preset || !preset.form) {
                return overrides;
            }
            preset.form.forEach(field => {
                const input = document.querySelector(`[data-field="${field.name}"]`);
                if (!input) return;
                const raw = input.value;
                if (raw === '' || raw === null || raw === undefined) return;
                if (field.input_type === 'text') {
                    overrides[field.name] = raw;
                } else if (field.value_type === 'int') {
                    overrides[field.name] = parseInt(raw, 10);
                } else if (field.value_type === 'float') {
                    overrides[field.name] = parseFloat(raw);
                } else {
                    overrides[field.name] = Number(raw);
                }
            });
            return overrides;
        }

        async function runQuestion() {
            const select = document.getElementById('question-select');
            const preset = questionPresets.find(item => item.id === select.value);
            if (!preset) {
                logStatus('请选择题目后再运行。', 'error');
                return;
            }
            const overrides = gatherQuestionOverrides(preset);
            updateSummary('question', null, null);
            logStatus(`正在运行 ${preset.label}...`);
            const response = await fetch('/api/run_question', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: preset.id, overrides })
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: '运行失败' }));
                logStatus(error.error || '题目运行失败。', 'error');
                return;
            }
            const result = await response.json();
            updateSummary('question', result.summary, result.summary ? result.summary.downloads : null);
            if (result.summary && result.summary.plan_name) {
                await refreshDashboard(result.summary.plan_name, true);
            }
            logStatus(`${preset.label} 运行完成。`, 'success');
        }

        async function verifySelectedPlan() {
            const plan = document.getElementById('verify-plan').value;
            if (!plan) {
                logStatus('请选择需要验证的方案。', 'error');
                return;
            }
            const missiles = Array.from(document.getElementById('verify-missiles').selectedOptions).map(opt => opt.value);
            logStatus(`正在验证 ${plan} ...`);
            updateSummary('verify', null, null);
            const response = await fetch('/api/verify_plan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ plan, missiles })
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: '验证失败' }));
                logStatus(error.error || '验证失败。', 'error');
                return;
            }
            const data = await response.json();
            updateSummary('verify', data.summary, data.summary ? data.summary.downloads : null);
            logStatus(`${plan} 遮挡验证完成。`, 'success');
        }

        function setupEventListeners() {
            document.getElementById('plan-select').addEventListener('change', async (event) => {
                const plan = event.target.value;
                logStatus(`正在加载方案 ${plan} ...`);
                const data = await refreshDashboard(plan, true);
                if (data && data.summary) {
                    setVerifySelection(data.summary.targeted_missiles);
                }
            });
            document.getElementById('run-question').addEventListener('click', runQuestion);
            document.getElementById('verify-button').addEventListener('click', verifySelectedPlan);
        }

        function bootstrap() {
            populateMissileOptions(missileOptions);
            populatePlanSelectors(initialData.plans);
            if (initialData.selected_plan) {
                document.getElementById('plan-select').value = initialData.selected_plan;
                document.getElementById('verify-plan').value = initialData.selected_plan;
            }
            if (initialData.summary) {
                updateSummary('plan', initialData.summary, initialData.summary.downloads);
                setVerifySelection(initialData.summary.targeted_missiles);
            } else {
                updateSummary('plan', null, null);
            }
            initQuestions();
            setupEventListeners();
            logStatus('系统初始化完成。', 'success');
        }

        bootstrap();
    </script>
</body>
</html>
"""

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


def _question_metadata() -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for key, value in QUESTION_PRESETS.items():
        form_config = []
        for name, config in value.get("form", {}).items():
            field = {
                "name": name,
                "label": config.get("label", name),
                "input_type": config.get("input_type", "number"),
                "value_type": config.get("value_type", "float"),
                "default": value.get("params", {}).get(name),
                "min": config.get("min"),
                "max": config.get("max"),
                "step": config.get("step"),
                "placeholder": config.get("placeholder"),
            }
            form_config.append(field)
        metadata.append({
            "id": key,
            "label": value.get("label", key),
            "category": value.get("category", "其它"),
            "description": value.get("description", ""),
            "type": value.get("type", "optimisation"),
            "info": value.get("info", {}),
            "form": form_config,
        })
    return metadata


def _available_plans() -> list[dict[str, Any]]:
    log_dir = OUTPUT_DIR / "log"
    if not log_dir.exists():
        return []

    plans: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        targeted = payload.get("targeted_missiles") or []
        metadata = {key: payload.get(key) for key in ("fitness", "generated_at", "question", "parameters") if key in payload}
        plan = {
            "name": path.name,
            "path": str(path),
            "size_kb": round(path.stat().st_size / 1024.0, 2),
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "targeted_missiles": targeted,
            "metadata": metadata,
            "downloads": {
                "json": f"/download/log/{path.name}",
                "excel": f"/download/excel/{path.stem}.xlsx",
            },
        }
        plans.append(plan)
    return plans


def _load_plan(path: Path) -> tuple[Plan, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan_data = payload.get("plan", payload)
    plan = Plan.from_dict(plan_data)
    plan.clamp()
    return plan, payload


def _default_missiles() -> list[str]:
    data = json.loads(Path(POSITIONS_PATH).read_text(encoding="utf-8"))
    return list(data.get("missiles", {}).keys())


def _simulation_payload(simulation, targeted: Iterable[str]) -> Dict[str, Any]:
    targeted_list = list(targeted)
    payload = {
        "durations": simulation.durations,
        "intervals": {key: [[float(start), float(end)] for start, end in values] for key, values in simulation.intervals.items()},
        "total_duration": simulation.total_duration,
        "plan": serialize_plan(simulation.plan),
        "targeted_missiles": targeted_list,
        "drone_count": len(simulation.plan.drones),
        "jammers": jam_interference_summary(simulation.system, targeted_list),
    }
    return payload


def _build_dashboard_payload(selected: Optional[str]) -> Dict[str, Any]:
    plans = _available_plans()
    plan_names = [plan["name"] for plan in plans]
    selected_plan = selected or (plan_names[0] if plan_names else None)

    summary: Optional[Dict[str, Any]] = None
    if selected_plan and selected_plan in plan_names:
        plan_path = OUTPUT_DIR / "log" / selected_plan
        plan, payload = _load_plan(plan_path)
        targeted = payload.get("targeted_missiles") or _default_missiles()
        simulation = run_simulation(POSITIONS_PATH, FORWARD_VECTOR_PATH, plan, targeted)
        summary = _simulation_payload(simulation, targeted)
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
        occlusion = system.detect_occlusion_single_jammer(time_point, system.missiles["M1"], jammer)

    image_dir = OUTPUT_DIR / "photos" / "webapp"
    image_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    image_path = image_dir / f"Q1_scene_{timestamp}.png"
    plot_scene(system, time_point, save_path=image_path, show=False)
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")

    duration, missile_id = compute_interference_stats(system, jammer, targeted)

    payload = {
        "durations": durations,
        "intervals": {key: [[float(start), float(end)] for start, end in values] for key, values in intervals.items()},
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


def _run_optimisation_question(question_id: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    preset = QUESTION_PRESETS[question_id]
    params = deepcopy(preset.get("params", {}))

    for name, config in preset.get("form", {}).items():
        if name not in overrides:
            continue
        value = overrides[name]
        if value is None:
            continue
        params[name] = value

    targeted = params.get("targeted_missile_ids", [])
    qname = params.get("Qname", question_id)
    json_path = OUTPUT_DIR / "log" / f"{qname}.json"
    excel_path = OUTPUT_DIR / "excel" / f"{qname}.xlsx"

    workflow = run_genetic_workflow(
        params.get("drone_ids", []),
        params.get("n_jammers", 1),
        params.get("population_size", 100),
        params.get("generations", 50),
        targeted,
        random_seed=params.get("random_seed"),
        save_json=False,
        export_excel_path=excel_path,
    )

    history = workflow.fitness_history
    simulation = workflow.optimisation_result

    params_with_history = dict(params)
    params_with_history["history"] = history
    _save_plan_with_metadata(simulation, params_with_history, json_path)

    summary = _simulation_payload(simulation, targeted)
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


def _parse_overrides(payload: Dict[str, Any], question_id: str) -> Dict[str, Any]:
    overrides = {}
    form = QUESTION_PRESETS.get(question_id, {}).get("form", {})
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


@app.route("/")
def index() -> str:
    selected = request.args.get("plan")
    dashboard = _build_dashboard_payload(selected)
    return render_template_string(
        TEMPLATE,
        initial_payload=dashboard,
        question_metadata=_question_metadata(),
        missile_options=_default_missiles(),
    )


@app.get("/api/dashboard")
def dashboard_api():
    selected = request.args.get("plan")
    return jsonify(_build_dashboard_payload(selected))


@app.post("/api/run_question")
def run_question_api():
    payload = request.get_json(force=True, silent=True) or {}
    question_id = payload.get("question")
    if not question_id or question_id not in QUESTION_PRESETS:
        return jsonify({"error": "未知的题目编号"}), 400

    try:
        overrides = _parse_overrides(payload.get("overrides", {}), question_id)
        if QUESTION_PRESETS[question_id].get("type") == "analysis":
            summary = _run_q1_analysis(overrides)
        else:
            summary = _run_optimisation_question(question_id, overrides)
        return jsonify({"summary": summary})
    except Exception as exc:  # pragma: no cover - unexpected runtime errors converted to JSON
        return jsonify({"error": str(exc)}), 500


@app.post("/api/verify_plan")
def verify_plan_api():
    payload = request.get_json(force=True, silent=True) or {}
    plan_name = payload.get("plan")
    if not plan_name:
        return jsonify({"error": "缺少方案名称"}), 400
    plan_path = OUTPUT_DIR / "log" / plan_name
    if not plan_path.exists():
        return jsonify({"error": "方案不存在"}), 404

    plan, raw_payload = _load_plan(plan_path)
    missiles = payload.get("missiles") or raw_payload.get("targeted_missiles") or _default_missiles()
    simulation = verify_plan(plan, missiles)
    summary = _simulation_payload(simulation, missiles)
    summary["plan_name"] = plan_name
    summary["downloads"] = {
        "json": f"/download/log/{plan_name}",
        "excel": f"/download/excel/{Path(plan_name).stem}.xlsx",
    }
    summary["metadata"] = {
        "fitness": raw_payload.get("fitness"),
        "generated_at": raw_payload.get("generated_at"),
        "question": raw_payload.get("question"),
        "parameters": raw_payload.get("parameters"),
    }
    return jsonify({"summary": summary})


@app.route("/download/<kind>/<path:filename>")
def download_file(kind: str, filename: str):
    safe_parts = Path(filename)
    if safe_parts.is_absolute() or ".." in safe_parts.parts:
        abort(400)

    if kind == "log":
        directory = OUTPUT_DIR / "log"
    elif kind == "excel":
        directory = OUTPUT_DIR / "excel"
    elif kind == "photos":
        directory = OUTPUT_DIR / "photos"
    else:
        abort(404)

    return send_from_directory(directory, filename, as_attachment=True)


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    app.run(debug=True)
