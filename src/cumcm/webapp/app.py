"""Interactive Flask web application for optimisation and verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
    send_from_directory,
    stream_with_context,
)

from ..questions import QUESTION_PRESETS, parse_overrides, question_metadata
from ..services import (
    build_dashboard_payload,
    default_missiles,
    load_plan,
    run_question,
    simulation_payload,
    stream_question,
)
from ..workflows import OUTPUT_DIR, verify_plan

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["JSON_SORT_KEYS"] = False

TEMPLATE = """
<!DOCTYPE html>
<html lang=\"zh-cn\">
<head>
    <meta charset=\"utf-8\" />
    <title>遮挡优化控制台</title>
    <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
    <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
    <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap\" rel=\"stylesheet\" />
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js\"></script>
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f3f6fb; color: #1f2933; }
        header { background: linear-gradient(135deg, #14213d, #274060); color: white; padding: 1.6rem 2.4rem; }
        header h1 { margin: 0; font-weight: 600; font-size: 1.8rem; }
        main { padding: 2.4rem; max-width: 1280px; margin: 0 auto; display: flex; flex-direction: column; gap: 1.6rem; }
        .layout { display: grid; grid-template-columns: minmax(320px, 380px) minmax(0, 1fr); gap: 1.6rem; align-items: stretch; }
        .card { background: white; border-radius: 16px; box-shadow: 0 12px 30px rgba(15, 35, 95, 0.08); padding: 1.8rem; display: flex; flex-direction: column; gap: 1.2rem; }
        h2 { margin: 0; font-weight: 600; color: #1b2a4b; }
        label { font-weight: 600; color: #37425b; display: block; margin-bottom: 0.4rem; }
        select, input { width: 100%; padding: 0.6rem 0.75rem; border: 1px solid #d4d9e6; border-radius: 10px; font-size: 0.95rem; }
        button { border: none; border-radius: 10px; padding: 0.65rem 1.2rem; background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; font-weight: 600; cursor: pointer; transition: transform 0.2s ease, box-shadow 0.2s ease; }
        button:hover:not([disabled]) { transform: translateY(-1px); box-shadow: 0 12px 22px rgba(37, 99, 235, 0.2); }
        button[disabled] { opacity: 0.6; cursor: not-allowed; box-shadow: none; }
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
        .section-title { font-size: 1.05rem; font-weight: 600; color: #1c2a4d; margin: 0; }
        .context-block { background: #f4f7ff; border: 1px solid #d9e2ff; border-radius: 12px; padding: 0.9rem 1.1rem; }
        .context-block img { max-width: 100%; border-radius: 12px; margin-top: 0.6rem; }
        .question-info { background: #f5f7fa; border-radius: 12px; padding: 0.8rem 1rem; border: 1px solid #e1e7f5; }
        .question-info ul { padding-left: 1.2rem; margin: 0.4rem 0 0; }
        .question-info li { margin-bottom: 0.2rem; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.8rem; }
        .status-log { background: #0f172a; color: #f1f5f9; padding: 1rem; border-radius: 12px; font-size: 0.9rem; max-height: 240px; overflow-y: auto; }
        .status-log strong { color: #38bdf8; }
        .tab-header { display: flex; gap: 0.6rem; background: #eef2ff; border-radius: 12px; padding: 0.4rem; }
        .tab-button { flex: 1; padding: 0.55rem 0.75rem; background: transparent; border-radius: 10px; color: #1b2a4b; font-weight: 600; border: none; cursor: pointer; transition: background 0.2s ease, color 0.2s ease; }
        .tab-button.active { background: white; color: #1d4ed8; box-shadow: 0 8px 18px rgba(37, 99, 235, 0.15); }
        .tab-panel { display: none; flex-direction: column; gap: 1rem; }
        .tab-panel.active { display: flex; }
        .plan-card { gap: 1.4rem; }
        .run-card { gap: 1.4rem; }
        .question-actions { display: flex; align-items: center; gap: 1rem; }
        .question-progress { font-size: 0.9rem; color: #47516b; }
        .question-progress[data-tone="error"] { color: #ef4444; }
        .question-progress[data-tone="success"] { color: #16a34a; }
        .history-wrapper { display: none; }
        .actions-row { display: flex; flex-direction: column; gap: 0.8rem; }
        @media (min-width: 640px) {
            .actions-row { flex-direction: row; align-items: center; justify-content: space-between; }
        }
        @media (max-width: 880px) {
            .layout { grid-template-columns: 1fr; }
        }
        @media (max-width: 720px) {
            main { padding: 1.6rem; }
            header { padding: 1.4rem 1.6rem; }
        }
    </style>
</head>
<body>
    <header>
        <h1>烟幕干扰全流程控制台</h1>
        <p class=\"muted\">一站式管理：题目运行、方案浏览与验证、结果下载</p>
    </header>
    <main>
        <div class=\"layout\">
            <section class=\"card plan-card\">
                <div class=\"tab-header\">
                    <button class=\"tab-button active\" data-target=\"plan-browser\">方案浏览</button>
                    <button class=\"tab-button\" data-target=\"plan-verify\">方案验证</button>
                </div>
                <div class=\"tab-panel active\" id=\"plan-browser\">
                    <div>
                        <label for=\"plan-select\">选择已有方案</label>
                        <select id=\"plan-select\"></select>
                    </div>
                    <div class=\"downloads\" id=\"plan-downloads\"></div>
                    <div id=\"plan-summary\" class=\"summary-block\">
                        <div class=\"metrics\" id=\"plan-metrics\"></div>
                        <canvas id=\"plan-chart\" height=\"120\"></canvas>
                        <div id=\"plan-history-wrapper\" class=\"history-wrapper\">
                            <p class=\"section-title\">优化历史</p>
                            <canvas id=\"plan-history\" height=\"160\"></canvas>
                        </div>
                        <div>
                            <p class=\"section-title\">干扰弹详情</p>
                            <div id=\"plan-jammers\"></div>
                        </div>
                        <div>
                            <p class=\"section-title\">遮挡区间</p>
                            <div id=\"plan-intervals\"></div>
                        </div>
                        <div id=\"plan-context\"></div>
                    </div>
                </div>
                <div class=\"tab-panel\" id=\"plan-verify\">
                    <div>
                        <label for=\"verify-plan\">待验证方案</label>
                        <select id=\"verify-plan\"></select>
                    </div>
                    <div>
                        <label for=\"verify-missiles\">目标导弹（可多选）</label>
                        <select id=\"verify-missiles\" multiple size=\"6\"></select>
                    </div>
                    <div class=\"actions-row\">
                        <button id=\"verify-button\">重新计算遮挡</button>
                        <div class=\"downloads\" id=\"verify-downloads\"></div>
                    </div>
                    <div id=\"verify-result\" class=\"summary-block\">
                        <div class=\"metrics\" id=\"verify-metrics\"></div>
                        <canvas id=\"verify-chart\" height=\"120\"></canvas>
                        <div id=\"verify-history-wrapper\" class=\"history-wrapper\">
                            <p class=\"section-title\">优化历史</p>
                            <canvas id=\"verify-history\" height=\"160\"></canvas>
                        </div>
                        <div id=\"verify-jammers\"></div>
                        <div id=\"verify-intervals\"></div>
                        <div id=\"verify-context\"></div>
                    </div>
                </div>
            </section>
            <section class=\"card run-card\">
                <h2>题目运行</h2>
                <div>
                    <label for=\"question-select\">选择题目</label>
                    <select id=\"question-select\">
                        {% for preset in question_metadata %}
                        <option value=\"{{ preset.id }}\"{% if loop.first %} selected{% endif %}>{{ preset.category }} · {{ preset.label }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class=\"question-info\" id=\"question-info\"></div>
                <div class=\"form-grid\" id=\"question-parameters\"></div>
                <div class=\"question-actions\">
                    <button id=\"run-question\">运行题目</button>
                    <div class=\"question-progress\" id=\"question-progress\"></div>
                </div>
                <div class=\"downloads\" id=\"question-downloads\"></div>
                <div id=\"question-result\" class=\"summary-block\">
                    <div class=\"metrics\" id=\"question-metrics\"></div>
                    <canvas id=\"question-chart\" height=\"120\"></canvas>
                    <div id=\"question-history-wrapper\" class=\"history-wrapper\">
                        <p class=\"section-title\">实时优化曲线</p>
                        <canvas id=\"question-history\" height=\"220\"></canvas>
                    </div>
                    <div id=\"question-jammers\"></div>
                    <div id=\"question-intervals\"></div>
                    <div id=\"question-context\"></div>
                </div>
            </section>
        </div>
        <section class=\"card\">
            <h2>系统日志</h2>
            <div class=\"status-log\" id=\"status-log\"></div>
        </section>
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
            return `<div class=\"metric\"><span>${label}</span><strong>${value}</strong></div>`;
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
            const historyWrapper = document.getElementById(`${prefix}-history-wrapper`);
            if (!summary) {
                if (metricContainer) {
                    metricContainer.innerHTML = '<p class="muted">暂无数据，请先运行任务。</p>';
                }
                renderJammerTable(`${prefix}-jammers`, []);
                renderIntervals(`${prefix}-intervals`, {});
                renderContext(`${prefix}-context`, null);
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
            if (metricContainer) {
                metricContainer.innerHTML = metrics.join('');
            }
            updateBarChart(`${prefix}-chart`, '遮挡时间 (s)', summary.durations || {});
            renderJammerTable(`${prefix}-jammers`, summary.jammers || []);
            renderIntervals(`${prefix}-intervals`, summary.intervals || {});
            renderContext(`${prefix}-context`, summary.context);
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
            if (!select) return;

            const presets = Array.isArray(questionPresets) ? questionPresets : [];
            if (!presets.length) {
                if (!select.options.length) {
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = '暂无题目';
                    select.appendChild(option);
                }
                select.disabled = true;
                renderQuestionInfo(null);
                renderQuestionForm(null);
                updateQuestionProgress('');
                return;
            }

            select.disabled = false;
            const previousValue = select.value;
            select.innerHTML = '';
            presets.forEach(preset => {
                const option = document.createElement('option');
                option.value = preset.id;
                option.textContent = `${preset.category} · ${preset.label}`;
                select.appendChild(option);
            });

            let defaultPreset = presets.find(item => item.id === previousValue);
            if (!defaultPreset) {
                defaultPreset = presets[0];
            }

            if (defaultPreset) {
                select.value = defaultPreset.id;
            }
            renderQuestionInfo(defaultPreset);
            renderQuestionForm(defaultPreset);
            select.addEventListener('change', () => {
                const preset = presets.find(item => item.id === select.value);
                renderQuestionInfo(preset);
                renderQuestionForm(preset);
                updateSummary('question', null, null);
                updateQuestionProgress('');
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

        function setQuestionRunning(running) {
            const button = document.getElementById('run-question');
            if (!button) return;
            button.disabled = running;
            button.textContent = running ? '运行中…' : '运行题目';
        }

        function updateQuestionProgress(message, tone = 'info') {
            const el = document.getElementById('question-progress');
            if (!el) return;
            el.textContent = message || '';
            el.dataset.tone = tone;
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
            updateQuestionProgress('准备启动优化...');
            logStatus(`正在运行 ${preset.label}...`);
            setQuestionRunning(true);
            try {
                const response = await fetch('/api/run_question_stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: preset.id, overrides })
                });
                if (!response.ok || !response.body) {
                    const error = await response.json().catch(() => ({ error: '运行失败' }));
                    updateQuestionProgress(error.error || '题目运行失败。', 'error');
                    logStatus(error.error || '题目运行失败。', 'error');
                    return;
                }
                const decoder = new TextDecoder('utf-8');
                const reader = response.body.getReader();
                let buffer = '';
                let completed = false;
                const historyWrapper = document.getElementById('question-history-wrapper');
                if (historyWrapper) historyWrapper.style.display = 'block';
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    let idx;
                    while ((idx = buffer.indexOf('\n')) >= 0) {
                        const line = buffer.slice(0, idx).trim();
                        buffer = buffer.slice(idx + 1);
                        if (!line) continue;
                        let message;
                        try {
                            message = JSON.parse(line);
                        } catch (err) {
                            console.error('无法解析进度消息', err);
                            continue;
                        }
                        if (message.event === 'progress') {
                            updateLineChart('question-history', message.history || []);
                            updateQuestionProgress(`第 ${message.generation + 1} 代 · 当前最优 ${formatNumber(message.generation_best)} / 历史最优 ${formatNumber(message.best_fitness)}`);
                        } else if (message.event === 'complete') {
                            completed = true;
                            if (message.summary) {
                                updateSummary('question', message.summary, message.summary.downloads);
                                updateQuestionProgress(`优化完成 · 最佳适应度 ${formatNumber(message.summary.metadata && message.summary.metadata.fitness)}`, 'success');
                                await refreshDashboard(message.summary.plan_name, true);
                                logStatus('题目运行完成。', 'success');
                            }
                        } else if (message.event === 'error') {
                            completed = true;
                            updateQuestionProgress(message.error || '题目运行失败。', 'error');
                            logStatus(message.error || '题目运行失败。', 'error');
                        }
                    }
                }
                if (!completed) {
                    updateQuestionProgress('优化流程提前结束。', 'error');
                    logStatus('优化流程提前结束。', 'error');
                }
            } catch (error) {
                updateQuestionProgress('题目运行异常中断。', 'error');
                logStatus('题目运行异常中断。', 'error');
                console.error(error);
            } finally {
                setQuestionRunning(false);
            }
        }

        async function verifyPlan() {
            const plan = document.getElementById('verify-plan').value;
            const selectedOptions = Array.from(document.getElementById('verify-missiles').selectedOptions).map(opt => opt.value);
            if (!plan) {
                logStatus('请先选择需要验证的方案。', 'error');
                return;
            }
            updateQuestionProgress('');
            updateSummary('verify', null, null);
            logStatus(`正在验证 ${plan}...`);
            const response = await fetch('/api/verify_plan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ plan, missiles: selectedOptions })
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: '验证失败' }));
                logStatus(error.error || '方案验证失败。', 'error');
                return;
            }
            const result = await response.json();
            updateSummary('verify', result.summary, result.summary ? result.summary.downloads : null);
            logStatus('方案验证完成。', 'success');
        }

        function attachTabEvents() {
            document.querySelectorAll('.tab-button').forEach(button => {
                button.addEventListener('click', () => {
                    const target = button.dataset.target;
                    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
                    button.classList.add('active');
                    const panel = document.getElementById(target);
                    if (panel) panel.classList.add('active');
                });
            });
        }

        function initialise() {
            attachTabEvents();
            populatePlanSelectors(initialData.plans);
            populateMissileOptions(missileOptions);
            initQuestions();
            if (initialData.summary) {
                updateSummary('plan', initialData.summary, initialData.summary.downloads);
                setVerifySelection(initialData.summary.targeted_missiles);
            } else {
                updateSummary('plan', null, null);
            }
            document.getElementById('plan-select').addEventListener('change', async (event) => {
                const value = event.target.value;
                const data = await refreshDashboard(value, true);
                if (data && data.summary) {
                    logStatus(`已加载方案 ${value}。`, 'success');
                }
            });
            document.getElementById('run-question').addEventListener('click', runQuestion);
            document.getElementById('verify-button').addEventListener('click', verifyPlan);
            document.getElementById('verify-plan').addEventListener('change', (event) => {
                const value = event.target.value;
                document.getElementById('plan-select').value = value;
            });
        }

        document.addEventListener('DOMContentLoaded', initialise);
    </script>
</body>
</html>
"""


@app.route("/")
def index() -> str:
    selected = request.args.get("plan")
    dashboard = build_dashboard_payload(selected)
    return render_template_string(
        TEMPLATE,
        initial_payload=dashboard,
        question_metadata=question_metadata(),
        missile_options=default_missiles(),
    )


@app.get("/api/dashboard")
def dashboard_api():
    selected = request.args.get("plan")
    return jsonify(build_dashboard_payload(selected))


@app.post("/api/run_question")
def run_question_api():
    payload = request.get_json(force=True, silent=True) or {}
    question_id = payload.get("question")
    if not question_id or question_id not in QUESTION_PRESETS:
        return jsonify({"error": "未知的题目编号"}), 400
    try:
        overrides = parse_overrides(payload.get("overrides", {}), question_id)
        summary = run_question(question_id, overrides)
        return jsonify({"summary": summary})
    except Exception as exc:  # pragma: no cover - unexpected runtime errors converted to JSON
        return jsonify({"error": str(exc)}), 500


@app.post("/api/run_question_stream")
def run_question_stream():
    payload = request.get_json(force=True, silent=True) or {}
    question_id = payload.get("question")
    if not question_id or question_id not in QUESTION_PRESETS:
        return jsonify({"error": "未知的题目编号"}), 400
    overrides = parse_overrides(payload.get("overrides", {}), question_id)

    def _generate():
        for item in stream_question(question_id, overrides):
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return Response(stream_with_context(_generate()), mimetype="application/x-ndjson")


@app.post("/api/verify_plan")
def verify_plan_api():
    payload = request.get_json(force=True, silent=True) or {}
    plan_name = payload.get("plan")
    if not plan_name:
        return jsonify({"error": "缺少方案名称"}), 400
    plan_path = OUTPUT_DIR / "log" / plan_name
    if not plan_path.exists():
        return jsonify({"error": "方案不存在"}), 404

    plan, raw_payload = load_plan(plan_path)
    missiles = payload.get("missiles") or raw_payload.get("targeted_missiles") or default_missiles()
    simulation = verify_plan(plan, missiles)
    summary = simulation_payload(simulation, missiles)
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
