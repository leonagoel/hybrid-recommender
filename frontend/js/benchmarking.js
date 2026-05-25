// =============================================================================
// frontend/js/benchmarking.js — Model Performance Benchmarking Dashboard
//
// Fetches /api/evaluate, renders a grouped bar chart (pure SVG, no Chart.js
// dependency), shows a history table of last 5 runs, and exposes controls
// for K selector and mode picker.
//
// Called from app.js:
//   import { initBenchmarkingDashboard } from './js/benchmarking.js';
//   initBenchmarkingDashboard();
// =============================================================================

import { state } from './state.js';
import { showToast, setLoadingState } from './ui.js';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Initialise the dashboard panel — bind buttons, load history. Call once from app.js. */
export function initBenchmarkingDashboard() {
    _buildPanelHTML();
    _bindControls();
    _loadHistory();
}

// ---------------------------------------------------------------------------
// Panel HTML injection
// ---------------------------------------------------------------------------

function _buildPanelHTML() {
    // Look for an existing mount point in index.html; skip if already built
    if (document.getElementById('benchmarking-panel')) return;

    const panel = document.createElement('section');
    panel.id            = 'benchmarking-panel';
    panel.className     = 'benchmark-panel';
    panel.setAttribute('aria-label', 'Model performance benchmarking');
    panel.innerHTML = `
    <div class="benchmark-panel__header" role="button" tabindex="0"
         aria-expanded="false" aria-controls="benchmark-panel__body"
         id="benchmark-panel__toggle">
      <h2 class="benchmark-panel__title">
        <span class="benchmark-panel__icon">📊</span>
        Model Performance
      </h2>
      <span class="benchmark-panel__chevron" aria-hidden="true">▾</span>
    </div>

    <div class="benchmark-panel__body" id="benchmark-panel__body" hidden>

      <!-- Controls -->
      <div class="benchmark-controls">
        <div class="benchmark-controls__group">
          <label for="bench-k-select" class="benchmark-controls__label">K</label>
          <select id="bench-k-select" class="benchmark-controls__select">
            <option value="5">5</option>
            <option value="10" selected>10</option>
            <option value="20">20</option>
          </select>
        </div>

        <div class="benchmark-controls__group">
          <label for="bench-mode-select" class="benchmark-controls__label">Mode</label>
          <select id="bench-mode-select" class="benchmark-controls__select">
            <option value="all" selected>All modes</option>
            <option value="content">Content only</option>
            <option value="collaborative">Collaborative only</option>
            <option value="sentiment">Sentiment only</option>
            <option value="hybrid">Hybrid only</option>
          </select>
        </div>

        <button id="bench-run-btn" class="btn btn--primary"
                data-loading-btn="benchmark" aria-label="Run evaluation">
          Run Evaluation
        </button>
      </div>

      <!-- Chart area -->
      <div class="benchmark-chart" id="benchmark-chart"
           data-loading-area="benchmark" role="img"
           aria-label="Model performance bar chart">
        <p class="benchmark-chart__empty">
          Click <strong>Run Evaluation</strong> to benchmark the models.
        </p>
      </div>

      <!-- Legend -->
      <div class="benchmark-legend" id="benchmark-legend" hidden>
        <span class="benchmark-legend__item benchmark-legend__item--precision">Precision@K</span>
        <span class="benchmark-legend__item benchmark-legend__item--recall">Recall@K</span>
        <span class="benchmark-legend__item benchmark-legend__item--ndcg">NDCG@K</span>
      </div>

      <!-- History table -->
      <div class="benchmark-history" id="benchmark-history">
        <h3 class="benchmark-history__title">Recent Runs</h3>
        <div id="benchmark-history-table">
          <p class="benchmark-chart__empty">No runs yet.</p>
        </div>
      </div>

    </div>`;

    // Mount below the weight sliders section if it exists, else append to main
    const anchor = document.getElementById('weights-section')
        ?? document.querySelector('main')
        ?? document.body;
    anchor.after(panel);
}

// ---------------------------------------------------------------------------
// Event binding
// ---------------------------------------------------------------------------

function _bindControls() {
    // Collapsible toggle
    const toggle = document.getElementById('benchmark-panel__toggle');
    const body   = document.getElementById('benchmark-panel__body');
    toggle?.addEventListener('click', () => {
        const isHidden = body.hidden;
        body.hidden = !isHidden;
        toggle.setAttribute('aria-expanded', String(isHidden));
    });
    toggle?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle.click(); }
    });

    // Run button
    document.getElementById('bench-run-btn')
        ?.addEventListener('click', _runEvaluation);
}

// ---------------------------------------------------------------------------
// Evaluation fetch & render
// ---------------------------------------------------------------------------

async function _runEvaluation() {
    const k    = document.getElementById('bench-k-select')?.value   ?? '10';
    const mode = document.getElementById('bench-mode-select')?.value ?? 'all';
    const { alpha, beta, gamma } = state.weights;

    setLoadingState('benchmark', true);
    _showChartLoading();

    try {
        const params = new URLSearchParams({ k, mode, alpha, beta, gamma });
        const res    = await fetch(`/api/evaluate?${params}`);

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail ?? `Evaluation failed (${res.status})`);
        }

        const data = await res.json();
        _renderChart(data.results, parseInt(k, 10));
        _loadHistory(); // refresh history table after new run
        showToast('Evaluation complete!', 'success');
    } catch (err) {
        showToast(err.message, 'error');
        console.error('[benchmarking] _runEvaluation:', err);
        _showChartError(err.message);
    } finally {
        setLoadingState('benchmark', false);
    }
}

// ---------------------------------------------------------------------------
// SVG Bar Chart
// ---------------------------------------------------------------------------

const METRICS   = ['precision', 'recall', 'ndcg'];
const COLORS    = {
    precision: 'var(--color-precision, #4f8ef7)',
    recall:    'var(--color-recall,    #34c97e)',
    ndcg:      'var(--color-ndcg,      #f5a623)',
};
const BAR_W     = 18;   // px per bar
const BAR_GAP   = 4;    // px between bars in a group
const GROUP_GAP = 24;   // px between mode groups
const CHART_H   = 200;  // chart area height px
const PADDING   = { top: 16, right: 16, bottom: 48, left: 48 };

function _renderChart(results, k) {
    const container = document.getElementById('benchmark-chart');
    const legend    = document.getElementById('benchmark-legend');
    if (!container) return;

    const modes      = Object.keys(results);
    const groupW     = METRICS.length * (BAR_W + BAR_GAP) - BAR_GAP;
    const totalW     = modes.length * (groupW + GROUP_GAP) - GROUP_GAP
                       + PADDING.left + PADDING.right;
    const totalH     = CHART_H + PADDING.top + PADDING.bottom;

    const svgW = Math.max(totalW, 300);

    let svgContent = `<svg viewBox="0 0 ${svgW} ${totalH}"
         xmlns="http://www.w3.org/2000/svg"
         role="img" aria-label="Model performance chart at K=${k}"
         style="width:100%;max-width:${svgW}px;display:block;margin:0 auto">
      <title>Model Performance @ K=${k}</title>`;

    // Y-axis gridlines at 0, 0.25, 0.5, 0.75, 1.0
    for (let tick = 0; tick <= 4; tick++) {
        const val = tick / 4;
        const y   = PADDING.top + CHART_H - val * CHART_H;
        const label = val.toFixed(2);
        svgContent += `
      <line x1="${PADDING.left - 6}" y1="${y}" x2="${svgW - PADDING.right}" y2="${y}"
            stroke="var(--color-border, #e0e0e0)" stroke-width="${tick === 0 ? 1.5 : 0.5}"
            stroke-dasharray="${tick === 0 ? 'none' : '4 4'}"/>
      <text x="${PADDING.left - 10}" y="${y + 4}" text-anchor="end"
            font-size="10" fill="var(--color-text-muted, #888)">${label}</text>`;
    }

    // Bars per mode
    modes.forEach((modeName, modeIdx) => {
        const groupX = PADDING.left + modeIdx * (groupW + GROUP_GAP);

        METRICS.forEach((metric, metricIdx) => {
            const value  = results[modeName]?.[metric] ?? 0;
            const barH   = value * CHART_H;
            const x      = groupX + metricIdx * (BAR_W + BAR_GAP);
            const y      = PADDING.top + CHART_H - barH;
            const color  = COLORS[metric];

            svgContent += `
        <rect x="${x}" y="${y}" width="${BAR_W}" height="${barH}"
              fill="${color}" rx="3"
              aria-label="${modeName} ${metric} ${value.toFixed(4)}">
          <title>${modeName} — ${metric}: ${value.toFixed(4)}</title>
        </rect>`;

            // Value label above bar (only if bar is tall enough)
            if (barH > 20) {
                svgContent += `
        <text x="${x + BAR_W / 2}" y="${y - 4}" text-anchor="middle"
              font-size="9" fill="var(--color-text, #333)">${value.toFixed(2)}</text>`;
            }
        });

        // Mode label below group
        const labelX = groupX + groupW / 2;
        const labelY = PADDING.top + CHART_H + 20;
        const label  = modeName.charAt(0).toUpperCase() + modeName.slice(1);
        svgContent += `
      <text x="${labelX}" y="${labelY}" text-anchor="middle"
            font-size="11" font-weight="600"
            fill="var(--color-text, #333)">${label}</text>`;
    });

    svgContent += `</svg>`;
    container.innerHTML = svgContent;
    if (legend) legend.hidden = false;
}

function _showChartLoading() {
    const container = document.getElementById('benchmark-chart');
    if (container) container.innerHTML = `
      <div class="benchmark-chart__loading" role="status" aria-live="polite">
        <span class="spinner" aria-hidden="true"></span>
        Running evaluation…
      </div>`;
}

function _showChartError(message) {
    const container = document.getElementById('benchmark-chart');
    if (container) container.innerHTML = `
      <p class="benchmark-chart__error" role="alert">⚠ ${_esc(message)}</p>`;
}

// ---------------------------------------------------------------------------
// History table
// ---------------------------------------------------------------------------

async function _loadHistory() {
    try {
        const res  = await fetch('/api/evaluate/history?limit=5');
        if (!res.ok) return;
        const data = await res.json();
        _renderHistory(data.runs ?? []);
    } catch (err) {
        console.warn('[benchmarking] _loadHistory:', err);
    }
}

function _renderHistory(runs) {
    const container = document.getElementById('benchmark-history-table');
    if (!container) return;

    if (!runs.length) {
        container.innerHTML = '<p class="benchmark-chart__empty">No runs yet.</p>';
        return;
    }

    const rows = runs.map(run => {
        const ts      = new Date(run.created_at).toLocaleString();
        const w       = run.weights ?? {};
        const weights = `α${(w.alpha ?? 0.4).toFixed(2)} β${(w.beta ?? 0.4).toFixed(2)} γ${(w.gamma ?? 0.2).toFixed(2)}`;
        const hybrid  = run.results?.hybrid ?? run.results?.[Object.keys(run.results ?? {})[0]] ?? {};

        return `
      <tr>
        <td class="bench-hist__time">${_esc(ts)}</td>
        <td class="bench-hist__mode">${_esc(run.mode ?? 'all')}</td>
        <td class="bench-hist__k">${_esc(String(run.k ?? '?'))}</td>
        <td class="bench-hist__weights">${_esc(weights)}</td>
        <td class="bench-hist__precision">${(hybrid.precision ?? 0).toFixed(4)}</td>
        <td class="bench-hist__recall">${(hybrid.recall ?? 0).toFixed(4)}</td>
        <td class="bench-hist__ndcg">${(hybrid.ndcg ?? 0).toFixed(4)}</td>
      </tr>`;
    }).join('');

    container.innerHTML = `
    <table class="bench-hist-table" role="table">
      <thead>
        <tr>
          <th scope="col">Time</th>
          <th scope="col">Mode</th>
          <th scope="col">K</th>
          <th scope="col">Weights</th>
          <th scope="col">Precision</th>
          <th scope="col">Recall</th>
          <th scope="col">NDCG</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// ---------------------------------------------------------------------------
// Escape helper
// ---------------------------------------------------------------------------

function _esc(str) {
    return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
