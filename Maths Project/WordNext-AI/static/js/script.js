/**
 * script.js  —  WordNext AI
 * =========================
 * Shared frontend utilities:
 *   - API helpers (fetch wrappers)
 *   - Chart.js builders
 *   - Toast notifications
 *   - Number animation
 *   - Chart.js global config
 */

// ════════════════════════════════════════════════════════════
//  CHART.JS GLOBAL DEFAULTS
// ════════════════════════════════════════════════════════════
if (typeof Chart !== 'undefined') {
  Chart.defaults.color          = '#9AA5BC';
  Chart.defaults.borderColor    = 'rgba(255,255,255,0.06)';
  Chart.defaults.backgroundColor = 'rgba(214,163,74,0.7)';
  Chart.defaults.font.family    = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size      = 12;
}

// ════════════════════════════════════════════════════════════
//  API HELPERS
// ════════════════════════════════════════════════════════════

/**
 * GET request returning parsed JSON.
 * @param {string} url
 * @returns {Promise<object>}
 */
async function apiGet(url) {
  const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
  return res.json();
}

/**
 * POST request with JSON body, returns parsed JSON.
 * @param {string} url
 * @param {object} body
 * @returns {Promise<object>}
 */
async function apiPost(url, body = {}) {
  const res = await fetch(url, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body:    JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${url} failed: ${res.status}`);
  return res.json();
}

// ════════════════════════════════════════════════════════════
//  TOAST NOTIFICATIONS
// ════════════════════════════════════════════════════════════

let _toastTimer = null;

/**
 * Show a brief toast notification.
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 * @param {number} duration  ms to stay visible
 */
function showToast(message, type = 'info', duration = 2800) {
  const el = document.getElementById('toast');
  if (!el) return;

  el.textContent = message;
  el.className   = `toast ${type} show`;

  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => {
    el.classList.remove('show');
  }, duration);
}

// ════════════════════════════════════════════════════════════
//  ANIMATED NUMBER COUNTER
// ════════════════════════════════════════════════════════════

/**
 * Animate a number counting up from 0 to target.
 * @param {string} elementId
 * @param {number} target
 * @param {number} duration  animation ms
 */
function animNum(elementId, target, duration = 800) {
  const el = document.getElementById(elementId);
  if (!el) return;

  const start     = Date.now();
  const startVal  = 0;

  function tick() {
    const elapsed = Date.now() - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(startVal + (target - startVal) * ease);
    el.textContent = current.toLocaleString();
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ════════════════════════════════════════════════════════════
//  CHART BUILDERS
// ════════════════════════════════════════════════════════════

// Store chart instances to allow re-creation
const _charts = {};

function _destroyChart(canvasId) {
  if (_charts[canvasId]) {
    _charts[canvasId].destroy();
    delete _charts[canvasId];
  }
}

/**
 * Build a horizontal or vertical bar chart.
 * @param {string} canvasId
 * @param {string[]} labels
 * @param {number[]} values
 * @param {string} label
 * @param {string} color
 * @param {boolean} horizontal
 */
function buildBarChart(canvasId, labels, values, label = 'Count',
                       color = '#D6A34A', horizontal = false) {
  _destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  _charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label,
        data:            values,
        backgroundColor: labels.map((_, i) =>
          i === 0 ? color :
          i === 1 ? '#9B8AFB' : 'rgba(214,163,74,0.5)'
        ),
        borderColor:     'transparent',
        borderRadius:    6,
        borderSkipped:   false,
      }],
    },
    options: {
      indexAxis: horizontal ? 'y' : 'x',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(16,19,29,0.95)',
          borderColor:     'rgba(255,255,255,0.1)',
          borderWidth:     1,
          padding:         10,
          titleColor:      '#F0F4FF',
          bodyColor:       '#9AA5BC',
        },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9AA5BC' } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9AA5BC' } },
      },
    },
  });
}

/**
 * Build a grouped bar chart (multiple datasets side-by-side).
 * @param {string} canvasId
 * @param {string[]} labels  — x-axis categories
 * @param {Array<{label, data, color}>} datasets
 */
function buildGroupedBarChart(canvasId, labels, datasets) {
  _destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  _charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: datasets.map(d => ({
        label:           d.label,
        data:            d.data,
        backgroundColor: d.color,
        borderColor:     'transparent',
        borderRadius:    5,
        borderSkipped:   false,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#9AA5BC', boxWidth: 12, font: { size: 11 } },
        },
        tooltip: {
          backgroundColor: 'rgba(16,19,29,0.95)',
          borderColor:     'rgba(255,255,255,0.1)',
          borderWidth:     1,
          padding:         10,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}%`,
          },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9AA5BC' } },
        y: {
          grid:   { color: 'rgba(255,255,255,0.04)' },
          ticks:  { color: '#9AA5BC', callback: v => v + '%' },
          min: 0, max: 100,
        },
      },
    },
  });
}

/**
 * Build a multi-line chart.
 * @param {string} canvasId
 * @param {string[]} labels
 * @param {Array<{label, data, color}>} lines
 */
function buildMultiLineChart(canvasId, labels, lines) {
  _destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  _charts[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: lines.map(l => ({
        label:           l.label,
        data:            l.data,
        borderColor:     l.color,
        backgroundColor: l.color.includes('0.5') ? l.color :
                         l.color + '22',
        fill:            true,
        tension:         0.4,
        pointBackgroundColor: l.color,
        pointRadius:     5,
        pointHoverRadius: 7,
        borderWidth:     2,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#9AA5BC', boxWidth: 12, font: { size: 11 } },
        },
        tooltip: {
          backgroundColor: 'rgba(16,19,29,0.95)',
          borderColor:     'rgba(255,255,255,0.1)',
          borderWidth:     1,
          padding:         10,
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}%`,
          },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#9AA5BC' } },
        y: {
          grid:   { color: 'rgba(255,255,255,0.04)' },
          ticks:  { color: '#9AA5BC', callback: v => v + '%' },
          min: 0,
        },
      },
    },
  });
}
