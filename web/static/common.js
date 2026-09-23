// Shared helpers: theme, Plotly defaults that follow it, sortable/filterable tables.
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function stateColor(state) { return css('--s-' + state) || css('--muted'); }

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark'
    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
  window.dispatchEvent(new Event('themechange'));
}

function axis(extra) {
  return Object.assign({
    gridcolor: css('--grid'), zerolinecolor: css('--grid'), linecolor: css('--border'),
    tickfont: { color: css('--muted'), size: 11 }, automargin: true,
  }, extra || {});
}

function titled(text) { return { text: text, font: { size: 12, color: css('--muted') } }; }

function layout(extra) {
  const base = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: css('--sans'), size: 12, color: css('--text') },
    margin: { l: 54, r: 12, t: 10, b: 42 }, showlegend: false,
    hoverlabel: { bgcolor: css('--panel'), bordercolor: css('--border'), font: { color: css('--text') } },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 11, color: css('--muted') } },
  };
  const merged = Object.assign(base, extra || {});
  for (const key of Object.keys(merged)) {
    if (/^[xy]axis\d*$/.test(key)) merged[key] = axis(merged[key]);
  }
  if (!merged.xaxis) merged.xaxis = axis();
  if (!merged.yaxis) merged.yaxis = axis();
  return merged;
}

function plot(el, data, lay, config) {
  if (typeof el === 'string') el = document.getElementById(el);
  if (!el || !window.Plotly) return;
  return Plotly.react(el, data, layout(lay), Object.assign({
    displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
  }, config || {}));
}

function chartData() {
  const el = document.getElementById('charts');
  return el ? JSON.parse(el.textContent) : {};
}

// A base64 float32 image from the API as an array of rows.
function decodeImage(image) {
  const bytes = Uint8Array.from(atob(image.data), c => c.charCodeAt(0));
  const values = new Float32Array(bytes.buffer);
  const rows = [];
  for (let r = 0; r < image.rows; r++) rows.push(Array.from(values.subarray(r * image.columns, (r + 1) * image.columns), v => (isNaN(v) ? null : v)));
  return rows;
}

function fmt(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits === undefined ? 2 : digits);
}

function makeSortable(table) {
  table.querySelectorAll('th').forEach((th, column) => {
    th.classList.add('sortable');
    th.addEventListener('click', () => {
      const body = table.tBodies[0];
      const ascending = th.dataset.dir !== 'asc';
      table.querySelectorAll('th').forEach(h => delete h.dataset.dir);
      th.dataset.dir = ascending ? 'asc' : 'desc';
      const key = row => {
        const cell = row.cells[column];
        const raw = cell ? (cell.dataset.sort ?? cell.textContent.trim()) : '';
        const number = parseFloat(raw);
        return raw !== '' && !isNaN(number) && /^[-+]?[\d.]/.test(raw) ? number : raw.toLowerCase();
      };
      const rows = Array.from(body.rows);
      rows.sort((a, b) => {
        const x = key(a), y = key(b);
        if (x === y) return 0;
        if (x === '' || x === '—') return 1;
        if (y === '' || y === '—') return -1;
        return (x < y ? -1 : 1) * (ascending ? 1 : -1);
      });
      rows.forEach(r => body.appendChild(r));
    });
  });
}

function filterRows(table, predicate) {
  let shown = 0;
  Array.from(table.tBodies[0].rows).forEach(row => {
    const visible = predicate(row);
    row.style.display = visible ? '' : 'none';
    shown += visible;
  });
  return shown;
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('table.sortable').forEach(makeSortable);
  document.querySelectorAll('.state').forEach(el => {
    el.style.background = stateColor(el.dataset.state || el.textContent.trim());
  });
  document.querySelectorAll('[data-copy]').forEach(el => el.addEventListener('click', () => {
    navigator.clipboard && navigator.clipboard.writeText(el.dataset.copy);
    el.title = 'Copied'; setTimeout(() => (el.title = 'Click to copy'), 1500);
  }));
});
window.addEventListener('themechange', () => {
  document.querySelectorAll('.state').forEach(el => { el.style.background = stateColor(el.dataset.state || el.textContent.trim()); });
});
