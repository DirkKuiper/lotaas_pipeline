// The single-pulse viewer: dynamic spectrum at any DM, band-averaged series,
// on/off-pulse spectrum, and S/N against DM with what a real pulse would do.
(() => {
  const C = JSON.parse(document.getElementById('candidate').textContent);
  if (!C.has_snippet) return;
  const $ = id => document.getElementById(id);
  const state = { dm: C.dm, tscrunch: 1, nsub: 81, window: 2, clip: 99, mask: '', raw: false, step: 0.05 };
  let last = null, curve = null, pending = null;
  window.viewerDM = () => (state.raw ? 0 : state.dm);

  function params(extra) {
    const p = new URLSearchParams(Object.assign({ tscrunch: state.tscrunch, nsub: state.nsub, window: state.window,
      clip: state.clip, mask: state.mask }, extra || {}));
    return p.toString();
  }

  async function render() {
    const dm = state.raw ? 0 : state.dm;
    $('status').textContent = 'Dedispersing…';
    const token = {};
    pending = token;
    const response = await fetch(`/api/sp/${C.id}/view?${params({ dm })}`);
    if (pending !== token) return;
    if (!response.ok) { $('status').textContent = 'Could not load: ' + response.status; return; }
    last = await response.json();
    draw();
  }

  function draw() {
    const v = last;
    if (!v) return;
    const z = decodeImage(v.image);
    const traces = [
      { type: 'heatmap', x: v.times, y: v.freqs, z, zmin: v.zmin, zmax: v.zmax, colorscale: 'Viridis', showscale: false,
        hovertemplate: 't %{x:.3f} s<br>%{y:.2f} MHz<br>%{z:.2f} σ<extra></extra>', xaxis: 'x', yaxis: 'y' },
      { type: 'scatter', mode: 'lines', x: v.times, y: v.series, line: { width: 1, color: css('--muted') },
        hovertemplate: 't %{x:.3f} s<br>%{y:.2f} σ per sample<extra></extra>', xaxis: 'x', yaxis: 'y2', name: 'per sample' },
      { type: 'scatter', mode: 'lines', x: v.times, y: v.boxcar, line: { width: 1.6, color: css('--accent') },
        hovertemplate: 't %{x:.3f} s<br>boxcar S/N %{y:.2f}<extra></extra>', xaxis: 'x', yaxis: 'y2', name: 'boxcar' },
      { type: 'scatter', mode: 'lines', x: v.spectrum_off, y: v.freqs, line: { width: 1, color: css('--muted') },
        hovertemplate: '%{y:.2f} MHz<br>off-pulse %{x:.2f}<extra></extra>', xaxis: 'x2', yaxis: 'y', name: 'off pulse' },
      { type: 'scatter', mode: 'lines', x: v.spectrum_on, y: v.freqs, line: { width: 1.6, color: css('--accent') },
        hovertemplate: '%{y:.2f} MHz<br>on-pulse %{x:.2f}<extra></extra>', xaxis: 'x2', yaxis: 'y', name: 'on pulse' },
    ];
    if (state.raw) {
      // Where a pulse at the candidate DM would run through the undedispersed data.
      traces.push({ type: 'scatter', mode: 'lines', x: v.sweep, y: v.freqs, xaxis: 'x', yaxis: 'y', hoverinfo: 'skip',
        line: { color: 'rgba(255,80,80,0.9)', width: 1.5, dash: 'dot' }, name: 'expected sweep' });
    }
    plot('dynamic', traces, {
      margin: { l: 58, r: 12, t: 8, b: 44 },
      xaxis: { domain: [0, 0.8], anchor: 'y', title: titled('time relative to the candidate (s)'), showspikes: true, spikemode: 'across', spikethickness: 1 },
      yaxis: { domain: [0, 0.74], title: titled('frequency (MHz)') },
      yaxis2: { domain: [0.78, 1], anchor: 'x', title: titled('S/N'), zeroline: true },
      xaxis2: { domain: [0.84, 1], anchor: 'y', title: titled('S/N per subband') },
      shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: 0, x1: 0, y0: 0.78, y1: 1, line: { color: css('--muted'), width: 1, dash: 'dot' } }],
    });
    const masked = v.masked.length ? v.masked.join(', ') : 'none';
    $('m-masked').textContent = masked;
    $('m-peak').textContent = fmt(v.peak_snr, 1);
    $('m-width').textContent = v.best_width ? `${v.best_width} × ${fmt(v.tsamp * 1e3, 1)} ms (S/N ${fmt(v.best_snr, 1)})` : '—';
    $('status').textContent = `DM ${fmt(v.dm, 3)}${state.raw ? ' (not dedispersed)' : ''} · ${fmt(v.tsamp * 1e3, 2)} ms × ${v.nsub} subbands` +
      ` · boxcar S/N ${fmt(v.peak_snr, 1)} at the search width ${v.width}` +
      (v.dm !== C.dm && !state.raw ? ` · candidate DM ${fmt(C.dm, 2)}` : '');
  }

  async function loadCurve() {
    const response = await fetch(`/api/sp/${C.id}/dm?${new URLSearchParams({ mask: state.mask })}`);
    if (!response.ok) { $('dmplots').textContent = 'Could not compute the DM response.'; return; }
    showCurve(await response.json());
  }

  function showCurve(data) {
    curve = data;
    const holder = $('dmplots');
    if (!holder.dataset.bound) holder.innerHTML = '';
    state.step = Math.max(0.01, Math.round(curve.half_width_dm / 4 * 100) / 100);
    drawCurve();
    const [low, high] = curve.smearing;
    const minimum = Math.hypot(curve.meta.tsamp * 1e3, (low + high) / 2);
    $('m-dm').textContent = curve.best_dm === null ? '—' : `${fmt(curve.best_dm, 2)} (S/N ${fmt(curve.best_snr, 1)})`;
    $('m-smear').textContent = `${fmt(low, 1)} / ${fmt(high, 1)} ms`;
    $('m-min').textContent = `${fmt(minimum, 1)} ms`;
    const width = curve.width_samples * curve.meta.tsamp * 1e3;
    const notes = [];
    if (width + 1e-6 < 0.5 * minimum) {
      notes.push(`<span class="flag bad">Best width ${fmt(width, 1)} ms is under half the ${fmt(minimum, 1)} ms that dispersion within a channel imposes on any real pulse at this DM.</span>`);
    }
    const zero = curve.coarse_snr.length ? curve.coarse_snr[0] : null;
    if (zero !== null && curve.best_snr !== null && zero > curve.best_snr) {
      notes.push(`<span class="flag">S/N at DM 0 (${fmt(zero, 1)}) exceeds S/N at the candidate DM: undispersed signal nearby.</span>`);
    }
    $('m-verdict').innerHTML = notes.join('<br>');
  }

  function drawCurve() {
    if (!curve) return;
    const accent = css('--accent'), muted = css('--muted'), err = css('--err');
    const plane = decodeImage(curve.plane);
    const marker = (axis) => ({ type: 'line', xref: axis, yref: 'paper', x0: C.dm, x1: C.dm, y0: 0, y1: 1,
      line: { color: muted, width: 1, dash: 'dot' } });
    const current = (axis) => ({ type: 'line', xref: axis, yref: 'paper', x0: window.viewerDM(), x1: window.viewerDM(), y0: 0, y1: 1,
      line: { color: err, width: 1 } });
    plot('dmplots', [
      { type: 'scatter', mode: 'lines+markers', x: curve.fine_dms, y: curve.fine_snr, marker: { size: 4, color: accent },
        line: { color: accent, width: 1.5 }, name: 'measured', hovertemplate: 'DM %{x:.3f}<br>S/N %{y:.2f}<extra></extra>' },
      { type: 'scatter', mode: 'lines', x: curve.fine_dms, y: curve.expected_snr, line: { color: muted, dash: 'dash', width: 1.5 },
        name: `real pulse, ${fmt(curve.width_ms, 0)} ms`, hovertemplate: 'expected %{y:.2f}<extra></extra>' },
      { type: 'scatter', mode: 'lines+markers', x: curve.coarse_dms, y: curve.coarse_snr, marker: { size: 3, color: accent },
        line: { color: accent, width: 1.2 }, xaxis: 'x2', yaxis: 'y2', name: 'from DM 0',
        hovertemplate: 'DM %{x:.1f}<br>S/N %{y:.2f}<extra></extra>' },
      { type: 'heatmap', x: curve.plane_times, y: curve.fine_dms, z: plane, colorscale: 'Viridis', showscale: false,
        xaxis: 'x3', yaxis: 'y3', hovertemplate: 't %{x:.3f} s<br>DM %{y:.2f}<br>S/N %{z:.2f}<extra></extra>' },
    ], {
      showlegend: true, legend: { x: 0, y: 1.16, orientation: 'h', font: { size: 11, color: muted } },
      margin: { l: 50, r: 10, t: 26, b: 42 },
      xaxis: { domain: [0, 0.31], title: titled('DM (pc cm⁻³)') }, yaxis: { title: titled('peak boxcar S/N') },
      xaxis2: { domain: [0.37, 0.64], anchor: 'y2', title: titled('DM from 0') }, yaxis2: { anchor: 'x2' },
      xaxis3: { domain: [0.71, 1], anchor: 'y3', title: titled('time (s)') }, yaxis3: { anchor: 'x3', title: titled('DM') },
      shapes: [marker('x'), marker('x2'), current('x'), current('x2')],
    });
    const el = $('dmplots');
    if (el.on && !el.dataset.bound) {
      el.dataset.bound = '1';
      el.on('plotly_click', e => {
        const p = e.points[0];
        if (p.data.type === 'heatmap') setDM(p.y); else setDM(p.x);
      });
    }
  }

  function setDM(dm) {
    state.dm = Math.max(0, Math.round(dm * 1000) / 1000);
    state.raw = false;
    $('raw').checked = false;
    $('dm').value = state.dm.toFixed(3);
    render();
    drawCurve();
  }

  let timer = null;
  const later = () => { clearTimeout(timer); timer = setTimeout(render, 120); };
  document.querySelectorAll('[data-step]').forEach(b => b.addEventListener('click', () => setDM(state.dm + Number(b.dataset.step) * state.step)));
  $('dm-reset').addEventListener('click', () => setDM(C.dm));
  $('dm').addEventListener('change', e => setDM(parseFloat(e.target.value) || 0));
  for (const id of ['tscrunch', 'nsub', 'window', 'clip']) {
    $(id).addEventListener('change', e => { state[id] = Number(e.target.value); later(); });
  }
  $('mask').addEventListener('change', e => { state.mask = e.target.value.trim(); render(); loadCurve(); });
  $('raw').addEventListener('change', e => { state.raw = e.target.checked; render(); drawCurve(); });
  document.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      setDM(state.dm + (e.key === 'ArrowLeft' ? -1 : 1) * state.step * (e.shiftKey ? 10 : 1));
    } else if (e.key === 'd' || e.key === 'D') {
      $('raw').checked = !$('raw').checked; state.raw = $('raw').checked; render(); drawCurve();
    }
  });
  window.addEventListener('themechange', () => { draw(); drawCurve(); });
  document.addEventListener('DOMContentLoaded', () => {
    // The page carries the first render, and the DM response once it has been computed.
    const initial = JSON.parse(($('initial') || {}).textContent || '{}');
    if (initial.view) { last = initial.view; draw(); } else render();
    if (initial.dm) showCurve(initial.dm); else loadCurve();
  });
})();
