// The single-pulse viewer: dynamic spectrum at any DM, band-averaged series,
// on/off-pulse spectrum, and S/N against DM with what a real pulse would do.
(() => {
  const C = JSON.parse(document.getElementById('candidate').textContent);
  if (!C.has_snippet) return;
  const $ = id => document.getElementById(id);
  // The server picks the first display: bins of the boxcar width and as many
  // subbands as keep the pulse visible (about 2.5 sigma per pixel).
  const state = { dm: C.dm, tscrunch: C.tscrunch || 1, nsub: C.nsub || 81, window: -1, clip: 99, mask: '',
    auto_mask: true, raw: false, step: 0.05, maskDrag: false };
  let last = null, curve = null, pending = null;
  window.viewerDM = () => (state.raw ? 0 : state.dm);
  window.viewerMask = () => state.mask;

  function addMaskRange(f0, f1) {
    // Frequencies of the dragged band to file-order channel indices.
    if (C.fch1 === null || C.foff === null || !C.nchans) return;
    const a = (f0 - C.fch1) / C.foff, b = (f1 - C.fch1) / C.foff;
    const lo = Math.max(0, Math.floor(Math.min(a, b))), hi = Math.min(C.nchans - 1, Math.ceil(Math.max(a, b)));
    if (hi < lo) return;
    state.mask = state.mask ? `${state.mask},${lo}-${hi}` : `${lo}-${hi}`;
    $('mask').value = state.mask;
    render();
    loadCurve();
  }

  function params(extra) {
    const p = new URLSearchParams(Object.assign({ tscrunch: state.tscrunch, nsub: state.nsub, window: state.window,
      clip: state.clip, mask: state.mask, auto_mask: state.auto_mask }, extra || {}));
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
      dragmode: state.maskDrag ? 'select' : 'zoom', selectdirection: 'v',
      margin: { l: 58, r: 12, t: 8, b: 44 },
      xaxis: { domain: [0, 0.8], anchor: 'y', title: titled('time relative to the candidate (s)'), showspikes: true, spikemode: 'across', spikethickness: 1 },
      yaxis: { domain: [0, 0.74], title: titled('frequency (MHz)') },
      yaxis2: { domain: [0.78, 1], anchor: 'x', title: titled('S/N'), zeroline: true },
      xaxis2: { domain: [0.84, 1], anchor: 'y', title: titled('S/N per subband') },
      shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: 0, x1: 0, y0: 0.78, y1: 1, line: { color: css('--muted'), width: 1, dash: 'dot' } }],
    });
    const holder = $('dynamic');
    if (holder.on && !holder.dataset.maskBound) {
      holder.dataset.maskBound = '1';
      holder.on('plotly_selected', e => {
        if (state.maskDrag && e && e.range && e.range.y) addMaskRange(e.range.y[0], e.range.y[1]);
      });
    }
    drawProfiles(v);
    const warn = $('m-maskwarn');
    if (state.mask && v.unmasked_peak_snr !== null && v.peak_snr !== null) {
      const gain = v.peak_snr - v.unmasked_peak_snr;
      warn.innerHTML = `Local S/N ${fmt(v.unmasked_peak_snr, 1)} with the pipeline's mask, ${fmt(v.peak_snr, 1)} with yours.` +
        (gain > 1 ? ' <span class="flag">Your mask raised it by more than 1. Channels chosen while looking at the pulse raise S/N by chance alone; mask only channels that are bad away from the pulse.</span>' : '');
    } else {
      warn.textContent = '';
    }
    const masked = v.masked.length ? v.masked.join(', ') : 'none';
    $('m-masked').textContent = masked;
    $('m-peak').textContent = fmt(v.peak_snr, 1);
    $('m-width').textContent = v.best_width ? `${fmt(v.best_width * v.analysis_tsamp, 3)} s (S/N ${fmt(v.best_snr, 1)})` : '—';
    $('status').textContent = `DM ${fmt(v.dm, 3)}${state.raw ? ' (not dedispersed)' : ''} · ${fmt(v.tsamp * 1e3, 2)} ms × ${v.nsub} subbands` +
      ` · local S/N ${fmt(v.peak_snr, 1)} at ${fmt(v.width_seconds, 3)} s width (${v.reference_windows} reference windows)` +
      (v.pixel_snr !== null && v.pixel_snr !== undefined ? ` · pulse ≈ ${fmt(v.pixel_snr, 1)}σ per pixel` +
        (v.pixel_snr < 2 ? ' (too faint to see here: fewer subbands or wider bins)' : '') : '') +
      (state.auto_mask && v.automatic_bad.length ? ` · ${v.automatic_bad.length} persistently noisy channels masked` : '') +
      (v.dm !== C.dm && !state.raw ? ` · candidate DM ${fmt(C.dm, 2)}` : '');
  }

  function drawProfiles(v) {
    if (!v.profiles || !v.profiles.length) return;
    const n = v.profiles.length, step = 6;
    const traces = v.profiles.map((p, k) => ({
      type: 'scatter', mode: 'lines', x: v.times, y: p.map(y => (y === null ? null : y + step * (n - 1 - k))),
      line: { width: 1.2, color: k % 2 ? css('--accent') : css('--text') },
      hovertemplate: `${fmt(v.profile_freqs[k], 1)} MHz<br>t %{x:.3f} s<br>S/N %{customdata:.1f}<extra></extra>`,
      customdata: p,
    }));
    plot('profiles', traces, {
      height: 170, margin: { l: 58, r: 12, t: 4, b: 34 },
      xaxis: { domain: [0, 0.8], title: titled('time relative to the candidate (s)') },
      yaxis: { tickvals: v.profile_freqs.map((_, k) => step * (n - 1 - k)),
        ticktext: v.profile_freqs.map(f => `${fmt(f, 0)} MHz`), zeroline: false },
      shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: 0, x1: 0, y0: 0, y1: 1, line: { color: css('--muted'), width: 1, dash: 'dot' } }],
    });
  }

  async function loadCurve() {
    const response = await fetch(`/api/sp/${C.id}/dm?${new URLSearchParams({ mask: state.mask, auto_mask: state.auto_mask })}`);
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
      notes.push(`<span class="flag">Best width ${fmt(width, 1)} ms is below the estimated instrumental broadening. Treat this as a diagnostic; it is not an automatic rejection.</span>`);
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
  $('auto-mask').addEventListener('change', e => { state.auto_mask = e.target.checked; render(); loadCurve(); });
  $('raw').addEventListener('change', e => { state.raw = e.target.checked; render(); drawCurve(); });
  $('mask-drag').addEventListener('change', e => { state.maskDrag = e.target.checked; draw(); });
  $('mask-clear').addEventListener('click', () => { state.mask = ''; $('mask').value = ''; render(); loadCurve(); });
  document.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      setDM(state.dm + (e.key === 'ArrowLeft' ? -1 : 1) * state.step * (e.shiftKey ? 10 : 1));
    } else if (e.key === 'd' || e.key === 'D') {
      $('raw').checked = !$('raw').checked; state.raw = $('raw').checked; render(); drawCurve();
    } else if (e.key === 'm' || e.key === 'M') {
      $('mask-drag').checked = !$('mask-drag').checked; state.maskDrag = $('mask-drag').checked; draw();
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
