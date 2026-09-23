// The periodic-candidate viewer, from the fold archive the search keeps.
(() => {
  const C = JSON.parse(document.getElementById('candidate').textContent);
  let fold = null;

  function twice(row) { return row ? row.concat(row) : row; }

  function draw() {
    const f = fold;
    if (!f) return;
    const accent = css('--accent'), muted = css('--muted');
    const bins = f.profile ? f.profile.length : 0;
    const phase = Array.from({ length: 2 * bins }, (_, i) => (i + 0.5) / bins);
    const profile = twice(f.profile), errors = twice(f.errors);
    plot('profile', [
      { type: 'scatter', mode: 'lines', x: phase, y: profile.map((v, i) => v + (errors[i] || 0)), line: { width: 0 }, hoverinfo: 'skip' },
      { type: 'scatter', mode: 'lines', x: phase, y: profile.map((v, i) => v - (errors[i] || 0)), line: { width: 0 }, fill: 'tonexty',
        fillcolor: 'rgba(120,130,150,0.2)', hoverinfo: 'skip' },
      { type: 'scatter', mode: 'lines', x: phase, y: profile, line: { color: accent, width: 1.8, shape: 'hvh' },
        hovertemplate: 'phase %{x:.3f}<br>%{y:.3g}<extra></extra>' },
    ], { xaxis: { title: titled('pulse phase (two cycles)') }, yaxis: { title: titled('intensity') } });
    if (f.subints) {
      const n = f.subints.length;
      const minutes = Array.from({ length: n }, (_, i) => (i + 0.5) * (f.observation_seconds || 3600) / n / 60);
      plot('subints', [{ type: 'heatmap', x: phase, y: minutes, z: f.subints.map(twice), colorscale: 'Viridis', showscale: false,
        hovertemplate: 'phase %{x:.3f}<br>%{y:.1f} min<extra></extra>' }],
        { xaxis: { title: titled('pulse phase') }, yaxis: { title: titled('time (minutes)') } });
    }
    if (f.subbands) {
      const n = f.subbands.length;
      const [low, high] = f.band;
      const freqs = Array.from({ length: n }, (_, i) => low + (i + 0.5) * (high - low) / n);
      plot('subbands', [{ type: 'heatmap', x: phase, y: freqs, z: f.subbands.map(twice), colorscale: 'Viridis', showscale: false,
        hovertemplate: 'phase %{x:.3f}<br>%{y:.2f} MHz<extra></extra>' }],
        { xaxis: { title: titled('pulse phase') }, yaxis: { title: titled('frequency (MHz)') } });
    } else {
      document.getElementById('subbands').innerHTML = '<div class="muted">The search folds the filterbank by subband only for the top non-RFI candidate of a beam.</div>';
      document.getElementById('subbands-note').textContent = '';
    }
    const line = (x, axis) => ({ type: 'line', xref: axis || 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: muted, width: 1, dash: 'dot' } });
    if (f.dm_curve && f.dm_curve.length) {
      plot('dmcurve', [{ type: 'scatter', mode: 'markers', x: f.dm_curve.map(p => p[0]), y: f.dm_curve.map(p => p[1]),
        marker: { size: 5, color: accent }, hovertemplate: 'DM %{x}<br>−log₁₀ p %{y:.2f}<extra></extra>' }],
        { xaxis: { title: titled('DM (pc cm⁻³)') }, yaxis: { title: titled('search −log₁₀ p') }, shapes: [line(C.dm)] });
    }
    if (f.offsets_uhz && f.fold_chi2) {
      plot('chi2', [{ type: 'scatter', mode: 'lines+markers', x: f.offsets_uhz, y: f.fold_chi2, marker: { size: 4, color: accent },
        line: { color: accent, width: 1.2 }, hovertemplate: '%{x:.2f} µHz<br>χ² %{y:.1f}<extra></extra>' }],
        { xaxis: { title: titled('offset from the search frequency (µHz)') }, yaxis: { title: titled('fold χ²') },
          shapes: f.refined_offset_uhz !== null ? [line(f.refined_offset_uhz)] : [] });
    }
  }

  document.addEventListener('DOMContentLoaded', async () => {
    const initial = JSON.parse((document.getElementById('initial') || {}).textContent || '{}');
    if (initial.fold) { fold = initial.fold; draw(); return; }
    const response = await fetch(`/api/periodic/${C.id}`);
    if (!response.ok) { document.getElementById('profile').textContent = 'The fold archive could not be read.'; return; }
    fold = await response.json();
    draw();
  });
  window.addEventListener('themechange', draw);
})();
