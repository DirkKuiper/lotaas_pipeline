// Chart renderers for the overview, staging, coverage, SAP and beam pages.
const pages = {};

function hourAxis(start, count) {
  return Array.from({ length: count }, (_, i) => new Date((start + i * 3600) * 1000).toISOString());
}

function stackedBar(el, entries, label) {
  const traces = entries.filter(([, n]) => n > 0).map(([state, n]) => ({
    type: 'bar', orientation: 'h', x: [n], y: [label], name: state, marker: { color: stateColor(state) },
    hovertemplate: `${state}: %{x}<extra></extra>`,
  }));
  plot(el, traces, { barmode: 'stack', margin: { l: 10, r: 10, t: 4, b: 22 }, yaxis: { visible: false },
                     xaxis: { showgrid: false } }, { displayModeBar: false });
}

pages.overview = (d) => {
  stackedBar('sap-bar', d.saps, 'SAPs');
  stackedBar('file-bar', d.files, 'files');
  const start = d.now - d.hours * 3600;
  const x = hourAxis(start, d.hours);
  plot('throughput', [
    { type: 'bar', x, y: d.converted, name: 'files converted', marker: { color: stateColor('converted') },
      hovertemplate: '%{x|%d %b %H:00}<br>%{y} files converted<extra></extra>' },
    { type: 'bar', x, y: d.searched, name: 'beams searched', marker: { color: stateColor('searched') },
      hovertemplate: '%{x|%d %b %H:00}<br>%{y} beams searched<extra></extra>' },
  ], { barmode: 'group', showlegend: true, bargap: 0.15, yaxis: { title: titled('per hour') } });
};

pages.staging = (d) => {
  if (d.latency.length) {
    plot('latency', [{ type: 'histogram', x: d.latency, marker: { color: stateColor('online') },
      hovertemplate: '%{x} h: %{y} files<extra></extra>' }],
      { xaxis: { title: titled('hours from request to on disk') }, yaxis: { title: titled('files') } });
  }
  const kinds = [...new Set(d.kinds.map(k => k.kind))];
  const colors = { submitted: stateColor('staging'), restage: css('--warn'), refused: css('--bad'),
                   throttled: stateColor('pending'), retrieve_failed: css('--err'), file_failed: css('--err') };
  const x = hourAxis(d.start, 72);
  plot('event-kinds', kinds.map(kind => {
    const y = new Array(72).fill(0);
    d.kinds.filter(k => k.kind === kind).forEach(k => { if (k.hour >= 0 && k.hour < 72) y[k.hour] = k.n; });
    return { type: 'bar', x, y, name: kind, marker: { color: colors[kind] || css('--muted') } };
  }), { barmode: 'stack', showlegend: true, yaxis: { title: titled('events per hour') } });
};

pages.coverage = (d) => {
  const states = [...new Set(d.sky.map(s => s.state))];
  const draw = () => plot('sky', states.map(state => {
    const points = d.sky.filter(s => s.state === state);
    return { type: d.sky.length > 5000 ? 'scattergl' : 'scatter', mode: 'markers', name: state, x: points.map(p => p.ra), y: points.map(p => p.dec),
      customdata: points.map(p => [p.key, p.pointing || '', p.searched || 0]),
      marker: { size: 11, color: stateColor(state), line: { width: 1, color: css('--panel') } },
      hovertemplate: '%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]} beams searched<br>' +
        'RA %{x:.2f}°, Dec %{y:.2f}°<extra>' + state + '</extra>' };
  }), { showlegend: true, xaxis: { title: titled('RA (deg)'), autorange: 'reversed' },
        yaxis: { title: titled('Dec (deg)'), scaleanchor: 'x', scaleratio: 1 } });
  draw();
  const el = document.getElementById('sky');
  if (el && el.on) el.on('plotly_click', e => { location.href = '/sap/' + e.points[0].customdata[0]; });
  window.addEventListener('themechange', draw);

  const table = document.getElementById('sap-table');
  const chips = document.querySelectorAll('.chip[data-state]');
  const search = document.getElementById('sap-search');
  const apply = () => {
    const active = [...chips].filter(c => c.classList.contains('on')).map(c => c.dataset.state);
    const text = (search.value || '').toLowerCase();
    const shown = filterRows(table, row => (!active.length || active.includes(row.dataset.state))
      && (!text || row.textContent.toLowerCase().includes(text)));
    document.getElementById('sap-shown').textContent = shown;
  };
  chips.forEach(c => c.addEventListener('click', () => { c.classList.toggle('on'); apply(); }));
  search.addEventListener('input', apply);
};

pages.sap = (d) => {
  if (!d.layout.length) return;
  const draw = () => plot('layout', [{
    type: 'scatter', mode: 'markers+text', x: d.layout.map(b => b.ra), y: d.layout.map(b => b.dec),
    text: d.layout.map(b => String(b.beam)), textposition: 'middle center',
    textfont: { size: 8, color: '#fff' },
    customdata: d.layout.map(b => [b.item, b.snr, b.clusters, b.positives]),
    marker: { size: 17, color: d.layout.map(b => b.snr ?? 0), colorscale: 'Viridis', showscale: true,
              colorbar: { title: titled('max cluster S/N'), thickness: 10 },
              line: { width: d.layout.map(b => (b.positives ? 3 : 0)), color: css('--err') } },
    hovertemplate: 'beam %{text}<br>max cluster S/N %{customdata[1]}<br>%{customdata[2]} clusters<br>' +
      '%{customdata[3]} FETCH positives<extra></extra>',
  }], { xaxis: { title: titled('RA (deg)'), autorange: 'reversed' },
        yaxis: { title: titled('Dec (deg)'), scaleanchor: 'x', scaleratio: Math.cos(d.layout[0].dec * Math.PI / 180) } });
  draw();
  const el = document.getElementById('layout');
  if (el && el.on) el.on('plotly_click', e => { location.href = '/beam/' + e.points[0].customdata[0]; });
  window.addEventListener('themechange', draw);
};

pages.beam = (d) => {
  if (!d.clusters.length && !d.found.length) return;
  const draw = () => {
    const traces = [{
      type: d.clusters.length > 5000 ? 'scattergl' : 'scatter', mode: 'markers', name: 'clusters', x: d.clusters.map(c => c[2]), y: d.clusters.map(c => c[0]),
      customdata: d.clusters.map(c => [c[1], c[3]]),
      marker: { size: d.clusters.map(c => Math.max(4, Math.min(22, (c[1] - 4) * 2.5))), color: d.clusters.map(c => c[1]),
                colorscale: 'Viridis', showscale: true, colorbar: { title: titled('S/N'), thickness: 10 }, opacity: 0.8 },
      hovertemplate: 't %{x:.3f} s<br>DM %{y}<br>S/N %{customdata[0]}<br>width %{customdata[1]}<extra></extra>',
    }];
    const kinds = { candidate: css('--err'), known_pulsar: css('--l-known'), rejected: css('--muted') };
    for (const [type, color] of Object.entries(kinds)) {
      const f = d.found.filter(c => c.type === type);
      if (!f.length) continue;
      traces.push({ type: 'scatter', mode: 'markers', name: type, x: f.map(c => c.time), y: f.map(c => c.dm),
        customdata: f.map(c => [c.id, c.snr]),
        marker: { symbol: type === 'rejected' ? 'x-thin-open' : 'star', size: type === 'rejected' ? 9 : 15,
                  color, line: { width: 1.5, color } },
        hovertemplate: type + '<br>t %{x:.3f} s<br>DM %{y}<br>S/N %{customdata[1]}<extra>click to review</extra>' });
    }
    plot('clusters', traces, { showlegend: true, xaxis: { title: titled('time (s)') }, yaxis: { title: titled('DM (pc cm⁻³)') } });
  };
  draw();
  const el = document.getElementById('clusters');
  if (el && el.on) el.on('plotly_click', e => { const c = e.points[0].customdata; if (c && typeof c[0] === 'string') location.href = '/verify/' + c[0]; });
  window.addEventListener('themechange', draw);
};

document.addEventListener('DOMContentLoaded', () => {
  const name = document.body.dataset.page;
  if (!pages[name]) return;
  const data = chartData();
  pages[name](data);
  if (name !== 'coverage' && name !== 'sap' && name !== 'beam') window.addEventListener('themechange', () => pages[name](data));
});
