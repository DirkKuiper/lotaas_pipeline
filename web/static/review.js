// Saving a verdict: one key (R, N, K, A, U) or click saves it and, with 'next to review after
// saving', opens the next candidate without a verdict; ',' and '.' step through the list itself.
(() => {
  const C = JSON.parse(document.getElementById('candidate').textContent);
  const $ = id => document.getElementById(id);
  const keys = { r: 'rfi', n: 'noise', k: 'known', a: 'astro', u: 'unsure' };
  const names = {};
  document.querySelectorAll('[data-label]').forEach(b => { names[b.dataset.label] = b.textContent.trim().replace(/\s+\S$/, ''); });

  function remembered(id, fallback) { try { return localStorage.getItem(id) ?? fallback; } catch (e) { return fallback; } }
  function remember(id, value) { try { localStorage.setItem(id, value); } catch (e) {} }

  async function save(label) {
    const reviewer = $('reviewer').value.trim();
    if (!reviewer) { $('review-status').textContent = 'Enter your name first.'; $('reviewer').focus(); return; }
    remember('reviewer', reviewer);
    $('review-status').textContent = 'Saving…';
    const response = await fetch('/api/review', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: C.id, label, reviewer, note: $('note').value,
        dm: window.viewerDM ? window.viewerDM() : C.dm, mask: window.viewerMask ? window.viewerMask() : '' }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      $('review-status').textContent = 'Not saved: ' + (detail.detail || response.status);
      return;
    }
    const { reviews } = await response.json();
    document.querySelectorAll('[data-label]').forEach(b => b.classList.toggle('on', b.dataset.label === label));
    $('history').innerHTML = reviews.map(r => `<li><span class="label-tag ${r.label}">${names[r.label] || r.label}</span> ${escape(r.reviewer)} · ` +
      `<span class="muted small">${new Date(r.created * 1000).toISOString().slice(0, 16).replace('T', ' ')}` +
      `${r.dm !== null && r.dm !== undefined ? ' · DM ' + Number(r.dm).toFixed(2) : ''}</span>` +
      `${r.note ? '<div class="small">' + escape(r.note) + '</div>' : ''}</li>`).join('');
    $('note').value = '';
    $('review-status').textContent = 'Saved.';
    // Onwards to the next candidate still without a verdict, never to one already classified.
    if (!$('advance').checked) return;
    const next = $('next-unreviewed');
    if (next) { setTimeout(() => { location.href = next.href; }, 350); return; }
    const back = $('review-panel').dataset.back;
    $('review-status').innerHTML = 'Saved. Everything in this queue has a verdict: ' +
      `<a href="${back}">back to the list</a>.`;
  }

  function escape(text) { const d = document.createElement('div'); d.textContent = text; return d.innerHTML; }

  document.addEventListener('DOMContentLoaded', () => {
    $('reviewer').value = remembered('reviewer', '');
    $('advance').checked = remembered('advance', '1') === '1';
    $('advance').addEventListener('change', e => remember('advance', e.target.checked ? '1' : '0'));
    document.querySelectorAll('[data-label]').forEach(b => b.addEventListener('click', () => save(b.dataset.label)));
  });
  document.addEventListener('keydown', e => {
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
    const label = keys[e.key.toLowerCase()];
    if (label) { e.preventDefault(); save(label); return; }
    if (e.key === '.' && $('next')) location.href = $('next').href;
    if (e.key === ',' && $('previous')) location.href = $('previous').href;
  });
})();
