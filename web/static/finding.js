// Recording a finding, from a candidate page or a lead on /findings: the form's data attributes carry the source,
// its other names, the evidence (candidate keys) and, for a lead, the text of each claim.
(() => {
  function remembered(id) { try { return localStorage.getItem(id) || ''; } catch (e) { return ''; } }
  function remember(id, value) { try { localStorage.setItem(id, value); } catch (e) {} }

  async function record(form, event) {
    event.preventDefault();
    const data = new FormData(form);
    const status = form.querySelector('.status-line');
    const reviewer = (data.get('reviewer') || '').trim();
    if (!reviewer) { status.textContent = 'Enter your name first.'; return; }
    remember('reviewer', reviewer);
    const texts = JSON.parse(form.dataset.claims || '{}');
    const body = {
      source: (data.get('source') || form.dataset.source || '').trim(), status: data.get('status'),
      names: JSON.parse(form.dataset.names || '[]'), evidence: JSON.parse(form.dataset.evidence || '[]'),
      claims: data.getAll('claim').map(kind => [kind, texts[kind] || '']),
      summary: data.get('summary') || '', checked: data.get('checked') || '', reviewer,
    };
    status.textContent = 'Saving…';
    const response = await fetch('/api/finding', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      status.textContent = 'Not saved: ' + (detail.detail || response.status);
      return;
    }
    status.innerHTML = `Recorded: <a href="/findings#${encodeURIComponent(body.source)}">${body.source} on Findings</a>.`;
    if (location.pathname === '/findings') setTimeout(() => location.reload(), 500);
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('form.finding-form').forEach(form => {
      const reviewer = form.querySelector('.finding-reviewer');
      if (reviewer && !reviewer.value) reviewer.value = remembered('reviewer');
      form.addEventListener('submit', event => record(form, event));
    });
  });
})();
