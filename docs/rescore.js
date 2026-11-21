// Every model against the old and the new GA base, on the same fresh games (data/rescore_ga2.json, written by
// rl-v3's experiments/rescore_ga2.py). Rows whose label contains a word in data-mine are this project's models.
(function () {
  const box = document.getElementById('rescore');
  if (!box) return;
  const mine = (box.dataset.mine || '').split(',').filter(Boolean);
  const pc = (x) => (100 * x).toFixed(1) + '%';
  const e = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  fetch('data/rescore_ga2.json?t=' + Date.now()).then((r) => (r.ok ? r.json() : null)).then((d) => {
    if (!d || !d.models || !d.models.length) { box.innerHTML = '<p class="muted">The new scores are being measured.</p>'; return; }
    const rows = d.models.map((m) => {
      const a = m.vs_ga.score, b = m.vs_ga2.score, own = mine.some((w) => m.label.indexOf(w) >= 0);
      return `<tr${own ? ' style="font-weight:600"' : ''}><td>${e(m.label)}</td><td class="num">${pc(a)}</td>
        <td class="num">${pc(b)} <span class="muted small">${pc(m.vs_ga2.ci[0])}–${pc(m.vs_ga2.ci[1])}</span></td>
        <td class="num">${(b - a >= 0 ? '+' : '−') + Math.abs(100 * (b - a)).toFixed(1)}</td></tr>`;
    }).join('');
    const top = Object.entries(d.pick_counts || {}).slice(0, 5).map(([n, c]) => `${e(n)} ${Math.round(100 * c / d.kingdoms)}%`).join(', ');
    box.innerHTML = `<div class="tablewrap"><table class="t"><tr><th>Model</th><th>vs old GA base</th><th>vs new GA base (95% range)</th><th>Change, points</th></tr>${rows}</table></div>
      <p class="muted small">${e(d.games)} (${(2 * d.kingdoms).toLocaleString()} games per model and base), measured ${e(d.t)}.
      Old base: ${e(d.old_base)}. New base: ${e(d.new_base)}; its most common picks: ${top}.
      ${d.models.length < 11 ? 'More models are still being measured; this table fills in as they finish.' : ''}</p>`;
  });
})();
