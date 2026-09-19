// Pure keyword matching shared by the browser worker and parity tests. No network here.
export function phraseText(text) {
  return (text.normalize('NFKC').toLowerCase().match(/[a-z0-9]+/g) || [])
    .map(t => t.length > 4 && t.endsWith('s') && !/(ss|us|is)$/.test(t) ? t.slice(0, -1) : t).join(' ');
}
export function parseQuery(prompt) {
  if (typeof prompt !== 'string' || !prompt.trim() || [...prompt.trim()].length > 1500) throw Error('Enter 1–1,500 characters.');
  const phrases = [...new Set(prompt.split(',').map(s => s.trim()).filter(Boolean))];
  if (!phrases.length || phrases.length > 16 || phrases.some(s => [...s].length > 120 || !phraseText(s)))
    throw Error('Use 1–16 comma-separated phrases, each containing letters or numbers and at most 120 characters.');
  return phrases.map(s => ` ${phraseText(s)} `);
}
export function search(manifest, shards, query) {
  const terms = parseQuery(query.prompt);
  if (!query.years?.length || !query.venues?.length || !query.tracks?.length) throw Error('Select at least one year, conference and track.');
  if (query.years.some(y => !manifest.years.includes(y)) || query.venues.some(v => !manifest.venues[v]) || query.tracks.some(t => !manifest.tracks.includes(t)))
    throw Error('Unsupported search scope.');
  const matches = [], totals = new Map();
  for (const shard of shards) {
    if (shard.kind !== 'papers' || !query.years.includes(shard.year) || !query.venues.includes(shard.venue)) continue;
    for (const p of shard.data) {
      if (!query.tracks.includes(p.track) || !terms.every(t => p.text.includes(t))) continue;
      const {text, title_text, ...paper} = p;
      matches.push({...paper, year: shard.year, venue: shard.venue,
        title_hits: terms.filter(t => title_text.includes(t)).length});
      const key = `${shard.year}-${shard.venue}`;
      totals.set(key, (totals.get(key) || 0) + 1);
    }
  }
  // Ranking is deliberately simpler than Python's IDF ranking; membership is identical.
  matches.sort((a,b) => b.title_hits-a.title_hits || b.year-a.year || a.id.localeCompare(b.id));
  const coverage = manifest.coverage[[...query.tracks].sort().join(',')]
    .filter(r => query.years.includes(r.year) && query.venues.includes(r.venue));
  const counts = coverage.map(r => ({...r, candidates: ['complete','partial'].includes(r.status) ? totals.get(`${r.year}-${r.venue}`) || 0 : null}));
  const acceptance = query.venues.includes('iclr') ? query.years.map(year => {
    const base = manifest.acceptance[String(year)] || {available:false};
    const row = {year, ...base, topic_accepted:null, topic_submitted:null, topic_rate:null, conference_rate:null};
    if (!base.available || query.tracks.length !== 1 || query.tracks[0] !== 'research') return row;
    const pool = shards.find(s => s.kind === 'submissions' && s.year === year);
    if (!pool) throw Error(`Submission data missing for ${year}; no partial rates computed.`);
    const statuses = pool.data.filter(([text]) => terms.every(t => text.includes(t))).map(([,s]) => s);
    row.topic_submitted = statuses.length;
    row.topic_accepted = statuses.filter(s => s === 'accepted').length;
    row.topic_rate = statuses.length ? 100 * row.topic_accepted / statuses.length : null;
    row.conference_rate = 100 * base.accepted / base.submitted;
    return row;
  }) : [];
  return {query, matches, counts, acceptance, snapshot:manifest.snapshot, corpus_sha256:manifest.corpus_sha256};
}
