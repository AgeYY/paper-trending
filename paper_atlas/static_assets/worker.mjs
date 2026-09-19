import {parseQuery, phraseText, search} from './search.mjs';
let manifest;
const cache = new Map(); // Public dataset cache only; never query history.
async function load(descriptor) {
  if (cache.has(descriptor.file)) return cache.get(descriptor.file);
  const response = await fetch(descriptor.file);
  if (!response.ok) throw Error(`Data download failed (${response.status}). Please retry.`);
  const bytes = await response.arrayBuffer();
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2,'0')).join('');
  if (hash !== descriptor.sha256) throw Error('Snapshot integrity check failed. Reload the page to get the latest version.');
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  const data = JSON.parse(await new Response(stream).text());
  if (data.length !== descriptor.count) throw Error('Snapshot count mismatch; search cancelled.');
  if (descriptor.kind === 'papers') for (const p of data) {
    p.text = ` ${phraseText(p.title + '. ' + p.abstract)} `;
    p.title_text = ` ${phraseText(p.title)} `;
  }
  const shard = {...descriptor, data}; cache.set(descriptor.file, shard); return shard;
}
self.onmessage = async ({data: {id, query}}) => {
  try {
    parseQuery(query.prompt);
    if (!manifest) {
      const response = await fetch('manifest.json', {cache:'no-cache'});
      if (!response.ok) throw Error('Cannot load the dataset manifest.');
      manifest = await response.json();
    }
    const needed = manifest.shards.filter(s => query.years.includes(s.year) && query.venues.includes(s.venue) &&
      (s.kind === 'papers' || (query.tracks.length === 1 && query.tracks[0] === 'research')));
    const pending = needed.filter(s => !cache.has(s.file));
    const size = pending.reduce((n,s) => n+s.bytes,0);
    self.postMessage({id, progress:`Loading public paper data: ${(size/1e6).toFixed(1)} MB to download. Your query stays in this browser.`});
    // Bounded concurrency and incremental decompression keep mobile peak memory lower.
    const shards = [];
    for (const s of needed) shards.push(await load(s));
    const result = search(manifest, shards, query);
    self.postMessage({id, result});
  } catch (error) { self.postMessage({id, error:error.message || 'Search failed.'}); }
};
