"use strict";
const element = (tag, text) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node; };
const count = x => Number(x).toLocaleString("en-US");
function table(headers, rows) {
  const root = element("table"), head = element("tr");
  headers.forEach(x => head.append(element("th", x))); root.append(head);
  rows.forEach(values => { const row = element("tr"); values.forEach(x => row.append(element("td", x))); root.append(row); });
  return root;
}
(async () => {
  try {
    const response = await fetch("/api/algorithm");
    if (!response.ok) throw new Error("The algorithm details could not be loaded.");
    const info = await response.json(), meta = info.metadata;
    const date = new Intl.DateTimeFormat("en-US", {year: "numeric", month: "short", day: "numeric", timeZone: "UTC"}).format(new Date(meta.snapshot));
    document.getElementById("corpus-summary").textContent = `Current snapshot: ${date} (UTC). ${count(meta.papers)} research papers; ${count(info.catalog_publications)} publications across all indexed tracks.`;
    document.getElementById("algorithm-coverage").replaceChildren(table(["Conference", "Year", "Research papers", "Coverage"], meta.coverage.map(r => [meta.venues[r.venue], r.year, ["failed", "unavailable"].includes(r.status) ? "Unavailable" : count(r.papers), r.status])));
    document.getElementById("algorithm-ai").textContent = `Interpreter: ${info.ai.model} · Prompt version: ${info.ai.prompt_version} · AI ${info.ai.enabled ? "enabled" : "disabled on this server; Keyword mode is available"}.`;
    document.getElementById("algorithm-schema").replaceChildren(table(["Structured field", "Purpose"], [
      ["groups", `Up to ${info.ai.max_groups} concepts, each with a label and up to ${info.ai.max_terms} alternative phrases.`],
      ["supporting_groups", "Optional evidence: improves rank without excluding papers or changing counts."],
      ["exclusions", `Up to ${info.ai.max_exclusions} explicit phrases to exclude.`],
      ["needs_clarification", "If true, ask the user to clarify rather than run a search."],
      ["explanation", "A short explanation of the interpretation or a clarification question."]
    ]));
    document.getElementById("algorithm-example").replaceChildren(table(["Concept (AND)", "Alternative phrases (OR)"], info.example.groups.map(g => [g.label, g.terms.join(" · ")])));
    document.getElementById("algorithm-limit").textContent = `The app permits ${count(info.ai.weekly_limit)} outbound AI requests per calendar week, resetting Monday at 00:00 (${info.ai.timezone}). Retries and failed attempts count. Temporary cache hits, applying reviewed keywords, pagination and downloads do not. Only aggregate week/count totals persist across restarts, shared by servers using the same data root. It is not an account-wide or dollar spending limit.`;
    document.getElementById("algorithm-version").textContent = `AI prompt: ${info.ai.prompt_version} · Search rules: ${meta.rule_version} · Corpus SHA256: ${meta.corpus_hash}`;
    if (meta.semantic?.policy) {
      const p = meta.semantic.policy;
      const details = element("p", `Local semantic index: ${meta.semantic.enabled ? "enabled" : "not enabled on this server"} · ${p.model} · revision ${p.revision} · ${p.compute_dtype} computation / ${p.stored_dtype} vectors · core threshold ${p.core_cosine_threshold} · query threshold ${p.query_cosine_threshold}.`);
      document.getElementById("matching").append(details);
    }
    if (info.acceptance) document.getElementById("acceptance-availability").textContent = `Currently loaded: ${info.acceptance.public_snapshot_cohorts||0} official public-query snapshots; ${info.acceptance.verified_topic_cohorts} organizer-reconciled topic cohorts; ${info.acceptance.baseline_cohorts} published conference references. Acceptance snapshot SHA256: ${info.acceptance.sha256}`;
  } catch (error) {
    const target = document.getElementById("algorithm-error"); target.hidden = false; target.textContent = error.message;
  }
})();
