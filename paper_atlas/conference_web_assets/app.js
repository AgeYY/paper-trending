"use strict";
const $ = (id) => document.getElementById(id);
const number = (x) => Number(x || 0).toLocaleString("en-US");
const englishDate = (date) => new Intl.DateTimeFormat("en-US", {year: "numeric", month: "short", day: "numeric", timeZone: "UTC"}).format(new Date(date));
let meta = null, current = null, pending = false, pageRequest = 0;
let interpretation = null;
const keywordExamples = ["autoregressive, language model, reinforcement learning", "discrete diffusion, language model, reinforcement learning", "continuous diffusion, language model, reinforcement learning", "flow matching, image generation, reinforcement learning"];
function searchMode() { return $("search-mode").value; }
function updateMode() {
  const ai = searchMode() === "ai";
  interpretation = null; $("interpretation").hidden = true; error();
  $("prompt-label").textContent = ai ? "Describe your research topic" : "Enter keywords separated by commas";
  $("prompt").placeholder = ai ? "e.g. Reinforcement learning for flow matching image generation" : "e.g. reinforcement learning, language model";
  $("search-hint").textContent = ai ? "AI converts your description into keywords." : "Separate phrases with commas. All phrases must match. No AI is used.";
  loading(pending);
  if (ai && meta && !meta.ai?.enabled) error("AI search is unavailable on this server. Keyword mode is still available.");
}

function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}
function error(message = "") { $("error").textContent = message; $("error").hidden = !message; }
async function api(path, body) {
  const options = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-Local-Token": meta.csrf_token}, body: JSON.stringify(body)};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed. Please try again.");
  return data;
}
function selected(name) { return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(x => x.value); }
function setSelected(name, values) { document.querySelectorAll(`input[name="${name}"]`).forEach(x => { x.checked = values.map(String).includes(x.value); }); }
function loading(on) {
  pending = on; $("search-button").disabled = on || !meta || (searchMode() === "ai" && !meta.ai?.enabled);
  $("search-mode").disabled = on;
  $("apply-concepts").disabled = on;
  $("prompt").disabled = on;
  $("loading").hidden = !on;
  $("search-button").textContent = on ? "Working…" : "Explore topic ↗";
}
async function search(reviewed = false) {
  if (pending || !meta) return;
  error();
  const ai = searchMode() === "ai";
  if (ai && !meta.ai?.enabled) return error("AI search is unavailable on this server. Keyword mode is still available.");
  const prompt = $("prompt").value.trim();
  if (prompt.length < (ai ? 3 : 1)) return error(ai ? "Please describe a topic in at least three characters." : "Please enter at least one keyword.");
  const request = {prompt, search_mode: searchMode(), years: selected("year").map(Number), venues: selected("venue"), tracks: selected("track")};
  if (!request.years.length || !request.venues.length || !request.tracks.length) return error("Choose at least one year, conference and paper track.");
  loading(true); ++pageRequest;
  try {
    if (ai && !reviewed) {
      $("loading").textContent = "Interpreting your description with OpenAI…";
      interpretation = await api("/api/interpret", {prompt});
      showInterpretation();
      return;
    }
    if (ai && reviewed) {
      if (!interpretation || interpretation.prompt !== prompt) throw new Error("The description changed. Interpret it again first.");
      request.interpretation_id = interpretation.id;
      request.groups = [...$("concept-editor").children].map(row => ({label: row.querySelector("input").value,
        terms: row.querySelector("textarea").value.split("\n").map(x => x.trim()).filter(Boolean)}));
      if (meta.rule_version === "core-support-hybrid-v2") {
        request.supporting_groups = [...$("supporting-editor").children].map(row => ({label: row.querySelector("input").value,
          terms: row.querySelector("textarea").value.split("\n").map(x => x.trim()).filter(Boolean)}));
      }
      request.exclusions = $("exclusions-editor").value.split("\n").map(x => x.trim()).filter(Boolean);
    }
    $("loading").textContent = "Searching the local corpus…";
    const data = await api("/api/search", request);
    $("interpretation").hidden = true;
    $("sort").value = "relevance";
    render(data, true);
  } catch (e) { error(e.message); }
  finally { loading(false); $("apply-concepts").disabled = !!interpretation?.needs_clarification; }
}
function showInterpretation() {
  $("interpretation").hidden = false;
  $("interpretation-explanation").textContent = [interpretation.explanation,
    ...(interpretation.adjustments || []).filter(note => !interpretation.explanation.includes(note))].join(" ");
  $("concept-editor").replaceChildren();
  $("supporting-editor").replaceChildren();
  for (const [group, destination] of [...interpretation.groups.map(g => [g, "concept-editor"]), ...(interpretation.supporting_groups || []).map(g => [g, "supporting-editor"])]) {
    const row = node("div", "concept-row"), label = node("input"), terms = node("textarea"), remove = node("button", "secondary", "Remove concept");
    label.value = group.label; label.maxLength = 120; label.setAttribute("aria-label", "Concept label");
    terms.value = group.terms.join("\n"); terms.rows = Math.min(5, group.terms.length); terms.setAttribute("aria-label", `Alternatives for ${group.label}`);
    const move = node("button", "secondary", destination === "concept-editor" ? "Make supporting" : "Make core");
    move.type = "button";
    move.onclick = () => {
      const optional = row.parentElement.id === "concept-editor";
      $(optional ? "supporting-editor" : "concept-editor").append(row);
      move.textContent = optional ? "Make core" : "Make supporting";
    };
    remove.type = "button"; remove.onclick = () => row.remove(); row.append(label, terms, move, remove); $(destination).append(row);
  }
  $("exclusions-editor").value = interpretation.exclusions.join("\n");
  $("interpretation-source").textContent = `${interpretation.model} · ${interpretation.prompt_version} · ${interpretation.cached ? "cached interpretation; no new API call" : "new interpretation"}. Structured format does not guarantee correct meaning.`;
  $("interpretation").scrollIntoView({block: "nearest"});
}
async function loadPage(page = 1, searchId = current?.id, restore = false) {
  if (!searchId) return;
  error();
  const serial = ++pageRequest;
  try {
    const sort = restore ? "relevance" : $("sort").value;
    const data = await api(`/api/search/${searchId}?page=${page}&sort=${sort}`);
    if (serial === pageRequest) render(data, restore);
  } catch (e) { if (serial === pageRequest) error(e.message); }
}
function render(data, restore) {
  $("interpretation").hidden = true;
  current = data; $("welcome").hidden = true; $("results").hidden = false;
  document.body.classList.add("has-results");
  if (restore) {
    $("search-mode").value = data.config.search_mode || (data.config.interpretation ? "ai" : "keyword");
    updateMode();
    $("prompt").value = data.config.prompt;
    setSelected("year", data.config.years); setSelected("venue", data.config.venues); setSelected("track", data.config.tracks);
    document.querySelector(".scope").open = false;
  }
  $("sort").value = data.sort;
  $("result-title").textContent = data.config.prompt;
  $("scope-label").textContent = `${data.config.years.length} years · ${data.config.venues.length} conferences`;
  $("timing").textContent = `${number(data.elapsed_ms)} ms · local search`;
  $("result-concepts").replaceChildren(node("p", "", data.method));
  for (const group of data.config.groups) $("result-concepts").append(node("p", "", `Core — ${group.label}: ${group.terms.join(" OR ")}`));
  for (const group of data.config.supporting_groups || []) $("result-concepts").append(node("p", "", `Supporting only — ${group.label}: ${group.terms.join(" OR ")}`));
  if (data.config.retrieval) $("result-concepts").append(node("p", "muted", `Local embeddings: ${data.config.retrieval.model}. Semantic expansion requires cosine ≥ ${data.config.retrieval.query_cosine_threshold} for the core query and ≥ ${data.config.retrieval.core_cosine_threshold} for each core without keyword evidence. These are experimental thresholds, not probabilities.`));
  if (data.config.exclusions.length) $("result-concepts").append(node("p", "", `Excluding: ${data.config.exclusions.join(", ")}`));
  for (const warning of data.warnings || []) $("result-concepts").append(node("p", "muted", warning));
  const incomplete = data.coverage.filter(x => x.status !== "complete");
  $("coverage-warning").hidden = !incomplete.length;
  $("coverage-warning").textContent = `Incomplete coverage: ${incomplete.map(x => `${meta.venues[x.venue]} ${x.year} (${x.status})`).join(", ")}. Hatched years count available papers only; missing data is not zero research activity.`;
  drawChart();
  drawAcceptance();
  const table = node("table");
  const header = node("tr"); ["Conference", "Year", "Status", "Papers", "Candidates", "Per 1,000"].forEach(x => header.append(node("th", "", x))); table.append(header);
  for (const row of data.counts) {
    const tr = node("tr"); [meta.venues[row.venue], row.year, row.status, number(row.papers), row.candidates === null ? "Unavailable" : number(row.candidates), row.per_1000 === null ? "—" : row.per_1000.toFixed(2)].forEach(x => tr.append(node("td", "", x))); table.append(tr);
  }
  $("coverage-table").replaceChildren(table);
  $("download-csv").href = `/api/export/${data.id}.csv`; $("download-counts").href = `/api/export/${data.id}.counts`; $("download-json").href = `/api/export/${data.id}.json`;
  $("paper-count").textContent = `${number(data.total)} ${data.config.retrieval ? "hybrid candidates" : "automatic matches"} · counts use all qualifying papers, not top-k`;
  if (data.retrieval_summary) $("paper-count").textContent += ` · ${number(data.retrieval_summary.keyword_core)} keyword-core + ${number(data.retrieval_summary.semantic_expanded)} semantic-expanded`;
  $("papers").replaceChildren(...data.papers.map(paperCard));
  if (!data.papers.length) $("papers").append(node("div", "empty", "No keyword matches. Try a shorter topic description or different terminology. This does not prove that no relevant papers exist."));
  $("page-info").textContent = `Page ${data.page} of ${data.pages}`;
  $("previous").disabled = data.page <= 1; $("next").disabled = data.page >= data.pages;
  if (restore) requestAnimationFrame(() => {
    const chart = document.querySelector(".chart-card");
    const rect = chart.getBoundingClientRect();
    if (rect.bottom > window.innerHeight - 12 || rect.top < 12) chart.scrollIntoView({block: "nearest", behavior: "instant"});
  });
}
function paperCard(paper) {
  const card = node("article", "paper");
  const top = node("div", "paper-top");
  top.append(node("span", "venue-tag", `${meta.venues[paper.venue]} ${paper.year}`), node("span", "", paper.track));
  const title = node("h3"); const link = node("a", "", paper.title); link.href = paper.url; link.target = "_blank"; link.rel = "noopener noreferrer"; title.append(link);
  const authorText = paper.authors.slice(0, 5).join(", ") + (paper.authors.length > 5 ? ` +${paper.authors.length - 5} more` : "");
  const abstract = node("p", "abstract", paper.abstract || "Abstract unavailable.");
  const expand = node("button", "text-button", "Read full abstract ↓");
  expand.onclick = () => { card.classList.toggle("expanded"); expand.textContent = card.classList.contains("expanded") ? "Collapse abstract ↑" : "Read full abstract ↓"; };
  const evidence = node("div", "evidence");
  if (paper.match_kind) evidence.append(node("span", "", paper.match_kind === "semantic-expanded" ? "Semantic-expanded candidate" : "Core keyword match"));
  for (const item of paper.evidence) {
    const prefix = item.role === "supporting" ? "Supporting only · " : "";
    const detail = item.source === "semantic" ? `semantic cosine ${Number(item.cosine).toFixed(2)}` : item.terms.slice(0, 2).join(", ");
    const tag = node("span", "", `${prefix}${item.concept}: ${detail}`);
    tag.title = item.source === "semantic" ? "Embedding similarity, not a relevance probability or verified model architecture." : item.terms.join(" | ") + (item.in_title ? " · title match" : " · abstract match");
    evidence.append(tag);
  }
  const actions = node("div", "paper-actions"); const open = node("a", "", "Open paper ↗"); open.href = paper.url; open.target = "_blank"; open.rel = "noopener noreferrer";
  actions.append(open); card.append(top, title, node("p", "authors", authorText), abstract, expand, evidence, actions); return card;
}
function svgElement(tag, attributes = {}, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== undefined) element.textContent = text;
  return element;
}
function drawAcceptance() {
  if (!current) return;
  const data = current.acceptance;
  if (!data) { $("acceptance-notice").textContent = "Acceptance data not loaded. Restart the local server to enable this chart."; return; }
  const years = current.config.years, venues = current.config.venues.filter(v=>v==="iclr");
  const colors = {iclr: "#214b40", icml: "#72947d", neurips: "#8b7049"};
  const width = Math.max(360, years.length * 68, $("acceptance-chart").clientWidth), height = 310;
  const left = 48, right = 16, top = 42, bottom = 46, w = width-left-right, h = height-top-bottom;
  $("acceptance-chart").style.minWidth = Math.max(360, years.length * 68) + "px";
  const svg = svgElement("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Topic versus overall acceptance rates by conference and year; NA means unavailable"});
  svg.append(svgElement("title", {}, "Topic and conference acceptance rates (%)"));
  ["Topic", "ICLR overall"].forEach((label,i) => { const x=left+i*110; svg.append(svgElement("rect", {x,y:9,width:12,height:12,fill:i===0?colors.iclr:"white",stroke:colors.iclr,"stroke-width":1.7}),svgElement("text",{x:x+19,y:19,"font-size":12,fill:"#656d67"},label)); });
  for (let value=0; value<=100; value+=25) {
    const y=top+h-value*h/100;
    svg.append(svgElement("line",{x1:left,x2:width-right,y1:y,y2:y,stroke:"#e4e8df"}),svgElement("text",{x:left-8,y:y+4,"text-anchor":"end","font-size":11,fill:"#656d67"},`${value}%`));
  }
  for (const [i,year] of years.entries()) {
    const spacing=w/years.length, bw=Math.min(32,spacing*.78/(Math.max(1,venues.length)*2));
    for (const [j,venue] of venues.entries()) {
      const row=data.rows.find(r=>r.year===year && r.venue===venue);
      if (!row) continue;
      for (const [k,metric] of ["topic_rate","conference_rate"].entries()) {
        const value=row[metric], x=left+spacing*(i+.5)-bw+bw*(2*j+k);
        const label=k===0?"Topic":"Overall";
        const a=k===0?row.topic_accepted:row.conference_accepted, n=k===0?row.topic_submitted:row.conference_submitted;
        const hint=`${meta.venues[venue]} ${year} · ${label}: ${value===null?"Unavailable":value.toFixed(2)+"%"}\n${n===null?"Exact denominator unavailable":number(a)+" / "+number(n)}\n${k===0?row.topic_reason:row.baseline_reason}\n${row.population}`;
        const mark=value===null?svgElement("text",{x:x+bw*.42,y:top+h-6,"text-anchor":"middle",fill:colors[venue],"font-size":9},"NA"):
          svgElement("rect",{x,y:top+h-value*h/100,width:bw*.84,height:Math.max(1,value*h/100),fill:k===0?colors[venue]:"white",stroke:colors[venue],"stroke-width":1.7,"data-series":metric,"data-rate":value});
        mark.append(svgElement("title",{},hint));svg.append(mark);
        if(value===0)svg.append(svgElement("text",{x:x+bw*.42,y:top+h-6,"text-anchor":"middle","font-size":9},"0%"));
      }
    }
    svg.append(svgElement("text",{x:left+spacing*(i+.5),y:top+h+25,"text-anchor":"middle","font-size":12,fill:"#656d67"},year));
  }
  svg.append(svgElement("line",{x1:left,x2:left+w,y1:top+h,y2:top+h,stroke:"#62755f","stroke-width":1.8}));
  $("acceptance-chart").replaceChildren(svg);
  const available=data.rows.filter(r=>r.topic_rate!==null).length;
  $("acceptance-notice").textContent=available ? `${available} conference-years have verified topic rates. Hover over bars for counts; remaining NA slots have missing data or no topic submissions. Small topic counts can produce unstable rates.` :
    "Topic rates are unavailable for this selection: no matching, complete submission cohort is loaded. Outline bars show sourced conference-wide baselines only—not topic estimates. See counts and sources below.";
  if (!available && data.rows.some(r=>r.topic_status==="no_matches")) $("acceptance-notice").textContent = "No matching topic submissions in the verified cohorts (0/0 is undefined). Other NA slots may have unavailable submission data; see counts and sources below.";
  if (data.rows.some(r=>r.baseline_kind==="official_public_snapshot")) {
    const missing=data.rows.filter(r=>r.baseline_kind==="unavailable").map(r=>r.year);
    $("acceptance-notice").textContent = `Official OpenReview public-pool rates: ${available} years have matching topic submissions. Both bars use the same public pool, including withdrawals and desk rejections—not the organizer’s reported cohort. Published rates are separate references below. ${missing.length?`${missing.join(", ")}: incomplete or unavailable data; no rate shown. `:""}Small topic samples can give unstable rates.`;
  }
  if (current.config.tracks.length!==1 || current.config.tracks[0]!=="research") $("acceptance-notice").textContent = "Acceptance comparisons currently support the main research track only. Select Research alone in Search scope to view its baselines.";
  if (!venues.length) $("acceptance-notice").textContent = "Acceptance rates are available for ICLR only. Select ICLR in Search scope. Paper counts still support all three conferences.";
  const table=node("table"), header=node("tr");
  ["Conference / year","Topic accepted / submitted","Topic rate","Same-pool overall accepted / submitted","Overall rate","Difference","Published reference (not plotted)","Denominator & sources"].forEach(x=>header.append(node("th","",x)));table.append(header);
  const percent=x=>x===null?"NA":x.toFixed(2)+"%";
  for(const row of data.rows){
    const tr=node("tr");
    const values=[`${meta.venues[row.venue]} ${row.year}`,row.topic_submitted===null?"Unavailable":`${number(row.topic_accepted)} / ${number(row.topic_submitted)}`,percent(row.topic_rate),row.conference_submitted===null?"Not provided":`${number(row.conference_accepted)} / ${number(row.conference_submitted)}`,percent(row.conference_rate)+(row.baseline_kind==="reported_rounded"?" (rounded)":""),row.delta_pp===null?"—":`${row.delta_pp>=0?"+":""}${row.delta_pp.toFixed(2)} pp`];
    values.forEach(x=>tr.append(node("td","",x)));
    const official=node("td","",percent(row.official_conference_rate??null));
    if(row.official_submitted!=null)official.append(node("br"),document.createTextNode(`${number(row.official_accepted)} / ${number(row.official_submitted)}`));
    if(row.official_population)official.title=row.official_population;
    if(row.official_source && /^https?:\/\//.test(row.official_source)){const a=node("a","","Published source ↗");a.href=row.official_source;a.target="_blank";a.rel="noopener noreferrer";official.append(node("br"),a);}
    tr.append(official);
    const td=node("td","",row.population);
    for(const [url,label] of [[row.baseline_source,"Baseline source ↗"],...(row.additional_sources||[]).map(url=>[url,"Additional baseline source ↗"]),[row.topic_source?.source,"Submission source ↗"]]){
      if(url && /^https?:\/\//.test(url)){const a=node("a","",label);a.href=url;a.target="_blank";a.rel="noopener noreferrer";td.append(node("br"),a);}
    }
    if(row.topic_reason)td.append(node("p","rate-reason",row.topic_reason));
    if(row.baseline_reason)td.append(node("p","rate-reason",row.baseline_reason));
    tr.append(td);table.append(tr);
  }
  $("acceptance-table").replaceChildren(table);
  for(const ext of ["svg","png","pdf"])$("acceptance-"+ext).href=`/api/export/${current.id}.${ext}?chart=acceptance`;
  $("acceptance-csv").href=`/api/export/${current.id}.acceptance`;
}
function drawChart() {
  if (!current) return;
  const metric = "candidates";
  const width = Math.max(280, $("chart").clientWidth || 570), height = 310, left = 48, right = 16, top = 42, bottom = 46;
  const w = width - left - right, h = height - top - bottom;
  const years = current.config.years, venues = current.config.venues;
  const colors = {iclr: "#214b40", icml: "#72947d", neurips: "#b9cdb3"};
  const totals = years.map(y => current.counts.filter(r => r.year === y).reduce((sum, r) => sum + (r[metric] || 0), 0));
  const rawMax = Math.max(1, ...totals);
  const exponent = 10 ** Math.floor(Math.log10(rawMax / 4));
  const step = Math.max(1, Math.ceil(rawMax / 4 / exponent) * exponent);
  const maximum = Math.ceil(rawMax / step) * step || 1;
  const svg = svgElement("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Automatic paper matches by conference year"});
  svg.append(svgElement("title", {}, "Conference paper counts"));
  const defs = svgElement("defs"); const pattern = svgElement("pattern", {id: "partial-stripe", width: 8, height: 8, patternUnits: "userSpaceOnUse", patternTransform: "rotate(40)"}); pattern.append(svgElement("line", {x1: 0, y1: 0, x2: 0, y2: 8, stroke: "#354e39", "stroke-width": 2, opacity: .4})); defs.append(pattern); svg.append(defs);
  venues.forEach((v, i) => { const x = left + i * Math.min(110, w / venues.length); svg.append(svgElement("rect", {x, y: 8, width: 11, height: 11, fill: colors[v], rx: 2}), svgElement("text", {x: x + 18, y: 18, fill: "#656d67", "font-size": 12}, meta.venues[v])); });
  for (let value = 0; value <= maximum; value += step) {
    const y = top + h - value / maximum * h;
    svg.append(svgElement("line", {x1: left, y1: y, x2: left + w, y2: y, stroke: "#e4e8df", "stroke-width": 1}), svgElement("text", {x: left - 12, y: y + 4, "text-anchor": "end", fill: "#656d67", "font-size": 12}, number(value)));
  }
  years.forEach((year, i) => {
    const spacing = w / years.length, barWidth = Math.min(49, spacing * .55), x = left + spacing * (i + .5) - barWidth / 2;
    const rows = current.counts.filter(r => r.year === year); let cumulative = 0;
    for (const venue of venues) {
      const row = rows.find(r => r.venue === venue); const value = row?.[metric] || 0;
      const rectangle = svgElement("rect", {x, y: top + h - (cumulative + value) / maximum * h, width: barWidth, height: value / maximum * h, fill: colors[venue]});
      rectangle.append(svgElement("title", {}, `${meta.venues[venue]} ${year}: ${row?.[metric] === null ? "unavailable" : value}`)); svg.append(rectangle); cumulative += value;
    }
    const partial = rows.some(r => r.status !== "complete"), unavailable = rows.every(r => ["failed", "unavailable"].includes(r.status));
    if (partial) svg.append(svgElement("rect", {x, y: top + h - cumulative / maximum * h, width: barWidth, height: cumulative / maximum * h, fill: "url(#partial-stripe)"}));
    svg.append(svgElement("text", {x: x + barWidth / 2, y: Math.max(top - 5, top + h - cumulative / maximum * h - 8), "text-anchor": "middle", fill: "#253a35", "font-size": 13}, unavailable ? "NA" : number(cumulative)), svgElement("text", {x: x + barWidth / 2, y: top + h + 24, "text-anchor": "middle", fill: "#656d67", "font-size": 12}, `${year}${partial ? "*" : ""}`));
  });
  svg.append(svgElement("line", {x1: left, y1: top + h, x2: left + w, y2: top + h, stroke: "#62755f", "stroke-width": 1.8}), svgElement("line", {x1: left, y1: top, x2: left, y2: top + h, stroke: "#62755f", "stroke-width": 1.8}));
  $("chart").replaceChildren(svg);
  $("chart-caption").textContent = (current.config.retrieval ? "Hybrid candidates (keywords + local embeddings), not verified relevance. All qualifying papers counted; no top-k cutoff." : "Automatic keyword matches, not verified semantic relevance.") + (current.incomplete ? " * Partial conference coverage." : "");
  for (const ext of ["svg", "png", "pdf"]) $("download-" + ext).href = `/api/export/${current.id}.${ext}`;
}
$("search-form").onsubmit = e => { e.preventDefault(); search(); };
$("apply-concepts").onclick = () => search(true);
$("search-mode").onchange = updateMode;
$("prompt").addEventListener("input", () => { interpretation = null; $("interpretation").hidden = true; });
$("prompt").onkeydown = e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); search(); } };
document.querySelectorAll("[data-example]").forEach(button => { button.onclick = () => { if (!meta || pending) return; $("prompt").value = (searchMode() === "ai" ? meta.examples : keywordExamples)[Number(button.dataset.example)]; search(); }; });
$("previous").onclick = () => loadPage(current.page - 1); $("next").onclick = () => loadPage(current.page + 1);
$("sort").onchange = () => loadPage(1);
$("new-search").onclick = () => { if (pending) return; ++pageRequest; current = null; interpretation = null; $("interpretation").hidden = true; document.body.classList.remove("has-results"); $("results").hidden = true; $("welcome").hidden = false; $("prompt").value = ""; error(); window.scrollTo({top: 0, behavior: "instant"}); $("prompt").focus(); };
window.addEventListener("resize", () => { drawChart(); drawAcceptance(); });
async function boot() {
  try {
    meta = await api("/api/meta");
    for (const [name, choices, container] of [["venue", Object.entries(meta.venues), "venues"], ["year", meta.years.map(y => [y, y]), "years"]]) {
      for (const [value, label] of choices) { const wrap = node("label"); const box = node("input"); box.type = "checkbox"; box.name = name; box.value = value; box.checked = true; wrap.append(box, document.createTextNode(" " + label)); $(container).append(wrap); }
    }
    updateMode();
  } catch (e) { error("Could not load the local corpus: " + e.message); }
}
boot();
