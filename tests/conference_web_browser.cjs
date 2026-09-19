// Optional real-browser smoke test. No runtime Node dependency for the app.
// PLAYWRIGHT_MODULE may point to an existing Playwright installation.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

(async () => {
  const base = process.env.CONFERENCE_APP_URL || "http://127.0.0.1:8766";
  const out = process.env.CONFERENCE_APP_PREVIEW || "data/literature/conference_survey_20260916/web_app/compact_previews";
  fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE ? {executablePath: process.env.PLAYWRIGHT_EXECUTABLE} : {})});
  // Exercise the locale that caused the user's Chinese date screenshot.
  const page = await browser.newPage({viewport: {width: 1366, height: 768}, deviceScaleFactor: 1, locale: "zh-CN", timezoneId: "Asia/Shanghai"});
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error" && !message.text().includes("favicon")) errors.push(message.text()); });
  // Mock only the AI step. All counts/exports still use the real local corpus.
  const metadata = await (await page.request.get(base + "/api/meta")).json();
  assert.equal(metadata.ai.enabled, false, "Use an AI-disabled test server; never make paid calls");
  await page.route("**/api/meta", route => route.fulfill({json: {...metadata, ai: {...metadata.ai, enabled: true}}}));
  async function localResult(body) {
    const {prompt, years, venues, tracks} = body;
    return (await page.request.post(base + "/api/search", {headers: {"X-Local-Token": metadata.csrf_token},
      data: {prompt, years, venues, tracks}})).json();
  }
  await page.route("**/api/interpret", async route => {
    const prompt = route.request().postDataJSON().prompt, result = await localResult({prompt});
    await route.fulfill({json: {id: "mock-interpretation", prompt, groups: result.config.groups,
      exclusions: result.config.exclusions, needs_clarification: false, explanation: "Browser test concepts.",
      model: "gpt-5.4-nano", prompt_version: "test", cached: true}});
  });
  await page.route("**/api/search", async route => route.fulfill({json: await localResult(route.request().postDataJSON())}));
  async function submitSearch(button = page.locator("#search-button")) {
    await page.locator("#search-mode").selectOption("ai");
    await button.click();
    await page.locator("#interpretation").waitFor({state: "visible"});
    await page.locator("#apply-concepts").click();
    await page.waitForFunction(() => !document.getElementById("search-button").disabled);
  }
  async function assertSearchBrand() {
    assert.equal(await page.locator(".sidebar .brand").count(), 0);
    assert.equal(await page.locator(".search-brand .brand").count(), 1);
    const brand=await page.locator(".search-brand .brand").boundingBox();
    const form=await page.locator("#search-form").boundingBox();
    assert.ok(brand.y+brand.height<form.y, "Brand must be above the search box");
    assert.ok(Math.abs(brand.x+brand.width/2-form.x-form.width/2)<2, "Brand must be centered over search");
  }
  async function assertCompactResults() {
    await assertSearchBrand();
    const countBox=await page.locator(".chart-card").boundingBox();
    const rateBox=await page.locator(".acceptance-card").boundingBox();
    if(page.viewportSize().width>=1200){
      assert.ok(Math.abs(countBox.y-rateBox.y)<2, "Charts must share a row on desktop");
      assert.ok(rateBox.x>=countBox.x+countBox.width, "Acceptance chart must sit to the right");
      const countPlot=await page.locator("#chart svg").boundingBox();
      const ratePlot=await page.locator("#acceptance-chart svg").boundingBox();
      assert.ok(Math.abs(countPlot.y-ratePlot.y)<2, "Plotting areas must align");
      assert.ok(Math.abs(countPlot.height-ratePlot.height)<2, "Charts must have equal heights");
    }else{
      assert.ok(rateBox.y>=countBox.y+countBox.height, "Charts must stack on narrow screens");
    }
    await page.waitForFunction(() => {
      const r = document.querySelector(".chart-card").getBoundingClientRect();
      return r.top >= 0 && r.bottom <= window.innerHeight;
    });
    assert.equal(await page.locator(".concept-card,#concepts,#confirmed-count,#review-progress,.review-buttons,.review-status,#chart-mode,#review-filter,a[href='/report/']").count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    assert.equal(await page.locator(".hero,.library-stat,#library-size,#snapshot").count(), 0);
    assert.equal(await page.locator(".metrics,#candidate-count,#screened-count,#confirmed-count").count(), 0);
    const box = await page.locator("#prompt").boundingBox();
    assert.ok(box.height <= (page.viewportSize().width > 720 ? 44 : 62));
  }
  try {
    await page.goto(base);
    await page.waitForFunction(() => !document.getElementById("search-button").disabled);
    await assertSearchBrand();
    assert.equal(await page.locator(".hero,.library-stat,#library-size,#snapshot").count(), 0);
    await page.screenshot({path: path.join(out, "home-desktop.png"), fullPage: true});
    await submitSearch(page.locator('[data-example="3"]'));
    await page.waitForFunction(() => !document.getElementById("results").hidden);
    assert.ok(await page.locator(".paper").count() > 0);
    assert.match(await page.locator("#coverage-warning").innerText(), /NeurIPS 2026/);
    await assertCompactResults();
    await page.screenshot({path: path.join(out, "results-laptop-1366.png")});
    await page.setViewportSize({width: 1440, height: 900});
    await submitSearch();
    await page.waitForFunction(() => !document.getElementById("search-button").disabled);
    await assertCompactResults();
    await page.screenshot({path: path.join(out, "results-desktop.png")});
    assert.equal(await page.locator(".acceptance-card").count(), 1);
    assert.match(await page.locator(".acceptance-card h3").innerText(), /ICLR/);
    assert.equal(await page.evaluate(() => current.acceptance.rows.every(r => r.venue === "iclr")), true);
    assert.equal(await page.evaluate(() => new Set(current.counts.map(r=>r.venue)).size), 3);
    assert.match(await page.locator("#acceptance-notice").innerText(), /unavailable|public-pool/);
    assert.ok(await page.locator('#acceptance-chart rect[data-series="conference_rate"]').count() > 0);
    const topicYears=await page.evaluate(() => current.acceptance.rows.filter(r=>r.topic_rate!==null).length);
    assert.equal(await page.locator('#acceptance-chart rect[data-series="topic_rate"]').count(), topicYears);
    await page.locator(".acceptance-card").screenshot({path:path.join(out,"acceptance-desktop.png")});
    for (const ext of ["svg", "png"]) {
      const pendingDownload = page.waitForEvent("download");
      await page.locator("#acceptance-" + ext).click();
      await (await pendingDownload).saveAs(path.join(out, "acceptance-chart." + ext));
    }
    if (!(await page.locator("#next").isDisabled())) {
      await page.locator("#next").click();
      await page.waitForFunction(() => document.getElementById("page-info").textContent.startsWith("Page 2"));
      await page.locator("#previous").click();
      await page.waitForFunction(() => document.getElementById("page-info").textContent.startsWith("Page 1"));
    }
    const first = page.locator(".paper").first();
    await first.getByRole("button", {name: "Read full abstract ↓"}).click();
    assert.ok((await first.getAttribute("class")).includes("expanded"));
    const download = page.waitForEvent("download");
    await page.locator("#download-svg").click();
    await (await download).saveAs(path.join(out, "downloaded-chart.svg"));
    await page.locator("#prompt").fill("Graph neural networks for drug discovery");
    await submitSearch();
    await page.waitForFunction(() => document.getElementById("result-title").textContent.includes("Graph neural"));
    assert.ok(await page.locator(".paper").count() > 0);
    // With real official data, topic and baseline must use the same public pool.
    const publicRows=await page.evaluate(() => current.acceptance.rows.filter(r=>r.baseline_kind==="official_public_snapshot"));
    if(publicRows.length){
      assert.equal(publicRows.length,4);
      for(const row of publicRows){
        assert.ok(row.topic_submitted>0);
        assert.equal(row.conference_rate,100*row.conference_accepted/row.conference_submitted);
        assert.equal(row.delta_pp,row.topic_rate-row.conference_rate);
        assert.ok(row.official_submitted!==row.conference_submitted);
      }
      assert.equal(await page.evaluate(()=>current.acceptance.rows.find(r=>r.year===2026).conference_rate),null);
      await page.locator(".acceptance-card").screenshot({path:path.join(out,"acceptance-topic-desktop.png")});
    }
    await page.locator("#prompt").fill("zzzxxyy unheardtopic");
    await submitSearch();
    await page.waitForFunction(() => document.getElementById("paper-count").textContent.startsWith("0 automatic matches"));
    assert.match(await page.locator("#papers").innerText(), /does not prove/);
    await page.locator("#prompt").fill("Reinforcement learning for flow matching image generation");
    await submitSearch();
    await page.waitForFunction(() => document.getElementById("result-title").textContent.includes("flow matching"));
    await page.setViewportSize({width: 390, height: 844});
    await submitSearch();
    await page.waitForFunction(() => !document.getElementById("search-button").disabled);
    await assertCompactResults();
    await page.screenshot({path: path.join(out, "results-mobile.png")});
    await page.locator(".chart-card").screenshot({path: path.join(out, "chart-mobile.png")});
    await page.locator(".acceptance-card").screenshot({path:path.join(out,"acceptance-mobile.png")});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    // Synthetic in-memory rendering check only; never saved to the corpus/history.
    await page.evaluate(() => {
      for(const r of current.acceptance.rows)r.topic_rate=null;
      const row=current.acceptance.rows.find(r=>r.conference_rate!==null);
      row.topic_rate=50; row.topic_accepted=2; row.topic_submitted=4;
      row.topic_status="complete"; row.topic_reason=""; row.delta_pp=50-row.conference_rate;
      drawAcceptance();
    });
    assert.equal(await page.locator('#acceptance-chart rect[data-series="topic_rate"][data-rate="50"]').count(), 1);
    await page.locator(".acceptance-details summary").click();
    assert.match(await page.locator("#acceptance-table").innerText(), /50.00%/);
    await page.reload();
    await page.waitForFunction(() => !document.getElementById("search-button").disabled);
    assert.equal(await page.locator("#history,.sidebar-heading").count(), 0);
    assert.match(await page.locator("footer").innerText(), /Created by Zeyuan Ye/);
    await page.setViewportSize({width: 1366, height: 768});
    await page.locator("#algorithm-link").click();
    await page.waitForURL("**/algorithm");
    await page.waitForFunction(() => document.getElementById("algorithm-example").querySelectorAll("tr").length === 4);
    assert.equal(await page.locator(".algorithm-section").count(), 10);
    assert.equal(await page.locator("#algorithm-vocabulary").count(), 0);
    assert.match(await page.locator("#algorithm-ai").innerText(), /gpt-5.4-nano/);
    assert.match(await page.locator("#algorithm-limit").innerText(), /1,000 outbound AI requests/);
    assert.equal(await page.locator("#algorithm-schema tr").count(), 6);
    assert.match(await page.locator("#example").innerText(), /not a live AI response/);
    assert.equal(await page.locator("#algorithm-error").isVisible(), false);
    assert.match(await page.locator("#ranking").innerText(), /60 \+ keyword rank/);
    assert.match(await page.locator("#counts").innerText(), /No|no approval/);
    assert.match(await page.locator("#corpus-summary").innerText(), /44,401/);
    assert.doesNotMatch(await page.locator("#corpus-summary").innerText(), /[\u3400-\u9fff]/);
    await page.screenshot({path: path.join(out, "algorithm-desktop.png")});
    await page.locator("#example").screenshot({path: path.join(out, "algorithm-example.png")});
    await page.setViewportSize({width: 390, height: 844});
    await page.locator("#prompt").scrollIntoViewIfNeeded();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({path: path.join(out, "algorithm-ai-mobile.png")});
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    await page.locator(".brand").click();
    await page.waitForURL(base + "/");
    await page.waitForFunction(() => !document.getElementById("search-button").disabled);
    assert.deepEqual(errors, []);
    console.log("Browser checks passed: desktop/mobile layout, algorithm documentation, pagination, downloads, no history, creator credit, empty state.");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
