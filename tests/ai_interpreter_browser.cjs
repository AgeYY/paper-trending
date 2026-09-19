// Desktop/mobile AI review checks with mocked provider output; never a paid call.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");

(async () => {
  const base = process.env.CONFERENCE_APP_URL || "http://127.0.0.1:8766";
  const out = "data/ai-integration-browser/screenshots";
  fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE ? {executablePath: process.env.PLAYWRIGHT_EXECUTABLE} : {})});
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({viewport: {width, height: 900}});
      const errors = [];
      page.on("pageerror", e => errors.push(e.message));
      const metadata = await (await page.request.get(base + "/api/meta")).json();
      assert.equal(metadata.ai.enabled, false, "Use an AI-disabled server for this test");
      const result = await (await page.request.post(base + "/api/search", {
        headers: {"X-Local-Token": metadata.csrf_token}, data: {prompt: "graph neural networks"}
      })).json();
      const quota = {limit: 1000, used: 0, remaining: 1000, resets_at: "2026-09-21T00:00:00-05:00", timezone: "America/Chicago"};
      await page.route("**/api/meta", route => route.fulfill({json: {...metadata, ai: {enabled: true, model: "gpt-5.4-nano", weekly_quota: quota}}}));
      let interprets = 0, searches = 0, failure = false, clarify = false, submitted;
      await page.route("**/api/interpret", route => {
        interprets++;
        if (failure) return route.fulfill({status: 429, json: {error: "Weekly AI limit reached. Cached interpretations and saved results are still available.", weekly_quota: {...quota, used: 1000, remaining: 0}}});
        return route.fulfill({json: {id: "mock-id", model: "gpt-5.4-nano", prompt_version: "concepts-v1",
          prompt: route.request().postDataJSON().prompt, cached: false, weekly_quota: {...quota, used: 1, remaining: 999},
          groups: [{label: "Graph models", terms: ["graph neural networks", "graph transformers"]}],
          supporting_groups: [{label: "Molecular applications", terms: ["molecular", "drug discovery"]}],
          exclusions: [], explanation: clarify ? "Which graph application?" : "Focus on graph neural networks.", needs_clarification: clarify}});
      });
      await page.route("**/api/search", route => {
        searches++; submitted = route.request().postDataJSON();
        return route.fulfill({json: {...result, config: {...result.config, search_mode: "ai"}}});
      });
      await page.goto(base);
      await page.waitForFunction(() => !document.getElementById("search-button").disabled).catch(async e => {
        console.error("Boot errors:", errors, await page.locator("#error").innerText()); throw e;
      });
      assert.equal(await page.locator("#interpreter-mode,#ai-disclosure,#weekly-quota").count(), 0);
      assert.equal(await page.locator("#search-mode").inputValue(), "keyword");
      await page.locator("#search-mode").selectOption("ai");
      assert.equal(await page.locator(".ai-pitch").innerText(), "AI converts your description into keywords.");
      assert.equal(await page.locator(".sidebar-bottom,#privacy-notice").count(), 0);
      await page.screenshot({path: `${out}/ai-home-${width}.png`, fullPage: true});
      await page.locator("#prompt").fill("I want to study graph networks for molecular applications, not all neural networks.");
      await page.locator("#search-button").click();
      await page.locator("#interpretation").waitFor({state: "visible"});
      assert.equal(searches, 0, "Must review before searching");
      await page.locator("#concept-editor textarea").fill("graph neural networks\ngraph convolutional networks");
      assert.equal(await page.locator("#supporting-editor textarea").count(), 1);
      await page.locator("#supporting-editor button", {hasText: "Make core"}).first().click();
      assert.equal(await page.locator("#concept-editor textarea").count(), 2);
      await page.locator("#concept-editor .concept-row").last().getByRole("button", {name: "Make supporting"}).click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      await page.screenshot({path: `${out}/ai-review-${width}.png`, fullPage: true});
      await page.locator("#apply-concepts").click();
      await page.locator("#results").waitFor({state: "visible"});
      assert.equal(searches, 1);
      assert.deepEqual(submitted.groups[0].terms, ["graph neural networks", "graph convolutional networks"]);
      assert.deepEqual(submitted.supporting_groups[0].terms, ["molecular", "drug discovery"]);
      assert.equal(submitted.interpretation_id, "mock-id");
      assert.equal(interprets, 1, "Applying concepts must not call AI again");
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      failure = true;
      await page.locator("#prompt").fill("A new research description");
      await page.locator("#search-button").click();
      await page.locator("#error").waitFor({state: "visible"});
      assert.match(await page.locator("#error").innerText(), /Weekly AI limit/);
      assert.equal(searches, 1, "Never silently fall back after an AI error");
      failure = false; clarify = true;
      await page.locator("#search-button").click();
      await page.locator("#interpretation").waitFor({state: "visible"});
      await page.waitForFunction(() => !document.getElementById("search-button").disabled);
      assert.equal(await page.locator("#apply-concepts").isDisabled(), true);
      await page.unroute("**/api/meta");
      const before = interprets;
      await page.reload();
      await page.waitForFunction(() => !document.getElementById("search-button").disabled);
      assert.equal(await page.locator("#search-mode").inputValue(), "keyword");
      await page.locator("#search-mode").selectOption("ai");
      await page.locator("#error").waitFor({state: "visible"});
      assert.match(await page.locator("#error").innerText(), /AI search is unavailable/);
      assert.equal(await page.locator("#search-button").isDisabled(), true);
      await page.locator("#prompt").fill("No silent local fallback");
      await page.locator("#prompt").press("Control+Enter");
      assert.equal(interprets, before);
      assert.equal(searches, 1, "Disabled AI must not turn into a local search");
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log("AI review desktop/mobile browser checks passed (mocked provider; zero paid calls).");
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
