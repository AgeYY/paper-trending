// Real local keyword searches; no provider calls. Use an AI-disabled server.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
(async () => {
  const base = process.env.CONFERENCE_APP_URL || 'http://127.0.0.1:8766';
  const out = 'data/two-mode-browser'; fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({headless: true,
    ...(process.env.PLAYWRIGHT_EXECUTABLE ? {executablePath: process.env.PLAYWRIGHT_EXECUTABLE} : {})});
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({viewport: {width, height: 1000}});
      const errors = []; let aiCalls = 0;
      page.on('pageerror', e => errors.push(e.message));
      await page.route('**/api/interpret', route => { aiCalls++; return route.abort(); });
      await page.goto(base);
      await page.waitForFunction(() => !document.getElementById('search-button').disabled);
      assert.equal(await page.locator('#search-mode').inputValue(), 'keyword');
      assert.equal(await page.locator('#error').isVisible(), false);
      assert.equal(await page.locator('#history,.sidebar-heading').count(), 0);
      assert.match(await page.locator('footer').innerText(), /Created by Zeyuan Ye/);
      assert.equal(await page.locator('footer span a').count(), 0, 'Creator name has no personal-profile link');
      await page.locator('#prompt').fill('graph neural networks, drug discovery');
      const resultPromise = page.waitForResponse(r => r.url().endsWith('/api/search'));
      await page.locator('#search-button').click();
      const result = await (await resultPromise).json();
      assert.equal(result.config.search_mode, 'keyword');
      assert.equal(result.config.groups.length, 2);
      assert.equal(result.config.retrieval, undefined);
      assert.equal(result.config.interpretation, undefined);
      assert.ok(result.total > 0);
      await page.locator('#results').waitFor({state: 'visible'});
      await page.waitForFunction(() => !document.getElementById('search-button').disabled);
      assert.equal(await page.locator('#interpretation').isVisible(), false);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      await page.screenshot({path: `${out}/keyword-${width}.png`, fullPage: true});
      await page.locator('#search-mode').selectOption('ai');
      assert.equal(await page.locator('#search-button').isDisabled(), true);
      assert.match(await page.locator('#error').innerText(), /AI search is unavailable/);
      assert.equal(await page.locator('#search-hint').innerText(), 'AI converts your description into keywords.');
      await page.locator('#search-mode').selectOption('keyword');
      assert.equal(await page.locator('#search-button').isEnabled(), true);
      assert.equal(await page.locator('#error').isVisible(), false);
      await page.reload();
      await page.waitForFunction(() => !document.getElementById('search-button').disabled);
      assert.equal(await page.locator('#search-mode').inputValue(), 'keyword');
      await page.screenshot({path: `${out}/home-${width}.png`, fullPage: true});
      assert.equal(aiCalls, 0);
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Default keyword mode, real lexical search, AI-disabled switching, desktop/mobile passed; no AI calls.');
  } finally {await browser.close();}
})().catch(e => {console.error(e); process.exitCode = 1;});
