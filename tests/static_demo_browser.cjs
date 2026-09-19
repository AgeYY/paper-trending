// Usage: BASE_URL=http://127.0.0.1:8768/ PLAYWRIGHT_MODULE=... node tests/static_demo_browser.cjs
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_EXECUTABLE,args:['--disable-gpu']});
  try{
    for(const viewport of [{width:1440,height:1000},{width:390,height:844}]){
      const context=await browser.newContext({viewport,acceptDownloads:true});
      const page=await context.newPage(),errors=[],requests=[];
      page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
      await page.goto(process.env.BASE_URL||'http://127.0.0.1:8768/');
      await page.waitForFunction(()=>!document.getElementById('search').disabled);
      assert.equal(requests.some(u=>u.endsWith('.gz')),false,'No corpus download before a search');
      await page.locator('#prompt').fill('reinforcement learning, language model');
      await page.locator('#search').click();
      await page.locator('#results').waitFor({state:'visible',timeout:120000});
      assert.match(await page.locator('#paper-total').innerText(),/matching papers/);
      assert.equal(await page.locator('svg').count(),2);
      assert.equal(await page.locator('.paper-card').count(),20);
      const countBox=await page.locator('#counts-chart').boundingBox(),acceptBox=await page.locator('#acceptance-chart').boundingBox();
      if(viewport.width>1000)assert.equal(Math.round(countBox.y),Math.round(acceptBox.y));else assert.ok(acceptBox.y>countBox.y+countBox.height);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'No horizontal page overflow');
      for(const selector of ['#papers-csv','#counts-csv','#acceptance-csv','#metadata-json','[data-chart="counts-chart"][data-format="svg"]','[data-chart="acceptance-chart"][data-format="png"]']){
        const download=page.waitForEvent('download');await page.locator(selector).click();const d=await download;assert.ok(await d.path());assert.equal(await d.failure(),null);
      }
      await page.locator('#next').click();assert.match(await page.locator('#page').innerText(),/Page 2/);
      if(process.env.SCREENSHOT_DIR)await page.screenshot({path:`${process.env.SCREENSHOT_DIR}/demo-${viewport.width}.png`});
      await page.locator('#prompt').fill('zzzz-nonexistent-phrase');await page.locator('#search').click();
      await page.waitForFunction(()=>!document.getElementById('search').disabled);
      assert.match(await page.locator('#paper-total').innerText(),/^0 matching/);
      assert.equal(await page.evaluate(()=>localStorage.length+sessionStorage.length),0);
      assert.equal(requests.some(u=>u.includes('/api/')||u.includes('openai')||u.includes('reinforcement')),false);
      assert.deepEqual(errors,[]);
      await context.close();
    }
    console.log('Static demo desktop/mobile, charts, exports, pagination and privacy passed.');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
