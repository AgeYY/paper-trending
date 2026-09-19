const $ = id => document.getElementById(id);
const number = n => n == null ? 'NA' : n.toLocaleString();
const colors = {iclr:'#214b40',icml:'#72947d',neurips:'#b9cdb3'};
let manifest, result, page = 1, request = 0, worker;
const element = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
const svgNode = (tag, attrs={}, text) => { const e = document.createElementNS('http://www.w3.org/2000/svg',tag); for (const [k,v] of Object.entries(attrs)) e.setAttribute(k,v); if (text !== undefined) e.textContent=text; return e; };
function link(url, text) { const a=element('a',text); if (/^https?:\/\//.test(url || '')) {a.href=url;a.target='_blank';a.rel='noopener noreferrer';} return a; }
function selected(name) {return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(x=>name==='years'?Number(x.value):x.value);}
function controls(name, values) { for (const [value,label] of values) {const l=element('label'), i=element('input');i.type='checkbox';i.name=name;i.value=value;i.checked=name!=='tracks'||value==='research';l.append(i,document.createTextNode(' '+label));$(name).append(l);} }
function fail(message) {$('error').hidden=false;$('error').textContent=message;$('status').textContent='';$('search').disabled=false;}
function save(name, bytes, type) {const url=URL.createObjectURL(new Blob([bytes],{type}));const a=element('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csv(name, rows, fields) { const quote=v=>{let s=v==null?'':String(v);if (/^\s*[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';};save(name,'\ufeff'+[fields,...rows.map(r=>fields.map(k=>r[k]))].map(r=>r.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8'); }
function chart(target, acceptance=false) {
  const width=570,height=320,left=45,top=40,w=510,h=225;
  const svg=svgNode('svg',{xmlns:'http://www.w3.org/2000/svg',viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':acceptance?'ICLR topic and overall acceptance rates':'Paper counts by year'});
  svg.append(svgNode('rect',{width,height,fill:'white'}),svgNode('title',{},acceptance?'ICLR acceptance rates':'Accepted paper matches'));
  const years=result.query.years,space=w/years.length;
  const max=acceptance?100:Math.max(1,...years.map(y=>result.counts.filter(r=>r.year===y).reduce((n,r)=>n+(r.candidates||0),0)))*1.15;
  for(let i=0;i<=4;i++){const value=max*i/4,y=top+h-h*i/4;svg.append(svgNode('line',{x1:left,x2:left+w,y1:y,y2:y,stroke:'#d9e2d5'}),svgNode('text',{x:left-7,y:y+4,'text-anchor':'end','font-size':11,fill:'#214b40'},acceptance?`${value}%`:Math.round(value)));}
  const legends=acceptance?[['Topic','#214b40'],['Overall','white']]:result.query.venues.map(v=>[manifest.venues[v],colors[v]]);
  legends.forEach(([label,color],i)=>svg.append(svgNode('rect',{x:left+i*125,y:9,width:11,height:11,fill:color,stroke:'#214b40'}),svgNode('text',{x:left+17+i*125,y:19,'font-size':12,fill:'#214b40'},label)));
  years.forEach((year,i)=>{
    const rows=result.counts.filter(r=>r.year===year),partial=rows.some(r=>r.status!=='complete');
    const center=left+space*(i+.5),barWidth=Math.min(52,space*.6);
    svg.append(svgNode('text',{x:center,y:top+h+23,'text-anchor':'middle','font-size':12,fill:'#214b40'},`${year}${!acceptance&&partial?'*':''}`));
    if(acceptance){
      const row=result.acceptance.find(r=>r.year===year);
      ['topic_rate','conference_rate'].forEach((key,j)=>{
        const val=row?.[key], x=center+(j-1)*barWidth*.6;
        if(val==null){svg.append(svgNode('text',{x:x+barWidth*.23,y:top+h-6,'font-size':10,'text-anchor':'middle',fill:'#214b40'},'NA'));return;}
        const rect=svgNode('rect',{x,y:top+h-h*val/100,width:barWidth*.46,height:h*val/100,fill:j?'white':'#214b40',stroke:'#214b40'});
        rect.append(svgNode('title',{},`${year}: ${val.toFixed(2)}% (${j?row.accepted:row.topic_accepted}/${j?row.submitted:row.topic_submitted})`));svg.append(rect);
        if(val===0)svg.append(svgNode('text',{x:x+barWidth*.23,y:top+h-6,'font-size':10,'text-anchor':'middle',fill:'#214b40'},'0%'));
      });
    }else{
      let total=0;
      for(const venue of result.query.venues){const row=rows.find(r=>r.venue===venue),val=row?.candidates||0;const rect=svgNode('rect',{x:center-barWidth/2,y:top+h-h*(total+val)/max,width:barWidth,height:h*val/max,fill:colors[venue]});rect.append(svgNode('title',{},`${manifest.venues[venue]} ${year}: ${row?.candidates??'unavailable'} (${row?.status})`));svg.append(rect);total+=val;}
      svg.append(svgNode('text',{x:center,y:top+h-h*total/max-7,'text-anchor':'middle','font-size':12,fill:'#214b40'},rows.every(r=>r.candidates==null)?'NA':total));
    }
  });
  svg.append(svgNode('text',{x:width/2,y:height-8,'text-anchor':'middle','font-size':10,fill:'#66736a'},acceptance?'Same public pool · NA is unavailable, not zero':'Keyword matches · * incomplete coverage'));
  $(target).replaceChildren(svg);
}
function renderPapers(){
  const rows=[...result.matches],sort=$('sort').value;
  if(sort!=='relevance')rows.sort((a,b)=>(sort==='newest'?b.year-a.year:a.year-b.year));
  const pages=Math.max(1,Math.ceil(rows.length/20));page=Math.min(page,pages);
  $('papers').replaceChildren();
  for(const p of rows.slice((page-1)*20,page*20)){
    const card=element('article');card.className='paper-card';const title=element('h3');title.append(link(p.url,p.title));
    const meta=element('p',`${manifest.venues[p.venue]} ${p.year} · ${p.track} · ${(p.authors||[]).join(', ')}`);meta.className='muted';
    const details=element('details');details.append(element('summary','Abstract'),element('p',p.abstract||'Abstract unavailable.'));
    card.append(title,meta,details);$('papers').append(card);
  }
  if(!rows.length)$('papers').append(element('p','No keyword matches. Try fewer phrases or different terminology. This does not prove there are no relevant papers.'));
  $('page').textContent=`Page ${page} of ${pages}`;$('previous').disabled=page===1;$('next').disabled=page===pages;
}
function render(){
  $('results').hidden=false;$('result-title').textContent=result.query.prompt;$('paper-total').textContent=`${number(result.matches.length)} matching papers`;
  const partial=result.counts.filter(r=>r.status!=='complete');
  $('warning').textContent=partial.length?'Incomplete coverage: '+partial.map(r=>`${manifest.venues[r.venue]} ${r.year} (${r.status})`).join(', ')+'. Missing data is not zero research activity.':'All selected publication sources have complete coverage in this snapshot.';
  chart('counts-chart');chart('acceptance-chart',true);
  const table=element('table'),head=element('tr');for(const label of ['Year','Topic accepted / submitted','Topic rate','Same-pool overall','Published reference (not plotted)','Source / scope'])head.append(element('th',label));table.append(head);
  for(const r of result.acceptance){const tr=element('tr');for(const value of [r.year,r.topic_submitted==null?'NA':`${r.topic_accepted} / ${r.topic_submitted}`,r.topic_rate==null?'NA':r.topic_rate.toFixed(2)+'%',r.conference_rate==null?'NA':`${r.accepted} / ${r.submitted} (${r.conference_rate.toFixed(2)}%)`])tr.append(element('td',value));
    const official=element('td');official.append(link(r.official_source,r.official_rate==null?'NA':r.official_rate.toFixed(2)+'%'));tr.append(official);
    const source=element('td',r.available?r.population:r.reason);if(r.source)source.append(element('br'),link(r.source,'Official source ↗'));tr.append(source);table.append(tr);}
  $('denominators').replaceChildren(table);
  if(!result.acceptance.length)$('denominators').append(element('p','Select ICLR to see acceptance rates.'));
  if(result.query.tracks.join(',')!=='research')$('denominators').append(element('p','Select Research alone for acceptance comparisons.'));
  renderPapers();
}
async function exportChart(target,format){
  const svg=$(target).querySelector('svg');if(!svg)return;
  const xml=new XMLSerializer().serializeToString(svg);
  if(format==='svg')return save(`${target}.svg`,xml,'image/svg+xml');
  const url=URL.createObjectURL(new Blob([xml],{type:'image/svg+xml'}));
  try{const img=new Image();img.src=url;await img.decode();const canvas=document.createElement('canvas');canvas.width=1140;canvas.height=640;canvas.getContext('2d').drawImage(img,0,0,1140,640);canvas.toBlob(blob=>{if(blob)save(`${target}.png`,blob,'image/png');});}finally{URL.revokeObjectURL(url);}
}
$('search-form').onsubmit=e=>{
  e.preventDefault();if(!worker)return;
  $('error').hidden=true;$('search').disabled=true;$('results').hidden=true;
  worker.postMessage({id:++request,query:{prompt:$('prompt').value,years:selected('years'),venues:selected('venues'),tracks:selected('tracks')}});
};
$('sort').onchange=()=>{page=1;renderPapers();};$('previous').onclick=()=>{page--;renderPapers();};$('next').onclick=()=>{page++;renderPapers();};
$('papers-csv').onclick=()=>csv('papers.csv',result.matches,['id','title','year','venue','track','url','abstract']);
$('counts-csv').onclick=()=>csv('counts.csv',result.counts,['year','venue','status','papers','candidates','missing_abstracts']);
$('acceptance-csv').onclick=()=>csv('acceptance.csv',result.acceptance,['year','topic_accepted','topic_submitted','topic_rate','accepted','submitted','conference_rate','population','source','official_rate','official_source']);
$('metadata-json').onclick=()=>save('search.json',JSON.stringify({query:result.query,snapshot:result.snapshot,corpus_sha256:result.corpus_sha256,counts:result.counts,acceptance:result.acceptance},null,2),'application/json');
for(const button of document.querySelectorAll('[data-chart]'))button.onclick=()=>exportChart(button.dataset.chart,button.dataset.format).catch(e=>fail(e.message));
async function boot(){
  try{
    if(!globalThis.Worker||!globalThis.DecompressionStream)throw Error('Please use a current browser supporting Web Workers and gzip decompression, or run the Python app locally.');
    const response=await fetch('manifest.json',{cache:'no-cache'});if(!response.ok)throw Error('Snapshot unavailable. Please retry later.');manifest=await response.json();
    if(manifest.format!==1)throw Error('Unsupported snapshot format.');
    for(const id of ['source','repo'])$(id).href=manifest.repository;$('data-doc').href=manifest.repository+'/blob/main/docs/DATA.md';
    controls('years',manifest.years.map(y=>[y,y]));controls('venues',Object.entries(manifest.venues));controls('tracks',manifest.tracks.map(t=>[t,t]));
    $('snapshot').textContent=`Dataset snapshot: ${manifest.snapshot.slice(0,10)} · Up to ${(manifest.download_bytes/1e6).toFixed(1)} MB compressed public data, downloaded on demand.`;
    worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
    worker.onmessage=({data})=>{if(data.id!==request)return;if(data.progress){$('status').textContent=data.progress;return;}if(data.error){fail(data.error);return;}result=data.result;page=1;$('sort').value='relevance';render();$('status').textContent='Search completed in your browser. No query sent to a server.';$('search').disabled=false;};
    worker.onerror=()=>fail('Browser search worker failed. Reload the page or use the local Python app.');
    $('status').textContent='Your search stays in this browser.';$('search').disabled=false;
  }catch(e){fail(e.message);$('search').disabled=true;}
}
boot();
