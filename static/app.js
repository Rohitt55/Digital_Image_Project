'use strict';
const $ = id => document.getElementById(id);
let operations = [], selected = null, inputFile = null, originalFile = null;
let resultData = null, resultName = '', beforeStats = null, afterStats = null;
let busy = false, loading = false, revision = 0, history = [];
let labFigure = null;
let encodedArtifact = null;

function notify(message, success = false) {
  $('notice').textContent = message;
  $('notice').className = 'notice' + (success ? ' success' : '');
  $('notice').hidden = false;
}
function clearNotice() { $('notice').hidden = true; }
function controls() {
  $('run-button').disabled = busy || loading || !inputFile || !selected;
  $('run-button').textContent = busy ? 'Processing…' : 'Run operation';
  $('download-button').disabled = busy || !resultData;
  $('use-result').disabled = busy || loading || !resultData;
  $('reset-button').disabled = busy || loading || !originalFile;
  $('demo-button').disabled = busy || loading;
  $('image-upload').disabled = busy || loading;
  $('reference-upload').disabled = busy || loading;
  $('save-figure').disabled = busy || !labFigure;
  $('download-encoded').disabled = busy || !encodedArtifact;
  $('chapter-filter').disabled = busy;
  $('reset-settings').disabled = busy || loading || !selected?.fields.length;
  $('zoom-original').disabled = !inputFile || loading;
  $('zoom-result').disabled = busy || !resultData;
  $('result-busy').hidden = !busy;
  $('result-stage').setAttribute('aria-busy',String(busy));
  $('run-help').textContent = loading ? 'Loading image…' : busy ? 'Running operation and generating plots…' : !inputFile ? 'Upload an image or use the sample.' : (selected?.requires_reference || selected?.id==='matching') && !$('reference-upload').files[0] ? 'A reference image is required.' : 'Image loaded. Settings are optional.';
  $('result-status').textContent=busy?'Processing…':resultData?'Complete · Download available':inputFile?'Not processed yet':'No result yet';
  $('step-upload').className=inputFile?'done':'current';
  $('step-run').className=resultData?'done':inputFile?'current':'';
  $('step-result').className=resultData?'current':'';
  document.querySelectorAll('.quick-operation').forEach(el=>el.disabled=busy||loading);
  document.querySelectorAll('.operation-button, #parameters input, #parameters select').forEach(el => el.disabled = busy);
}
function clearResult() {
  resultData = null; afterStats = null;
  encodedArtifact = null; $('download-encoded').hidden = true;
  labFigure = null;
  $('lab-figure').hidden = true; $('lab-figure').removeAttribute('src');
  if ($('lab-figure-link').href.startsWith('blob:')) URL.revokeObjectURL($('lab-figure-link').href);
  $('lab-figure-link').hidden = true; $('lab-figure-link').removeAttribute('href');
  $('lab-empty').hidden = false; $('console-wrap').hidden = true; $('lab-console').textContent = '';
  $('result-image').hidden = true; $('result-empty').hidden = false;
  $('result-meta').textContent = 'No result yet'; $('result-label').textContent = 'OUTPUT';
  $('extra-details').replaceChildren();
  renderStats(beforeStats, null); drawHistogram(); controls();
}
function navigation() {
  const query = $('search').value.trim().toLowerCase();
  const list = $('operation-list'); list.replaceChildren();
  let group = '';
  const chapter = Number($('chapter-filter').value);
  const matches=operations.filter(op => (!chapter || op.chapters.includes(chapter)) && `${op.name} ${op.group} ${op.explanation}`.toLowerCase().includes(query));
  const groupOrder=['Color & intensity','Geometry','Histograms','Edges & segmentation','Filters & restoration','Transforms & comparison','Morphology','Fundamentals','Compression'];
  matches.sort((a,b)=>groupOrder.indexOf(a.group)-groupOrder.indexOf(b.group));
  $('operation-count').textContent=`${matches.length} operation${matches.length===1?'':'s'}`;
  $('clear-search').hidden=!query&&!chapter;
  matches.forEach(op => {
    if (op.group !== group) {
      const label = document.createElement('p'); label.className = 'nav-group'; label.textContent = op.group;
      list.append(label); group = op.group;
    }
    const button = document.createElement('button');
    button.className = 'operation-button' + (selected?.id === op.id ? ' active' : '');
    button.setAttribute('aria-pressed', selected?.id === op.id ? 'true' : 'false');
    button.textContent=op.name;
    button.disabled = busy; button.onclick = () => selectOperation(op);
    button.title=op.explanation;
    list.append(button);
  });
  if (!list.children.length) { const p = document.createElement('p'); p.className = 'muted'; p.textContent = 'No matching operations.'; list.append(p); }
}
function selectOperation(op) {
  selected = op; revision++; navigation();
  $('operation-name').textContent = op.name;
  $('operation-group').textContent = op.group;
  $('operation-explanation').textContent = op.explanation;
  $('formula').textContent = op.formula; $('source').textContent = op.source;
  $('python-code').textContent = op.code;
  $('implementation-file').textContent=op.implementation_file;
  $('code-type').textContent=op.code_type;
  $('code-libraries').textContent=op.module_libraries.join(', ') || 'Python standard library';
  document.querySelectorAll('.quick-operation').forEach(button=>{
    button.classList.toggle('active',button.dataset.operation===op.id);
    button.setAttribute('aria-pressed',String(button.dataset.operation===op.id));
  });
  const parameters = $('parameters'); parameters.replaceChildren();
  op.fields.forEach(field => {
    const label = document.createElement('label'); label.className = 'parameter'; label.textContent = field.label;
    let input;
    if (field.type === 'select') {
      input = document.createElement('select');
      field.options.forEach(value => { const option = document.createElement('option'); option.value = value; option.textContent = value; input.append(option); });
    } else {
      input = document.createElement('input'); input.type = field.type === 'text' ? 'text' : 'number';
      if (field.type === 'text') input.maxLength = 512;
      else { input.min = field.min; input.max = field.max; input.step = field.step; }
      input.required = true;
    }
    input.name = field.key; input.value = field.default;
    input.id=`setting-${field.key}`;
    const hint=document.createElement('small');hint.className='parameter-hint';hint.id=`hint-${field.key}`;
    hint.textContent=field.type==='number'?`${field.min}–${field.max} · Default ${field.default}${field.key==='kernel'?' · Use odd numbers':''}`:field.type==='select'?`Default: ${field.default}`:field.key==='coefficients'?'Enter 9 numbers, separated by commas.':field.key==='template'?'Use 1, -1 or 0 for each of the 9 positions.':'Enter the values for this setting.';
    input.setAttribute('aria-describedby',hint.id);
    input.addEventListener('input', () => { revision++; clearResult(); });
    label.append(input,hint); parameters.append(label);
  });
  if (!op.fields.length) { const p = document.createElement('p'); p.className = 'no-params'; p.textContent = 'No parameters for this operation.'; parameters.append(p); }
  $('reference-wrap').hidden = !(op.requires_reference || op.id === 'matching');
  clearResult(); clearNotice();
  if(window.matchMedia('(max-width:760px)').matches && document.querySelector('.sidebar').classList.contains('menu-open')){
    document.querySelector('.sidebar').classList.remove('menu-open');$('mobile-menu').setAttribute('aria-expanded','false');
    $('settings-panel').scrollIntoView({behavior:'smooth',block:'start'});
  }
}
async function readResponse(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'The request failed. Please try again.');
  return data;
}
function requestError(error) {
  return error instanceof TypeError
    ? 'Cannot connect to the Python backend. Run .\\start.bat, keep its terminal window open, then refresh this page.'
    : error.message;
}
async function upload(file, remember = true) {
  if (!file || busy || loading) return;
  if (file.size > 23 * 1024 * 1024) { notify('Please choose an image smaller than 23 MB.'); return; }
  loading = true; controls(); clearNotice();
  try {
    const form = new FormData(); form.append('image', file); form.append('operation', 'histogram');
    const data = await readResponse(await fetch('/api/process', {method:'POST', body:form}));
    inputFile = file; if (remember) originalFile = file;
    revision++; beforeStats = data.before;
    $('original-image').src = data.original; $('original-image').hidden = false; $('original-empty').hidden = true;
    $('filename').textContent = file.name;
    $('upload-details').textContent=`${data.before.width} × ${data.before.height} pixels · ${(file.size/1024/1024).toFixed(2)} MB`;
    $('upload-details').hidden=false;
    $('upload-description').textContent='Choose another file to replace the input image.';
    $('original-meta').textContent = `${data.before.width} × ${data.before.height} px`;
    clearResult();
  } catch (error) { notify(requestError(error)); }
  finally { loading = false; controls(); }
}
async function sample() {
  loading = true; controls(); clearNotice();
  try {
    const data = await readResponse(await fetch('/api/demo'));
    const blob = await (await fetch(data.image)).blob();
    loading = false;
    await upload(new File([blob], 'PixelLab-sample.png', {type:'image/png'}));
  } catch (error) { notify(requestError(error)); }
  finally { loading = false; controls(); }
}
async function run(event) {
  event.preventDefault();
  if (busy || loading || !inputFile || !selected) return;
  if ((selected.requires_reference || selected.id === 'matching') && !$('reference-upload').files[0]) { notify('Choose a reference image to compare with your input.'); return; }
  const params = {};
  $('parameters').querySelectorAll('input, select').forEach(input => params[input.name] = input.type === 'number' ? Number(input.value) : input.value);
  const form = new FormData(); form.append('image', inputFile); form.append('operation', selected.id); form.append('params', JSON.stringify(params)); form.append('lab_output', 'true');
  if (selected.requires_reference || selected.id === 'matching') form.append('reference', $('reference-upload').files[0]);
  const currentRevision = revision, currentOp = selected;
  busy = true; controls(); clearNotice();
  try {
    const data = await readResponse(await fetch('/api/process', {method:'POST', body:form}));
    if (revision !== currentRevision) return;
    resultData = data.result; resultName = currentOp.id;
    encodedArtifact = data.artifact || null; $('download-encoded').hidden = !encodedArtifact;
    beforeStats = data.before; afterStats = data.after;
    $('result-image').src = resultData; $('result-image').hidden = false; $('result-empty').hidden = true;
    $('result-image').style.width = '100%'; $('result-image').style.height = '100%';
    $('result-image').style.imageRendering = currentOp.id === 'pixel_matrix' ? 'pixelated' : 'auto';
    $('result-meta').textContent = `${data.after.width} × ${data.after.height} px`;
    $('result-label').textContent = currentOp.name.toUpperCase();
    renderStats(data.before, data.after, data.details.processing_ms); drawHistogram(); renderDetails(data.details);
    if (data.lab) {
      labFigure = data.lab.figure;
      $('lab-figure').src = labFigure; $('lab-figure').hidden = false; $('lab-empty').hidden = true;
      // Blob URLs let the full-size figure open reliably in a separate tab.
      if ($('lab-figure-link').href.startsWith('blob:')) URL.revokeObjectURL($('lab-figure-link').href);
      const bytes = Uint8Array.from(atob(labFigure.split(',')[1]), c => c.charCodeAt(0));
      $('lab-figure-link').href = URL.createObjectURL(new Blob([bytes], {type:'image/png'}));
      $('lab-figure-link').hidden = false;
      $('lab-console').textContent = data.lab.console; $('console-wrap').hidden = false;
      $('tab-lab-output').click();
    }
    history.unshift({name:currentOp.name, input:inputFile.name, params, time:new Date().toLocaleTimeString(), ms:data.details.processing_ms, image:resultData});
    history = history.slice(0, 12); renderHistory();
    notify(`${currentOp.name} · ${data.details.processing_ms} ms`, true);
  } catch (error) { notify(requestError(error)); }
  finally { busy = false; controls(); }
}
function renderStats(before, after, ms) {
  const stats = $('stats'); stats.replaceChildren();
  const latest = after || before;
  const values = [
    ['DIMENSIONS', latest ? `${latest.width} × ${latest.height}` : '—', after ? `Input: ${before.width} × ${before.height}` : 'Pixels'],
    ['MEAN INTENSITY', latest ? latest.mean.toFixed(1) : '—', after ? `Input: ${before.mean.toFixed(1)}` : 'Grayscale · 0–255'],
    ['RMS CONTRAST', latest ? latest.rms_contrast.toFixed(2) : '—', after ? `Input: ${before.rms_contrast.toFixed(2)}` : 'Grayscale standard deviation'],
    ['PROCESSING TIME', ms === undefined ? '—' : `${ms} ms`, after ? `${after.channels} output channel${after.channels === 1 ? '' : 's'}` : 'Runs in Python']
  ];
  values.forEach(([label, value, sub]) => { const div = document.createElement('div'); const span = document.createElement('span'); span.textContent = label; const strong = document.createElement('strong'); strong.textContent = value; const small = document.createElement('small'); small.textContent = sub; div.append(span, strong, small); stats.append(div); });
}
function drawHistogram() {
  const canvas = $('histogram-chart'), width = canvas.clientWidth;
  if (!width) return;
  const height = 150, ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio); ctx.clearRect(0,0,width,height);
  ctx.strokeStyle = '#edf0e7'; ctx.lineWidth = 1;
  for (let y=0; y<=height; y+=height/4) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(width,y); ctx.stroke(); }
  const distributions = [beforeStats, afterStats].map(s => s ? s.histogram.map(n => n / (s.width * s.height)) : null);
  const max = Math.max(.001, ...distributions.filter(Boolean).flat());
  distributions.forEach((hist, index) => {
    if (!hist) return;
    ctx.beginPath(); ctx.moveTo(0,height);
    hist.forEach((value,i) => ctx.lineTo(i / 255 * width, height - value / max * (height - 8)));
    ctx.lineTo(width,height); ctx.closePath(); ctx.fillStyle = index ? '#37725544' : '#b8c5ab77'; ctx.fill();
    ctx.strokeStyle = index ? '#377255' : '#a5b797'; ctx.lineWidth = 1.3; ctx.stroke();
  });
}
function renderDetails(details) {
  const wrap = $('extra-details'); wrap.replaceChildren();
  Object.entries(details).forEach(([key,value]) => {
    if (key === 'processing_ms' || key === 'download') return;
    if (key === 'pixel_matrix') {
      const table = document.createElement('table'); table.className = 'matrix';
      const caption = document.createElement('caption'); caption.textContent = 'Grayscale pixel matrix'; table.append(caption);
      value.forEach(row => { const tr = document.createElement('tr'); row.forEach(n => { const td = document.createElement('td'); td.textContent = n; tr.append(td); }); table.append(tr); }); wrap.append(table); return;
    }
    const span = document.createElement('span'); span.className = key === 'note' ? 'detail-note' : 'detail-pill';
    span.textContent = key === 'note' ? String(value) : `${key.replaceAll('_',' ')}: ${value}`;
    wrap.append(span);
  });
}
function renderHistory() {
  $('history-count').textContent = history.length; $('history-empty').hidden = !!history.length;
  const list = $('history-list'); list.replaceChildren();
  history.forEach(item => {
    const row = document.createElement('div'); row.className = 'history-row';
    const image = document.createElement('img'); image.src = item.image; image.alt = `${item.name} output thumbnail`;
    const info = document.createElement('div'), title = document.createElement('strong'), sub = document.createElement('small');
    title.textContent = item.name; sub.textContent = `${item.input} · ${Object.entries(item.params).map(([k,v]) => `${k}=${v}`).join(', ') || 'Default settings'}`;
    info.append(title, sub); const time = document.createElement('span'); time.textContent = `${item.time} · ${item.ms} ms`; row.append(image, info, time); list.append(row);
  });
}
$('search').addEventListener('input', navigation);
$('image-upload').addEventListener('change', e => upload(e.target.files[0]));
$('reference-upload').addEventListener('change', e => { $('reference-name').textContent = e.target.files[0]?.name || ''; revision++; clearResult(); });
$('demo-button').addEventListener('click', sample);
$('process-form').addEventListener('submit', run);
$('reset-button').addEventListener('click', () => upload(originalFile, false));
$('use-result').addEventListener('click', async () => {
  const blob = await (await fetch(resultData)).blob();
  await upload(new File([blob], `${resultName}-input.${blob.type === 'image/jpeg' ? 'jpg' : 'png'}`, {type:blob.type}), false);
});
$('download-button').addEventListener('click', () => {
  if (!resultData) return;
  const link = document.createElement('a'); link.href = resultData;
  link.download = `PixelLab-${resultName}.${resultData.startsWith('data:image/jpeg') ? 'jpg' : 'png'}`;
  document.body.append(link); link.click(); link.remove();
});
$('save-figure').addEventListener('click', () => {
  if (!labFigure) return;
  const link = document.createElement('a'); link.href = labFigure; link.download = `PixelLab-${resultName}-figure.png`;
  document.body.append(link); link.click(); link.remove();
});
$('download-encoded').addEventListener('click', () => {
  if (!encodedArtifact) return;
  const link=document.createElement('a'); link.href=encodedArtifact.data; link.download=encodedArtifact.name;
  document.body.append(link); link.click(); link.remove();
});
$('chapter-filter').addEventListener('change', navigation);
$('clear-search').addEventListener('click',()=>{$('search').value='';$('chapter-filter').value='0';navigation();$('search').focus();});
$('reset-settings').addEventListener('click',()=>{
  if(!selected||busy) return;
  selected.fields.forEach(field=>{const input=$(`setting-${field.key}`);if(input){input.value=field.default;input.setCustomValidity('');}});
  revision++;clearResult();clearNotice();
});
function browseOperations(){
  if(window.matchMedia('(max-width:760px)').matches){document.querySelector('.sidebar').classList.add('menu-open');$('mobile-menu').setAttribute('aria-expanded','true');}
  document.querySelector('.sidebar').scrollIntoView({behavior:'smooth',block:'start'});$('search').focus();
}
$('browse-all').addEventListener('click',browseOperations);
$('mobile-menu').addEventListener('click',()=>{
  const open=document.querySelector('.sidebar').classList.toggle('menu-open');$('mobile-menu').setAttribute('aria-expanded',String(open));
});
function showImage(source,title){
  $('viewer-image').src=source;$('viewer-image').alt=title;$('viewer-title').textContent=title;
  $('image-viewer').showModal();
}
$('zoom-original').addEventListener('click',()=>showImage($('original-image').src,'Original image'));
$('zoom-result').addEventListener('click',()=>showImage(resultData,selected?.name||'Processed image'));
$('close-viewer').addEventListener('click',()=>$('image-viewer').close());
$('image-viewer').addEventListener('click',event=>{if(event.target===$('image-viewer')){const rect=$('image-viewer').getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)$('image-viewer').close();}});
document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(button => { button.classList.toggle('active', button === tab); button.setAttribute('aria-selected', String(button === tab)); });
  document.querySelectorAll('.tab-panel').forEach(panel => panel.hidden = panel.id !== tab.dataset.tab);
  drawHistogram();
}));
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('keydown',event=>{
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  event.preventDefault();const tabs=[...document.querySelectorAll('.tab')];let index=tabs.indexOf(tab);
  index=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
  tabs[index].click();tabs[index].focus();tabs[index].scrollIntoView({block:'nearest',inline:'nearest'});
}));
const dropZone = $('drop-zone');
['dragenter','dragover'].forEach(type => dropZone.addEventListener(type, e => { e.preventDefault(); if (!busy && !loading) dropZone.classList.add('dragging'); }));
['dragleave','drop'].forEach(type => dropZone.addEventListener(type, e => { e.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', e => upload(e.dataTransfer.files[0]));
window.addEventListener('resize', drawHistogram);
(async () => {
  try {
    const [loadedOperations,chapters] = await Promise.all([fetch('/api/operations').then(readResponse),fetch('/api/chapters').then(readResponse)]);
    operations=loadedOperations;
    chapters.forEach(chapter=>{
      const option=document.createElement('option');option.value=chapter.id;option.textContent=`${chapter.id}. ${chapter.title}`;$('chapter-filter').append(option);
    });
    const quickTasks=[['grayscale','Grayscale'],['clahe','CLAHE'],['resize','Resize'],['canny','Canny'],['gaussian','Gaussian blur']];
    quickTasks.forEach(([id,name])=>{
      const button=document.createElement('button');button.className='quick-operation';button.dataset.operation=id;
      button.textContent=name;button.onclick=()=>{$('search').value='';$('chapter-filter').value='0';selectOperation(operations.find(op=>op.id===id));};
      $('quick-operations').append(button);
    });
    selectOperation(operations.find(op=>op.id==='grayscale'));
  }
  catch (error) { $('operation-list').textContent = 'Cannot reach the backend.'; notify('Could not load the Python backend. Restart the app and refresh this page.'); }
})();
