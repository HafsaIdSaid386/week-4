// app.js
/* ------------------------------------------------------------
   MovieLens 100K Two-Tower Demo (TF.js, Pure Client-Side)
   - Loads u.data + u.item
   - Trains Deep Two-Tower (MLP) with in-batch softmax
   - Also exposes a shallow (no-MLP, no-genre) baseline
   - Test view renders a 3-column table:
       History (Top-10 rated) | Baseline Recs | Deep Recs
   - After training, projects a sample of item embeddings with PCA
------------------------------------------------------------- */

let dataState = {
  interactions: [],           // [{userId, itemId, rating, ts}]
  items: new Map(),           // itemId -> { title, year, genres: Int8Array(19) }
  userToItems: new Map(),     // userId -> [{itemId, rating, ts}]
  userIds: [], itemIds: [],
  userIndex: new Map(), itemIndex: new Map(),
  indexToUser: [], indexToItem: [],
  genreCount: 19
};

let ui = {};
let model; // TwoTowerModel
let globalItemGenreTensor; // [numItems, numGenres] float32
let trainCfg = {
  epochs: 30,      // More epochs for better learning
  batchSize: 256,  // Smaller batches for better gradients
  embDim: 64,      // Larger embeddings
  hiddenDim: 128,  // Larger hidden layers
  learningRate: 0.01, // Higher learning rate
  maxInteractions: 80000,
  useBPR: false
};

// Simple status + chart helpers
function logStatus(msg) {
  const el = ui.status;
  el.textContent += `\n${msg}`;
  el.scrollTop = el.scrollHeight;
}
function clearStatus(msg='') {
  ui.status.textContent = msg;
}

class SimpleLine {
  constructor(canvas) {
    this.ctx = canvas.getContext('2d');
    this.w = canvas.width; this.h = canvas.height;
    this.data = [];
    this.reset();
  }
  reset(){ this.data.length=0; this.draw(); }
  push(v){ this.data.push(v); if(this.data.length>1024) this.data.shift(); this.draw(); }
  draw(){
    const {ctx,w,h} = this;
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle='#0b1229'; ctx.fillRect(0,0,w,h);
    ctx.strokeStyle='#1f2937'; ctx.lineWidth=1;
    for(let y=0;y<=h;y+=52){ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke(); }
    if(!this.data.length) return;
    const max = Math.max(...this.data), min = Math.min(...this.data);
    const range = (max-min)||1;
    ctx.strokeStyle='#22d3ee'; ctx.beginPath();
    const n = this.data.length;
    for(let i=0;i<n;i++){
      const x = (i/(n-1))*w;
      const y = h - ((this.data[i]-min)/range)*h;
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
  }
}

// Better PCA using TF.js SVD
async function performTFjsPCA(embeddings) {
  return tf.tidy(() => {
    const X = tf.tensor2d(embeddings);
    const centered = X.sub(X.mean(0));
    const covariance = tf.matMul(centered.transpose(), centered).div(X.shape[0] - 1);
    
    // Use SVD for more stable PCA
    const [u, s, v] = tf.svd(covariance);
    const components = v.slice([0, 0], [v.shape[0], 2]);
    
    const projected = tf.matMul(centered, components);
    return projected.arraySync();
  });
}

// Utility: get Top-K indices from Float32Array scores
function topKIndices(scores, k, excludeSet) {
  const arr = scores;
  const heap = [];
  for (let i=0;i<arr.length;i++){
    if(excludeSet && excludeSet.has(i)) continue;
    const val = arr[i];
    if(heap.length<k){ heap.push([val,i]); heap.sort((a,b)=>a[0]-b[0]); }
    else if(val>heap[0][0]){ heap[0]=[val,i]; heap.sort((a,b)=>a[0]-b[0]); }
  }
  return heap.sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
}

// Render 3-column table: History | Baseline | Deep
function renderResults(historyList, baselineList, deepList) {
  const toHTML = (items) =>
    `<ol>${items.map(t=>`<li>${t}</li>`).join('')}</ol>`;
  const html = `
    <table>
      <thead>
        <tr>
          <th>Top-10 Historically Rated</th>
          <th>Top-10 Recommended (Baseline: no-MLP, no-genres)</th>
          <th>Top-10 Recommended (Deep: MLP + genres)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>${toHTML(historyList)}</td>
          <td>${toHTML(baselineList)}</td>
          <td>${toHTML(deepList)}</td>
        </tr>
      </tbody>
    </table>
  `;
  ui.results.innerHTML = html;
}

/* ------------------------ Data Loading ------------------------ */
async function loadData() {
  clearStatus('Loading…');
  // u.item
  const itemTxt = await (await fetch('./data/u.item')).text();
  const linesI = itemTxt.split(/\r?\n/).filter(Boolean);
  const items = new Map();
  // u.item has 24+ columns; last 19 are genres
  for (const line of linesI) {
    const parts = line.split('|');
    const itemId = parseInt(parts[0],10);
    const title = parts[1] || `Item ${itemId}`;
    const yearMatch = title.match(/\((\d{4})\)/);
    const year = yearMatch ? parseInt(yearMatch[1],10) : null;
    const genreFlags = parts.slice(-19).map(x=>parseInt(x||'0',10));
    items.set(itemId, { title, year, genres: Int8Array.from(genreFlags) });
  }
  dataState.items = items;

  // u.data
  const dataTxt = await (await fetch('./data/u.data')).text();
  const rows = dataTxt.split(/\r?\n/).filter(Boolean);
  const interactions = [];
  for (let i=0;i<rows.length;i++){
    const [u,iid,r,ts] = rows[i].split('\t');
    interactions.push({ userId:+u, itemId:+iid, rating:+r, ts:+ts });
  }
  interactions.sort((a,b)=>a.userId-b.userId || b.rating-a.rating || b.ts-a.ts);
  dataState.interactions = interactions;

  // Build user→items
  const userToItems = new Map();
  for (const it of interactions) {
    if(!userToItems.has(it.userId)) userToItems.set(it.userId, []);
    userToItems.get(it.userId).push({ itemId: it.itemId, rating: it.rating, ts: it.ts });
  }
  dataState.userToItems = userToItems;

  // Indexers
  const userIds = Array.from(userToItems.keys()).sort((a,b)=>a-b);
  const itemIds = Array.from(items.keys()).sort((a,b)=>a-b);
  dataState.userIds = userIds; dataState.itemIds = itemIds;
  const userIndex = new Map(); const itemIndex = new Map();
  userIds.forEach((u,idx)=>userIndex.set(u, idx));
  itemIds.forEach((i,idx)=>itemIndex.set(i, idx));
  dataState.userIndex = userIndex; dataState.itemIndex = itemIndex;
  dataState.indexToUser = userIds.slice(); dataState.indexToItem = itemIds.slice();

  // Precompute global item-genre matrix
  const numItems = itemIds.length;
  const G = new Float32Array(numItems * dataState.genreCount);
  for (let r=0;r<numItems;r++){
    const genres = items.get(itemIds[r]).genres;
    for (let c=0;c<genres.length;c++) G[r*dataState.genreCount + c] = genres[c];
  }
  globalItemGenreTensor = tf.tensor2d(G, [numItems, dataState.genreCount]);

  ui.countsPill.textContent = `users: ${userIds.length}, items: ${itemIds.length}, interactions: ${interactions.length}`;
  logStatus('Loaded dataset. Click Train to start.');
  ui.trainBtn.disabled = false;
  ui.testBtn.disabled = true;
}

/* ------------------------ Training ------------------------ */
async function train() {
  const { interactions, userIndex, itemIndex, itemIds } = dataState;
  const limit = Math.min(trainCfg.maxInteractions, interactions.length);
  const triplets = interactions.slice(0, limit);

  // Build training indices
  const userIdx = new Int32Array(limit);
  const itemIdx = new Int32Array(limit);
  for (let i=0;i<limit;i++){
    userIdx[i] = userIndex.get(triplets[i].userId);
    itemIdx[i] = itemIndex.get(triplets[i].itemId);
  }

  // Two-tower model
  const numUsers = dataState.userIds.length;
  const numItems = dataState.itemIds.length;
  model = new TwoTowerModel(numUsers, numItems, dataState.genreCount,
    trainCfg.embDim, trainCfg.hiddenDim, trainCfg.learningRate, trainCfg.useBPR);

  const lossChart = new SimpleLine(ui.lossCanvas);
  clearStatus('Training…');

  const batchSize = trainCfg.batchSize;
  const stepsPerEpoch = Math.ceil(limit / batchSize);

  // Mini-batch loop (in-batch softmax)
  for (let epoch=0; epoch<trainCfg.epochs; epoch++){
    let epochLoss = 0;
    for (let s=0; s<stepsPerEpoch; s++){
      const start = s*batchSize;
      const end = Math.min(limit, start+batchSize);
      const uBatch = tf.tensor1d(userIdx.slice(start, end), 'int32');
      const iBatch = tf.tensor1d(itemIdx.slice(start, end), 'int32');
      const gBatch = tf.gather(globalItemGenreTensor, iBatch); // [B, numGenres]

      const loss = await model.trainStep(uBatch, iBatch, gBatch);
      epochLoss += loss;

      lossChart.push(loss);
      tf.dispose([uBatch, iBatch, gBatch]);
      await tf.nextFrame();
    }
    logStatus(`Epoch ${epoch+1}/${trainCfg.epochs} – avg loss: ${ (epochLoss/stepsPerEpoch).toFixed(4) }`);
  }
  logStatus('Training finished.');

  // PCA projection of a sample of item embeddings (post-MLP)
  await drawPCA();
  ui.testBtn.disabled = false;
}

async function drawPCA() {
  const canvas = ui.pcaCanvas, ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0b1229';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  const numItems = dataState.itemIds.length;
  const sampleSize = Math.min(800, numItems);
  
  // Sample diverse items for better visualization
  const step = Math.floor(numItems / sampleSize);
  const sampleIndices = [];
  for (let i = 0; i < sampleSize; i++) {
    sampleIndices.push(Math.min(i * step, numItems - 1));
  }

  const idxTensor = tf.tensor1d(sampleIndices, 'int32');
  const genreBatch = tf.gather(globalItemGenreTensor, idxTensor);

  try {
    const emb = await tf.tidy(() => model.itemForwardForPCA(idxTensor, genreBatch));
    const embArray = await emb.array();
    
    // Use TF.js built-in PCA for better results
    const proj = await performTFjsPCA(embArray);
    const xs = proj.map(p => p[0]);
    const ys = proj.map(p => p[1]);
    
    // Calculate visualization parameters
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    
    const padding = 0.1;
    const scaleX = (canvas.width * (1 - 2 * padding)) / xRange;
    const scaleY = (canvas.height * (1 - 2 * padding)) / yRange;
    const scale = Math.min(scaleX, scaleY);
    
    const offsetX = canvas.width * padding - xMin * scale;
    const offsetY = canvas.height * (1 - padding) - yMin * scale;
    
    // Draw clusters with different colors based on position
    const titles = [];
    
    for (let k = 0; k < proj.length; k++) {
      const x = offsetX + xs[k] * scale;
      const y = offsetY - ys[k] * scale; // Flip Y axis
      
      // Color points based on their quadrant for better visualization
      const quadrantX = Math.floor((xs[k] - xMin) / (xRange / 3));
      const quadrantY = Math.floor((ys[k] - yMin) / (yRange / 3));
      const colors = ['#22d3ee', '#a78bfa', '#fbbf24', '#34d399', '#f87171', '#60a5fa'];
      ctx.fillStyle = colors[(quadrantX + quadrantY) % colors.length];
      
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
      
      titles.push(dataState.items.get(dataState.itemIds[sampleIndices[k]]).title);
    }
    
    // Add grid and labels for better readability
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(offsetX + xMin * scale, offsetY - yMax * scale, xRange * scale, yRange * scale);
    
    setupHover(canvas, proj, titles, offsetX, offsetY, scale, xMin, yMin);
    
  } catch (error) {
    console.error('PCA Error:', error);
    logStatus('PCA failed: ' + error.message);
  } finally {
    tf.dispose([idxTensor, genreBatch]);
  }
}

// Update the hover function:
function setupHover(canvas, points, titles, offsetX, offsetY, scale, xMin, yMin) {
  const ctx = canvas.getContext('2d');
  const hoverLabel = ui.hoverTitle;
  
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    let closestIndex = -1;
    let minDist = 15;
    
    for (let k = 0; k < points.length; k++) {
      const x = offsetX + points[k][0] * scale;
      const y = offsetY - points[k][1] * scale;
      const dist = Math.sqrt((x - mx) ** 2 + (y - my) ** 2);
      if (dist < minDist) {
        minDist = dist;
        closestIndex = k;
      }
    }
    
    hoverLabel.textContent = closestIndex >= 0 ? titles[closestIndex] : 'Hover over points to see movie titles';
    hoverLabel.style.color = closestIndex >= 0 ? '#22d3ee' : '#9ca3af';
  };

  canvas.onmouseleave = () => {
    hoverLabel.textContent = 'Hover over points to see movie titles';
    hoverLabel.style.color = '#9ca3af';
  };
}

/* ------------------------ Testing / Recommendation ------------------------ */
async function testOnce() {
  if(!model){ alert('Train the model first.'); return; }
  const { userToItems, userIndex, itemIndex, itemIds, items } = dataState;

  // Pick user with ≥20 ratings
  const candidates = Array.from(userToItems.entries()).filter(([,arr])=>arr.length>=20);
  const [userId, history] = candidates[Math.floor(Math.random()*candidates.length)];
  const historySorted = history.slice().sort((a,b)=> b.rating-a.rating || b.ts-a.ts).slice(0,10);
  const historyTitles = historySorted.map(x=>items.get(x.itemId).title);

  // Exclude set (indices)
  const exclude = new Set(history.map(x=> itemIndex.get(x.itemId)));

  const uIdx = tf.tensor1d([userIndex.get(userId)], 'int32');

  // Deep (MLP + genres)
  const deepScores = await tf.tidy(() => {
    const uEmb = model.userForward(uIdx);              // [1, E]
    // Compute scores vs all items in batches to save memory
    const B = 1024;
    const total = itemIds.length;
    const out = new Float32Array(total);
    for (let s=0, p=0; s<total; s+=B){
      const end = Math.min(total, s+B);
      const idx = tf.tensor1d(new Int32Array(Array.from({length:end-s},(_,k)=>s+k)), 'int32');
      const g = tf.gather(globalItemGenreTensor, idx);
      const iEmb = model.itemForward(idx, g);          // [b, E]
      const logits = tf.matMul(iEmb, uEmb.transpose()).reshape([end-s]); // [b]
      const arr = logits.dataSync();
      for(let k=0;k<arr.length;k++) out[p++]=arr[k];
      tf.dispose([idx,g,iEmb,logits]);
    }
    return out;
  });

  // Baseline (no-MLP, no-genres) — raw embedding dot product
  const baseScores = await tf.tidy(() => {
    const uRaw = tf.gather(model.userEmbedding, uIdx);       // [1,E]
    const iRaw = model.itemEmbedding;                         // [I,E]
    const logits = tf.matMul(iRaw, uRaw.transpose()).reshape([iRaw.shape[0]]);
    return logits.dataSync();
  });

  const k = 10;
  const deepIdxs = topKIndices(deepScores, k, exclude);
  const baseIdxs = topKIndices(baseScores, k, exclude);

  const deepTitles = deepIdxs.map(i=>items.get(itemIds[i]).title);
  const baseTitles = baseIdxs.map(i=>items.get(itemIds[i]).title);

  renderResults(historyTitles, baseTitles, deepTitles);
  logStatus(`Tested user ${userId}. History vs Baseline vs Deep rendered.`);
}

/* ------------------------ Wire-up ------------------------ */
window.addEventListener('DOMContentLoaded', ()=>{
  ui = {
    loadBtn: document.getElementById('loadBtn'),
    trainBtn: document.getElementById('trainBtn'),
    testBtn: document.getElementById('testBtn'),
    status: document.getElementById('status'),
    lossCanvas: document.getElementById('lossCanvas'),
    pcaCanvas: document.getElementById('pcaCanvas'),
    results: document.getElementById('results'),
    countsPill: document.getElementById('countsPill'),
    hoverTitle: document.getElementById('hoverTitle')
  };
  ui.loadBtn.onclick = ()=> loadData().catch(e=>logStatus('Error: '+e.message));
  ui.trainBtn.onclick = ()=> train().catch(e=>logStatus('Error: '+e.message));
  ui.testBtn.onclick = ()=> testOnce().catch(e=>logStatus('Error: '+e.message));
});
