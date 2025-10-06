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
  epochs: 10,
  batchSize: 512,
  embDim: 32,
  hiddenDim: 64,
  learningRate: 0.001,
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

// PCA (power iteration for top-2 eigenvectors of covariance)
async function pca2D(tensor2d /* [N, D] */) {
  return tf.tidy(() => {
    const X = tensor2d;                       // [N,D]
    const mean = tf.mean(X, 0, true);         // [1,D]
    const Xc = tf.sub(X, mean);               // centered
    const cov = tf.matMul(Xc.transpose(), Xc).div(X.shape[0]-1); // [D,D]

    function powerVec(init, iters=30){
      let v = init;
      for(let i=0;i<iters;i++){
        v = tf.matMul(cov, v);
        v = tf.div(v, tf.norm(v));
      }
      return v; // [D,1]
    }
    const D = X.shape[1];
    let v1 = powerVec(tf.randomNormal([D,1],0,1));
    const lambda1 = tf.sum(tf.mul(v1, tf.matMul(cov, v1)));
    // Deflate
    const covDef = tf.sub(cov, tf.matMul(v1, v1.transpose()).mul(lambda1));
    let v2 = v1; // reuse var; recompute with deflated cov
    for(let i=0;i<30;i++){
      v2 = tf.matMul(covDef, v2);
      v2 = tf.div(v2, tf.norm(v2));
    }
    const W = tf.concat([v1, v2], 1); // [D,2]
    const proj = tf.matMul(Xc, W);    // [N,2]
    return proj;
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
  const numItems = dataState.itemIds.length;
  const sampleSize = Math.min(1000, numItems);
  const step = Math.floor(numItems / sampleSize) || 1;
  const idxs = [];
  for (let i=0;i<numItems;i+=step) idxs.push(i);
  const idxTensor = tf.tensor1d(new Int32Array(idxs), 'int32');
  const genreBatch = tf.gather(globalItemGenreTensor, idxTensor);

  const emb = await tf.tidy(() => model.itemForward(idxTensor, genreBatch).array());
  const proj = await pca2D(tf.tensor2d(emb));
  const points = await proj.array();
  const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
  const minX = Math.min(...xs), maxX=Math.max(...xs);
  const minY = Math.min(...ys), maxY=Math.max(...ys);
  const norm = (v, a, b, w) => ( (v-a)/(b-a) ) * w;

  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#0b1229'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle = '#93c5fd88';
  const titles = [];
  for (let k=0;k<points.length;k++){
    const x = norm(xs[k], minX, maxX, canvas.width);
    const y = canvas.height - norm(ys[k], minY, maxY, canvas.height);
    ctx.beginP
