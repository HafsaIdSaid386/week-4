// app.js
/* ------------------------------------------------------------
   MovieLens 100K Two-Tower Demo (TensorFlow.js, Pure Client-Side)
   FINAL VERSION ✅
   - Works with final two-tower.js
   - Shows user ID
   - Stable loss decreasing
   - Reset button added
------------------------------------------------------------- */

let dataState = {
  interactions: [],
  items: new Map(),
  userToItems: new Map(),
  userIds: [],
  itemIds: [],
  userIndex: new Map(),
  itemIndex: new Map(),
  indexToUser: [],
  indexToItem: [],
  genreCount: 19,
};

let ui = {};
let model;
let globalItemGenreTensor;

// ⚙️ Training configuration — tuned for stable convergence
let trainCfg = {
  epochs: 20,
  batchSize: 256,
  embDim: 64,
  hiddenDim: 128,
  learningRate: 0.0005,
  maxInteractions: 80000,
  useBPR: false, // set true to use BPR pairwise loss instead of softmax
};

/* ---------------------- UI Helpers ---------------------- */
function logStatus(msg) {
  ui.status.textContent += `\n${msg}`;
  ui.status.scrollTop = ui.status.scrollHeight;
}
function clearStatus(msg = "") {
  ui.status.textContent = msg;
}

/* ---------------------- Simple Line Chart ---------------------- */
class SimpleLine {
  constructor(canvas) {
    this.ctx = canvas.getContext("2d");
    this.w = canvas.width;
    this.h = canvas.height;
    this.data = [];
    this.reset();
  }
  reset() {
    this.data.length = 0;
    this.draw();
  }
  push(v) {
    this.data.push(v);
    if (this.data.length > 1024) this.data.shift();
    this.draw();
  }
  draw() {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0b1229";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#1f2937";
    for (let y = 0; y <= h; y += 52) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    if (!this.data.length) return;
    const max = Math.max(...this.data),
      min = Math.min(...this.data),
      range = max - min || 1;
    ctx.strokeStyle = "#22d3ee";
    ctx.beginPath();
    const n = this.data.length;
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const y = h - ((this.data[i] - min) / range) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

/* ---------------------- PCA Helper ---------------------- */
async function pca2D(tensor2d) {
  return tf.tidy(() => {
    const X = tensor2d;
    const mean = tf.mean(X, 0, true);
    const Xc = tf.sub(X, mean);
    const cov = tf.matMul(Xc.transpose(), Xc).div(X.shape[0] - 1);

    function powerVec(mat, iters = 30) {
      let v = tf.randomNormal([mat.shape[0], 1], 0, 1);
      for (let i = 0; i < iters; i++) {
        v = tf.matMul(mat, v);
        v = tf.div(v, tf.norm(v));
      }
      return v;
    }

    const v1 = powerVec(cov);
    const lambda1 = tf.sum(tf.mul(v1, tf.matMul(cov, v1)));
    const covDef = tf.sub(cov, tf.matMul(v1, v1.transpose()).mul(lambda1));
    const v2 = powerVec(covDef);
    const W = tf.concat([v1, v2], 1);
    return tf.matMul(Xc, W);
  });
}

/* ---------------------- Utility Helpers ---------------------- */
function topKIndices(scores, k, excludeSet) {
  const heap = [];
  for (let i = 0; i < scores.length; i++) {
    if (excludeSet && excludeSet.has(i)) continue;
    const val = scores[i];
    if (heap.length < k) {
      heap.push([val, i]);
      heap.sort((a, b) => a[0] - b[0]);
    } else if (val > heap[0][0]) {
      heap[0] = [val, i];
      heap.sort((a, b) => a[0] - b[0]);
    }
  }
  return heap.sort((a, b) => b[0] - a[0]).map((x) => x[1]);
}

function renderResults(userId, historyList, baselineList, deepList) {
  const toHTML = (items) => `<ol>${items.map((t) => `<li>${t}</li>`).join("")}</ol>`;
  const html = `
    <div style="margin-bottom:8px;color:#93c5fd;">Showing recommendations for <b>User ${userId}</b></div>
    <table>
      <thead>
        <tr>
          <th>Top-10 Historically Rated</th>
          <th>Top-10 Recommended (Baseline)</th>
          <th>Top-10 Recommended (Deep Two-Tower)</th>
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

/* ---------------------- Load Dataset ---------------------- */
async function loadData() {
  clearStatus("Loading data...");
  const itemTxt = await (await fetch("./data/u.item")).text();
  const linesI = itemTxt.split(/\r?\n/).filter(Boolean);
  const items = new Map();

  for (const line of linesI) {
    const parts = line.split("|");
    const itemId = parseInt(parts[0], 10);
    const title = parts[1] || `Item ${itemId}`;
    const yearMatch = title.match(/\((\\d{4})\\)/);
    const year = yearMatch ? parseInt(yearMatch[1], 10) : null;
    const genreFlags = parts.slice(-19).map((x) => parseInt(x || "0", 10));
    items.set(itemId, { title, year, genres: Int8Array.from(genreFlags) });
  }
  dataState.items = items;

  const dataTxt = await (await fetch("./data/u.data")).text();
  const rows = dataTxt.split(/\r?\n/).filter(Boolean);
  const interactions = rows.map((r) => {
    const [u, iid, rating, ts] = r.split("\t");
    return { userId: +u, itemId: +iid, rating: +rating, ts: +ts };
  });
  dataState.interactions = interactions;

  const userToItems = new Map();
  for (const it of interactions) {
    if (!userToItems.has(it.userId)) userToItems.set(it.userId, []);
    userToItems.get(it.userId).push(it);
  }
  dataState.userToItems = userToItems;

  const userIds = Array.from(userToItems.keys()).sort((a, b) => a - b);
  const itemIds = Array.from(items.keys()).sort((a, b) => a - b);
  dataState.userIds = userIds;
  dataState.itemIds = itemIds;

  const userIndex = new Map();
  const itemIndex = new Map();
  userIds.forEach((u, i) => userIndex.set(u, i));
  itemIds.forEach((i, j) => itemIndex.set(i, j));
  dataState.userIndex = userIndex;
  dataState.itemIndex = itemIndex;

  const G = new Float32Array(itemIds.length * dataState.genreCount);
  for (let r = 0; r < itemIds.length; r++) {
    const genres = items.get(itemIds[r]).genres;
    for (let c = 0; c < genres.length; c++) G[r * dataState.genreCount + c] = genres[c];
  }
  globalItemGenreTensor = tf.tensor2d(G, [itemIds.length, dataState.genreCount]);

  ui.countsPill.textContent = `users: ${userIds.length}, items: ${itemIds.length}, interactions: ${interactions.length}`;
  logStatus("Dataset loaded. Click Train to begin.");
  ui.trainBtn.disabled = false;
  ui.testBtn.disabled = true;
}

/* ---------------------- Training ---------------------- */
async function train() {
  const { interactions, userIndex, itemIndex } = dataState;
  const limit = Math.min(trainCfg.maxInteractions, interactions.length);
  const triplets = interactions.slice(0, limit);

  const userIdx = new Int32Array(limit);
  const itemIdx = new Int32Array(limit);
  for (let i = 0; i < limit; i++) {
    userIdx[i] = userIndex.get(triplets[i].userId);
    itemIdx[i] = itemIndex.get(triplets[i].itemId);
  }

  const numUsers = dataState.userIds.length;
  const numItems = dataState.itemIds.length;
  model = new TwoTowerModel(
    numUsers,
    numItems,
    dataState.genreCount,
    trainCfg.embDim,
    trainCfg.hiddenDim,
    trainCfg.learningRate,
    trainCfg.useBPR
  );

  const lossChart = new SimpleLine(ui.lossCanvas);
  clearStatus("Training started...");

  const stepsPerEpoch = Math.ceil(limit / trainCfg.batchSize);
  for (let epoch = 0; epoch < trainCfg.epochs; epoch++) {
    let epochLoss = 0;
    for (let s = 0; s < stepsPerEpoch; s++) {
      const start = s * trainCfg.batchSize;
      const end = Math.min(limit, start + trainCfg.batchSize);
      const uBatch = tf.tensor1d(userIdx.slice(start, end), "int32");
      const iBatch = tf.tensor1d(itemIdx.slice(start, end), "int32");
      const gBatch = tf.gather(globalItemGenreTensor, iBatch);
      const loss = await model.trainStep(uBatch, iBatch, gBatch);
      epochLoss += loss;
      lossChart.push(loss);
      tf.dispose([uBatch, iBatch, gBatch]);
      await tf.nextFrame();
    }
    logStatus(`Epoch ${epoch + 1}/${trainCfg.epochs} – avg loss: ${(epochLoss / stepsPerEpoch).toFixed(4)}`);
  }

  logStatus("✅ Training finished.");
  await drawPCA();
  ui.testBtn.disabled = false;
}

/* ---------------------- PCA Visualization ---------------------- */
async function drawPCA() {
  const canvas = ui.pcaCanvas;
  const ctx = canvas.getContext("2d");
  const numItems = dataState.itemIds.length;
  const sampleSize = Math.min(1000, numItems);
  const step = Math.floor(numItems / sampleSize) || 1;
  const idxs = [];
  for (let i = 0; i < numItems; i += step) idxs.push(i);
  const idxTensor = tf.tensor1d(new Int32Array(idxs), "int32");
  const genreBatch = tf.gather(globalItemGenreTensor, idxTensor);

  const emb = await tf.tidy(() => model.itemForward(idxTensor, genreBatch).array());
  const proj = await pca2D(tf.tensor2d(emb));
  const points = await proj.array();
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs),
    maxX = Math.max(...xs),
    minY = Math.min(...ys),
    maxY = Math.max(...ys);
  const norm = (v, a, b, w) => ((v - a) / (b - a)) * w;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0b1229";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#93c5fd88";

  for (let k = 0; k < points.length; k++) {
    const x = norm(xs[k], minX, maxX, canvas.width);
    const y = canvas.height - norm(ys[k], minY, maxY, canvas.height);
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  tf.dispose([idxTensor, genreBatch, proj]);
}

/* ---------------------- Testing ---------------------- */
async function testOnce() {
  if (!model) return alert("Train the model first!");
  const { userToItems, userIndex, itemIndex, itemIds, items } = dataState;
  const candidates = Array.from(userToItems.entries()).filter(([, arr]) => arr.length >= 20);
  const [userId, history] = candidates[Math.floor(Math.random() * candidates.length)];
  const historySorted = history.slice().sort((a, b) => b.rating - a.rating || b.ts - a.ts).slice(0, 10);
  const historyTitles = historySorted.map((x) => items.get(x.itemId).title);
  const exclude = new Set(history.map((x) => itemIndex.get(x.itemId)));

  const uIdx = tf.tensor1d([userIndex.get(userId)], "int32");

  // Deep tower
  const deepScores = await tf.tidy(() => {
    const uEmb = model.userForward(uIdx);
    const B = 1024;
    const total = itemIds.length;
    const out = new Float32Array(total);
    for (let s = 0, p = 0; s < total; s += B) {
      const end = Math.min(total, s + B);
      const idx = tf.tensor1d(new Int32Array(Array.from({ length: end - s }, (_, k) => s + k)), "int32");
      const g = tf.gather(globalItemGenreTensor, idx);
      const iEmb = model.itemForward(idx, g);
      const logits = tf.matMul(iEmb, uEmb.transpose()).reshape([end - s]);
      const arr = logits.dataSync();
      for (let k = 0; k < arr.length; k++) out[p++] = arr[k];
      tf.dispose([idx, g, iEmb, logits]);
    }
    return out;
  });

  // Baseline (no-MLP)
  const baseScores = await tf.tidy(() => {
    const uRaw = tf.gather(model.userEmbedding, uIdx);
    const iRaw = model.itemEmbedding;
    const logits = tf.matMul(iRaw, uRaw.transpose()).reshape([iRaw.shape[0]]);
    return logits.dataSync();
  });

  const k = 10;
  const deepIdxs = topKIndices(deepScores, k, exclude);
  const baseIdxs = topKIndices(baseScores, k, exclude);
  const deepTitles = deepIdxs.map((i) => items.get(itemIds[i]).title);
  const baseTitles = baseIdxs.map((i) => items.get(itemIds[i]).title);

  renderResults(userId, historyTitles, baseTitles, deepTitles);
  logStatus(`Tested user ${userId}. History vs Baseline vs Deep rendered.`);
}

/* ---------------------- Reset ---------------------- */
function resetAll() {
  if (model) model = null;
  if (ui.lossCanvas) {
    const ctx = ui.lossCanvas.getContext("2d");
    ctx.clearRect(0, 0, ui.lossCanvas.width, ui.lossCanvas.height);
  }
  if (ui.results) ui.results.innerHTML = "";
  clearStatus("Reset complete. You can Train again.");
}

/* ---------------------- Wire-up ---------------------- */
window.addEventListener("DOMContentLoaded", () => {
  ui = {
    loadBtn: document.getElementById("loadBtn"),
    trainBtn: document.getElementById("trainBtn"),
    testBtn: document.getElementById("testBtn"),
    status: document.getElementById("status"),
    lossCanvas: document.getElementById("lossCanvas"),
    pcaCanvas: document.getElementById("pcaCanvas"),
    results: document.getElementById("results"),
    countsPill: document.getElementById("countsPill"),
  };

  // Add Reset button dynamically
  const resetBtn = document.createElement("button");
  resetBtn.textContent = "Reset";
  resetBtn.onclick = resetAll;
  ui.loadBtn.parentElement.appendChild(resetBtn);

  ui.loadBtn.onclick = () => loadData().catch((e) => logStatus("Error: " + e.message));
  ui.trainBtn.onclick = () => train().catch((e) => logStatus("Error: " + e.message));
  ui.testBtn.onclick = () => testOnce().catch((e) => logStatus("Error: " + e.message));
});
