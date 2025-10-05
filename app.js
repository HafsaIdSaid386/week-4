// ===========================================================
// app.js - Data loading, training, testing, and visualization
// ===========================================================

let interactions = [];
let items = new Map();
let users = new Set();
let model;
let user2idx = new Map(), item2idx = new Map();
let idx2user = [], idx2item = [];
let userRatings = new Map();
const embDim = 32, hiddenDim = 64;

// DOM elements
const statusEl = document.getElementById('status');
const ctxLoss = document.getElementById('lossChart').getContext('2d');
const ctxProj = document.getElementById('embeddingPlot').getContext('2d');
const resultsDiv = document.getElementById('results');

// Utility
function log(msg) { console.log(msg); statusEl.textContent = 'Status: ' + msg; }

// ===========================================================
// 1. Load MovieLens data
// ===========================================================
async function loadData() {
  log('Loading MovieLens data...');
  const [uData, uItem] = await Promise.all([
    fetch('data/u.data').then(r => r.text()),
    fetch('data/u.item').then(r => r.text())
  ]);

  // Parse u.item
  const itemLines = uItem.trim().split('\n');
  for (const line of itemLines) {
    const parts = line.split('|');
    const itemId = parts[0];
    const title = parts[1];
    const yearMatch = title.match(/\((\d{4})\)/);
    const year = yearMatch ? yearMatch[1] : 'NA';
    items.set(itemId, { title, year });
  }

  // Parse u.data
  const lines = uData.trim().split('\n');
  for (const line of lines) {
    const [u, i, r, t] = line.split('\t');
    interactions.push({ userId: u, itemId: i, rating: Number(r), ts: Number(t) });
    users.add(u);
  }

  // Indexers
  idx2user = [...users];
  idx2item = [...items.keys()];
  idx2user.forEach((u, i) => user2idx.set(u, i));
  idx2item.forEach((i, j) => item2idx.set(i, j));

  // User → rated items map
  for (const { userId, itemId, rating } of interactions) {
    if (!userRatings.has(userId)) userRatings.set(userId, []);
    userRatings.get(userId).push({ itemId, rating });
  }

  log(`Loaded ${interactions.length} interactions, ${users.size} users, ${items.size} items.`);
}

// ===========================================================
// 2. Training pipeline
// ===========================================================
async function trainModel() {
  if (!interactions.length) return log('Load data first.');
  log('Initializing model...');
  const numUsers = users.size;
  const numItems = items.size;
  const numGenres = 1; // placeholder (not used in this demo)
  model = new TwoTowerModel(numUsers, numItems, numGenres, embDim, hiddenDim);
  const optimizer = tf.train.adam(0.005);

  const epochs =10, batchSize = 256;
  const numBatches = Math.ceil(interactions.length / batchSize);
  const losses = [];

  log('Training started...');
  for (let epoch = 0; epoch < epochs; epoch++) {
    tf.util.shuffle(interactions);
    for (let b = 0; b < numBatches; b++) {
      const batch = interactions.slice(b * batchSize, (b + 1) * batchSize);
      const userIdx = batch.map(x => user2idx.get(x.userId));
      const itemIdx = batch.map(x => item2idx.get(x.itemId));

      const uTensor = tf.tensor1d(userIdx, 'int32');
      const iTensor = tf.tensor1d(itemIdx, 'int32');

      const lossVal = optimizer.minimize(() => {
        const uEmb = model.userForward(uTensor);
        const iEmb = model.itemForward(iTensor, tf.zeros([batch.length, numGenres]));
        const logits = tf.matMul(uEmb, iEmb, false, true);
        const labels = tf.range(0, batch.length, 1, 'int32');
        const loss = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, batch.length), logits);
        return loss;
      }, true).dataSync()[0];

      losses.push(lossVal);
      drawLoss(losses);
      if (b % 10 === 0) log(`Epoch ${epoch + 1}/${epochs} - Batch ${b}/${numBatches} - Loss: ${lossVal.toFixed(4)}`);
      await tf.nextFrame();
    }
  }

  log('Training complete. Visualizing embeddings...');
  visualizeEmbeddings();
}

// ===========================================================
// 3. Visualization (loss chart + PCA projection)
// ===========================================================
function drawLoss(losses) {
  const w = ctxLoss.canvas.width, h = ctxLoss.canvas.height;
  ctxLoss.clearRect(0, 0, w, h);
  ctxLoss.beginPath();
  ctxLoss.moveTo(0, h - losses[0]);
  for (let i = 1; i < losses.length; i++) {
    const x = (i / losses.length) * w;
    const y = h - (losses[i] * 10);
    ctxLoss.lineTo(x, y);
  }
  ctxLoss.strokeStyle = '#0077cc';
  ctxLoss.stroke();
}

// Simple PCA for 2D projection
async function visualizeEmbeddings() {
  const numSample = 300;
  const allEmb = model.itemEmbedding.slice([0, 0], [numSample, embDim]).arraySync();
  const mean = Array(embDim).fill(0);
  for (const v of allEmb) for (let i = 0; i < embDim; i++) mean[i] += v[i];
  for (let i = 0; i < embDim; i++) mean[i] /= numSample;
  const centered = allEmb.map(v => v.map((x, i) => x - mean[i]));

  const cov = tf.matMul(tf.tensor2d(centered).transpose(), tf.tensor2d(centered)).div(numSample);
  const { eigenValues, eigenVectors } = tf.linalg.eigh(cov);
  const top2 = eigenVectors.slice([0, embDim - 2], [embDim, 2]).arraySync();

  const proj = centered.map(v => [
    v.reduce((s, x, i) => s + x * top2[i][0], 0),
    v.reduce((s, x, i) => s + x * top2[i][1], 0)
  ]);

  ctxProj.clearRect(0, 0, ctxProj.canvas.width, ctxProj.canvas.height);
  for (let i = 0; i < proj.length; i++) {
    const x = 300 + proj[i][0] * 20;
    const y = 200 - proj[i][1] * 20;
    ctxProj.fillRect(x, y, 2, 2);
  }
  log('Embedding projection drawn.');
}

// ===========================================================
// 4. Test (recommendations)
// ===========================================================
function testModel() {
  if (!model) return log('Train model first.');
  const validUsers = [...userRatings.keys()].filter(u => userRatings.get(u).length >= 20);
  const userId = validUsers[Math.floor(Math.random() * validUsers.length)];
  const uIdx = tf.tensor1d([user2idx.get(userId)], 'int32');
  const userEmb = model.userForward(uIdx);

  const allItems = tf.range(0, items.size, 1, 'int32');
  const itemEmb = model.itemForward(allItems, tf.zeros([items.size, 1]));
  const scores = model.score(userEmb, itemEmb).flatten().arraySync();

  const rated = new Set(userRatings.get(userId).map(x => x.itemId));
  const sorted = scores
    .map((s, i) => ({ itemId: idx2item[i], score: s }))
    .filter(x => !rated.has(x.itemId))
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  renderResults(userId, sorted);
}

function renderResults(userId, recs) {
  const hist = userRatings.get(userId)
    .sort((a, b) => b.rating - a.rating)
    .slice(0, 10)
    .map(x => items.get(x.itemId)?.title || 'Unknown');

  const recTitles = recs.map(x => items.get(x.itemId)?.title || 'Unknown');
  let html = `<h3>User ${userId}</h3><table><tr><th>Top Rated</th><th>Model Recommendations</th></tr>`;
  for (let i = 0; i < 10; i++) html += `<tr><td>${hist[i] || ''}</td><td>${recTitles[i] || ''}</td></tr>`;
  html += '</table>';
  resultsDiv.innerHTML = html;
  log('Displayed Top-10 recommendations.');
}

// ===========================================================
// 5. Event listeners
// ===========================================================
document.getElementById('loadBtn').onclick = loadData;
document.getElementById('trainBtn').onclick = trainModel;
document.getElementById('testBtn').onclick = testModel;
