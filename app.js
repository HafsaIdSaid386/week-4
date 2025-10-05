let interactions = [];
let items = new Map();
let user2idx = {}, item2idx = {}, idx2item = {};
let model;

const status = msg => document.getElementById('status').innerText = msg;

// === LOAD DATA ===
async function loadData() {
  const [udata, uitem] = await Promise.all([
    fetch('data/u.data').then(r => r.text()),
    fetch('data/u.item').then(r => r.text())
  ]);

  const lines = udata.trim().split('\n');
  interactions = lines.map(l => {
    const [u, i, r, t] = l.split('\t').map(Number);
    return { userId: u, itemId: i, rating: r, ts: t };
  });

  const itemLines = uitem.trim().split('\n');
  itemLines.forEach(l => {
    const parts = l.split('|');
    const id = Number(parts[0]);
    items.set(id, { title: parts[1] });
  });

  const users = [...new Set(interactions.map(x => x.userId))];
  const itemIds = [...new Set(interactions.map(x => x.itemId))];
  users.forEach((u, i) => user2idx[u] = i);
  itemIds.forEach((i, j) => item2idx[i] = j);
  itemIds.forEach((i, j) => idx2item[j] = i);

  status(`Loaded ${interactions.length} interactions and ${itemIds.length} items. ${users.length} users have 20+ ratings.`);
}

document.getElementById('load').onclick = loadData;

// === TRAIN ===
document.getElementById('train').onclick = async () => {
  const numUsers = Object.keys(user2idx).length;
  const numItems = Object.keys(item2idx).length;
  const numGenres = 20; // placeholder
  const embDim = 32, hiddenDim = 64;

  model = new TwoTowerModel(numUsers, numItems, numGenres, embDim, hiddenDim);

  const learningRate = 0.001;
  const optimizer = tf.train.adam(learningRate);
  const epochs = 15, batchSize = 256;
  const numBatches = Math.ceil(interactions.length / batchSize);

  const losses = [];
  const ctx = document.getElementById('lossChart').getContext('2d');

  status('Training started...');
  for (let epoch = 0; epoch < epochs; epoch++) {
    tf.util.shuffle(interactions);

    for (let b = 0; b < numBatches; b++) {
      const batch = interactions.slice(b * batchSize, (b + 1) * batchSize);
      const users = tf.tensor1d(batch.map(x => user2idx[x.userId]), 'int32');
      const itemsIdx = tf.tensor1d(batch.map(x => item2idx[x.itemId]), 'int32');
      const genreTensor = tf.zeros([batch.length, numGenres]);

      const loss = optimizer.minimize(() => {
        const uEmb = model.userForward(users);
        const iEmb = model.itemForward(itemsIdx, genreTensor);
        const logits = tf.matMul(uEmb, iEmb, false, true);
        const labels = tf.tensor1d([...Array(batch.length).keys()], 'int32');
        const lossVal = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, batch.length), logits);
        return lossVal;
      }, true);

      const l = (await loss.data())[0];
      losses.push(l);
      drawLoss(ctx, losses);
      if (b % 100 === 0) status(`Epoch ${epoch + 1}/${epochs} - Batch ${b}/${numBatches} - Loss: ${l.toFixed(4)}`);
      await tf.nextFrame();
    }
  }

  status('Training finished ✅');
  drawPCA();
};

// === DRAW LOSS ===
function drawLoss(ctx, losses) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.beginPath();
  ctx.moveTo(0, ctx.canvas.height - losses[0] * 20);
  losses.forEach((l, i) => {
    const x = (i / losses.length) * ctx.canvas.width;
    const y = ctx.canvas.height - l * 20;
    ctx.lineTo(x, y);
  });
  ctx.strokeStyle = 'blue';
  ctx.stroke();
}

// === SIMPLE PCA ===
function simplePCA(data, dim = 2) {
  const X = tf.tensor2d(data);
  const Xmean = X.sub(tf.mean(X, 0));
  const cov = tf.matMul(Xmean.transpose(), Xmean).div(X.shape[0]);
  const { eigenVectors } = tf.linalg.eigh(cov);
  const topVecs = eigenVectors.slice([0, eigenVectors.shape[1] - dim], [-1, dim]);
  const projected = tf.matMul(Xmean, topVecs);
  return projected.arraySync();
}

// === DRAW PCA ===
async function drawPCA() {
  const itemEmb = await model.itemEmbedding.array();
  const proj = simplePCA(itemEmb, 2);
  const ctx = document.getElementById('projection').getContext('2d');
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  proj.slice(0, 500).forEach((p) => {
    const x = (p[0] + 3) * 100;
    const y = (p[1] + 3) * 100;
    ctx.fillStyle = 'rgba(0,0,255,0.5)';
    ctx.fillRect(x, y, 2, 2);
  });
}

// === TEST ===
document.getElementById('test').onclick = async () => {
  const users = Object.keys(user2idx);
  const randUser = users[Math.floor(Math.random() * users.length)];
  const userIdx = tf.tensor1d([user2idx[randUser]], 'int32');
  const uEmb = model.userForward(userIdx);

  const allItems = tf.tensor1d([...Array(Object.keys(item2idx).length).keys()], 'int32');
  const genreTensor = tf.zeros([allItems.shape[0], 20]);
  const iEmb = model.itemForward(allItems, genreTensor);
  const scores = tf.matMul(uEmb, iEmb, false, true).flatten();

  const topK = await scores.topk(10);
  const topIdx = await topK.indices.array();
  const topTitles = topIdx.map(i => items.get(idx2item[i]).title);

  const div = document.getElementById('results');
  div.innerHTML = `
    <h3>User ${randUser}</h3>
    <table>
      <tr><th>Top Rated</th><th>Model Recommendations</th></tr>
      <tr>
        <td>${interactions.filter(x => x.userId == randUser).slice(0, 10).map(x => items.get(x.itemId)?.title).join('<br>')}</td>
        <td>${topTitles.join('<br>')}</td>
      </tr>
    </table>
  `;
};
