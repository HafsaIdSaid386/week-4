let model, interactions = [], items = new Map();
let userToItems = {}, userIndex = {}, itemIndex = {}, reverseItems = {};
let numUsers = 0, numItems = 0, numGenres = 1;
const embDim = 32, hiddenDim = 64;

const status = msg => document.getElementById('status').innerText = msg;

// Load MovieLens data
async function loadData() {
  status('Loading data...');
  const dataTxt = await fetch('data/u.data').then(r => r.text());
  const itemTxt = await fetch('data/u.item').then(r => r.text());

  dataTxt.trim().split('\n').forEach(line => {
    const [user, item, rating, ts] = line.split('\t').map(Number);
    interactions.push({ userId: user, itemId: item, rating, ts });
    if (!userToItems[user]) userToItems[user] = [];
    userToItems[user].push({ item, rating, ts });
  });

  itemTxt.trim().split('\n').forEach(line => {
    const parts = line.split('|');
    const itemId = Number(parts[0]);
    const title = parts[1];
    items.set(itemId, { title });
  });

  numUsers = Math.max(...interactions.map(i => i.userId)) + 1;
  numItems = Math.max(...interactions.map(i => i.itemId)) + 1;
  interactions = interactions.slice(0, 80000);

  let u = 0, it = 0;
  for (const uid of Object.keys(userToItems)) userIndex[uid] = u++;
  for (const iid of items.keys()) itemIndex[iid] = it++;
  reverseItems = Object.fromEntries(Object.entries(itemIndex).map(([k, v]) => [v, items.get(Number(k)).title]));

  status('Data loaded. Users: ' + numUsers + ', Items: ' + numItems);
}

// Train Two-Tower model
async function trainModel() {
  model = new TwoTowerModel(numUsers, numItems, numGenres, embDim, hiddenDim);

  const learningRate = 0.01;
  const optimizer = tf.train.adam(learningRate);
  const epochs = 15, batchSize = 256;
  const losses = [];

  const lossCanvas = document.getElementById('lossChart');
  const ctx = lossCanvas.getContext('2d');

  status('Training started...');
  for (let epoch = 0; epoch < epochs; epoch++) {
    tf.util.shuffle(interactions);
    for (let i = 0; i < interactions.length; i += batchSize) {
      const batch = interactions.slice(i, i + batchSize);
      const userIdx = tf.tensor1d(batch.map(b => userIndex[b.userId]), 'int32');
      const itemIdx = tf.tensor1d(batch.map(b => itemIndex[b.itemId]), 'int32');

      const lossVal = optimizer.minimize(() => {
        const uEmb = model.userForward(userIdx);
        const iEmb = model.itemForward(itemIdx, tf.zeros([batch.length, numGenres]));
        const logits = tf.matMul(uEmb, iEmb, false, true);
        const labels = tf.oneHot(tf.range(0, batch.length, 1, 'int32'), batch.length);
        const loss = tf.losses.softmaxCrossEntropy(labels, logits);
        return loss;
      }, true);

      const val = (await lossVal.data())[0];
      losses.push(val);
      if (i % (batchSize * 4) === 0) {
        drawLoss(ctx, losses);
        status(`Epoch ${epoch + 1}/${epochs} - Batch ${i}/${interactions.length} - Loss: ${val.toFixed(4)}`);
        await tf.nextFrame();
      }
    }
  }

  status('Training complete!');
  await drawEmbeddingProjection(model.itemEmbedding);
}

// Draw loss curve
function drawLoss(ctx, losses) {
  ctx.clearRect(0, 0, 500, 300);
  ctx.beginPath();
  ctx.moveTo(0, 300 - losses[0] * 20);
  for (let i = 1; i < losses.length; i++) {
    const x = (i / losses.length) * 500;
    const y = 300 - losses[i] * 20;
    ctx.lineTo(x, y);
  }
  ctx.strokeStyle = 'blue';
  ctx.stroke();
}

// PCA projection visualization
async function drawEmbeddingProjection(itemEmbedding) {
  const emb = await itemEmbedding.array();
  const sample = emb.slice(0, 1000);
  const mean = tf.mean(sample, 0);
  const centered = tf.sub(sample, mean);
  const cov = tf.matMul(centered.transpose(), centered);
  const { u } = tf.linalg.svd(cov);
  const pc = tf.matMul(centered, u.slice([0, 0], [centered.shape[1], 2]));
  const points = await pc.array();

  const canvas = document.getElementById('embeddingChart');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'steelblue';
  for (const [x, y] of points) {
    ctx.fillRect(canvas.width / 2 + x * 10, canvas.height / 2 + y * 10, 2, 2);
  }
  status('PCA projection drawn.');
}

// Test model
async function testModel() {
  const users = Object.keys(userToItems).filter(u => userToItems[u].length >= 20);
  const randUser = users[Math.floor(Math.random() * users.length)];
  const userIdx = userIndex[randUser];
  const topRated = userToItems[randUser]
    .sort((a, b) => b.rating - a.rating)
    .slice(0, 10)
    .map(x => items.get(x.item).title);

  const userEmb = model.userForward(tf.tensor1d([userIdx], 'int32'));
  const allItemsTensor = tf.range(0, numItems, 1, 'int32');
  const itemEmbeddings = model.itemForward(allItemsTensor, tf.zeros([numItems, numGenres]));
  const scores = tf.matMul(userEmb, itemEmbeddings.transpose());
  const topIndices = tf.topk(scores, 10).indices.dataSync();
  const deepRecs = Array.from(topIndices).map(i => reverseItems[i]);

  renderResults(randUser, topRated, deepRecs, deepRecs);
}

// Show result tables
function renderResults(userId, topRated, modelRecs, deepRecs) {
  const tableHTML = `
    <h3>User ${userId}</h3>
    <table>
      <tr><th>Top Rated</th><th>Model Recommendations</th><th>Deep Learning Recommendations</th></tr>
      ${Array.from({ length: 10 }).map((_, i) => `
        <tr>
          <td>${topRated[i] || ''}</td>
          <td>${modelRecs[i] || ''}</td>
          <td>${deepRecs[i] || ''}</td>
        </tr>`).join('')}
    </table>`;
  document.getElementById('results').innerHTML = tableHTML;
}
