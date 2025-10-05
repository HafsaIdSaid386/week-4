// app.js
class MovieLensApp {
  constructor() {
    // Data structures
    this.interactions = [];                 // [{userId,itemId,rating,ts}]
    this.items = new Map();                 // itemId -> {title, year}
    this.userMap = new Map();               // raw userId -> 0-based idx
    this.itemMap = new Map();               // raw itemId -> 0-based idx
    this.reverseUserMap = new Map();        // idx -> raw userId
    this.reverseItemMap = new Map();        // idx -> raw itemId
    this.userTopRated = new Map();          // userId -> sorted interactions
    this.qualifiedUsers = [];               // users with >=20 ratings
    this.model = null;

    // Config (matches prof)
    this.config = {
      maxInteractions: 80000,
      embeddingDim: 32,
      batchSize: 512,
      epochs: 20,
      learningRate: 0.001
    };

    this.lossHistory = [];
    this.wireUI();
  }

  wireUI() {
    document.getElementById('loadData').addEventListener('click', () => this.loadData());
    document.getElementById('train').addEventListener('click', () => this.train());
    document.getElementById('test').addEventListener('click', () => this.test());
    this.updateStatus('Click "Load Data" to start');
  }

  updateStatus(msg) {
    document.getElementById('status').textContent = msg;
  }

  // ------------------ LOAD DATA ------------------
  async loadData() {
    this.updateStatus('Loading data…');
    try {
      // u.data
      const udata = await fetch('data/u.data').then(r => {
        if (!r.ok) throw new Error(`u.data fetch failed: ${r.status}`);
        return r.text();
      });
      const lines = udata.trim().split('\n').slice(0, this.config.maxInteractions);
      this.interactions = lines.map(l => {
        const [u, i, r, t] = l.split('\t');
        return { userId: +u, itemId: +i, rating: +r, ts: +t };
      });

      // u.item (only id|title|…; year from title if present)
      const uitem = await fetch('data/u.item').then(r => {
        if (!r.ok) throw new Error(`u.item fetch failed: ${r.status}`);
        return r.text();
      });
      uitem.trim().split('\n').forEach(l => {
        const parts = l.split('|');
        const id = +parts[0];
        const title = parts[1] || '';
        const yearMatch = title.match(/\((\d{4})\)$/);
        const year = yearMatch ? +yearMatch[1] : 'N/A';
        this.items.set(id, { title: title.replace(/\(\d{4}\)$/, '').trim(), year });
      });

      this.buildIndexersAndTopRated();
      this.findQualifiedUsers();

      this.updateStatus(
        `Loaded ${this.interactions.length} interactions and ${this.items.size} items. ` +
        `${this.qualifiedUsers.length} users have 20+ ratings.`
      );
      document.getElementById('train').disabled = false;
    } catch (err) {
      this.updateStatus(
        `Load failed: ${err.message}. ` +
        `If you're opening file://, host via GitHub Pages or local server.`
      );
    }
  }

  buildIndexersAndTopRated() {
    // indexers
    const users = [...new Set(this.interactions.map(x => x.userId))];
    const itemIds = [...new Set(this.interactions.map(x => x.itemId))];
    users.forEach((u, i) => { this.userMap.set(u, i); this.reverseUserMap.set(i, u); });
    itemIds.forEach((it, j) => { this.itemMap.set(it, j); this.reverseItemMap.set(j, it); });

    // group per user & sort (rating desc, recency desc)
    const perUser = new Map();
    this.interactions.forEach(rec => {
      if (!perUser.has(rec.userId)) perUser.set(rec.userId, []);
      perUser.get(rec.userId).push(rec);
    });
    perUser.forEach(arr => arr.sort((a, b) => (b.rating - a.rating) || (b.ts - a.ts)));
    this.userTopRated = perUser;
  }

  findQualifiedUsers() {
    this.qualifiedUsers = [...this.userTopRated.keys()].filter(u => this.userTopRated.get(u).length >= 20);
  }

  // ------------------ TRAIN ------------------
  async train() {
    if (!this.interactions.length) return this.updateStatus('Load data first.');
    this.model = new TwoTowerModel(this.userMap.size, this.itemMap.size, this.config.embeddingDim);
    this.lossHistory = [];
    document.getElementById('test').disabled = true;
    this.updateStatus('Training…');

    const userIdx = this.interactions.map(x => this.userMap.get(x.userId));
    const itemIdx = this.interactions.map(x => this.itemMap.get(x.itemId));
    const opt = tf.train.adam(this.config.learningRate);
    const batches = Math.ceil(userIdx.length / this.config.batchSize);
    const ctx = document.getElementById('lossChart').getContext('2d');

    for (let e = 0; e < this.config.epochs; e++) {
      let epochLoss = 0;
      // Shuffle each epoch
      const order = tf.util.createShuffledIndices(userIdx.length);
      const uShuf = order.map(i => userIdx[i]);
      const iShuf = order.map(i => itemIdx[i]);

      for (let b = 0; b < batches; b++) {
        const s = b * this.config.batchSize;
        const eidx = Math.min(s + this.config.batchSize, uShuf.length);
        const loss = await this.model.trainStep(uShuf.slice(s, eidx), iShuf.slice(s, eidx), opt);
        epochLoss += loss;
        this.lossHistory.push(loss);
        this.drawLoss(ctx, this.lossHistory);
        if (b % 10 === 0) this.updateStatus(`Epoch ${e + 1}/${this.config.epochs} • Batch ${b}/${batches} • Loss ${loss.toFixed(4)}`);
        await tf.nextFrame();
      }
      this.updateStatus(`Epoch ${e + 1} avg loss: ${(epochLoss / batches).toFixed(4)}`);
    }

    document.getElementById('test').disabled = false;
    this.updateStatus('Training complete. Click "Test".');
    await this.visualizeEmbeddings();
  }

  drawLoss(ctx, losses) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    if (!losses.length) return;
    const min = Math.min(...losses), max = Math.max(...losses);
    const range = Math.max(1e-8, max - min);
    ctx.beginPath();
    losses.forEach((l, i) => {
      const x = (i / (losses.length - 1)) * ctx.canvas.width;
      const y = ctx.canvas.height - ((l - min) / range) * ctx.canvas.height;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#0077cc';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // ------------------ PCA VIS ------------------
  async visualizeEmbeddings() {
    this.updateStatus('Computing PCA of item embeddings…');
    const canvas = document.getElementById('embeddingChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const allItemEmb = await this.model.getItemEmbeddings().array();
    const sample = allItemEmb.slice(0, Math.min(1000, allItemEmb.length));

    // PCA via SVD for broad TF.js compatibility
    const X = tf.tensor2d(sample);                     // [N, E]
    const Xm = X.sub(tf.mean(X, 0));                   // center
    const { v } = tf.svd(Xm);                          // right singular vectors [E,E]
    const top2 = v.slice([0, 0], [v.shape[0], 2]);     // first 2 PCs
    const proj = tf.matMul(Xm, top2);                  // [N,2]
    const points = await proj.array();

    // Normalize & draw
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    ctx.fillStyle = 'rgba(0,0,255,0.55)';
    points.forEach(([x, y]) => {
      const px = ((x - minX) / Math.max(1e-8, maxX - minX)) * (canvas.width - 20) + 10;
      const py = ((y - minY) / Math.max(1e-8, maxY - minY)) * (canvas.height - 20) + 10;
      ctx.fillRect(px, py, 2, 2);
    });

    tf.dispose([X, Xm, v, top2, proj]); // (u,s auto-disposed by GC)
    this.updateStatus('PCA projection drawn.');
  }

  // ------------------ TEST ------------------
  async test() {
    if (!this.model || !this.qualifiedUsers.length) return this.updateStatus('Train first.');
    this.updateStatus('Scoring…');

    const rawUser = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
    const uIdx0 = this.userMap.get(rawUser);
    const history = this.userTopRated.get(rawUser);

    // Deep model recs (MLP heads)
    const uDeep = this.model.getUserEmbedding(uIdx0);                  // [1,E]
    const scoresDeep = await this.model.getScoresForAllItems(uDeep);   // [N]
    uDeep.dispose();

    // Baseline recs (raw embeddings, no MLP)
    const uRaw = this.model.getRawUserEmbedding(uIdx0);                // [1,E]
    const scoresRaw = await this.model.getScoresRawForAllItems(uRaw);  // [N]
    uRaw.dispose();

    // Filter out already-rated
    const rated = new Set(history.map(x => x.itemId));
    const candidatesDeep = [];
    const candidatesRaw = [];
    scoresDeep.forEach((s, j) => {
      const itemId = this.reverseItemMap.get(j);
      if (!rated.has(itemId)) candidatesDeep.push({ itemId, score: s });
    });
    scoresRaw.forEach((s, j) => {
      const itemId = this.reverseItemMap.get(j);
      if (!rated.has(itemId)) candidatesRaw.push({ itemId, score: s });
    });
    candidatesDeep.sort((a, b) => b.score - a.score);
    candidatesRaw.sort((a, b) => b.score - a.score);

    const topRated = history.slice(0, 10);
    const topModel = candidatesRaw.slice(0, 10); // "Model" = baseline (no MLP)
    const topDeep = candidatesDeep.slice(0, 10); // Deep Learning (MLP)

    this.renderTables(rawUser, topRated, topModel, topDeep);
    this.updateStatus('Recommendations generated.');
  }

  renderTables(userId, topRated, topModel, topDeep) {
    const results = document.getElementById('results');
    const makeRows = (arr, isHist=false) => arr.map((r, i) => {
      const item = this.items.get(r.itemId);
      const left = item ? item.title : '(unknown)';
      const val = isHist ? r.rating.toFixed(1) : r.score.toFixed(4);
      const yr = item?.year ?? 'N/A';
      return `<tr><td>${i+1}</td><td>${left}</td><td>${val}</td><td>${yr}</td></tr>`;
    }).join('');

    const table = (title, body) => `
      <div>
        <h3>${title}</h3>
        <table>
          <thead><tr><th>#</th><th>Movie</th><th>${title.includes('Rated') ? 'Rating' : 'Score'}</th><th>Year</th></tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>`;

    results.innerHTML = `
      <h2>User ${userId}</h2>
      <div class="side-by-side">
        ${table('Top 10 Rated (History)', makeRows(topRated, true))}
        ${table('Model Recommendations (Baseline)', makeRows(topModel))}
        ${table('Deep Learning Recommendations (MLP)', makeRows(topDeep))}
      </div>`;
  }
}

// Boot
let app;
document.addEventListener('DOMContentLoaded', () => { app = new MovieLensApp(); });
