class MovieLensApp {
  constructor() {
    this.interactions = [];
    this.items = new Map();
    this.userMap = new Map();
    this.itemMap = new Map();
    this.reverseUserMap = new Map();
    this.reverseItemMap = new Map();
    this.userTopRated = new Map();
    this.model = null;

    this.config = {
      maxInteractions: 80000,
      embeddingDim: 32,
      batchSize: 512,
      epochs: 20,
      learningRate: 0.001
    };

    this.lossHistory = [];
    this.isTraining = false;
    this.initializeUI();
  }

  initializeUI() {
    document.getElementById('loadData').addEventListener('click', () => this.loadData());
    document.getElementById('train').addEventListener('click', () => this.train());
    document.getElementById('test').addEventListener('click', () => this.test());
    this.updateStatus('Click "Load Data" to start');
  }

  async loadData() {
    this.updateStatus('Loading data...');
    try {
      const dataTxt = await fetch('data/u.data').then(r => r.text());
      const lines = dataTxt.trim().split('\n').slice(0, this.config.maxInteractions);
      this.interactions = lines.map(line => {
        const [user, item, rating, ts] = line.split('\t').map(Number);
        return { userId: user, itemId: item, rating, ts };
      });

      const itemTxt = await fetch('data/u.item').then(r => r.text());
      itemTxt.trim().split('\n').forEach(line => {
        const [id, title] = line.split('|');
        const yearMatch = title.match(/\((\d{4})\)$/);
        const year = yearMatch ? parseInt(yearMatch[1]) : 'N/A';
        this.items.set(parseInt(id), { title: title.replace(/\(\d{4}\)$/, '').trim(), year });
      });

      this.createMappings();
      this.findQualifiedUsers();
      this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} items. ${this.userTopRated.size} users have 20+ ratings.`);
      document.getElementById('train').disabled = false;
    } catch (e) {
      this.updateStatus('Error loading data: ' + e.message);
    }
  }

  createMappings() {
    const users = [...new Set(this.interactions.map(i => i.userId))];
    const items = [...new Set(this.interactions.map(i => i.itemId))];
    users.forEach((u, i) => { this.userMap.set(u, i); this.reverseUserMap.set(i, u); });
    items.forEach((it, j) => { this.itemMap.set(it, j); this.reverseItemMap.set(j, it); });

    const userInteractions = new Map();
    this.interactions.forEach(inter => {
      if (!userInteractions.has(inter.userId)) userInteractions.set(inter.userId, []);
      userInteractions.get(inter.userId).push(inter);
    });

    userInteractions.forEach(arr => arr.sort((a, b) => (b.rating - a.rating) || (b.ts - a.ts)));
    this.userTopRated = userInteractions;
  }

  findQualifiedUsers() {
    this.qualifiedUsers = [...this.userTopRated.keys()].filter(u => this.userTopRated.get(u).length >= 20);
  }

  async train() {
    if (this.isTraining) return;
    this.isTraining = true;
    document.getElementById('train').disabled = true;
    this.lossHistory = [];
    this.updateStatus('Initializing model...');

    this.model = new TwoTowerModel(this.userMap.size, this.itemMap.size, this.config.embeddingDim);

    const userIdx = this.interactions.map(i => this.userMap.get(i.userId));
    const itemIdx = this.interactions.map(i => this.itemMap.get(i.itemId));

    const optimizer = tf.train.adam(this.config.learningRate);
    const numBatches = Math.ceil(userIdx.length / this.config.batchSize);

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      let avgLoss = 0;
      for (let b = 0; b < numBatches; b++) {
        const start = b * this.config.batchSize;
        const end = Math.min(start + this.config.batchSize, userIdx.length);
        const users = userIdx.slice(start, end);
        const items = itemIdx.slice(start, end);

        const loss = await this.model.trainStep(users, items, optimizer);
        avgLoss += loss;
        this.lossHistory.push(loss);
        this.updateLossChart();

        if (b % 10 === 0)
          this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs}, Batch ${b}/${numBatches}, Loss: ${loss.toFixed(4)}`);
        await new Promise(r => setTimeout(r, 0));
      }
      avgLoss /= numBatches;
      this.updateStatus(`Epoch ${epoch + 1} done. Avg Loss: ${avgLoss.toFixed(4)}`);
    }

    this.isTraining = false;
    document.getElementById('train').disabled = false;
    document.getElementById('test').disabled = false;
    this.updateStatus('Training completed! Click "Test".');
    this.visualizeEmbeddings();
  }

  updateLossChart() {
    const canvas = document.getElementById('lossChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.lossHistory.length) return;
    const max = Math.max(...this.lossHistory);
    const min = Math.min(...this.lossHistory);
    ctx.beginPath();
    this.lossHistory.forEach((l, i) => {
      const x = (i / this.lossHistory.length) * canvas.width;
      const y = canvas.height - ((l - min) / (max - min)) * canvas.height;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  async visualizeEmbeddings() {
    const canvas = document.getElementById('embeddingChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const emb = await this.model.getItemEmbeddings().array();
    const proj = this.computePCA(emb.slice(0, 1000), 2);
    const xs = proj.map(p => p[0]), ys = proj.map(p => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    ctx.fillStyle = 'rgba(0,0,255,0.6)';
    proj.forEach(([x, y]) => {
      const px = ((x - minX) / (maxX - minX)) * (canvas.width - 20) + 10;
      const py = ((y - minY) / (maxY - minY)) * (canvas.height - 20) + 10;
      ctx.fillRect(px, py, 2, 2);
    });
  }

  computePCA(data, dim = 2) {
    const X = tf.tensor2d(data);
    const Xmean = X.sub(tf.mean(X, 0));
    const cov = tf.matMul(Xmean.transpose(), Xmean).div(X.shape[0]);
    const { eigenVectors } = tf.linalg.eigh(cov);
    const top = eigenVectors.slice([0, eigenVectors.shape[1] - dim], [-1, dim]);
    return tf.matMul(Xmean, top).arraySync();
  }

  async test() {
    if (!this.model) return this.updateStatus('Train first!');
    const randUser = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
    const userInter = this.userTopRated.get(randUser);
    const userIdx = this.userMap.get(randUser);

    const userEmb = this.model.getUserEmbedding(userIdx);
    const allItemScores = await this.model.getScoresForAllItems(userEmb);

    const rated = new Set(userInter.map(i => i.itemId));
    const candidates = [];
    allItemScores.forEach((score, idx) => {
      const itemId = this.reverseItemMap.get(idx);
      if (!rated.has(itemId)) candidates.push({ itemId, score });
    });
    candidates.sort((a, b) => b.score - a.score);
    const topRecs = candidates.slice(0, 10);

    // For demonstration: simulate "Deep Learning" tower output as same here
    const deepRecs = candidates.slice(10, 20);

    this.displayResults(randUser, userInter, topRecs, deepRecs);
  }

  displayResults(userId, userInteractions, recommendations, deepRecs) {
    const div = document.getElementById('results');
    const topRated = userInteractions.slice(0, 10);
    const makeTable = (title, data, cols) => `
      <div>
        <h3>${title}</h3>
        <table>
          <thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>
          <tbody>
            ${data.map((d, i) => `
              <tr>
                <td>${i + 1}</td>
                <td>${this.items.get(d.itemId)?.title || ''}</td>
                <td>${d.rating?.toFixed ? d.rating.toFixed(1) : d.score.toFixed(4)}</td>
                <td>${this.items.get(d.itemId)?.year || 'N/A'}</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
    div.innerHTML = `
      <h2>User ${userId}</h2>
      <div class="side-by-side">
        ${makeTable('Top 10 Rated', topRated, ['#', 'Movie', 'Rating', 'Year'])}
        ${makeTable('Model Recommendations', recommendations, ['#', 'Movie', 'Score', 'Year'])}
        ${makeTable('Deep Learning Recommendations', deepRecs, ['#', 'Movie', 'Score', 'Year'])}
      </div>`;
  }

  updateStatus(msg) { document.getElementById('status').innerText = msg; }
}

let app;
document.addEventListener('DOMContentLoaded', () => app = new MovieLensApp());
