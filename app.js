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
      embeddingDim: 32,
      batchSize: 512,
      epochs: 5,
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
      // --- Load interactions (u.data) ---
      const interactionsResponse = await fetch('data/u.data');
      if (!interactionsResponse.ok) throw new Error(`Cannot load data/u.data (${interactionsResponse.status})`);
      let interactionsText = await interactionsResponse.text();
      interactionsText = interactionsText.replace(/\r\n/g, '\n');
      const interactionsLines = interactionsText.trim().split('\n');

      this.interactions = interactionsLines.map(line => {
        // Prefer tabs, fallback to any whitespace
        let cols = line.split('\t');
        if (cols.length < 4) cols = line.trim().split(/\s+/);
        const [userId, itemId, rating, timestamp] = cols;
        return {
          userId: parseInt(userId, 10),
          itemId: parseInt(itemId, 10),
          rating: parseFloat(rating),
          timestamp: parseInt(timestamp, 10)
        };
      }).filter(r => Number.isFinite(r.userId) && Number.isFinite(r.itemId));

      // --- Load items (u.item) ---
      const itemsResponse = await fetch('data/u.item');
      if (!itemsResponse.ok) throw new Error(`Cannot load data/u.item (${itemsResponse.status})`);
      let itemsText = await itemsResponse.text();
      itemsText = itemsText.replace(/\r\n/g, '\n');
      const itemsLines = itemsText.trim().split('\n');

      itemsLines.forEach(line => {
        // Prefer pipes, fallback to tab, then comma
        let parts = line.split('|');
        if (parts.length < 2) parts = line.split('\t');
        if (parts.length < 2) parts = line.split(',');
        if (parts.length < 2) return;

        const itemId = parseInt(parts[0], 10);
        if (!Number.isFinite(itemId)) return;

        const titleRaw = (parts[1] || '').trim();
        const yearMatch = titleRaw.match(/\((\d{4})\)$/);
        const year = yearMatch ? parseInt(yearMatch[1], 10) : null;

        this.items.set(itemId, {
          title: titleRaw.replace(/\(\d{4}\)$/, '').trim(),
          year
        });
      });

      // Build maps
      this.createMappings();
      this.findQualifiedUsers();

      // Quick sanity log (visible in DevTools console)
      console.log('Interactions:', this.interactions.length);
      console.log('Items:', this.items.size);
      console.log('Sample item #1:', this.items.get(1));

      this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} items. ${this.userTopRated.size} users have 20+ ratings.`);
      document.getElementById('train').disabled = false;
    } catch (error) {
      this.updateStatus(`Error loading data: ${error.message}`);
    }
  }

  createMappings() {
    const userSet = new Set(this.interactions.map(i => i.userId));
    const itemSet = new Set(this.interactions.map(i => i.itemId));

    Array.from(userSet).forEach((userId, index) => {
      this.userMap.set(userId, index);
      this.reverseUserMap.set(index, userId);
    });

    Array.from(itemSet).forEach((itemId, index) => {
      this.itemMap.set(itemId, index);
      this.reverseItemMap.set(index, itemId);
    });

    const byUser = new Map();
    this.interactions.forEach(interaction => {
      const userId = interaction.userId;
      if (!byUser.has(userId)) byUser.set(userId, []);
      byUser.get(userId).push(interaction);
    });

    byUser.forEach(list => {
      list.sort((a, b) => (b.rating !== a.rating) ? (b.rating - a.rating) : (b.timestamp - a.timestamp));
    });

    this.userTopRated = byUser;
  }

  findQualifiedUsers() {
    const qualified = [];
    this.userTopRated.forEach((list, userId) => {
      if (list.length >= 20) qualified.push(userId);
    });
    this.qualifiedUsers = qualified;
  }

  async train() {
    if (this.isTraining) return;
    if (this.items.size === 0 || this.interactions.length === 0) {
      this.updateStatus('Load data first.');
      return;
    }

    this.isTraining = true;
    document.getElementById('train').disabled = true;
    this.lossHistory = [];

    this.updateStatus('Initializing model...');

    this.model = new TwoTowerModel(
      this.userMap.size,
      this.itemMap.size,
      this.config.embeddingDim,
      this.config.learningRate
    );

    const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
    const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));

    this.updateStatus('Starting training...');
    const numBatches = Math.ceil(userIndices.length / this.config.batchSize);

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      let epochLoss = 0;

      for (let batch = 0; batch < numBatches; batch++) {
        const start = batch * this.config.batchSize;
        const end = Math.min(start + this.config.batchSize, userIndices.length);

        const batchUsers = userIndices.slice(start, end);
        const batchItems = itemIndices.slice(start, end);

        try {
          const loss = await this.model.trainStep(batchUsers, batchItems);
          epochLoss += loss;
          this.lossHistory.push(loss);
          this.updateLossChart();

          if (batch % 10 === 0) {
            this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches}, Loss: ${loss.toFixed(4)}`);
          }
        } catch (err) {
          this.updateStatus(`Training error: ${err.message}`);
        }

        await new Promise(r => setTimeout(r, 0));
      }

      epochLoss /= numBatches;
      this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs} completed. Avg loss: ${epochLoss.toFixed(4)}`);
    }

    this.isTraining = false;
    document.getElementById('train').disabled = false;
    document.getElementById('test').disabled = false;

    this.updateStatus('Training completed! Click "Test" to see recommendations.');
    this.visualizeEmbeddings();
  }

  updateLossChart() {
    const canvas = document.getElementById('lossChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (this.lossHistory.length === 0) return;

    const maxLoss = Math.max(...this.lossHistory);
    const minLoss = Math.min(...this.lossHistory);
    const range = maxLoss - minLoss || 1;

    ctx.strokeStyle = '#007acc';
    ctx.lineWidth = 2;
    ctx.beginPath();

    this.lossHistory.forEach((loss, index) => {
      const x = (index / this.lossHistory.length) * canvas.width;
      const y = canvas.height - ((loss - minLoss) / range) * canvas.height;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });

    ctx.stroke();
  }

  async visualizeEmbeddings() {
    if (!this.model) return;
    this.updateStatus('Computing embedding visualization...');

    const canvas = document.getElementById('embeddingChart');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
      const sampleSize = Math.min(500, this.itemMap.size);
      const sampleIndices = Array.from({ length: sampleSize }, (_, i) =>
        Math.floor(i * this.itemMap.size / sampleSize)
      );

      const embeddingsTensor = this.model.getItemEmbeddings();
      const embeddings = embeddingsTensor.arraySync(); // [[D], ...]
      const sampleEmbeddings = sampleIndices.map(i => embeddings[i]);

      const projected = this.computePCA(sampleEmbeddings, 2);
      const xs = projected.map(p => p[0]);
      const ys = projected.map(p => p[1]);

      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      const xRange = xMax - xMin || 1;
      const yRange = yMax - yMin || 1;

      ctx.fillStyle = 'rgba(0, 122, 204, 0.6)';
      sampleIndices.forEach((_, i) => {
        const x = ((projected[i][0] - xMin) / xRange) * (canvas.width - 40) + 20;
        const y = ((projected[i][1] - yMin) / yRange) * (canvas.height - 40) + 20;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      });

      this.updateStatus('Embedding visualization completed.');
    } catch (error) {
      this.updateStatus(`Error in visualization: ${error.message}`);
    }
  }

  computePCA(embeddings, dimensions) {
    const n = embeddings.length;
    const dim = embeddings[0].length;

    const mean = Array(dim).fill(0);
    embeddings.forEach(emb => emb.forEach((v, i) => mean[i] += v));
    for (let i = 0; i < dim; i++) mean[i] /= n;

    const centered = embeddings.map(emb => emb.map((v, i) => v - mean[i]));

    const covariance = Array.from({ length: dim }, () => Array(dim).fill(0));
    centered.forEach(emb => {
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          covariance[i][j] += emb[i] * emb[j];
        }
      }
    });
    for (let i = 0; i < dim; i++) for (let j = 0; j < dim; j++) covariance[i][j] /= n;

    const components = [];
    for (let d = 0; d < dimensions; d++) {
      let v = Array(dim).fill(1 / Math.sqrt(dim));
      for (let it = 0; it < 10; it++) {
        const nv = Array(dim).fill(0);
        for (let i = 0; i < dim; i++) {
          for (let j = 0; j < dim; j++) nv[i] += covariance[i][j] * v[j];
        }
        const norm = Math.sqrt(nv.reduce((s, x) => s + x * x, 0)) || 1;
        v = nv.map(x => x / norm);
      }
      components.push(v);
      for (let i = 0; i < dim; i++) for (let j = 0; j < dim; j++) covariance[i][j] -= v[i] * v[j];
    }

    return embeddings.map(emb => components.map(comp => emb.reduce((s, val, i) => s + val * comp[i], 0)));
  }

  async test() {
    if (!this.model || this.qualifiedUsers.length === 0) {
      this.updateStatus('Model not trained or no qualified users found.');
      return;
    }

    this.updateStatus('Generating recommendations...');

    try {
      const randomUser = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
      const userInteractions = this.userTopRated.get(randomUser);
      const userIndex = this.userMap.get(randomUser);

      const userEmb = this.model.getUserEmbedding(userIndex);
      const allItemScores = await this.model.getScoresForAllItems(userEmb); // Float32Array

      const ratedItemIds = new Set(userInteractions.map(i => i.itemId));
      const candidateScores = [];

      allItemScores.forEach((score, itemIndex) => {
        const itemId = this.reverseItemMap.get(itemIndex);
        if (!ratedItemIds.has(itemId)) candidateScores.push({ itemId, score, itemIndex });
      });

      candidateScores.sort((a, b) => b.score - a.score);
      const topRecommendations = candidateScores.slice(0, 10);

      this.displayResults(randomUser, userInteractions, topRecommendations);
    } catch (error) {
      this.updateStatus(`Error generating recommendations: ${error.message}`);
    }
  }

  displayResults(userId, userInteractions, recommendations) {
    const resultsDiv = document.getElementById('results');
    const topRated = userInteractions.slice(0, 10);

    let html = `
      <h2>Recommendations for User ${userId}</h2>
      <div class="side-by-side">
        <div>
          <h3>Top 10 Rated Movies (Historical)</h3>
          <table>
            <thead><tr><th>Rank</th><th>Movie</th><th>Rating</th><th>Year</th></tr></thead>
            <tbody>
    `;

    topRated.forEach((interaction, idx) => {
      const item = this.items.get(interaction.itemId);
      html += `<tr><td>${idx + 1}</td><td>${item?.title ?? 'N/A'}</td><td>${interaction.rating}</td><td>${item?.year ?? 'N/A'}</td></tr>`;
    });

    html += `</tbody></table></div><div><h3>Top 10 Recommended Movies</h3><table><thead><tr><th>Rank</th><th>Movie</th><th>Score</th><th>Year</th></tr></thead><tbody>`;

    recommendations.forEach((rec, idx) => {
      const item = this.items.get(rec.itemId);
      html += `<tr><td>${idx + 1}</td><td>${item?.title ?? rec.itemId}</td><td>${rec.score.toFixed(4)}</td><td>${item?.year ?? 'N/A'}</td></tr>`;
    });

    html += `</tbody></table></div></div>`;
    resultsDiv.innerHTML = html;

    this.updateStatus('Recommendations generated successfully!');
  }

  updateStatus(message) {
    document.getElementById('status').textContent = message;
  }
}

let app;
document.addEventListener('DOMContentLoaded', () => {
  app = new MovieLensApp();
});
