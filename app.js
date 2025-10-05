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
      epochs: 15,
      learningRate: 0.001
    };

    this.lossHistory = [];
    this.initializeUI();
  }

  initializeUI() {
    document.getElementById("loadData").addEventListener("click", () => this.loadData());
    document.getElementById("train").addEventListener("click", () => this.train());
    document.getElementById("test").addEventListener("click", () => this.test());
  }

  updateStatus(msg) {
    document.getElementById("status").textContent = msg;
  }

  async loadData() {
    this.updateStatus("Loading data...");
    const dataTxt = await fetch("data/u.data").then(r => r.text());
    const lines = dataTxt.trim().split("\n").slice(0, this.config.maxInteractions);
    this.interactions = lines.map(line => {
      const [u, i, r, t] = line.split("\t").map(Number);
      return { userId: u, itemId: i, rating: r, ts: t };
    });

    const itemTxt = await fetch("data/u.item").then(r => r.text());
    itemTxt.trim().split("\n").forEach(line => {
      const [id, title] = line.split("|");
      const year = title.match(/\((\\d{4})\\)/) ? parseInt(title.match(/\((\\d{4})\\)/)[1]) : "N/A";
      this.items.set(parseInt(id), { title: title.replace(/\(\d{4}\)/, "").trim(), year });
    });

    this.createMappings();
    this.findQualifiedUsers();
    this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} items.`);
    document.getElementById("train").disabled = false;
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
    this.updateStatus("Training...");
    this.model = new TwoTowerModel(this.userMap.size, this.itemMap.size, this.config.embeddingDim);
    const userIdx = this.interactions.map(i => this.userMap.get(i.userId));
    const itemIdx = this.interactions.map(i => this.itemMap.get(i.itemId));
    const opt = tf.train.adam(this.config.learningRate);
    const batches = Math.ceil(userIdx.length / this.config.batchSize);

    this.lossHistory = [];
    for (let e = 0; e < this.config.epochs; e++) {
      let epochLoss = 0;
      for (let b = 0; b < batches; b++) {
        const s = b * this.config.batchSize;
        const eidx = Math.min(s + this.config.batchSize, userIdx.length);
        const loss = await this.model.trainStep(userIdx.slice(s, eidx), itemIdx.slice(s, eidx), opt);
        epochLoss += loss;
        this.lossHistory.push(loss);
        this.updateLossChart();
        if (b % 10 === 0) this.updateStatus(`Epoch ${e + 1}/${this.config.epochs}, Batch ${b}/${batches}, Loss=${loss.toFixed(4)}`);
        await tf.nextFrame();
      }
      this.updateStatus(`Epoch ${e + 1} avg loss: ${(epochLoss / batches).toFixed(4)}`);
    }

    this.updateStatus("Training done!");
    document.getElementById("test").disabled = false;
    this.visualizeEmbeddings();
  }

  updateLossChart() {
    const c = document.getElementById("lossChart"), ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    if (!this.lossHistory.length) return;
    const max = Math.max(...this.lossHistory), min = Math.min(...this.lossHistory);
    ctx.beginPath();
    this.lossHistory.forEach((l, i) => {
      const x = (i / this.lossHistory.length) * c.width;
      const y = c.height - ((l - min) / (max - min)) * c.height;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  async visualizeEmbeddings() {
    const c = document.getElementById("embeddingChart"), ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const emb = await this.model.getItemEmbeddings().array();
    const proj = this.computePCA(emb.slice(0, 500), 2);
    const xs = proj.map(p => p[0]), ys = proj.map(p => p[1]);
    const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
    const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
    ctx.fillStyle = "rgba(0,0,255,0.6)";
    proj.forEach(([x, y]) => {
      const px = ((x - minX) / (maxX - minX)) * (c.width - 20) + 10;
      const py = ((y - minY) / (maxY - minY)) * (c.height - 20) + 10;
      ctx.fillRect(px, py, 3, 3);
    });
  }

  computePCA(data, k = 2) {
    const X = tf.tensor2d(data);
    const Xm = X.sub(tf.mean(X, 0));
    const cov = tf.matMul(Xm.transpose(), Xm).div(X.shape[0]);
    const { eigenVectors } = tf.linalg.eigh(cov);
    const top = eigenVectors.slice([0, eigenVectors.shape[1] - k], [-1, k]);
    return tf.matMul(Xm, top).arraySync();
  }

  async test() {
    const user = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
    const userIdx = this.userMap.get(user);
    const userInter = this.userTopRated.get(user);
    const userEmb = this.model.getUserEmbedding(userIdx);
    const scores = await this.model.getScoresForAllItems(userEmb);
    const rated = new Set(userInter.map(i => i.itemId));
    const candidates = [];
    scores.forEach((s, i) => {
      const itemId = this.reverseItemMap.get(i);
      if (!rated.has(itemId)) candidates.push({ itemId, score: s });
    });
    candidates.sort((a, b) => b.score - a.score);
    const topRecs = candidates.slice(0, 10);
    const deepRecs = candidates.slice(10, 20);
    this.displayResults(user, userInter, topRecs, deepRecs);
  }

  displayResults(userId, userInteractions, recs, deepRecs) {
    const div = document.getElementById("results");
    const topRated = userInteractions.slice(0, 10);
    const makeTable = (title, arr, col) => `
      <div><h3>${title}</h3>
      <table><thead><tr>${col.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>
      ${arr.map((r, i) => {
        const item = this.items.get(r.itemId);
        return `<tr><td>${i + 1}</td><td>${item?.title || ""}</td>
                <td>${r.rating?.toFixed ? r.rating.toFixed(1) : r.score.toFixed(4)}</td>
                <td>${item?.year || "N/A"}</td></tr>`;
      }).join("")}
      </tbody></table></div>`;
    div.innerHTML = `
      <h2>User ${userId}</h2>
      <div class="side-by-side">
        ${makeTable("Top 10 Rated", topRated, ["#", "Movie", "Rating", "Year"])}
        ${makeTable("Model Recommendations", recs, ["#", "Movie", "Score", "Year"])}
        ${makeTable("Deep Learning Recommendations", deepRecs, ["#", "Movie", "Score", "Year"])}
      </div>`;
  }
}

let app;
document.addEventListener("DOMContentLoaded", () => app = new MovieLensApp());
