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

    this.config = { embeddingDim: 32, batchSize: 512, epochs: 5, learningRate: 0.001 };

    this.lossHistory = [];
    this.isTraining = false;
    this.pageSize = 50; // for data explorer

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
      // --- Load interactions ---
      const interactionsResponse = await fetch('data/u.data');
      let interactionsText = await interactionsResponse.text();
      interactionsText = interactionsText.replace(/\r\n/g, '\n');
      const interactionsLines = interactionsText.trim().split('\n');
      this.interactions = interactionsLines.map(line => {
        let cols = line.split('\t');
        if (cols.length < 4) cols = line.trim().split(/\s+/);
        const [userId, itemId, rating, timestamp] = cols;
        return { userId: +userId, itemId: +itemId, rating: +rating, timestamp: +timestamp };
      }).filter(r => Number.isFinite(r.userId) && Number.isFinite(r.itemId));

      // --- Load items ---
      const itemsResponse = await fetch('data/u.item');
      let itemsText = await itemsResponse.text();
      itemsText = itemsText.replace(/\r\n/g, '\n');
      const itemsLines = itemsText.trim().split('\n');
      itemsLines.forEach(line => {
        let parts = line.split('|');
        if (parts.length < 2) parts = line.split('\t');
        if (parts.length < 2) parts = line.split(',');
        const itemId = parseInt(parts[0], 10);
        if (!Number.isFinite(itemId)) return;
        const titleRaw = (parts[1] || '').trim();
        const yearMatch = titleRaw.match(/\((\d{4})\)$/);
        const year = yearMatch ? parseInt(yearMatch[1], 10) : null;
        this.items.set(itemId, { title: titleRaw.replace(/\(\d{4}\)$/, '').trim(), year });
      });

      this.createMappings();
      this.findQualifiedUsers();

      console.log('Interactions:', this.interactions.length);
      console.log('Items:', this.items.size);

      this.updateStatus(`Loaded ${this.interactions.length} interactions and ${this.items.size} items. ${this.userTopRated.size} users have 20+ ratings.`);
      document.getElementById('train').disabled = false;
    } catch (error) {
      this.updateStatus(`Error loading data: ${error.message}`);
    }
  }

  createMappings() {
    const userSet = new Set(this.interactions.map(i => i.userId));
    const itemSet = new Set(this.interactions.map(i => i.itemId));
    Array.from(userSet).forEach((u, i) => { this.userMap.set(u, i); this.reverseUserMap.set(i, u); });
    Array.from(itemSet).forEach((it, i) => { this.itemMap.set(it, i); this.reverseItemMap.set(i, it); });
    const byUser = new Map();
    this.interactions.forEach(inter => {
      if (!byUser.has(inter.userId)) byUser.set(inter.userId, []);
      byUser.get(inter.userId).push(inter);
    });
    byUser.forEach(list => list.sort((a, b) => (b.rating !== a.rating) ? (b.rating - a.rating) : (b.timestamp - a.timestamp)));
    this.userTopRated = byUser;
  }

  findQualifiedUsers() {
    const qualified = [];
    this.userTopRated.forEach((list, userId) => { if (list.length >= 20) qualified.push(userId); });
    this.qualifiedUsers = qualified;
  }

  // === DATA EXPLORER ===
  showInteractions(page = 1) {
    const container = document.getElementById('dataView');
    const start = (page - 1) * this.pageSize;
    const slice = this.interactions.slice(start, start + this.pageSize);
    let html = `<h3>Interactions (page ${page})</h3><table><tr><th>User</th><th>Item</th><th>Rating</th><th>Timestamp</th></tr>`;
    slice.forEach(r => {
      html += `<tr><td>${r.userId}</td><td>${r.itemId}</td><td>${r.rating}</td><td>${r.timestamp}</td></tr>`;
    });
    html += `</table>`;
    html += this.makePagination(page, Math.ceil(this.interactions.length / this.pageSize), 'app.showInteractions');
    container.innerHTML = html;
  }

  showItems(page = 1) {
    const container = document.getElementById('dataView');
    const itemsArray = Array.from(this.items.entries());
    const start = (page - 1) * this.pageSize;
    const slice = itemsArray.slice(start, start + this.pageSize);
    let html = `<h3>Items (page ${page})</h3><table><tr><th>ItemId</th><th>Title</th><th>Year</th></tr>`;
    slice.forEach(([id, item]) => {
      html += `<tr><td>${id}</td><td>${item.title}</td><td>${item.year ?? ''}</td></tr>`;
    });
    html += `</table>`;
    html += this.makePagination(page, Math.ceil(itemsArray.length / this.pageSize), 'app.showItems');
    container.innerHTML = html;
  }

  makePagination(current, total, fnName) {
    let html = `<div class="pagination">`;
    if (current > 1) html += `<button onclick="${fnName}(${current - 1})">Prev</button>`;
    if (current < total) html += `<button onclick="${fnName}(${current + 1})">Next</button>`;
    html += `</div>`;
    return html;
  }

  // === EXISTING training, test, visualization methods stay unchanged ===
  // (keep your full training, visualizeEmbeddings, updateLossChart, test, displayResults, updateStatus methods here!)
}

let app;
document.addEventListener('DOMContentLoaded', () => { app = new MovieLensApp(); });
