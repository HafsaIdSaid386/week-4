/* app.js — Handles data loading, training, testing, and visualization */
(() => {
  const btnLoad = document.getElementById('btnLoad');
  const btnTrain = document.getElementById('btnTrain');
  const btnTest = document.getElementById('btnTest');
  const statusEl = document.getElementById('status');
  const lossCanvas = document.getElementById('lossCanvas');
  const projCanvas = document.getElementById('projCanvas');
  const resultsEl = document.getElementById('results');

  const state = {
    interactions: [],
    items: new Map(),
    userToRated: new Map(),
    userIds: [], itemIds: [],
    userIdToIdx: new Map(), itemIdToIdx: new Map(),
    idxToUserId: [], idxToItemId: [],
    model: null, tfItemGenreMatrix: null,
    config: { epochs: 3, batchSize: 1024, embDim: 32, hiddenDim: 64, learningRate: 0.003, maxInteractions: 80000 }
  };

  const setStatus = m => statusEl.textContent = `Status: ${m}`;

  async function loadData() {
    setStatus('Loading MovieLens data...');
    const [uDataTxt, uItemTxt] = await Promise.all([
      fetch('data/u.data').then(r=>r.text()),
      fetch('data/u.item').then(r=>r.text())
    ]);

    const itemLines = uItemTxt.split('\n').filter(Boolean);
    const NUM_GENRES = 19;
    for (const line of itemLines) {
      const cols = line.split('|');
      const itemId = parseInt(cols[0]);
      const titleRaw = cols[1] || '';
      const yearMatch = titleRaw.match(/\((\d{4})\)/);
      const title = yearMatch ? titleRaw.replace(/\(\d{4}\)/,'').trim() : titleRaw;
      const year = yearMatch ? parseInt(yearMatch[1]) : null;
      const genres = cols.slice(-NUM_GENRES).map(x=>parseInt(x||'0'));
      state.items.set(itemId, {title, year, genres});
    }

    const interLines = uDataTxt.split('\n').filter(Boolean);
    for (const line of interLines) {
      const [u,i,r,t] = line.split('\t').map(Number);
      if (!state.items.has(i)) continue;
      state.interactions.push({userId:u, itemId:i, rating:r, ts:t});
      if (!state.userToRated.has(u)) state.userToRated.set(u,[]);
      state.userToRated.get(u).push({itemId:i, rating:r, ts:t});
    }

    for (const [u, arr] of state.userToRated.entries())
      arr.sort((a,b)=>b.rating-a.rating || b.ts-a.ts);

    state.userIds = Array.from(state.userToRated.keys()).sort((a,b)=>a-b);
    state.itemIds = Array.from(state.items.keys()).sort((a,b)=>a-b);
    state.userIds.forEach((u,idx)=>{state.userIdToIdx.set(u,idx);state.idxToUserId[idx]=u;});
    state.itemIds.forEach((i,idx)=>{state.itemIdToIdx.set(i,idx);state.idxToItemId[idx]=i;});

    const numItems = state.itemIds.length;
    const numGenres = 19;
    const genreData = new Float32Array(numItems*numGenres);
    for (let r=0;r<numItems;r++){
      const g = state.items.get(state.idxToItemId[r]).genres;
      for (let c=0;c<numGenres;c++) genreData[r*numGenres+c]=g[c];
    }
    state.tfItemGenreMatrix = tf.tensor2d(genreData,[numItems,numGenres]);

    setStatus(`Loaded ${state.interactions.length} interactions, ${state.userIds.length} users, ${state.itemIds.length} items`);
    btnTrain.disabled = false;
  }

  async function train() {
    const {epochs,batchSize,embDim,hiddenDim,learningRate,maxInteractions} = state.config;
    const inter = tf.util.shuffle([...state.interactions]).slice(0,maxInteractions);
    const userIdx = inter.map(x=>state.userIdToIdx.get(x.userId));
    const itemIdx = inter.map(x=>state.itemIdToIdx.get(x.itemId));
    const numUsers = state.userIds.length, numItems = state.itemIds.length, numGenres = 19;

    state.model = new TwoTowerModel(numUsers,numItems,numGenres,embDim,hiddenDim);
    const opt = tf.train.adam(learningRate);
    const losses=[], ctx=lossCanvas.getContext('2d');

    function drawLoss() {
      ctx.clearRect(0,0,lossCanvas.width,lossCanvas.height);
      ctx.strokeStyle='#60a5fa'; ctx.beginPath();
      losses.forEach((l,i)=>{
        const x=i/(losses.length-1)*lossCanvas.width;
        const y=lossCanvas.height*(1-l/Math.max(...losses));
        i?ctx.lineTo(x,y):ctx.moveTo(x,y);
      }); ctx.stroke();
    }

    for (let e=0;e<epochs;e++){
      for (let i=0;i<userIdx.length;i+=batchSize){
        const u=tf.tensor1d(userIdx.slice(i,i+batchSize),'int32');
        const it=tf.tensor1d(itemIdx.slice(i,i+batchSize),'int32');
        const loss=opt.minimize(()=>state.model.inBatchSoftmaxLoss(
          state.model.userForward(u),
          state.model.itemForward(it,tf.gather(state.tfItemGenreMatrix,it))
        ),true).dataSync()[0];
        losses.push(loss); drawLoss(); await tf.nextFrame();
      }
      setStatus(`Epoch ${e+1}/${epochs} loss=${losses.at(-1).toFixed(4)}`);
    }
    btnTest.disabled=false;
    await projectEmbeddings();
  }

  async function projectEmbeddings(){
    const emb=state.model.getAllItemEmbeddings();
    const mean=tf.mean(emb,0,true);
    const X=tf.sub(emb,mean);
    const svd=tf.linalg.svd(X,true);
    const V2=svd.v.slice([0,0],[svd.v.shape[0],2]);
    const proj=tf.matMul(X,V2);
    const pts=await proj.array();
    const ctx=projCanvas.getContext('2d');
    ctx.clearRect(0,0,projCanvas.width,projCanvas.height);
    const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
    const xMin=Math.min(...xs),xMax=Math.max(...xs),yMin=Math.min(...ys),yMax=Math.max(...ys);
    for(let i=0;i<pts.length;i++){
      const x=(pts[i][0]-xMin)/(xMax-xMin)*projCanvas.width;
      const y=projCanvas.height-(pts[i][1]-yMin)/(yMax-yMin)*projCanvas.height;
      ctx.fillStyle='#60a5fa'; ctx.fillRect(x,y,2,2);
    }
    tf.dispose([emb,mean,X,svd.v,proj]);
  }

  async function testOnce(){
    const users=state.userIds.filter(u=>(state.userToRated.get(u)||[]).length>=20);
    const uRaw=users[Math.floor(Math.random()*users.length)];
    const uIdx=state.userIdToIdx.get(uRaw);
    const uEmb=state.model.getUserEmbedding(uIdx);
    const allEmb=state.model.getAllItemEmbeddings();
    const scores=tf.matMul(uEmb.expandDims(0),allEmb,false,true).squeeze().arraySync();
    const rated=new Set(state.userToRated.get(uRaw).map(x=>state.itemIdToIdx.get(x.itemId)));
    const recIdx=[...scores.keys()].filter(i=>!rated.has(i)).sort((a,b)=>scores[b]-scores[a]).slice(0,10);
    const recs=recIdx.map(i=>{
      const id=state.idxToItemId[i],meta=state.items.get(id);
      return `${meta.title} (${meta.year||''})`;
    });
    const hist=state.userToRated.get(uRaw).slice(0,10).map(x=>{
      const m=state.items.get(x.itemId);return `${m.title} (${x.rating})`;
    });
    resultsEl.innerHTML=`<h3>User ${uRaw}</h3>
      <table><tr><th>Top-10 Rated</th><th>Top-10 Recommended</th></tr>
      <tr><td>${hist.join('<br>')}</td><td>${recs.join('<br>')}</td></tr></table>`;
  }

  btnLoad.onclick=loadData;
  btnTrain.onclick=train;
  btnTest.onclick=testOnce;
})();
