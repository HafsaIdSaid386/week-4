// two-tower.js
/* ------------------------------------------------------------
   TwoTowerModel (TensorFlow.js)
   - User tower: user_id embedding → MLP
   - Item tower: item_id embedding + genre projection → MLP
   - Score: dot product
   - Loss: default in-batch softmax cross-entropy (labels are diagonal)
           (optional BPR pairwise if useBPR=true)
------------------------------------------------------------- */

class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim=32, hiddenDim=64, lr=0.001, useBPR=false) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.numGenres = numGenres;
    this.embDim = embDim;
    this.hiddenDim = hiddenDim;
    this.useBPR = useBPR;

    // Embedding tables
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    // Genre projection to embedding space
    this.genreWeights = tf.variable(tf.randomNormal([numGenres, embDim], 0, 0.05));

    // User MLP
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu', useBias: true });
    this.userDense2 = tf.layers.dense({ units: embDim, activation: 'linear', useBias: true });

    // Item MLP
    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu', useBias: true });
    this.itemDense2 = tf.layers.dense({ units: embDim, activation: 'linear', useBias: true });

    this.optimizer = tf.train.adam(lr);
  }

  /* ---------------- Forwards ---------------- */
  // User forward pass: gather embedding → MLP → normalized vector
  userForward(userIdxTensor /* [B] int32 */) {
    return tf.tidy(() => {
      const userEmb = tf.gather(this.userEmbedding, userIdxTensor); // [B,E]
      const h1 = this.userDense1.apply(userEmb);
      const out = this.userDense2.apply(h1);
      // Replace tf.linalg.l2Normalize with manual normalization
      const normed = tf.div(out, tf.norm(out, 'euclidean', -1, true));
      return normed;
    });
  }

  // Item forward pass: gather embedding + genre projection → MLP → normalized vector
  itemForward(itemIdxTensor /* [B] int32 */, genreTensor /* [B, G] float */) {
    return tf.tidy(() => {
      const itemEmb = tf.gather(this.itemEmbedding, itemIdxTensor); // [B,E]
      const genreEmb = tf.matMul(genreTensor, this.genreWeights);   // [B,E]
      const combined = tf.add(itemEmb, genreEmb);                   // fuse ID + content
      const h1 = this.itemDense1.apply(combined);
      const out = this.itemDense2.apply(h1);
      // Replace tf.linalg.l2Normalize with manual normalization
      const normed = tf.div(out, tf.norm(out, 'euclidean', -1, true));
      return normed;
    });
  }

  // Raw (no-MLP, no-genre) embeddings for baseline scoring
  userRaw(userIdxTensor){ return tf.gather(this.userEmbedding, userIdxTensor); }
  itemRaw(){ return this.itemEmbedding; }

  // Score is dot product between normalized vectors
  score(userEmb /* [B,E] */, itemEmb /* [B,E] or [I,E] */) {
    return tf.tidy(() => tf.sum(tf.mul(userEmb, itemEmb), -1)); // [B] or [I]
  }

  /* ---------------- Training Step ---------------- */
  async trainStep(userIdx /* [B] */, itemIdx /* [B] */, genreBatch /* [B,G] */) {
    const lossVal = await this.optimizer.minimize(() => {
      const u = this.userForward(userIdx);           // [B,E]
      const ip = this.itemForward(itemIdx, genreBatch); // [B,E]

      if (!this.useBPR) {
        // In-batch sampled softmax:
        // logits = U @ I^T; labels are diagonal (each user pairs with its positive item)
        const logits = tf.matMul(u, ip, false, true);          // [B,B]
        const labels = tf.oneHot(tf.range(0, logits.shape[0], 1, 'int32'), logits.shape[1]); // [B,B]
        const loss = tf.losses.softmaxCrossEntropy(labels, logits).mean();
        return loss;
      } else {
        // BPR pairwise: sample negatives by rolling positives within batch
        const rolled = tf.concat([itemIdx.slice(1), itemIdx.slice(0,1)], 0);
        const genreNeg = tf.gather(genreBatch, tf.tensor1d([...Array(genreBatch.shape[0]).keys()].map(i=>(i+1)%genreBatch.shape[0]), 'int32'));
        const ineg = this.itemForward(rolled, genreNeg);
        const posScore = this.score(u, ip);          // [B]
        const negScore = this.score(u, ineg);        // [B]
        const x = tf.sub(posScore, negScore);
        const loss = tf.neg(tf.mean(tf.logSigmoid(x)));
        return loss;
      }
    }, true);

    // Return scalar
    const v = Array.isArray(lossVal) ? lossVal[0] : lossVal;
    const num = (await v.data())[0];
    tf.dispose(lossVal);
    return num;
  }

  /* ---------------- Inference helpers ---------------- */
  getUserEmbedding(uIdx /* number */) {
    const t = tf.tensor1d([uIdx], 'int32');
    const out = this.userForward(t); // [1,E]
    t.dispose();
    return out;
  }

  // Scores for a single user vs all items (deep path)
  scoresAllItemsDeep(userIdxTensor /* [1] */, allGenres /* [I,G] */) {
    return tf.tidy(() => {
      const u = this.userForward(userIdxTensor);          // [1,E]
      const I = this.itemForward(tf.range(0, this.numItems, 1, 'int32'), allGenres); // [I,E]
      const scores = tf.matMul(I, u.transpose()).reshape([this.numItems]); // [I]
      return scores; // caller disposes
    });
  }
}

window.TwoTowerModel = TwoTowerModel;
