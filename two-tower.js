// two-tower.js
/* ------------------------------------------------------------
   TwoTowerModel (TensorFlow.js)
   Final Optimized Version — works with all TF.js ≥3.0
   ✅ Stable normalization
   ✅ Temperature-scaled softmax loss
   ✅ Optional BPR loss (ranking)
------------------------------------------------------------- */

class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim = 32, hiddenDim = 64, lr = 0.001, useBPR = false) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.numGenres = numGenres;
    this.embDim = embDim;
    this.hiddenDim = hiddenDim;
    this.useBPR = useBPR;

    // ---- Embedding tables ----
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));
    this.genreWeights = tf.variable(tf.randomNormal([numGenres, embDim], 0, 0.05));

    // ---- User tower ----
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu", useBias: true });
    this.userDense2 = tf.layers.dense({ units: embDim, activation: "linear", useBias: true });

    // ---- Item tower ----
    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu", useBias: true });
    this.itemDense2 = tf.layers.dense({ units: embDim, activation: "linear", useBias: true });

    // ---- Optimizer ----
    this.optimizer = tf.train.adam(lr);
  }

  /* ------------------- Normalization ------------------- */
  _l2Normalize(x) {
    const square = tf.mul(x, x);
    const sum = tf.sum(square, -1, true);
    const norm = tf.sqrt(tf.maximum(sum, 1e-8)); // avoid divide by zero
    return tf.div(x, norm);
  }

  /* ------------------- Forwards ------------------- */
  userForward(userIdxTensor) {
    return tf.tidy(() => {
      const userEmb = tf.gather(this.userEmbedding, userIdxTensor);
      const h1 = this.userDense1.apply(userEmb);
      const out = this.userDense2.apply(h1);
      return this._l2Normalize(out);
    });
  }

  itemForward(itemIdxTensor, genreTensor) {
    return tf.tidy(() => {
      const itemEmb = tf.gather(this.itemEmbedding, itemIdxTensor);
      const genreEmb = tf.matMul(genreTensor, this.genreWeights);
      const combined = tf.add(itemEmb, genreEmb);
      const h1 = this.itemDense1.apply(combined);
      const out = this.itemDense2.apply(h1);
      return this._l2Normalize(out);
    });
  }

  /* ------------------- Scoring ------------------- */
  score(userEmb, itemEmb) {
    return tf.sum(tf.mul(userEmb, itemEmb), -1);
  }

  userRaw(userIdxTensor) {
    return tf.gather(this.userEmbedding, userIdxTensor);
  }
  itemRaw() {
    return this.itemEmbedding;
  }

  /* ------------------- Training Step ------------------- */
  async trainStep(userIdx, itemIdx, genreBatch) {
    const lossVal = await this.optimizer.minimize(() => {
      const u = this.userForward(userIdx); // [B,E]
      const ip = this.itemForward(itemIdx, genreBatch); // [B,E]

      // ----- OPTION 1: In-batch softmax with temperature scaling -----
      if (!this.useBPR) {
        // Compute logits (user-item similarity matrix)
        const logits = tf.matMul(u, ip, false, true).div(Math.sqrt(this.embDim)); // temperature scaling

        // Diagonal = positive pairs
        const labels = tf.oneHot(
          tf.range(0, logits.shape[0], 1, "int32"),
          logits.shape[1]
        );

        // Cross-entropy loss with in-batch negatives
        const loss = tf.losses.softmaxCrossEntropy(labels, logits).mean();
        return loss;
      }

      // ----- OPTION 2: BPR pairwise loss -----
      else {
        // Sample negatives by rolling items within batch
        const rolled = tf.concat([itemIdx.slice(1), itemIdx.slice(0, 1)], 0);
        const genreNeg = tf.gather(
          genreBatch,
          tf.tensor1d(
            [...Array(genreBatch.shape[0]).keys()].map((i) => (i + 1) % genreBatch.shape[0]),
            "int32"
          )
        );

        // Compute forward passes
        const ineg = this.itemForward(rolled, genreNeg);
        const posScore = this.score(u, ip);
        const negScore = this.score(u, ineg);

        // BPR loss = -log σ(pos - neg)
        const x = tf.sub(posScore, negScore);
        const loss = tf.neg(tf.mean(tf.logSigmoid(x)));
        return loss;
      }
    }, true);

    // Return scalar number for chart
    const v = Array.isArray(lossVal) ? lossVal[0] : lossVal;
    const num = (await v.data())[0];
    tf.dispose(lossVal);
    return num;
  }

  /* ------------------- Inference ------------------- */
  getUserEmbedding(uIdx) {
    const t = tf.tensor1d([uIdx], "int32");
    const out = this.userForward(t);
    t.dispose();
    return out;
  }

  scoresAllItemsDeep(userIdxTensor, allGenres) {
    return tf.tidy(() => {
      const u = this.userForward(userIdxTensor);
      const I = this.itemForward(tf.range(0, this.numItems, 1, "int32"), allGenres);
      const scores = tf.matMul(I, u.transpose()).reshape([this.numItems]);
      return scores;
    });
  }
}

// Expose to window
window.TwoTowerModel = TwoTowerModel;
