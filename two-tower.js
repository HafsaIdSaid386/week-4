// two-tower.js
/* ------------------------------------------------------------
   TwoTowerModel (TensorFlow.js)
   Fixed version with proper PCA support
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

  /* ------------------- Normalization (only for training) ------------------- */
  _l2Normalize(x) {
    const square = tf.mul(x, x);
    const sum = tf.sum(square, -1, true);
    const norm = tf.sqrt(tf.maximum(sum, 1e-8));
    return tf.div(x, norm);
  }

  /* ------------------- Forwards ------------------- */
  userForward(userIdxTensor, normalize = true) {
    return tf.tidy(() => {
      const userEmb = tf.gather(this.userEmbedding, userIdxTensor);
      const h1 = this.userDense1.apply(userEmb);
      const out = this.userDense2.apply(h1);
      return normalize ? this._l2Normalize(out) : out;
    });
  }

  itemForward(itemIdxTensor, genreTensor, normalize = true) {
    return tf.tidy(() => {
      const itemEmb = tf.gather(this.itemEmbedding, itemIdxTensor);
      const genreEmb = tf.matMul(genreTensor, this.genreWeights);
      const combined = tf.add(itemEmb, genreEmb);
      const h1 = this.itemDense1.apply(combined);
      const out = this.itemDense2.apply(h1);
      return normalize ? this._l2Normalize(out) : out;
    });
  }

  /* ------------------- PCA-specific method (without strong normalization) ------------------- */
  itemForwardForPCA(itemIdxTensor, genreTensor) {
    return tf.tidy(() => {
      const itemEmb = tf.gather(this.itemEmbedding, itemIdxTensor);
      const genreEmb = tf.matMul(genreTensor, this.genreWeights);
      const combined = tf.add(itemEmb, genreEmb);
      const h1 = this.itemDense1.apply(combined);
      const out = this.userDense2.apply(h1);
      
      // Apply mild normalization for stability, but not full L2 normalization
      const square = tf.mul(out, out);
      const sum = tf.sum(square, -1, true);
      const norm = tf.sqrt(tf.maximum(sum, 1e-8));
      return tf.div(out, tf.add(norm, 0.1)); // Mild normalization
    });
  }

  /* ------------------- Scoring ------------------- */
  score(userEmb, itemEmb) {
    return tf.sum(tf.mul(userEmb, itemEmb), -1);
  }

  /* ------------------- Training Step ------------------- */
  async trainStep(userIdx, itemIdx, genreBatch) {
    const lossVal = await this.optimizer.minimize(() => {
      const u = this.userForward(userIdx); // [B,E] - normalized for training
      const ip = this.itemForward(itemIdx, genreBatch); // [B,E] - normalized for training

      if (!this.useBPR) {
        // In-batch softmax with temperature scaling
        const logits = tf.matMul(u, ip, false, true).div(Math.sqrt(this.embDim));
        const labels = tf.oneHot(
          tf.range(0, logits.shape[0], 1, "int32"),
          logits.shape[1]
        );
        const loss = tf.losses.softmaxCrossEntropy(labels, logits).mean();
        return loss;
      } else {
        // BPR pairwise loss
        const rolled = tf.concat([itemIdx.slice(1), itemIdx.slice(0, 1)], 0);
        const genreNeg = tf.gather(
          genreBatch,
          tf.tensor1d(
            [...Array(genreBatch.shape[0]).keys()].map((i) => (i + 1) % genreBatch.shape[0]),
            "int32"
          )
        );
        const ineg = this.itemForward(rolled, genreNeg);
        const posScore = this.score(u, ip);
        const negScore = this.score(u, ineg);
        const x = tf.sub(posScore, negScore);
        const loss = tf.neg(tf.mean(tf.logSigmoid(x)));
        return loss;
      }
    }, true);

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
}

// Expose to window
window.TwoTowerModel = TwoTowerModel;
