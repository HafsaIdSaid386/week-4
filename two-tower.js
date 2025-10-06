// two-tower.js - REPLACE THIS ENTIRE FILE
class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim = 32, hiddenDim = 64, lr = 0.001, useBPR = false) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.numGenres = numGenres;
    this.embDim = embDim;
    this.hiddenDim = hiddenDim;
    this.useBPR = useBPR;

    // SIMPLER embedding initialization - this was the main problem!
    this.userEmbedding = tf.variable(tf.randomUniform([numUsers, embDim], -0.05, 0.05));
    this.itemEmbedding = tf.variable(tf.randomUniform([numItems, embDim], -0.05, 0.05));
    this.genreWeights = tf.variable(tf.randomUniform([numGenres, embDim], -0.05, 0.05));

    // SIMPLER network - remove batch normalization
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu" });
    this.userDense2 = tf.layers.dense({ units: embDim, activation: "linear" });

    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu" });
    this.itemDense2 = tf.layers.dense({ units: embDim, activation: "linear" });

    this.optimizer = tf.train.adam(lr);
  }

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

  // FOR PCA - no normalization
  itemForwardForPCA(itemIdxTensor, genreTensor) {
    return tf.tidy(() => {
      const itemEmb = tf.gather(this.itemEmbedding, itemIdxTensor);
      const genreEmb = tf.matMul(genreTensor, this.genreWeights);
      const combined = tf.add(itemEmb, genreEmb);
      const h1 = this.itemDense1.apply(combined);
      return this.itemDense2.apply(h1); // NO NORMALIZATION for PCA
    });
  }

  _l2Normalize(x) {
    return tf.tidy(() => {
      const norm = tf.norm(x, 2, -1, true);
      return tf.div(x, tf.maximum(norm, 1e-8));
    });
  }

  async trainStep(userIdx, itemIdx, genreBatch) {
    const lossVal = await this.optimizer.minimize(() => {
      const u = this.userForward(userIdx);
      const ip = this.itemForward(itemIdx, genreBatch);

      // FIXED LOSS CALCULATION - this was the problem!
      const logits = tf.matMul(u, ip, false, true); // [B, B]
      
      // Labels: diagonal should be positive (1), others negative (0)
      const labels = tf.oneHot(
        tf.range(0, logits.shape[0], 1, "int32"),
        logits.shape[1]
      );
      
      // Use sparseCategoricalCrossEntropy - much more stable!
      const loss = tf.losses.softmaxCrossEntropy(labels, logits);
      return loss;
    }, true);

    const loss = (await lossVal.data())[0];
    tf.dispose(lossVal);
    return loss;
  }
}

window.TwoTowerModel = TwoTowerModel;
