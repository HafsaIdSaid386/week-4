// two-tower.js - WORKING VERSION
class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim = 32, hiddenDim = 64, lr = 0.001, useBPR = false) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.numGenres = numGenres;
    this.embDim = embDim;
    this.hiddenDim = hiddenDim;
    this.useBPR = useBPR;

    // Embeddings
    this.userEmbedding = tf.variable(tf.randomUniform([numUsers, embDim], -0.05, 0.05));
    this.itemEmbedding = tf.variable(tf.randomUniform([numItems, embDim], -0.05, 0.05));
    this.genreWeights = tf.variable(tf.randomUniform([numGenres, embDim], -0.05, 0.05));

    // MLP layers
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu" });
    this.userDense2 = tf.layers.dense({ units: embDim, activation: "linear" });

    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu" });
    this.itemDense2 = tf.layers.dense({ units: embDim, activation: "linear" });

    this.optimizer = tf.train.adam(lr);
  }

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

  itemForwardForPCA(itemIdxTensor, genreTensor) {
    return this.itemForward(itemIdxTensor, genreTensor, false);
  }

  _l2Normalize(x) {
    return tf.tidy(() => {
      const norm = tf.norm(x, 2, -1, true);
      return tf.div(x, tf.maximum(norm, 1e-8));
    });
  }

  async trainStep(userIdx, itemIdx, genreBatch) {
    const lossVal = await this.optimizer.minimize(() => {
      return tf.tidy(() => {
        const u = this.userForward(userIdx);
        const i_pos = this.itemForward(itemIdx, genreBatch);

        if (!this.useBPR) {
          // WORKING LOSS: Simple dot product + margin ranking loss
          const pos_scores = tf.sum(tf.mul(u, i_pos), 1);
          
          // Sample negatives - different items from the batch
          const neg_indices = tf.randomUniform(itemIdx.shape, 0, this.numItems, 'int32');
          const neg_genres = tf.gather(globalItemGenreTensor, neg_indices);
          const i_neg = this.itemForward(neg_indices, neg_genres);
          const neg_scores = tf.sum(tf.mul(u, i_neg), 1);
          
          // Margin ranking loss: max(0, - (pos - neg) + margin)
          const margin = 1.0;
          const loss = tf.mean(tf.maximum(0, tf.add(tf.sub(neg_scores, pos_scores), margin)));
          return loss;
        } else {
          // BPR loss
          const rolled = tf.concat([itemIdx.slice(1), itemIdx.slice(0, 1)], 0);
          const genreNeg = tf.gather(genreBatch, 
            tf.range(0, genreBatch.shape[0]).add(1).mod(genreBatch.shape[0]));
          const i_neg = this.itemForward(rolled, genreNeg);
          const pos_score = tf.sum(tf.mul(u, i_pos), 1);
          const neg_score = tf.sum(tf.mul(u, i_neg), 1);
          return tf.neg(tf.mean(tf.logSigmoid(tf.sub(pos_score, neg_score))));
        }
      });
    }, true);

    const loss = (await lossVal.data())[0];
    tf.dispose(lossVal);
    return loss;
  }
}

window.TwoTowerModel = TwoTowerModel;
