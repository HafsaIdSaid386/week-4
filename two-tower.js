// two-tower.js - Replace the entire file with this:

class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim = 64, hiddenDim = 128, lr = 0.01, useBPR = false) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.numGenres = numGenres;
    this.embDim = embDim;
    this.hiddenDim = hiddenDim;
    this.useBPR = useBPR;

    // Larger embedding dimensions for better representation
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.1));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.1));
    this.genreWeights = tf.variable(tf.randomNormal([numGenres, embDim], 0, 0.1));

    // Deeper network with batch normalization
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu" });
    this.userBn1 = tf.layers.batchNormalization();
    this.userDense2 = tf.layers.dense({ units: embDim, activation: "linear" });

    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu" });
    this.itemBn1 = tf.layers.batchNormalization();
    this.itemDense2 = tf.layers.dense({ units: embDim, activation: "linear" });

    // Higher learning rate for faster convergence
    this.optimizer = tf.train.adam(lr);
  }

  userForward(userIdxTensor, normalize = true) {
    return tf.tidy(() => {
      const userEmb = tf.gather(this.userEmbedding, userIdxTensor);
      const h1 = this.userDense1.apply(userEmb);
      const h1bn = this.userBn1.apply(h1);
      const out = this.userDense2.apply(h1bn);
      return normalize ? this._l2Normalize(out) : out;
    });
  }

  itemForward(itemIdxTensor, genreTensor, normalize = true) {
    return tf.tidy(() => {
      const itemEmb = tf.gather(this.itemEmbedding, itemIdxTensor);
      const genreEmb = tf.matMul(genreTensor, this.genreWeights);
      const combined = tf.add(itemEmb, genreEmb);
      const h1 = this.itemDense1.apply(combined);
      const h1bn = this.itemBn1.apply(h1);
      const out = this.itemDense2.apply(h1bn);
      return normalize ? this._l2Normalize(out) : out;
    });
  }

  itemForwardForPCA(itemIdxTensor, genreTensor) {
    return this.itemForward(itemIdxTensor, genreTensor, false); // No normalization for PCA
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

      if (!this.useBPR) {
        // Better temperature scaling
        const temperature = 0.1; // Lower temperature for sharper distributions
        const logits = tf.matMul(u, ip, false, true).div(temperature);
        const labels = tf.oneHot(tf.range(0, logits.shape[0], 1, "int32"), logits.shape[1]);
        return tf.losses.softmaxCrossEntropy(labels, logits).mean();
      } else {
        const rolled = tf.concat([itemIdx.slice(1), itemIdx.slice(0, 1)], 0);
        const genreNeg = tf.gather(genreBatch, 
          tf.range(0, genreBatch.shape[0]).add(1).mod(genreBatch.shape[0]));
        const ineg = this.itemForward(rolled, genreNeg);
        const posScore = tf.sum(tf.mul(u, ip), -1);
        const negScore = tf.sum(tf.mul(u, ineg), -1);
        return tf.neg(tf.mean(tf.logSigmoid(tf.sub(posScore, negScore))));
      }
    }, true);

    const loss = (await lossVal.data())[0];
    tf.dispose(lossVal);
    return loss;
  }
}

window.TwoTowerModel = TwoTowerModel;
