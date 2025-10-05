// ======================================================
// two-tower.js - Deep Learning Two-Tower Model (MLP)
// ======================================================

class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim, hiddenDim) {
    // User & item embeddings
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    // Optional genre weights
    this.genreWeights = tf.variable(tf.randomNormal([numGenres, embDim], 0, 0.05));

    // MLP layers for both towers
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.userDense2 = tf.layers.dense({ units: embDim, activation: 'linear' });

    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.itemDense2 = tf.layers.dense({ units: embDim, activation: 'linear' });
  }

  userForward(userIdxTensor) {
    const uEmb = tf.gather(this.userEmbedding, userIdxTensor);
    const h = this.userDense1.apply(uEmb);
    return this.userDense2.apply(h);
  }

  itemForward(itemIdxTensor, genreTensor) {
    const iEmb = tf.gather(this.itemEmbedding, itemIdxTensor);
    const gEmb = tf.matMul(genreTensor, this.genreWeights);
    const combined = tf.add(iEmb, gEmb);
    const h = this.itemDense1.apply(combined);
    return this.itemDense2.apply(h);
  }

  score(userEmb, itemEmb) {
    return tf.sum(tf.mul(userEmb, itemEmb), -1);
  }
}
