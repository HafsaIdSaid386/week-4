// Deep Learning Two-Tower Model for MovieLens
class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim, hiddenDim) {
    // Embedding layers
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));
    this.genreWeights = tf.variable(tf.randomNormal([numGenres, embDim], 0, 0.05));

    // MLP towers
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.userDense2 = tf.layers.dense({ units: embDim, activation: 'linear' });

    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.itemDense2 = tf.layers.dense({ units: embDim, activation: 'linear' });
  }

  userForward(userIdxTensor) {
    const userEmb = tf.gather(this.userEmbedding, userIdxTensor);
    const h1 = this.userDense1.apply(userEmb);
    return this.userDense2.apply(h1);
  }

  itemForward(itemIdxTensor, genreTensor) {
    const itemEmb = tf.gather(this.itemEmbedding, itemIdxTensor);
    const genreEmb = tf.matMul(genreTensor, this.genreWeights);
    const combined = tf.add(itemEmb, genreEmb);
    const h1 = this.itemDense1.apply(combined);
    return this.itemDense2.apply(h1);
  }

  score(userEmb, itemEmb) {
    return tf.sum(tf.mul(userEmb, itemEmb), -1);
  }
}
