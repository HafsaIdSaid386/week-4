class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim = 32, hiddenDim = 64) {
    // Embeddings
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));
    this.genreWeights = tf.variable(tf.randomNormal([numGenres, embDim], 0, 0.05));

    // MLPs for user and item towers
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.userDense2 = tf.layers.dense({ units: embDim, activation: 'linear' });

    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.itemDense2 = tf.layers.dense({ units: embDim, activation: 'linear' });
  }

  userForward(userIdx) {
    const emb = tf.gather(this.userEmbedding, userIdx);
    const h1 = this.userDense1.apply(emb);
    return this.userDense2.apply(h1);
  }

  itemForward(itemIdx, genreTensor) {
    const itemEmb = tf.gather(this.itemEmbedding, itemIdx);
    const genreEmb = tf.matMul(genreTensor, this.genreWeights);
    const combined = tf.add(itemEmb, genreEmb);
    const h1 = this.itemDense1.apply(combined);
    return this.itemDense2.apply(h1);
  }

  score(uEmb, iEmb) {
    return tf.sum(tf.mul(uEmb, iEmb), -1);
  }
}
