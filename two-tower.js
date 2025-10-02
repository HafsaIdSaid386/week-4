class TwoTowerModel {
  constructor(numUsers, numItems, embeddingDim = 32, lr = 0.001) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embeddingDim = embeddingDim;
    this.userEmbeddings = tf.variable(tf.randomNormal([numUsers, embeddingDim], 0, 0.05));
    this.itemEmbeddings = tf.variable(tf.randomNormal([numItems, embeddingDim], 0, 0.05));
    this.optimizer = tf.train.adam(lr);
  }

  userForward(userIndices) { return tf.gather(this.userEmbeddings, userIndices); }
  itemForward(itemIndices) { return tf.gather(this.itemEmbeddings, itemIndices); }
  score(userEmbeddings, itemEmbeddings) { return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1); }

  async trainStep(userIndices, itemIndices) {
    const userTensor = tf.tensor1d(userIndices, 'int32');
    const itemTensor = tf.tensor1d(itemIndices, 'int32');
    let lossTensor;
    this.optimizer.minimize(() => {
      const userEmbs = this.userForward(userTensor);
      const itemEmbs = this.itemForward(itemTensor);
      const scores = this.score(userEmbs, itemEmbs);
      const labels = tf.onesLike(scores);
      const loss = tf.losses.meanSquaredError(labels, scores);
      const meanLoss = tf.mean(loss);
      lossTensor = meanLoss;
      return meanLoss;
    });
    const lossVal = lossTensor.dataSync()[0];
    userTensor.dispose(); itemTensor.dispose(); lossTensor.dispose();
    return lossVal;
  }

  getUserEmbedding(userIndex) { return tf.tidy(() => this.userForward(tf.tensor1d([userIndex], 'int32')).squeeze()); }
  getItemEmbeddings() { return this.itemEmbeddings; }
  async getScoresForAllItems(userEmb) {
    return tf.tidy(() => {
      const scores = tf.matMul(userEmb.reshape([1, this.embeddingDim]), this.itemEmbeddings.transpose());
      return scores.dataSync();
    });
  }
}
