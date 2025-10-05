// two-tower.js — fully working version with training + inference

class TwoTowerModel {
  constructor(numUsers, numItems, embDim = 32, hiddenDim = 64) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;

    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu" });
    this.userDense2 = tf.layers.dense({ units: embDim, activation: "linear" });

    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: "relu" });
    this.itemDense2 = tf.layers.dense({ units: embDim, activation: "linear" });
  }

  userForward(userIdx) {
    const uEmb = tf.gather(this.userEmbedding, userIdx);
    return this.userDense2.apply(this.userDense1.apply(uEmb));
  }

  itemForward(itemIdx) {
    const iEmb = tf.gather(this.itemEmbedding, itemIdx);
    return this.itemDense2.apply(this.itemDense1.apply(iEmb));
  }

  score(uEmb, iEmb) {
    return tf.sum(tf.mul(uEmb, iEmb), -1);
  }

  async trainStep(userIdxArr, itemIdxArr, optimizer) {
    const uIdx = tf.tensor1d(userIdxArr, "int32");
    const iIdx = tf.tensor1d(itemIdxArr, "int32");

    const lossTensor = optimizer.minimize(() => {
      const uEmb = this.userForward(uIdx);
      const iEmb = this.itemForward(iIdx);

      const logits = tf.matMul(uEmb, iEmb, false, true); // [B,B]
      const labels = tf.oneHot(tf.range(0, logits.shape[0], 1, "int32"), logits.shape[0]);
      return tf.losses.softmaxCrossEntropy(labels, logits);
    }, true);

    const lossVal = (await lossTensor.data())[0];
    tf.dispose([uIdx, iIdx, lossTensor]);
    return lossVal;
  }

  getUserEmbedding(uIndex) {
    const idx = tf.tensor1d([uIndex], "int32");
    const emb = this.userForward(idx);
    idx.dispose();
    return emb;
  }

  getItemEmbeddings() {
    const idx = tf.range(0, this.numItems, 1, "int32");
    const emb = this.itemForward(idx);
    idx.dispose();
    return emb;
  }

  async getScoresForAllItems(uEmb) {
    const iEmb = this.getItemEmbeddings();
    const scores = tf.matMul(uEmb, iEmb, false, true).squeeze();
    const arr = await scores.array();
    tf.dispose([iEmb, scores]);
    return arr;
  }
}
