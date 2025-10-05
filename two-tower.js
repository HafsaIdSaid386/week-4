// two-tower.js
// Two-Tower model with MLP heads + all helpers the app calls.

class TwoTowerModel {
  constructor(numUsers, numItems, embDim = 32, hiddenDim = 64) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;

    // Trainable embedding tables
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    // MLP heads
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.userDense2 = tf.layers.dense({ units: embDim,   activation: 'linear' });

    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.itemDense2 = tf.layers.dense({ units: embDim,   activation: 'linear' });
  }

  // ------- Forward (MLP heads) -------
  userForward(userIdxTensor) {
    const emb = tf.gather(this.userEmbedding, userIdxTensor); // [B,E]
    return this.userDense2.apply(this.userDense1.apply(emb));  // [B,E]
  }
  itemForward(itemIdxTensor) {
    const emb = tf.gather(this.itemEmbedding, itemIdxTensor); // [B,E]
    return this.itemDense2.apply(this.itemDense1.apply(emb));  // [B,E]
  }

  // ------- Baseline (raw tables, no MLP) -------
  getRawUserEmbedding(uIndex) {
    const idx = tf.tensor1d([uIndex], 'int32');
    const out = tf.gather(this.userEmbedding, idx); // [1,E]
    idx.dispose();
    return out;
  }
  getRawItemEmbeddings() {
    const idx = tf.range(0, this.numItems, 1, 'int32');
    const out = tf.gather(this.itemEmbedding, idx); // [N,E]
    idx.dispose();
    return out;
  }
  async getScoresRawForAllItems(uRawEmb) {
    const iEmb = this.getRawItemEmbeddings();                            // [N,E]
    const scores = tf.matMul(uRawEmb, iEmb, false, true).squeeze();       // [N]
    const arr = await scores.array();
    tf.dispose([iEmb, scores]);
    return arr;
  }

  // ------- Dot product -------
  score(uEmb, iEmb) { return tf.sum(tf.mul(uEmb, iEmb), -1); }

  // ------- Training step (in-batch softmax) -------
  async trainStep(userIdxArr, itemIdxArr, optimizer) {
    const uIdx = tf.tensor1d(userIdxArr, 'int32');
    const iIdx = tf.tensor1d(itemIdxArr, 'int32');

    const lossTensor = optimizer.minimize(() => {
      const U = this.userForward(uIdx);    // [B,E]
      const V = this.itemForward(iIdx);    // [B,E]
      const logits = tf.matMul(U, V, false, true); // [B,B]
      const B = logits.shape[0];
      const labels = tf.oneHot(tf.range(0, B, 1, 'int32'), B); // diag
      return tf.losses.softmaxCrossEntropy(labels, logits);
    }, true);

    const loss = (await lossTensor.data())[0];
    tf.dispose([uIdx, iIdx, lossTensor]);
    return loss;
  }

  // ------- Inference (MLP heads) -------
  getUserEmbedding(uIndex) {
    const idx = tf.tensor1d([uIndex], 'int32');
    const emb = this.userForward(idx); // [1,E]
    idx.dispose();
    return emb;
  }
  getItemEmbeddings() {
    const idx = tf.range(0, this.numItems, 1, 'int32');
    const emb = this.itemForward(idx); // [N,E]
    idx.dispose();
    return emb;
  }
  async getScoresForAllItems(uEmb) {
    const iEmb = this.getItemEmbeddings();                          // [N,E]
    const scores = tf.matMul(uEmb, iEmb, false, true).squeeze();     // [N]
    const arr = await scores.array();
    tf.dispose([iEmb, scores]);
    return arr;
  }
}
