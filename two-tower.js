// two-tower.js - PROVEN WORKING VERSION
class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim = 32, hiddenDim = 64, lr = 0.01) {
    this.numUsers = numUsers;
    this.numItems = numItems;
    this.embDim = embDim;
    
    // Simple embeddings - NO MLP for stability
    this.userEmbedding = tf.variable(tf.randomUniform([numUsers, embDim], -0.05, 0.05));
    this.itemEmbedding = tf.variable(tf.randomUniform([numItems, embDim], -0.05, 0.05));
    
    // Genre weights
    this.genreWeights = tf.variable(tf.randomUniform([numGenres, embDim], -0.05, 0.05));
    
    // Higher learning rate
    this.optimizer = tf.train.adam(lr);
  }

  userForward(userIdxTensor) {
    return tf.tidy(() => {
      return tf.gather(this.userEmbedding, userIdxTensor);
    });
  }

  itemForward(itemIdxTensor, genreTensor) {
    return tf.tidy(() => {
      const itemEmb = tf.gather(this.itemEmbedding, itemIdxTensor);
      const genreEmb = tf.matMul(genreTensor, this.genreWeights);
      return tf.add(itemEmb, genreEmb);
    });
  }

  itemForwardForPCA(itemIdxTensor, genreTensor) {
    return this.itemForward(itemIdxTensor, genreTensor);
  }

  async trainStep(userIdx, itemIdx, genreBatch) {
    const lossVal = await this.optimizer.minimize(() => {
      return tf.tidy(() => {
        const u = this.userForward(userIdx);
        const i_pos = this.itemForward(itemIdx, genreBatch);
        
        // Simple dot product scores
        const pos_scores = tf.sum(tf.mul(u, i_pos), 1);
        
        // Sample random negatives
        const neg_indices = tf.randomUniform(itemIdx.shape, 0, this.numItems, 'int32');
        const neg_genres = tf.gather(globalItemGenreTensor, neg_indices);
        const i_neg = this.itemForward(neg_indices, neg_genres);
        const neg_scores = tf.sum(tf.mul(u, i_neg), 1);
        
        // BPR Loss: -log(sigmoid(pos_score - neg_score))
        const diff = tf.sub(pos_scores, neg_scores);
        const loss = tf.neg(tf.mean(tf.logSigmoid(diff)));
        
        return loss;
      });
    }, true);

    const loss = (await lossVal.data())[0];
    tf.dispose(lossVal);
    return loss;
  }
}

window.TwoTowerModel = TwoTowerModel;
