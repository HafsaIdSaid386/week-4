/* two-tower.js — Deep Learning Two-Tower (MLP) model in TensorFlow.js */
class TwoTowerModel {
  constructor(numUsers,numItems,numGenres,embDim=32,hiddenDim=64){
    this.userEmbedding=tf.variable(tf.randomNormal([numUsers,embDim],0,0.05));
    this.itemEmbedding=tf.variable(tf.randomNormal([numItems,embDim],0,0.05));
    this.genreWeights=tf.variable(tf.randomNormal([numGenres,embDim],0,0.05));
    this.userDense1=tf.layers.dense({units:hiddenDim,activation:'relu'});
    this.userDense2=tf.layers.dense({units:embDim,activation:'linear'});
    this.itemDense1=tf.layers.dense({units:hiddenDim,activation:'relu'});
    this.itemDense2=tf.layers.dense({units:embDim,activation:'linear'});
  }
  userForward(userIdx){
    const e=tf.gather(this.userEmbedding,userIdx);
    return this.userDense2.apply(this.userDense1.apply(e));
  }
  itemForward(itemIdx,genreVec){
    const e=tf.gather(this.itemEmbedding,itemIdx);
    const g=tf.matMul(genreVec,this.genreWeights);
    const h=tf.add(e,g);
    return this.itemDense2.apply(this.itemDense1.apply(h));
  }
  score(u,i){ return tf.sum(tf.mul(u,i),-1); }
  inBatchSoftmaxLoss(U,I){
    const logits=tf.matMul(U,I,false,true);
    const labels=tf.oneHot(tf.range(0,U.shape[0],'int32'),U.shape[0]);
    return tf.losses.softmaxCrossEntropy(labels,logits).mean();
  }
  bprLoss(U,Ip,In){
    const diff=tf.sub(this.score(U,Ip),this.score(U,In));
    return tf.neg(tf.mean(tf.logSigmoid(diff)));
  }
  getUserEmbedding(idx){ return tf.tidy(()=>this.userForward(tf.tensor1d([idx],'int32')).squeeze()); }
  getAllItemEmbeddings(){ const g=tf.zeros([this.itemEmbedding.shape[0],this.genreWeights.shape[0]]);
    return tf.tidy(()=>this.itemForward(tf.range(0,this.itemEmbedding.shape[0],1,'int32'),g)); }
  dispose(){ this.userEmbedding.dispose(); this.itemEmbedding.dispose(); this.genreWeights.dispose(); }
}
window.TwoTowerModel=TwoTowerModel;
