Role

-   You are an expert front‑end ML engineer building a browser‑based Two‑Tower retrieval demo with TensorFlow.js for the MovieLens 100K dataset (u.data, u.item), suitable for static GitHub Pages hosting.[classic.d2l+2](https://classic.d2l.ai/chapter_recommender-systems/movielens.html)
    

Context

-   Dataset: MovieLens 100K
    
    -   u.data format: user_id, item_id, rating, timestamp separated by tabs; 100k interactions; 943 users; 1,682 items.[kaggle+2](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
        
    -   u.item format: item_id|title|release_date|…; use item_id and title, optionally year parsed from title. [info.univ-tours](https://www.info.univ-tours.fr/~vperalta/publications/trapmd2-vp.pdf)
        
-   Goal: Build an in‑browser Two‑Tower model:
    
    -   User tower: user_id → embedding
        
    -   Item tower: item_id → embedding
        
    -   Scoring: dot product
        
    -   Loss: sampled‑softmax (in‑batch negatives) or BPR‑style pairwise; acceptable to use a simple contrastive loss with in‑batch negatives for clarity.[tensorflow+1](https://www.tensorflow.org/recommenders/examples/basic_retrieval)
        
-   UX requirements:
    
    -   Buttons: “Load Data”, “Train”, “Test”.
        
    -   Training shows live loss chart and epoch progress; after training, render 2D projection (PCA or t‑SNE via numeric approximation) of a sample of item embeddings.
        
    -   Test action: randomly select a user who has at least 20 ratings; show:
        
        -   Left: that user’s top‑10 historically rated movies (by rating, then recency).
            
        -   Right: model’s top‑10 recommended movies (exclude items the user already rated).
            
    -   Present the two lists in a single side‑by‑side HTML table.
        
-   Constraints:
    
    -   Pure client‑side (no server), runs on GitHub Pages. Fetch u.data and u.item via relative paths (place files under data/).
        
    -   Use TensorFlow.js only; no Python, no build step.
        
    -   Keep memory in check: allow limiting interactions (e.g., max 80k) and embedding dim (e.g., 32).
        
    -   Deterministic seeding optional; browsers vary.
        
-   References for correctness:
    
    -   Two‑tower retrieval on MovieLens in TF/TFRS (concepts and loss)[tensorflow+1](https://blog.tensorflow.org/2020/09/introducing-tensorflow-recommenders.html)
        
    -   MovieLens 100K format details[fontaine618.github+2](https://fontaine618.github.io/publication/fontaine-movielens-2020/fontaine-movielens-2020.pdf)
        
    -   TensorFlow.js in‑browser training guidance[techhub.iodigital+1](https://techhub.iodigital.com/articles/on-the-fly-machine-learning-in-the-browser-with-tensor-flow-js)
        

Instructions

-   Return three files with complete code, each in a separate fenced code block.
    
-   Implement clean, commented JavaScript with clear sections.
    

a) index.html

-   Include:
    
    -   Title and minimal CSS.
        
    -   Buttons: Load Data, Train, Test.
        
    -   Status area, loss chart canvas, and embedding projection canvas.
        
    -   A <div id="results"> to hold the side‑by‑side table of Top‑10 Rated vs Top‑10 Recommended.
        
    -   Scripts: load TensorFlow.js from CDN, then app.js and two-tower.js.
        
-   Add usability tips (how long training takes, how to host files on GitHub Pages).
    

b) app.js

-   Data loading:
    
    -   Fetch data/u.data and data/u.item with fetch(); parse lines; build:
        
        -   interactions: [{userId, itemId, rating, ts}]
            
        -   items: Map itemId → {title, year}
            
    -   Build user→rated items and user→top‑rated (compute once).
        
    -   Create integer indexers for userId and itemId to 0‑based indices; store reverse maps.
        
-   Train pipeline:
    
    -   Build batches: for each (u, i_pos), sample negatives from global item set or use in‑batch negatives.
        
    -   Normalize user/item counts; allow config: epochs, batch size, embeddingDim, learningRate, maxInteractions.
        
    -   Show a live line chart of loss per batch/epoch using a simple canvas 2D plotter (no external chart lib).
        
-   Test pipeline:
    
    -   Pick a random user with ≥20 ratings.
        
    -   Compute user embedding via user tower; compute scores vs all items using matrix ops (batched for memory).
        
    -   Exclude items the user already rated; return top‑10 titles.
        
    -   Render a side‑by‑side HTML table: left = user’s historical top‑10; right = model recommendations top‑10.
        
-   Visualization:
    
    -   After training, take a sample (e.g., 1,000 items), project item embeddings to 2D with PCA (simple power method or SVD via numeric approximation) and draw scatter with titles on hover.
        

c) two-tower.js

-   Implement a minimal Two‑Tower in TF.js:
    
    -   Class TwoTowerModel:
        
        c) two-tower.js

- Implement a minimal Two-Tower in TF.js (Deep Learning version):

```js
// ======================================================
// TwoTowerModel (Deep Learning version)
// ======================================================
//
// This replaces the previous static embedding model with
// a Deep Learning MLP-based architecture for both user
// and item towers. It also integrates genre information
// as item-side features.
//

class TwoTowerModel {
  constructor(numUsers, numItems, numGenres, embDim, hiddenDim) {
    // User and item embeddings
    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.05));
    this.itemEmbedding = tf.variable(tf.randomNormal([numItems, embDim], 0, 0.05));

    // Item genre feature input (one-hot or multi-hot vectors)
    this.genreWeights = tf.variable(tf.randomNormal([numGenres, embDim], 0, 0.05));

    // Define MLP layers (shared architecture for both towers)
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

                
        -   userForward(userIdxTensor) → embeddings gather
            
        -   itemForward(itemIdxTensor) → embeddings gather
            
        -   score(uEmb, iEmb): dot product along last dim
            
    -   Loss:
        
        -   Option 1 (default): in‑batch sampled softmax
            
            -   For a batch of user embeddings U and positive item embeddings I+, compute logits = U @ I^T, labels = diagonal; apply softmax cross‑entropy.
                
        -   Option 2: BPR pairwise loss
            
            -   Sample negative items I−; loss = −log σ(score(U, I+) − score(U, I−)).
                
        -   Provide a flag to switch.
            
    -   Training step:
        
        -   Adam optimizer; gradient tape to update both embedding tables.
            
        -   Return scalar loss for UI plotting.
            
    -   Inference:
        
        -   getUserEmbedding(uIdx)
            
        -   getScoresForAllItems(uEmb, itemEmbMatrix) with batched matmul; return top‑K indices.
            
-   Comments:
    
    -   Add short comments above each key block explaining the idea (why two‑towers, how in‑batch negatives work, why dot product).
        

Format

-   Return three code blocks only, labeled exactly:
    
    -   index.html
        
    -   app.js
        
    -   two-tower.js
        
-   No extra prose outside the code blocks.
    
-   Ensure the code runs when the repository structure is:
    
    -   /index.html
        
    -   /app.js
        
    -   /two-tower.js
        
    -   /data/u.data
        
    -   /data/u.item
        
-   The UI must:
    
    -   Load Data → parse and index.
        
    -   Train → run epochs, update loss chart, then draw embedding projection.
        
    -   Test → pick a random qualified user, render a side‑by‑side table of Top‑10 Rated vs Top‑10 Recommended.
