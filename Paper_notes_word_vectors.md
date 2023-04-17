# Word Vectors

## 1 Word2Vec

### 1.1 L1

1. **less than a day** to learn high quality word vectors from a **1.6 billion words** data set

2. **N-gram** model is popular before this paper.
3. **distributed** representations of words
4. **millions** of words in the vocabulary.
5. **multiple** degrees of similarity 
6. word offset
7. **syntactic** and **semantic** regularities
8. how **training time** and **accuracy** depends on the dimensionality of the word vectors and on the amount of the training data
9. **log-bilinear** model 
10. well-known Latent Semantic Analysis (**LSA**) and Latent Dirichlet Allocation (**LDA**)
11. training complexity is proportional to $ O = E \times T \times Q $.
12. Feedforward Neural Net Language Model (**NNLM**)
13. **hierarchical** versions of the softmax
14. **Huffman binary tree**
15. obtaining classes
16. $ log_2(Unigram\_perplexity(V )) $
17. time complextiy depends heavily on the efficiency of the **softmax normalization**
18. Models are implemented on top of a large-scale distributed framework called DistBelief.
19. One hundred or more model **replicas**
20. **log-linear** models
21. **CBOW**: $ Q = N \times D + D \times \log_2(V ) $
22. **CSM**: $ Q = C \times (D + D \times log_2(V )) $
23. Previous work **evaluate** the similarity of different words intuitively.
24. Evaluate the ability of analogy: $ X = vector(``biggest") − vector(``big") + vector(``small") $
25. Dataset: Google News corpus, **6B** tokens, vocabulary size: **1M** most frequent words.
26. We call this archi- tecture a **bag-of-words** model as the order of words in the history does not influence the projection.

### 1.2 L2

1. Introduce the original two Word2Vec methods: Continuous Bag-Of-Word Model (CBOW) and Continuous Skip-gram Model (CSM).

## 2 Negative Sampling

### 2.1 L1

1. subsampling
2. We also describe a simple alternative to the **hierarchical softmax** called negative sampling.
3. Skip-gram model: an **optimized** **single-machine** implementation can train on more than **100 billion words** in **one day**.
4. **subsampling** speedups training and improves accuracy of representations of **less** frequent words.
5. Noise Contrastive Estimation (**NCE**) for faster training and better vector representations for **frequent** words, compared to more complex hierarchical softmax.
6. phrase vectors
7. analogical reasoning tasks
8. **hinge loss**
9. the vector representations of frequent words **do not change significantly** after training on several million examples.
10. Each word $ w_i $ in the training set is discarded with probability computed by the formula $ P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} $
11. Most frequent words can easily occur hundreds of million times, but they provide less information than the rare words.
12. syntactic and semantic in Chinese
13. the best representations of phrases are learned by a model with the hierarchical softmax and subsampling
14. additive property of the vectors
15. Why the score be strucured in this way and how's it used? $ score(w_i, w_j) = \frac{count(w_iw_j) - \delta}{count(w_i) \times count(w_j)} $

### 2.2 L2

1. Introduce four extensions of the original proposed methods: 1) Negative sampling; 2) Subsampling; 3) Phrase vectors; 4) Explaining the additive property of the vectors.

2. It performs significantly well for that training 1000d vectors on phrases using over 30 billion words takes only 1 day.

3. Negative sampling: three steps why the author came up with this.

   1. The original loss function is **impractical** because computing $ \nabla{\log{p(w_O \mid w_I )}}$ is proportional to W, which is often large($ 10^5 - 10^7 $ terms).
      $$
      \frac{1}{T} \sum_{t=1}^T \sum_{-c \le j \le c, ~ j \ne 0} \log{p(w_{t + j} \mid w_t)} \\
      p(w_O | w_I) = \frac{\exp{{v_{w_O}^\prime}^T v_{w_I}}}{\sum_{w=1}^W \exp{{v_{w}^\prime}^T v_{w_I}}}
      $$
      
   2. Then the paper employs **Hierachical Softmax**, it is only needed to evaluate $ \log_2(W) $ nodes. It uses a binary tree representation of the output layer with W words as its leaves. Let $ n(w, j) $ be the $ j $-th node on the path from the root to $ w $, let $ L(w) $ be the length of the path, so $ n(w, 1) = root $ and $ n(w, L(w)) = w $. For inner node $ n $, let $ ch(n) $ be an arbitrary fixed child of n and let $ [x] $ be 1 if x is true and -1 otherwise. It learns representations not only for words but also for every inner nodes. I understand it in a intuitive way that inner nodes are virtual or distilled context nodes, we randomly select some of them as our current context nodes and others as noises.
      $$
      p(w \mid w_I )= \prod_{j = 1}^{L(w) - 1} \sigma([n(w, j + 1) \, = ch(n(w, j))] \cdot {v_{n(w,j)}^\prime}^T v_{w_I} )
      $$
   
   3. An alternative to hierachical softmax is **Noise Contrastive Estimation (NCE)**, **Negative sampling (NEG)** is a simplified version of it and it works better as shown in empirical results. $ k $ is in the range 5-20 for small datasets and 2-5 for large datasets. The number of nodes evaluated is typically smaller than Hierachical Softmax. $ U(w) $ is unigram distribution and Z is a normalization constant.
      $$
      \log{p(w_O \mid w_I) = \log{\sigma({v_{w_O}^\prime}^\intercal v_{w_I})}} ~ + ~ \sum_{i = 1}^k{\mathbb{E}_{w_i \sim P_n(w)} [\log{\sigma({-{v_{w_i}^\prime}}^\intercal v_{w_I})}]}
      \\
      P_n(w) = \frac{{U(w)}^{3/4}}{Z}
      $$
   
4. Subsampling is for the reason that most frequent words can easily occur **hundreds of millions** of times while it provides less context information than the rare words. And their representations don't change too much after trained on **several million** examples. The method is that for each word $$ w_i $$ in the training set, we discard it with probability. $$ f(w_i) $$ is its frequency and $ t $ is a chosen threshold. It both speedups training and improves representations of rare words.
   $$
   P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}
   $$

5. Representations of phrases now can be learnt with the following steps: 1) Decide phrases based on the unigram and bigram counts, using $ score(w_i, w_j) = \frac{count(w_i w_j) - \delta}{count(w_i) \times count(w_j)} $; 2) Run the procedure 2-4 passes with decreasing the threshold, allowing longer phrases that consists of several words to be formed.

6. An angle why additive property of the vectors work is that the sum of two word vectors is related to the product of the two context distributions.

## 3 GloVe

### 3.1 L1

1. origin of these regularities
2. global matrix factorization
3. local context window
4. A fusion of count-based and prediction-based methods?
5. produce dimensions of meaning capturing the multi-clustering idea of meaning.
6. latent semantic analysis (LSA)
7. weighted least squares model
8. have roots stretching as far back as LSA
9. low-rank approximations
10. thermodynamic phase
11. desiderata
12. $ F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}} $
13. vector spaces are inherently linear structures
14. obfuscate
15. homomorphism
16. $ (\mathbb{R},+) $
17. $ (\mathbb{R}_{\ge{0}}, \times) $
18. $ w_i^{\intercal}\tilde{w}_k + b_i + \tilde{b}_k = \log(X_{ik}) $
19. weighted least squares regression model
20. least squares problem
21. $ J = \sum_{i,j=1}^V f(X_{ij})(w_i^{\intercal}\tilde{w}_j + b_i + \tilde{b}_j - \log{X_{ij}})^2 $
22. $ f(x) = \begin{cases} (x/x_{max})^\alpha & \text{if } x < x_{max} \\ 1 & \text{otherwise .} \end{cases} $
23. global skip-gram
24. **cross entropy error** has the unfortunate property that distributions with long tails are of- ten modeled poorly with too much weight given to the unlikely events
25. A natural choice would be a **least squares** objective in which **normalization** factors in *Q* and *P* are **discarded**
26. $ \hat{J} = \sum_{i, j}f(X_{ij})(w_i^\intercal \tilde{w}_j - logX_{ij})^2 $
27. $ \mathcal{O}(|V|^2) $ vs. $ \mathcal{O}(|C|) $
28. power-law function
29. $ X_{ij} = \frac{k}{(r_{ij})^\alpha} $
30. generalized harmonic number
31. Riemann zeta function
32. $ \mathcal{O}(|C|^{0.8}) $
33. In all cases we use a decreasing weighting function, so that word pairs that are *d* words apart contribute 1/*d* to the total count.
34. Skip-gram is to capture cross-entropy error as GloVe is to capture least square error.
35. Spearman’s rank correlation coefficient on word similarity task
36. F1 score on NER
37. Performance is better on the syntactic subtask for small and asymmetric context windows, which aligns with the intuition that syntactic infor- mation is mostly drawn from the immediate context and can depend strongly on word order. Semantic information, on the other hand, is more frequently non-local, and more of it is captured with larger window sizes.
38. assimilate
39. harmonic mean
40. $ r_s = 1 - \frac{6\sum{d_i^2}}{n(n^2 - 1)} $

### 3.2 L2

1. The paper found that the evaluation scheme for analogy task favors models that produce dimensions of meanings, thereby capturing the multi-cluster idea of distributed representations.

2. $ F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}} $ characterize that if the $ \tilde{w}_k $ is more relevant to $ w_i $, the ratio is larger than 1; if the $ \tilde{w}_k $ is more relevant to $ w_j $, the ratio is smaller than 1. So the ratio is mainly decided by the $ \tilde{w}_k $ and the difference of $ w_i $ and $ w_j $. A new formula out of this motivation is $ F((w_i - w_j)^\intercal \tilde{w}_k) = \frac{P_{ik}}{P_{jk}} $. To make our final model invariant under relabling, we require that F be a homomorphism between the groups $ (\mathbb{R},+) $ and $ (\mathbb{R}_{\ge{0}}, \times) $, i.e., $ F((w_i - w_j)^\intercal \tilde{w}_k) = \frac{F(w_i^\intercal\tilde{w}_k)}{F(w_j^\intercal\tilde{w}_k)} $. Then we have $ F(w_i^\intercal\tilde{w}_k) = P_{ik} = \frac{X_{ik}}{X_i} $. Exp is one of solution to F to keep it homomorphism. So we have $ \exp{w_i^\intercal\tilde{w}_k} = \frac{X_{ik}}{X_i} $, which can be transformed to $ w_i^\intercal\tilde{w}_k + \log{X_i} = \log{X_{ik}} $. We add a bias to restore the symmetry, $ w_i^{\intercal}\tilde{w}_k + b_i + \tilde{b}_k = \log(X_{ik}) $, left hand is an approximation to right hand, our objective would be decrease the gap as much as possible. And we also concentrate more on frequent co-occurences. So the final objective function would be
   $$
   J = \sum_{i,j=1}^V f(X_{ij})(w_i^{\intercal}\tilde{w}_j + b_i + \tilde{b}_j - \log{X_{ij}})^2
   \\
   \text{where } f(x) = \begin{cases} (x/x_{max})^\alpha & \text{if } x < x_{max} \\ 1 & \text{otherwise .} \end{cases}
   $$

3. The objective function of skip-gram is $ J = -\sum_{i \in corpus, ~ j \in context(i)} \log{Q_{ij}} $, where  $ Q_{ij} = \frac{\exp(w_i^\intercal\tilde{w}_j)}{\sum_{k=1}^V \exp(w_i^\intercal\tilde{w}_k)} $. The relation between skip-gram and GloVe is that the objective function of skip-gram can be **rewritten** to $ J = \sum_{i=1}^V X_i H(P_i, Q_i) $, it's a weighted sum of cross entropy. Considering that cross entropy attends too much to unlikely events, if we instead use least square error, it can be $ \hat{J} = \sum_{i, j} X_i(\hat{P}_{ij} - \hat{Q}_{ij})^2 $, where $ \hat{P}_{ij} = X_{ij} $ and $ \hat{Q}_{ij} = \exp(w_i^\intercal \tilde{w}_j) $. In the fact of that $ X_i $ often takes large value, which complicate the optimization, we further modify this equation to $ \hat{J} = \sum_{i, j}f(X_{ij})(w_i^\intercal \tilde{w}_j - logX_{ij})^2 $, which is equivalent to the cost function of GloVe.

4. Complexity of the model: the computational complexity of the model depends  on the number of nonzero elements in the matrix $ X $. $ X_{ij} $ can be modeled as a power-law function of the frequency rank of that word pair, $ r_{ij} $:  $ X_{ij} = \frac{k}{(r_{ij})^\alpha} $. After some inductions, we have
   $$
   |X| = \begin{cases} \mathcal{O}(|C|) & \text{ if } \alpha \lt 1, \\
    \mathcal{O}(|C|^{1/\alpha}) & \text{ if } \alpha \gt 1
    \end{cases}
   $$
   For the corpora studied in this article, $ \alpha = 1.25$, then $ |X| = \mathcal{O}(|C|^{0.8}) $, it's better than "global skip-gram" model which scale like $ \mathcal{O}(|C|) $.

5. Experiments:

   1. When construcing co-occurence matrix X,  they use a decreasing weighting function, so that word pairs that are *d* words apart contribute 1/*d* to the total count.
   2. A context window that extends to both side of the word would be called symmetric, whereas extends to only the left side would be called asymmetric. Asymmetric works better than symmetric on **syntactic** subtask when the window is still small; Asymmetric works better than symmetric on **semantic** subtask when the window is enough big. Generally speaking, Bi-directional is not a good choice although BERT is prevalent for time being.
   3. Evaluated on three tasks: 1) analogy tasks with accuracy as the metric; 2) similarity tasks with Spearman’s rank correlation as the metric; 3) NER tasks with F1 score as the metric.
   4. Spearman’s rank correlation: it measures the correlation between the rankings of the word pairs based on their human-annotated similarity scores and the rankings based on the cosine similarity of their embeddings. It's modeled as $ \rho = 1 - \frac{6\sum{d_i^2}}{n(n^2 - 1)} $.
   5. F1 score: it's a harmonic mean of precision and recall, which is $ \text{F1 score} = 2 * (precision * recall) / (precision + recall) $. Formula for harmonic mean is $ H_n = \frac{n}{\frac{1}{x_1} + \frac{1}{x_2} + \dots + \frac{1}{x_n}} $.

## 4 DistSim-WordEmb

### 4.1 L1

1. plethora
2. controlled-variable experiments
3. commend
4. culminated
5. SGNS
6. pointwise mutual information (PMI)
7. implicitly factorizing a word-context PMI matrix
8. thereafter
9. dynamically-sized
10. smoothing of the negative sampling distribution
11. consistent advantage
12. explicit PPMI matrix
13. SVD factorization
14. essentially bag-of-words models
15. All vectors are normalized to unit length before they are used for similarity calculation, making cosine similarity and dot-product equivalent (see Section 3.3 for further discussion). Won't it affect performance? Vectors in the same direction is normalized to the same thing.
16. marginal probabilities
17. $ M^{\text{PMI}} $ and $ M_0^{\text{PMI}} $
18. *positive* PMI (PPMI)
19. $ PPMI(w, c) = \log\frac{\hat{P}(w, c)}{\hat{P}(w)\hat{P}(c)} = \log\frac{\sharp(w, c) \cdot |D|}{\sharp(w) \cdot \sharp(c)} $
20. $ PPMI(w, c) = max(PMI(w, c), 0) $
21. Known defect of PMI which persists in PPMI is its bias towards infrequent events.
22. A common method of doing so is truncated Singular Value Decomposition (SVD), which finds the optimal rank d factorization with respect to L2 loss
23. SVD factorizes M into the product of three matrices U·Σ·V⊤,where U and V are orthonormal and Σ is a diagonal matrix of eigenvalues in decreasing order. By keeping only the top d elements of Σ,we obtain Md =Ud·Σd·V⊤.The d dot-products between the rows of W = Ud · Σd are equal to the dot-products between rows of Md.
24. The dot-products between the rows of $ W = U_d \cdot \Sigma_d $ are equal to the dot-products between rows of $ M_d $, $ M_d = U_d \cdot \Sigma_d \cdot V_d^\intercal $
25. If we were to fix bw = log#(w) and bc = log #(c), this would be almost equivalent to factorizing the PMI matrix shifted by log(|D|).
26. Note that column normalization is akin to dismissing the eigenvalues in SVD. While the hyperparameter setting eig = 0 has an important positive impact on SVD, the same cannot be said of column normalization on other methods.
27. 3CosAdd, 3CosMul
28. UKWaC

### 4.2 L2

1. This paper argues that much of the performance gains of word embeddings over count-based models are due to certain system design choices and hyperparameter optimization. These modifications can be transferred to traditional distributional models, yielding similar gains.

2. Notation: we assume a collection of words $ w \in V_W $ and their contexts $ c \in V_C $, where $ V_W $ and $ V_C $ are the word and context vocabularies, and denote the collection of observed word-context pairs as $ D $. We use $ \#(w,c) $ to denote the number of times the pair $ (w, c) $ appears in $ D $. Similarly, $ \#(w) = \sum_{c^\prime \in V_C} \#(w, c^\prime) $ and $ \#(c) = \sum_{w^\prime \in V_W} \#(w^\prime, c) $ are the number of times $ w $ and $ c $ occured in $ D $, respectively. Each word $ w \in V_W $ is associated with a vector $ \vec{w} \in \mathbb{R}^d $ and similarly each contex $ c \in V_C $ is represented as $ \vec{c} \in \mathbb{R}^d $. We sometimes refer to the vectors $ \vec{w} $ as rows in a $ |V_W| \times d $ matrix $ W $, and to the vectors $ \vec{c} $ as rows in a $ |V_C| \times d $ matrix $ C $. When referred to embeddings produced by a specific method $ x $, we may use $ W^x $ and $ C^x $.

3. PPMI matrix: Each matrix cell $ M_{ij} $ represents the association between the word $ w_i $ and the context $ c_j $. A popular measure of this association is pointwise mutual information (PMI) (Church and Hanks, 1990). Posivitve PMI(PPMI) replaces all negatives by zero. A well-known shortcoming is its bias towards infrequent events.
   $$
   PMI(w, c) = \log{\frac{\hat{P}(w, c)}{\hat{P}(w)\hat{P}(c)}} = \log{\frac{\#(w, c) \cdot |D|}{\#(w) \cdot \#(c)}} \\
   PPMI(w, c) = \max(PMI(w, c), 0)
   $$

4. SVD: SVD factorizes M into the product of three matrices $ U \cdot \Sigma \cdot V^\intercal $, where $ U $ and $ V $ are orthonormal and $ \Sigma $ is a diagonal matrix of eigenvalues in decreasing order. By keeping only the top $ d $ elements of $ \Sigma $, we obtain $ M_d =U_d \cdot \Sigma_d \cdot V_d^\intercal $. The $ d $

   dot-products between the rows of $ W = U_d · \Sigma_d $ are equal to the dot-products between rows of $ M_d $. It can be proved:
   $$
   \text{We eliminate d suffix for simplicity.} \\
   M_i \cdot M_j^\intercal = (W_i \cdot V^\intercal) \cdot (W_j \cdot V^\intercal)^\intercal 
   = (W_i \cdot V^\intercal) \cdot (V \cdot W_j^\intercal)
   = W_i \cdot W_j^\intercal
   $$
   In this way, d-dimensional of $ W $ can **substitute** the very high-dimensional rows of $ M $. A common approach in NLP literature is factorizing the PPMI matrix $ M^{\text{PPMI}} $ with SVD, and then:
   $$
   W^{SVD} = U_d \cdot \Sigma_d \\
   C^{SVD} = V_d
   $$

5. SGNS: SGNS's corpus level objective achieve its optimal value when:
   $$
   \vec{w} \cdot \vec{c} = \text{PMI}(w, c) - \log{k} \\
   \text{Hence, } W \cdot C^\intercal = M^{\text{PMI}} - \log{k} 
   $$

6. GloVe: It try to represent $ \vec{w} $ and $ \vec{c} $ such that:
   $$
   \vec{w} \cdot \vec{c} + b_w + b_c = \log(\#(w, c)) ~ \forall(w, c) \in D \\
   \text{Hence, } M^{\log{\#(w, c)}} \approx W \cdot C^\intercal + \vec{b^w} + \vec{b^c}
   $$
   If we were to fix $ b_w = \log\#(w) $ and $ b_c = \log\#(c) $, this would be almost equivalent to factorizing the PMI matrix shifted by $ \log|D| $.

7. Transferable parameters: The first three are preprocessing prameters, next two are association metrics, last three are for post-processing.

   1. Dynamic Context Window: GloVe's weights contexts using the harmonic function. Word2Vec uniformly samples the actual window size between $1$ and $L$.

   2. Subsampling: Dilute very frequent words with a probability of $ p $, where $ f $ marks the word's corpus frequency:
      $$
      p = 1 - \sqrt{\frac{t}{f}}
      $$

   3. Deleting Rare Words: Preliminary experiments showed the effect on performance is small.

   4. Shifted PMI: SGNS is trying to optimize for each $ (w, c) $: $ PMI(w, c) - \log{k} $. This shift method can be applied to PPMI:
      $$
      SPPMI(w, c) = max(PMI(w, c) - \log{k}, 0)
      $$

   5. Context distribution Smoothing: In Word2Vec, negative examples are sampled accroding to a $ smoothed $ unigram distribution. This can be applied to PPMI by:
      $$
      PMI_\alpha(w, c) = \log\frac{\hat{P}(w, c)}{\hat{P}(w)\hat{P}_\alpha(c)} \\
      \hat{P}_\alpha(c) = \frac{\#(c)^\alpha}{\sum_c\#(c)^\alpha}
      $$
      

   6. Adding Context Vectors: GloVe add the word embedding and the context embedding together. Consider the similarity of two word:
      $$
      cos(x, y) = \frac{\vec{v_x} \cdot \vec{v_y}}{\sqrt{\vec{v_x} \cdot \vec{v_x}} \sqrt{\vec{v_y} \cdot \vec{v_y}}}
      $$
      We have $ \vec{v_x} = \vec{w_x} + \vec{c_x} $, then
      $$
      cos(x, y) = \frac{\vec{w_x}\cdot\vec{w_y} + \vec{c_x}\cdot\vec{c_y}
      + \vec{w_x}\cdot\vec{c_y} + \vec{c_x}\cdot\vec{w_y}}
      {2\sqrt{{\vec{w_x}\cdot\vec{c_x} + 1}}\sqrt{{\vec{w_y}\cdot\vec{c_y} + 1}}}
      $$
      $ (w_x \cdot w_y, c_x \cdot c_y) $ is second order similarity and $ (w_* \cdot c_*) $ is first order similarity, then
      $$
      sim(x, y) = \frac{sim_2(x,y) + sim_1(x,y)}{\sqrt{sim_1(x,x) + 1}\sqrt{sim_1(y,y) + 1}}
      $$
      This similarity measure states that words are similar if they tend to appear in similar contexts, or if they tend to appear in the contexts of each other.

   7. Eigenvalue Weighting: The following isn't the necessarily the optimal construction of $ W^{SVD} $ for word similarity tasks.
      $$
      W^{\text{SVD}} = U_d \cdot \Sigma_d \\
      C^{\text{SVD}} = V_d
      $$
      We can add a parameter $ p $ to control the eigenvalue matrix $ \Sigma $:
      $$
      W^{\text{SVD}_p} = U_d \cdot \Sigma_d^p
      $$
      The exponent $ p $ can have a significant effect on performance.

   8. Vector Normalization: We can normalize the rows, the columns or both of them for $ W $.

8. Rules of thumb from empirical results:

   - Always use context distribution smoothing (cds = 0.75) to modify PMI.

   - Do not use SVD “correctly” (eig = 1).

   - SGNS is a robust baseline.

   - With SGNS, prefer many negative samples.

   - For both SGNS and GloVe, it is worthwhile to experiment with the $ \vec{w} + \vec{c} $ variant.

   - Prefer 3CosMul over 3CosAdd for analogy tasks. The latter can be written as
     $$
     arg\max_{b^*\in V_W \textbackslash \{{a^*, b, a}\} } \cos(b^*, a^*-a+b) = 
     arg\max_{b^*\in V_W \textbackslash \{{a^*, b, a}\} } (\cos(b^*, a^*)-\cos(b^*, a)+\cos(b^*, b))
     $$
     The former can be written as
     $$
     arg\max_{b^*\in V_W \textbackslash \{{a^*, b, a}\} } \frac{\cos(b^*, a^*) \cdot \cos(b^*, b)}{\cos(b^*, a) + \epsilon}
     $$
     