# Week 1: Word Vectors

## Vocabs

synonym, hypernym, WordNet

## Key Notes

1. **WordNet**: Previously commonest NLP solution, a thesaurus containing lists of synonym sets and hypernyms. Cons are listed as below:

   - missing nuance: "proficient" is listed as synonym for "good" while it's only correct in some contexts.
   - missing new meaning of words: impossible to keep up-to-date.
   - subjective
   - require human labor to create and adapt
   - can't be used to accurately compute word similarity

2. Representing words by their context: *"You shall know a word by the company it keeps"* (J. R. Firth 1957: 11)

3. **Word Vectors**: measuring similarity as the vector dot (scalar) product

4. **Word2Vec** (Mikolov et al. 2013): framework for learning word vectors. For each position $t = 1, ..., T$, predict context words within a window of fixed size *m*, given center word $w_t$,

   1. Maximize the Likelihood = $L(\theta) = \prod_{t=1}^T \prod_{-m \le j \le m ; j \ne 0} P(w_{t+j} \mid w_t; \theta) $
   2. Minimize objective function = $J(\theta) = -\frac 1 T \log L(\theta) = -\frac 1 T \sum_{t=1}^T \sum_{-m \le j \le m ; j \ne 0} \log P(w_{t+j} \mid w_t; \theta) $

5. Computing $ P(w_{t+j} \mid w_t; \theta) $: We will use **two vectors** per word w, $ v_w $ when $w$ is a center word; $ u_w $ when w is a context word. Then for a center word c and a context word o:
   $$
   P(o \mid c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}
   $$

6. Parameters: $ \theta $ represents all the model parameters, in one long vector. In our case, with d-dimensional vectors and V-many words, we have:
   $$
   \theta = 
   \begin{bmatrix}
   v_{aardvark} \\
   v_{a} \\
   \vdots \\
   v_{zebra} \\
   u_{aardvark} \\
   u_{a} \\
   \vdots \\
   u_{zebra} \\
   \end{bmatrix}
   \in R^{2dV}
   $$

7. **TODO**: $ \frac{\partial\log(P(o \mid c))}{\partial v_c} = u_o - \sum_{x=1}^V{P(x \mid c) \, u_x} = observed - expected $, what does obeserved and expected mean? One explanation: average over all context vectors weighted by their probability.

8. **Negative sampling**: training a **skip-gram model** is expensive considering we need to compute $ \sum_{w \in V} \exp(u_w^T v_c) $, so we introduce SGNS (skip-gram negative sampling):
   $$
   J_{neg-sample} \, (u_o, v_c, U) = -\log{\sigma(u_o^T v_c)} \, - \, \sum_{k \,  \in \, {K \, sampled \, indices}}\log{\sigma(u_k^T v_c)} \\
   where ~ \sigma(x) = \frac{1}{1 + e^{-x}} \\
   $$
   k is sampled with $ P(w) = \frac{U(w)^{3/4}}{Z} $ where $ U(w) $ is unigram frequency, Z is a normalization constant.

9. **TODO**: GloVe is something to combine the usage of co-occurrence matrix and word vectors.

10. Word vector **evaluation**: Intrinsic and Extrinsic

    1. Intrinsic:

       1. Evaluate by ask syntactic **analogy** questions. cons: information is not always linear. Some insight to get good accuracy: 1) more data helps; 2) good dimension is ~300.

       $$
       d = arg\max_i{\frac{(x_b - x_a + x_c)^Tx_i}{\Vert{x_b - x_a + x_c}\Vert}}
       $$

       2. **Correlation** of two words.

    2. Extrinsic: All subsequent NLP tasks in this class.

11. **TODO**: learn word vectors w.r.t. different senses and add them together, check the paper *Linear Algebraic Structure of Word Senses, with Applications to Polysemy*.
    $$
    v_{pike} = \alpha_1 v_{pike_1} + \alpha_2 v_{pike_2} + \alpha_3 v_{pike_3} \\
    where ~ \alpha_n = \frac{f_n}{f_1+f_2+f_3}
    $$
    