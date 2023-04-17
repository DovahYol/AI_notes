# Neural Machine Translation

## 1 Sequence to Sequence Learning with Neural Networks

### 1.1 L1

1. The paper read, "Our main result is that on an English to French translation task from the WMT’14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set". What's the best BLEU score as you know?
2. The paper read, "When we used the LSTM to rerank the 1000 hypotheses produced by the aforementioned SMT system, its BLEU score increases to 36.5, which is close to the previous best result on this task.". How did he use LSTM to rerank the hypotheses? Accept the token in the hypothese?
3. The paper read, "Finally, we found that reversing the order of the words in all source sentences (but not target sentences) improved the LSTM’s performance markedly, because doing so introduced many short term dependencies between the source and the target sentence which made the optimization problem easier.". Why? Average distance between source token and corresponding target token doesn't change.
4. The paper read, "Despite their flexibility and power, DNNs can only be applied to problems whose inputs and targets can be sensibly encoded with vectors of fixed dimensionality.". Transformer still has this limitation, right? Its time sequence is limited.
5. phrase-based SMT system
6. The paper read, "A qualitative evaluation supports this claim, showing that our model is aware of word order and is fairly invariant to the active and passive voice.". It seems to indicate it's a good thing that vector representations are invariant to the active and passive voice. I don't think so, they have two different meanings.
7.  deep LSTM with four layers
8. The paper read, "We search for the most likely translation using a simple left-to-right beam search decoder which maintains a small number B of partial hypotheses, where a partial hypothesis is a prefix of some translation.". Does beam search only apply in reference phase?
9. The paper read, "By doing so, the LSTM’s test perplexity dropped from 5.8 to 4.7, and the test BLEU scores of its decoded translations increased from 25.9 to 30.6.". It seems perlexity and BLEU score are two important metrics for LMs. Why did the paper compare with other models only by BLEU score, not perlexity?
10. As a result, the problem has a large “minimal time lag” [17]. By reversing the words in the source sentence, the average distance between corresponding words in the source and target language is unchanged. However, the first few words in the source language are now very close to the first few words in the target language, so the problem’s minimal time lag is greatly reduced.
11. The paper read, "To address this problem, we made sure that all sentences in a minibatch are roughly of the same length, yielding a 2x speedup.". Why does this trick help?

### 1.2 L2

1. This paper employs deep LSTM based sequence to sequence model to firstly outperfroms a phrase-based SMT baseline.
2. The model layer number is 4, each additional layer reduced perplexity by nearly 10%.
3. The model employs beam search when doing reference. A beam of size 2 provides most of the benefits of this technique.
4. The model reverse the source sentences, by doing so, LSTM's test perplexity dropped from 5.8 to 4.7, and the test BLEU scores of its decoded translations increased from 25.9 to 30.6. Although the average distance between corresponding words in source and target language doesn't change, **minimal time lag** hugely decreased, the first few words are now very close.
5. After analysis of LSTM hidden states by 2-dimensional PCA projection, it's clear that sentences have similar meanings are close to each other, no matter it's active voice or passive voice.

## 2 BLEU: a Method for Automatic Evaluation of Machine Translation

### 2.1 L1

1. adequacy, fidelity and fluency.
2. The paper reads, "The primary programming task for a BLEU implementor is to compare *n*-grams of the candidate with the *n*-grams of the reference translation and count the number of matches. ". It seems so simple.
3. $ Count_{clip} = \text{min}(Count, Max\_Ref\_Count) $
4. Modified n-gram precision
5. $ p_n = \frac{\sum_{C \in {Candidates}} \sum_{n-gram \in C} Count_{clip}(n-gram)}{\sum_{C^\prime \in {Candidates}} \sum_{n-gram^\prime \in C^\prime} Count(n-gram^\prime)} $
6. The paper reads, "Currently, case folding is the only text normalization performed before computing the precision.", what's case folding?
7. The paper reads, "We performed four pairwise t-test comparisons between adjacent systems as ordered by their aggregate average score.". What's t-test?

### 2.2 L2

1. Human evaluations of machine translation weigh many aspects including adequacy, fidelity and fluency.

2. BLEU is method for automatic evaluation of MT, basically it measures the **n-gram similarity**.

3. BLEU score composes of two parts, modified n-gram precision and sentence brevity penalty.
   $$
   Count_{clip} = \text{min}(Count, Max\_Ref\_Count) \\
   
   p_n = \frac{\sum_{C \in {Candidates}} \sum_{n-gram \in C} Count_{clip}(n-gram)}{\sum_{C^\prime \in {Candidates}} \sum_{n-gram^\prime \in C^\prime} Count(n-gram^\prime)} \\
   
   BP = \begin{cases}
   1 && \text{if } c \gt r \\
   e^{(1-r/c)} && \text{if } c \le r
   \end{cases} \\
   
   \text{BLEU=BP} \cdot \exp{(\sum_{n=1}^N w_n \log p_n)} \\
   
   \log\text{BLEU=} \min{(1 - \frac{r}{c}, 0)} + \sum_{n = 1}^N w_n \log p_n
   
   $$

4. Modified n-gram precision already penalizes longer sentences than the reference. Sentence brevity penalizes shorter sentences.

5. We call the closest reference sentence length the "best match length". $ r $ is the test corpus' effective reference length, which is by summing the best match lengths for each candidate sentence in the corpus. $ c $ is the total length of the candidate translation corpus.

## 3 N-gram Language Models

### 3.1 L2

It's a textbook concept from *Speech and Language Processing*. So we just do the L2 notes.

1. Language models offer a way to assign a probability to a sentence or other sequence of words, and to predict a word from preceding words.

2. n-grams are Markov models that estimate words from a fixed window of previous words. n-gram probabilities can be estimated by counting in a corpus and normalizing (the **maximum likelihood estimate**)
   $$
   P(w_n \mid w_{n - N + 1: n - 1}) = \frac{C(w_{n - N + 1: n - 1}w_n)}{C(w_{n - N + 1: n - 1})}
   $$

3. n-gram **language models** are evaluated extrinsically in some task, or intrinsically using **perplexity**. Minimizing perplexity is equivalent to maximizing the test set probability according to the language model. Perplexity can be treated as **weighted average branching factor**. The branching factor of a language is the number of possible next words that can follow any word.
   $$
   \text{perplexity}(W) = \sqrt[N]{\prod_{i = 1}^N \frac{1}{P(w_i \mid w_1 \dots w_{i - 1})}}
   $$
   

4. **Smoothing** algorithms provide a more sophisticated way to estimate the probability of n-grams. Commonly used smoothing algorithms for n-grams rely on lower-order n-gram counts through **backoff** or **interpolation**.

   1. **Laplace Smoothing**: simply add one to all the n-gram counts.

   2. **Add-k Smoothing**: simply add k to all the n-gram counts.

   3. **Interpolation**: 
      $$
      \hat{P}(w_n \mid w_{n - 2} w_{n - 1}) = \lambda_1(w_{n-2:n-1})P(w_n) \\
      + \lambda_2(w_{n-2:n-1})P(w_n \mid w_{n - 1}) \\
      + \lambda_3(w_{n-2:n-1})P(w_n \mid w_{n - 2} w_{n - 1}) \\
      \sum_i \lambda_i = 1
      $$
      

   4. **Backoff**: In order to give a correct probability distribution, **discounting** should be involved.
      $$
      P_{\text{BO}}(w_n \mid w_{n-N+1:n-1}) = \begin{cases} P^*(w_n \mid w_{n - N + 1: n - 1}), && \text{if } C(w_{n-N+1:n}) \gt 0 \\
      \alpha(w_{n-N+1:n-1}) P_{\text{BO}}(w_n \mid w_{n - N + 2:n - 1}), && otherwise
      \end{cases}
      $$

5. **Kneser-Ney** smoothing makes use of the probability of a word being a novel **continuation**. The interpolated **Kneser-Ney** smoothing algorithm mixes a discounted probability with a lower-order continuation probability.

