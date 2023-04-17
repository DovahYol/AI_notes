# Dependency Parsing

## 1 Incrementality in Deterministic Dependency Parsing



## 2 A Fast and Accurate Dependency Parser using Neural Networks

### 2.1 L1

1. How does unlabeled and labeled attachment scores compute?
2. parse more than 1000 sentences per second at 92.2% unlabeled attachment score on the English Penn Treebank, what about the SOTA?
3. The neural network learns compact dense vector representations of words, part-of-speech (POS) tags, and dependency labels.
4. Transition-based dependency parsing aims to predict a transition sequence from an initial configuration to some terminal configuration, which derives a target dependency parse tree.
5. The paper uses a classifier to predict the correct transition based on features extracted from the configuration.
6. The paper employ the arc-standard system.
7. The paper reads "The essential goal of a greedy parser is to predict a correct transition from T , based on one given configuration.", is it like RNN?
8. lexicalized features are indispensable
9. cube activation function
10. In the labeled version of parsing, there are in total $ |\mathcal{T}| = 2N_l + 1 $ transitions, where $ N_l $ is number of different arc labels. Why? Can you give me some examples?
11. gold parse trees
12. The training examples are synthetic data, I don't it's a good way to do evaluation.

### 2.2 L2

1. The paper proposes neural network classifier for use in a greedy, transition-based dependency parser.

2. The parser can parse more than 1000 sentences per second at 92.2% unlabeled attachment score on the English Penn Treebank.

3. Treebank: In linguistics, a treebank is a parsed text corpus that annotates syntactic or semantic structure.

4. Evaluation metrics:

   1. Labeled attachment score (LAS) = Percentage of words that get the correct head and label
   2. Unlabeled attachment score (UAS) = Percantage of words that get the correct head

5. Arc-standard system: In the arc-standard system, a *configuration* $ c = (s, b, A) $ consists of a *stack* $ s $, a *buffer* $ b $, and a set of *dependency arcs* $ A $. The initial configuration for a sentence $ w_1, \dots, w_n $ is $ s = [\text{ROOT}], b = [w_1,\dots,w_n], A = \emptyset $. A configuration $ c $ is terminal if the buffer is empty and the stack contains the single node $ \text{ROOT} $, and the parse tree is given by $ A_c $. Denoting $ s_i (i = 1,2,\dots) $ as the $ i^{\text{th}} $ top element on the stack, and $ b_i (i = 1,2,\dots) $ as the $ i^{\text{th}} $ element on the buffer, the arc-standard system defines three types of transition:

   - LEFT-ARC(l): adds an arc $ s_1 \to s_2 $ with label $ l $ and removes $ s_2 $ from the stack. Precondition: $ |s| \ge 2 $.
   - RIGHT-ARC(l): adds an arc $ s_2 \to s_1 $ with label $ l $ and removes $ s_1 $ from the stack. Precondition: $ |s| \ge 2 $.
   - SHIFT: moves $ b_1 $ from the buffer to the stack. Precondition: $ |b| \ge 1 $.

6. In the labeled version of parsing, there are in total $ |\mathcal{T}| = 2N_l + 1 $ transitions, where $ N_l $ is number of different arc labels.

7. Conventional approaches which involves manual feature engineering suffer from three problems: sparsity, incompleteness and expensive feature computation.

8. Model structure: $ x^w, x^t, x^l $ are elements from $ S^w, S^t, S^l $. In detail, $ S^w $ contains $ n_w = 18 $ elements: (1) The top 3 words on the stack and buffer: $ s_1, s_2, s_3, b_1, b_2, b_3 $; (2) The first and second leftmost / rightmost children of the top two words on the stack: $ lc_1(s_i), rc_1(s_i), lc_2(s_i), rc_2(s_i), i = 1, 2 $. (3) The leftmost of leftmost / rightmost of rightmost children of the top two words on the stack: $ lc_1(lc_1(s_i)), rc_1(rc_1(s_i)), i = 1, 2 $.

   We use the corresponding POS tags for $ S^t (n_t = 18) $, and the corresponding arc labels of words excluding those 6 words on the stack/buffer for $ S^l (n_l = 12) $.

   ![image-20230411083306483](/Users/dovahyol/Library/Application Support/typora-user-images/image-20230411083306483.png)

9. 