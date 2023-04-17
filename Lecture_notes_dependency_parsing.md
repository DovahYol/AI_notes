# Dependency Parsing

## Preparation

1. Phrase structure
2. Dependency structure
3. sentence structure
4. Prepositional phrase attachment ambiguity
5. PP attachment ambiguities multiply
6. Coordination scope ambiguity
7. Adjectival/Adverbial Modifier Ambiguity
8. Verb Phrase (VP) attachment ambiguity
9. Treebanks
10. **transition-based parsers**, graph-based parsers, and feature-based parsers
11. **A Fast and Accurate Dependency Parser using Neural Networks**



## Video

1. the(determiner) cat(noun) cuddly(adjective) by(preposition) door(noun)
2. *the cuddly cat* is noun phrase, whereas *by the door* is prepositional phrase.
3. *the cuddly cat by the door* is noun phrase.
4. All they are non-terminals in context-free grammars (CFG).
5. NP -> Det N
6. NP -> Det (Adj)* N (PP), where ()* means 0, 1, ..., infinite, () means 0 or 1.
7. PP -> P NP
8. VP -> V PP
9. S -> NP VP
10. Above are **constituency grammars**.
11. It turns out that in modern NLP, starting I guess around 2000, NLP people are really swung behind **dependency grammars**.
12. Look in the large crate in the kitchen by the door. crate is **head**, large is **dependent**; door is **head**, the is **dependent**.
13. Universal dependency
14. A model need to understand sentence structure in order to be able to interpret language correctly.
15. Prepositional phrase(PP) attachment ambiguity / Coordination scope ambiguity / Adjectival/Adverbial Modifier Ambiguity / Verb Phrase (VP) attachment ambiguity
16. Dependency syntax postulates that syntatic structure consists of relations between lexical items, normally binary asymmetric relations("arrows") called dependencies.
17. Dependency grammars are much natrual for free-er word order, such as Russian, Latin.
18. Dependency grammars were also very prominent in the very beginnings of computational linguistics.
19. Universal Dependency **Treebank**: https://universaldependencies.org/
20. The rise of treebanks turn parser building into an empirical science, where people can then compete rigorously on the basis of accuracy difference.
21. How do we build a parser once we've got dependencies?
22. Methods of Denpency Parsing
    1. Dynamic Programming
    2. Graph Algorithms
       1. MSTParser
       2. TurboParser
    3. Constraint Satisfaction
    4. "Transition-based parsing" or "deterministic dependency parsing"
       1. Becomes prominent since "Greedy transition-based parsing [Nivre 2003]"
       2. MaltParser [Nivre and Hall 2005] is fractionally below the SOTA but provides very fast linear time parsing.
    5. Neural dependency parser
       1. A neural dependency parser [Chen and Manning 2014]
       2. A neural graph-based dependency parser [Dozat and Manning 2017; Dozat, Qi, and Manning 2017]
23. Evaludation: UAS vs. LAS