# Spelling and Grammatical Error Detection
Built a system from scratch in Python which can detect spelling and grammatical errors in a word and sentence respectively using N-gram based Smoothed-Language Model, Levenshtein Distance, Hidden Markov Model and Naive Bayes Classifier.

# Brief Overview of Problem
The task is to build a system from scratch which can detect spelling and grammatical errors of given word and a sentence respectively. This whole task is split into two parts: (i) Spelling Error Detection-Correction Task and (ii) Grammatical Error Detection-Correction Task. 

## Spelling Error Detection-Correction Task
The goal of this task is to build a system which can correct spelling-error (if there exists) in an input single-word and also can judge a sentence regarding its grammatical correctness. This judging is based on some kind of score aka 'score of grammaticality' which can tell how grammatical a sentence is. For example : '_I have a red apple_' should have a higher score than '_apple a have I red_'. This whole module can be divided into following modules.

**1. Tokenisation:** Given a text, task is to tokenise into sentences and words. For various delimiters and formats refer to [this](https://www.ibm.com/developerworks/community/blogs/nlp/entry/tokenization?lang=en).

**2. Language Model:** Using tokenisation build a model which can predict probability of occurence of next word given its history/context in prior. Take n=3 for building language model at word and character level. Different smoothing methods like Laplace, Witten-Bell, Good-Turing, Kneser-Ney, Backoff and Deleted Interpolation has to be used. Plot zipf's curve for all ngrams and do analysis using above mentioned smoothing methods and use best language model in final submission at both charatcer and word level. Split the corpus into train-test-dev with 8:1:1 ratio. Use 'Perplexity' for measuring the performance of LM. Dealing with open-vocabulary is must. 

**3. Spelling Detection-Correction:** Using LM, build a model that can recognise and suggest the correct spelling of a word. The model should output top-5 suggestions for a mis-spelled word. Make use of 'Levenshtein distance' to find the candidates words for source mis-spelled word.

**4. Grammaticality Test:** Using LM and spelling suggestion system to build a model from data which can give a score of grammaticality for a given sentence. This score/metric is a measure which indicates how grammatical a sentence is. For example, '_I have a red apple_' should have a higher score than '_apple a have I red_'. 

# NOTE :- 
For part 1 "gutenberg" corpus is provided and for part 2 data from this [paper](https://www.comp.nus.edu.sg/~nlp/conll14st/CoNLLST01.pdf) is provided. For task 2 you have to use LM and Naive Bayes classifier. 

## Grammatical Error Detection-Correction Task
The goal of this task is to build a system which can detect grammatical errors in an input sentence and can give the correction along with the error type. For the classification part, the constraint is to use 'Naive Bayes' which was taught in the class upto that point. Just like previous task, this too is divided into two sub-tasks:

**1. Error Detection:** Given tokens of a sentence, determine if a grammatical error exists due to that token. If errors exists, then classify each of them into one of 28 classes of error types whose description is given in table1 of the [paper](https://www.comp.nus.edu.sg/~nlp/conll14st/CoNLLST01.pdf). 

**2. Error Correction:** Once all the errors are identified and classified, give the suitable correction against each of the grammatical errors present in the sentence. 
