# Spelling and Grammatical Error Detection and Correction [Incomplete Project]
Built a system from scratch in Python, which can detect and correct spelling and grammatical errors in a user-given word and sentence, respectively.

# Brief Overview of Problem
The task is to build a system from scratch that can detect and correct spelling and grammatical errors in a user-given word and sentence. The whole project is divided into two parts: (i) Spelling Error Detection-Correction Task and (ii) Grammatical Error Detection-Correction Task. 

## Spelling Error Detection-Correction Task
The aim is to build a system that can correct spelling error (if there exists) in a user-given word and judge a sentence based on its grammatical correctness by the 'score of grammaticality', which can tell how grammatical a sentence is. For example: '_I have a red apple_' should have a higher score than '_apple a have I red_'. Different sub-tasks are mentioned:

**1. Tokenization:** Given a text, tokenize it into sentences and words. For various delimiters and formats, refer to [this](https://www.ibm.com/developerworks/community/blogs/nlp/entry/tokenization?lang=en).

**2. Language Model:** Using Tokenizer, build a model which can predict the probability of occurrence of the next word given some of its previous words (context/history). Use ```Gutenberg``` corpus from this [paper](https://www.comp.nus.edu.sg/~nlp/conll14st/CoNLLST01.pdf). Experiment with different N-grams (N=1,2,3) based-language models at word and character level. Implement different smoothing methods such as Laplace, Witten-Bell, Good-Turing, Kneser-Ney, Backoff, and Deleted Interpolation. Plot Zipf's curve for all N-grams (N=1,2,3) and do performance analysis on unseen data which can be part of Gutenberg corpus (Split the corpus into train-test-dev with 7:1:2 ratio).

**3. Spelling Detection-Correction:** Using Language Model, build a model that can recognize and suggest the correct spelling of a given misspelled word. The model should output top-5 suggestions for the correct spelling.

**4. Grammaticality Test:** Using Language Model and spelling correction system, designed and build a model to determine how grammatical a given sentence is. For example, '_I have a red apple_' should have a higher score than '_apple a have I red_'. 

## Grammatical Error Detection-Correction Task
The aim is to build a system that can detect grammatical errors in a user-given sentence and correct the error type. For the classification of different error types, use 'Naive Bayes'. The sub-tasks are mentioned below.

**1. Error Detection:** Given sentence in a tokenized form (see problem statement in ```p1b``` for more detail). For each token, determine if a grammatical error exists and classify the identified error category into one of 28 classes of error types whose description is given in 'table1' of the [paper](https://www.comp.nus.edu.sg/~nlp/conll14st/CoNLLST01.pdf). 

**2. Error Correction:** Once all the errors are identified and classified, give the suitable correction for each of the grammatical errors present in the sentence. 
