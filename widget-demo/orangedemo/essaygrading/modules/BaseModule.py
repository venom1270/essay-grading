import string
import nltk
import numpy as np


class BaseModule:

    def __init__(self, corpus, corpus_sentences):
        self.word_length_threshold = 7
        self.sentence_length_threshold = 40

        self.corpus = corpus
        self.corpus_sentences = corpus_sentences

        self.filtered_tokens = [[token.lower() for token in doc if token not in string.punctuation] for doc in
                                corpus.tokens]
        self.num_of_characters = np.array([len(doc) for doc in corpus.documents])
        self.num_of_words = np.array([len(doc) for doc in self.filtered_tokens])
        self.num_of_sentences = np.array([len(doc) for doc in corpus_sentences.tokens])
        self.num_of_long_words = [len([x for x in doc if len(x) >= self.word_length_threshold])
                                  for doc in corpus.tokens]
        self.num_of_different_words = np.array([len(set(doc)) for doc in corpus.tokens])
        self.freq_words = [nltk.probability.FreqDist(doc) for doc in self.filtered_tokens]

    def _update_progressbar(self, callback, proportions, i):
        if callback is not None and proportions is not None and i is not None:
            callback(proportions[i])
            return i + 1
        else:
            return i
