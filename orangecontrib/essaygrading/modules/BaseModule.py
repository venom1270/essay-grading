import string
import nltk
import numpy as np

# https://charactercounttool.com/

class BaseModule:

    def __init__(self, corpus=None, corpus_sentences=None):
        """
        Initializes module.
        :param corpus: Tokenized essay Corpus.
        :param corpus_sentences: Tokenized (by sentence) essay Corpus.
        """
        if corpus is not None and corpus_sentences is not None:
            self._load(corpus, corpus_sentences)

    def _update_progressbar(self, callback, proportions, i):
        """
        Helper function for updating widget progressbar in Orange.
        :param callback: Callback function which updates progressbar.
        :param proportions: Array of possible progress-bar progress.
        :param i: Current progress (proportions index).
        :return: Returns next progress index for calling this method next time.
        """
        if callback is not None and proportions is not None and i is not None:
            callback(proportions[i])
            return i + 1
        else:
            return i

    def _init_processing(self, corpus, corpus_sentences):
        """
        Called by _load(), initializes basic attributes shared across all modules (counting, frequency).
        :param corpus: Tokenized essay Corpus.
        :param corpus_sentences: Tokenized (by sentence) essay Corpus.
        """
        self.filtered_tokens = [[token.lower() for token in doc if token not in string.punctuation] for doc in
                                corpus.tokens]
        self.num_of_characters = np.array([len(doc) for doc in corpus.documents])
        self.num_of_characters_no_spaces = np.array([len(doc.replace(" ", "")) for doc in corpus.documents])
        self.num_of_words = np.array([len(doc) for doc in self.filtered_tokens])
        self.num_of_sentences = np.array([len(doc) for doc in corpus_sentences.tokens])
        self.num_of_long_words = [len([x for x in doc if len(x) >= self.long_word_threshold])
                                  for doc in corpus.tokens]
        self.num_of_different_words = np.array([len(set(doc)) for doc in corpus.tokens])
        self.freq_words = [nltk.probability.FreqDist(doc) for doc in self.filtered_tokens]

    def _load(self, corpus, corpus_sentences):
        """
        Sets thresholds and calls _init_processing() for calculation of basic attributes shared across modules.
        :param corpus: Tokenized essay Corpus.
        :param corpus_sentences: Tokenized (by sentence) essay Corpus.
        """
        self.word_length_threshold = 5  # old?

        self.short_word_threshold = 3
        self.long_word_threshold = 7

        self.sentence_length_threshold = 90  # old?
        self.short_sentence_threshold = 5
        self.long_sentence_threshold = 30

        self.corpus = corpus
        self.corpus_sentences = corpus_sentences

        self._init_processing(corpus, corpus_sentences)


