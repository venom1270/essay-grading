from Orange.widgets import settings
import nltk
import collections
from orangecontrib.essaygrading.modules.BaseModule import BaseModule

name = "Basic measures"


class BasicMeasures(BaseModule):
    name = "Basic measures"

    def calculate_all(self, selected_attributes, attribute_dictionary, callback=None, proportions=None, i=None):
        """
        Calculates all attributes in this module.
        :param selected_attributes: Object with attributes to calculate (boolean flags). If None, calculate all.
        :param attribute_dictionary: Attribute dicitionary which will be filled with calculated attributes.
        :param callback: Callback update function for progressbar.
        :param proportions: List of possible progressbar values.
        :param i: Index of current progressbar value.
        :return: i (index of progressbar value).
        """
        if selected_attributes is None or selected_attributes.cbNumberOfCharacters:
            num_of_characters = self.calculate_num_characters()
            print("Number of characters:", num_of_characters)
            attribute_dictionary["numberOfCharacters"] = num_of_characters

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfCharacters:
            num_of_characters_no_spaces = self.calculate_num_characters_no_spaces()
            print("Number of characters (no spaces):", num_of_characters_no_spaces)
            attribute_dictionary["numberOfCharactersNoSpaces"] = num_of_characters_no_spaces

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfWords:
            num_of_words = self.calculate_num_words()
            print("Number of words:", num_of_words)
            attribute_dictionary["numberOfWords"] = num_of_words

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfShortWords:
            num_of_short_words = self.calculate_num_short_words()
            print("Number of short words:", num_of_short_words)
            attribute_dictionary["numberOfShortWords"] = num_of_short_words

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfLongWords:
            num_of_long_words = self.calculate_num_long_words()
            print("Number of long words:", num_of_long_words)
            attribute_dictionary["numberOfLongWords"] = num_of_long_words

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbMostFrequentWordLength:
            most_frequent_lengths = self.calculate_most_frequent_word_length()
            print("Most frequent word length:", most_frequent_lengths)
            attribute_dictionary["mostFrequentWordLength"] = most_frequent_lengths

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbAverageWordLength:
            averages = self.calculate_average_word_length()
            print("Average word length:", averages)
            attribute_dictionary["averageWordLength"] = averages

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfSentences:
            num_of_sentences = self.calculate_num_sentences()
            print("Number of sentences:", num_of_sentences)
            attribute_dictionary["numberOfSentences"] = num_of_sentences

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfShortSentences:
            num_of_short_sentences = self.calculate_num_short_sentences()
            print("Number of short sentences:", num_of_short_sentences)
            attribute_dictionary["numberOfShortSentences"] = num_of_short_sentences

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfLongSentences:
            num_of_long_sentences = self.calculate_num_long_sentences()
            print("Number of long sentences:", num_of_long_sentences)
            attribute_dictionary["numberOfLongSentences"] = num_of_long_sentences

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbMostFrequentSentenceLength:
            most_frequent_lengths = self.calculate_most_frequent_sentence_length()
            print("Most frequent sentence length:", most_frequent_lengths)
            attribute_dictionary["mostFrequentSentenceLength"] = most_frequent_lengths

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbAverageSentenceLength:
            averages = self.calculate_average_sentence_length()
            print("Average sentence length:", averages)
            attribute_dictionary["averageSentenceLength"] = averages

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfDifferentWords:
            num_of_different_words = self.calculate_num_different_words()
            print("Number of different words:", num_of_different_words)
            attribute_dictionary["numberOfDifferentWords"] = num_of_different_words

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfStopwords:
            num_of_stopwords = self.calculate_num_stopwords()
            print("Number of stopwords:", num_of_stopwords)
            attribute_dictionary["numberOfStopwords"] = num_of_stopwords

        # i = self._update_progressbar(callback, proportions, i)

        return i

    def calculate_num_characters(self):
        """
        Returns number of characters in each essay.
        :return: Array of number of characters in each essay.
        """
        return self.num_of_characters

    def calculate_num_characters_no_spaces(self):
        """
        Returns number of characters in each essay.
        :return: Array of number of characters in each essay.
        """
        return self.num_of_characters_no_spaces

    def calculate_num_words(self):
        """
        Returns number of words in each essay.
        :return: Array of number of words in each essay.
        """
        return self.num_of_words

    def calculate_num_short_words(self):
        """
        Calculates number of short words in each essay.
        :return: Array of number of short words in each essay.
        """
        # With normalization: / len(doc)
        return [len([x for x in doc if len(x) <= self.short_word_threshold]) / len(doc) for doc in self.corpus.tokens]

    def calculate_num_long_words(self):
        """
        Calculates number of long words in each essay.
        :return: Array of number of long words in each essay.
        """
        # With normalization: / len(doc)
        return [len([x for x in doc if len(x) >= self.long_word_threshold]) / len(doc) for doc in self.corpus.tokens]

    def calculate_most_frequent_word_length(self):
        """
        Calculates most frequent word length in each essay.
        :return: Array of most frequent word lengths in each essay.
        """
        counters = [collections.Counter([len(x) for x in doc]) for doc in self.corpus.tokens]
        most_frequent_lengths = [max(c, key=c.get) for c in counters]
        return most_frequent_lengths

    def calculate_average_word_length(self):
        """
        Calculates average word length in each essay.
        :return: Array of average word lengths in each essay.
        """
        return [sum([len(x) for x in doc]) / len(doc) for doc in self.corpus.tokens]

    def calculate_num_sentences(self):
        """
        Returns number of sentences in each essay.
        :return: Array of number of sentences in each essay.
        """
        return self.num_of_sentences

    def calculate_num_short_sentences(self):
        """
        Calculates number of short sentences in each essay.
        :return: Array of number of short sentences in each essay.
        """
        # With normalization: / len(doc)
        return [len([x for x in doc if len(x.split(" ")) <= self.short_sentence_threshold]) / len(doc)
                for doc in self.corpus_sentences.tokens]

    def calculate_num_long_sentences(self):
        """
        Calculates number of long sentences in each essay.
        :return: Array of number of long sentences in each essay.
        """
        # With normalization: / len(doc)
        return [len([x for x in doc if len(x.split(" ")) >= self.long_sentence_threshold]) / len(doc)
                for doc in self.corpus_sentences.tokens]

    def calculate_most_frequent_sentence_length(self):
        """
        Calculates most frequent sentence length in each essay. (words)
        :return: Array of most frequent sentence lengths in each essay.
        """
        counters = [collections.Counter([len(x.split(" ")) for x in doc]) for doc in self.corpus_sentences.tokens]
        most_frequent_lengths = [max(c, key=c.get) for c in counters]
        return most_frequent_lengths

    def calculate_average_sentence_length(self):
        """
        Calculates average sentence length in each essay (words).
        :return: Array of average sentence lengths in each essay.
        """
        # return [sum([len(x) for x in doc]) / len(doc) for doc in self.corpus_sentences.tokens]
        return [self.num_of_words[i] / self.num_of_sentences[i] for i in range(len(self.corpus))]

    def calculate_num_different_words(self):
        """
        Returns number of different words in each essay.
        :return: Array of number of different words in each essay.
        """
        return self.num_of_different_words

    def calculate_num_stopwords(self):
        """
        Calculates number of stopwords in each essay.
        :return: Array of number of stopwords in each essay.
        """
        # Normalized with / len(doc)
        stopwords = nltk.corpus.stopwords.words("english")
        num_of_stopwords = [sum([1 for tok in doc if tok in stopwords]) / len(doc) for doc in self.corpus.tokens]
        return num_of_stopwords
