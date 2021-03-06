import numpy as np
import nltk
import math
import collections
import string
import os
from orangecontrib.essaygrading.modules.BaseModule import BaseModule
from orangecontrib.essaygrading.utils.syllables_util import get_syllable_count, get_syllable_count_word

name = "Readability measures"


class ReadabilityMeasures(BaseModule):
    name = "Readability measures"

    def __init__(self, corpus, corpus_sentences):
        """
        Overrides parent __init__ and calls _load().
        :param corpus: Tokenized essay Corpus.
        :param corpus_sentences: Tokenized (by sentence) essay Corpus.
        """
        super().__init__(corpus, corpus_sentences)
        if corpus is not None and corpus_sentences is not None:
            self._load(corpus, corpus_sentences)

    def _load(self, corpus, corpus_sentences):
        """
        Calls parent _load() and sets additional parameters - caluclates number of syllables in each essay.
        :param corpus: Tokenized essay Corpus.
        :param corpus_sentences: Tokenized (by sentence) essay Corpus.
        """
        super()._load(corpus, corpus_sentences)
        self.syllable_count = get_syllable_count(self.filtered_tokens)

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
        if selected_attributes is None or selected_attributes.cbGunningFogIndex:
            gunning_fog_index = self.calculate_gunning_fog()
            print("Gunning Fog index:", gunning_fog_index)
            attribute_dictionary["gunningFogIndex"] = gunning_fog_index

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbFleschReadingEase:
            flesch_reading_ease = self.calculate_flesch_reading_ease()
            print("Flesch reading ease:", flesch_reading_ease)
            attribute_dictionary["fleschReadingEase"] = flesch_reading_ease

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbFleschKincaidGradeLevel:
            flesch_kincaid_grade_level = self.calculate_flesch_kincaid_grade()
            print("Flesch Kincaid grade level:", flesch_kincaid_grade_level)
            attribute_dictionary["fleschKincaidGradeLevel"] = flesch_kincaid_grade_level

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbDaleChallReadabilityFormula:
            dale_chall_readability_formula = self.calculate_dale_chall_readability()
            print("Dale Chall readability formula:", dale_chall_readability_formula)
            attribute_dictionary["daleChallReadabilityFormula"] = dale_chall_readability_formula

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbAutomatedReadabilityIndex:
            automated_readability_index = self.calculate_automated_readability_index()
            print("Automated readability index:", automated_readability_index)
            attribute_dictionary["automatedReadabilityIndex"] = automated_readability_index

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbSimpleMeasureOfGobbledygook:
            simple_measure_of_gobbledygook = self.calculate_simple_measure_gobbledygook()
            print("Simple measure of Gobbledygook: ", simple_measure_of_gobbledygook)
            attribute_dictionary["simpleMeasureOfGobbledygook"] = simple_measure_of_gobbledygook

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbLix:
            lix = self.calculate_lix()
            print("LIX:", lix)
            attribute_dictionary["lix"] = lix

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbWordVariationIndex:
            ovix = self.calculate_word_variation_index()
            print("Ovix: ", ovix)
            attribute_dictionary["wordVariationIndex"] = ovix

        # i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNominalRatio:
            nominal_ratio = self.calculate_nominal_ratio()
            print("nominalRatio: ", nominal_ratio)
            attribute_dictionary["nominalRatio"] = nominal_ratio

        # i = self._update_progressbar(callback, proportions, i)

        return i

    def calculate_gunning_fog(self):
        """
        Calculates Gunning fog index. (https://en.wikipedia.org/wiki/Gunning_fog_index)
        :return: Gunning fog index for each essay.
        """
        complex_words = np.array([len([w for w in doc if len(w) > 7]) for doc in self.corpus.tokens])
        gunning_fog_index = 0.4 * ((self.num_of_words / self.num_of_sentences) +
                                   100 * (complex_words / self.num_of_words))
        return gunning_fog_index

    def calculate_flesch_reading_ease(self):
        """
        Calculates Flesch reading ease. (https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
        :return: Flesch reading ease for each essay.
        """
        flesch_reading_ease = 206.835 - 1.015 * (self.num_of_words / self.num_of_sentences) - \
                              84.6 * (self.syllable_count / self.num_of_words)
        return flesch_reading_ease

    def calculate_flesch_kincaid_grade(self):
        """
        Calculates Flesch-Kincaid grade level. (https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
        :return: Flesch-Kincaid grade level for each essay.
        """
        flesch_kincaid_grade_level = 0.39 * (self.num_of_words / self.num_of_sentences) + \
                                     11.8 * (self.syllable_count / self.num_of_words) - 15.59
        return flesch_kincaid_grade_level

    def calculate_dale_chall_readability(self):
        """
        Calculates Dale-Chall readability formula. (https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula)
        A set of about 3000 common words is used.
        :return: Dale-Chall readability formula for each essay.
        """
        stemmer = nltk.stem.PorterStemmer()
        word_list = []
        # TODO: lematizacija, trenutno mislim da so nekolk previsoki rezultati; tudi ce to resim so potem problem
        #  utf-8 punctuationi...

        file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/dale_chall_word_list.txt"
        # "C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/dale_chall_word_list.txt"
        with open(file_path, "r") as word_list_file:
            word_list = [word.replace("\n", "").lower() for word in word_list_file]
        num_difficult_words = np.array([sum([1 for token in doc if token not in word_list])
                                        for doc in self.filtered_tokens])
        dale_chall_readability_formula = 0.1579 * (100 * num_difficult_words / self.num_of_words) + \
                                         0.0496 * (self.num_of_words / self.num_of_sentences)
        # add 3.6365 if percentage of difficult words exceeds 5%
        dale_chall_readability_formula += np.array([3.6365 if percentage > 0.05 else 0
                                                    for percentage in (num_difficult_words / self.num_of_words)])
        return dale_chall_readability_formula

    def calculate_automated_readability_index(self):
        """
        Calculates Automated redability index. (https://en.wikipedia.org/wiki/Automated_readability_index)
        :return: Automated readability index for each essay.
        """
        # TODO: nekolk velke score dobim (+1)
        num_letters_numbers = np.array([sum([len(token) for token in doc]) for doc in self.filtered_tokens])
        automated_readability_index = 4.71 * (num_letters_numbers / self.num_of_words) + \
                                      0.5 * (self.num_of_words / self.num_of_sentences) - 21.43
        automated_readability_index = np.array([math.ceil(val) for val in automated_readability_index])
        return automated_readability_index

    def calculate_simple_measure_gobbledygook(self):
        """
        Calculates Simple measure of Gobbledygook ("SMOG"). (https://en.wikipedia.org/wiki/SMOG)
        :return: Simple measure of Gobbledygook for each essay.
        """
        # TODO: zlogi se cist nedelajo ampak mislm da je dost dobra ocena dejanske vrednosti
        num_of_polysyllables = np.array([len([token for token in doc if get_syllable_count_word(token) >= 3])
                                         for doc in self.filtered_tokens])
        simple_measure_of_gobbledygook = 1.0430 * np.sqrt(num_of_polysyllables * (30 / self.num_of_sentences)) + 3.1291
        return simple_measure_of_gobbledygook

    def calculate_lix(self):
        """
        Calculates LIX readability test. (https://en.wikipedia.org/wiki/Lix_(readability_test))
        :return: LIX for each essay.
        """
        # Number of periods (defined by period, colon or capital first letter)
        num_of_periods = np.array([len([c for c in doc if c == '.' or c == ':']) for doc in self.corpus.documents])
        num_of_periods += np.array([len([token for token in doc if token[0].isupper()])
                                    for doc in self.corpus.tokens])
        lix = (self.num_of_words / self.num_of_sentences) + 100 * (self.num_of_long_words / self.num_of_words)
        return lix

    def calculate_word_variation_index(self):
        """
        Calculates word variation index. (https://www.aclweb.org/anthology/W11-4627.pdf)
        :return: Word variation index for each essay.
        """
        # TODO: REFERENCE: https://www.semanticscholar.org/paper/Automatic-summarization-as-means-of-simplifying-an-Sm
        #  ith-J%C3%B6nsson/76d562cbda4d4bfc74bbc9488570615b3f78841e
        # http://aclweb.org/anthology/W/W11/W11-4627.pdf
        # TODO: division by zero -napou poravlen
        result = np.log(self.num_of_words) / \
                 np.log(2 - (np.log(self.num_of_different_words) / np.log(self.num_of_words)))
        # problem ker ma en kratek stavk enako stevilo uniqe besed kot vseh besed - popravek
        ovix = np.array([val if val < 99999 else 0 for val in result])
        return ovix

    def calculate_nominal_ratio(self):
        """
        Calculates nominal ratio. Uses POS tags. (https://www.aclweb.org/anthology/W11-4627.pdf)
        :return: Nominal ratio for each essay.
        """
        pos_tag_counter = [collections.Counter([x for x in doc]) for doc in self.corpus.pos_tags]
        # clean stopwords
        pos_tag_counter = [{key: value for key, value in doc.items() if key not in string.punctuation and key != "''"}
                           for doc in pos_tag_counter]
        num_of_nouns = np.array([tags.get("NN", 0) + tags.get("NNP", 0) + tags.get("NNS", 0) + tags.get("NNPS", 0)
                                 for tags in pos_tag_counter])
        num_of_prepositions = np.array([tags.get("IN", 0) + tags.get("TO", 0)
                                        for tags in pos_tag_counter])
        num_of_participles = np.array([tags.get("VBG", 0) + tags.get("VBN", 0)
                                       for tags in pos_tag_counter])
        num_of_pronouns = np.array([tags.get("PRP", 0) + tags.get("PRP$", 0) + tags.get("WP", 0) + tags.get("WP$", 0)
                                    for tags in pos_tag_counter])
        num_of_adverbs = np.array([tags.get("RB", 0) + tags.get("RBR", 0) + tags.get("RBS", 0)
                                   for tags in pos_tag_counter])
        num_of_verbs = np.array([tags.get("VB", 0) + tags.get("VBD", 0) + tags.get("VBG", 0) + tags.get("VBN", 0) +
                                 tags.get("VBP", 0) + tags.get("VBZ", 0) for tags in pos_tag_counter])
        numerator = num_of_nouns + num_of_prepositions + num_of_participles
        denominator = num_of_pronouns + num_of_adverbs + num_of_verbs
        zeros = np.where(denominator == 0)[0]
        denominator[zeros] = 1
        numerator[zeros] = 0
        return numerator / denominator
