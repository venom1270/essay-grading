import numpy as np
import nltk
import os
import collections
from orangecontrib.essaygrading.modules.BaseModule import BaseModule

name = "Lexical diversity"


class LexicalDiversity(BaseModule):

    name = "Lexical diversity"

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
        if selected_attributes is None or selected_attributes.cbTypeTokenRatio:
            type_token_ratio = self.calculate_type_token_ratio()
            print("Type Token Ratio: ", type_token_ratio)
            attribute_dictionary["typeTokenRatio"] = type_token_ratio

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbGuiraudsIndex:
            guirauds_index = self.calculate_guirauds_index()
            print("Guirauds Index: ", guirauds_index)
            attribute_dictionary["guiraudsIndex"] = guirauds_index

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbYulesK:
            yules_k = self.calculate_yules_k()
            print("Yule's K: ", yules_k)
            attribute_dictionary["yulesK"] = yules_k

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbTheDEstimate:
            d_estimate = self.calculate_d_estimate()
            print("D Estimate: ", d_estimate)
            attribute_dictionary["theDEstimate"] = d_estimate

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbHapaxLegomena:
            num_of_words_once = self.calculate_hapax_legomena()
            print("Hapax Legomena: ", num_of_words_once)
            attribute_dictionary["hapaxLegomena"] = num_of_words_once

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbAdvancedGuirardIndex:
            advanced_guiraud = self.calculate_advanced_guirauds_index()
            print("Advanced Guirauds index: ", advanced_guiraud)
            attribute_dictionary["advancedGuiraudIndex"] = advanced_guiraud

        #i = self._update_progressbar(callback, proportions, i)

        return i

    def calculate_type_token_ratio(self):
        """
        Calculates Type-token ratio.
        :return: Type-token ratio for each essay.
        """
        return self.num_of_different_words / self.num_of_words

    # https://pdfs.semanticscholar.org/a93b/a9b8f746800dc06ebeda02284cd8148d238a.pdf
    def calculate_guirauds_index(self):
        """
        Calculates Guiraud's index. (https://pdfs.semanticscholar.org/a93b/a9b8f746800dc06ebeda02284cd8148d238a.pdf)
        :return: Guiraud's index for each essay.
        """
        return self.num_of_different_words / np.sqrt(self.num_of_words)

    # https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00228
    # https://swizec.com/blog/measuring-vocabulary-richness-with-python/swizec/2528
    def calculate_yules_k(self):
        """
        Calculates Yule's K. (https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00228)
        :return:
        """
        m1 = np.array([len(fw) for fw in self.freq_words])
        stemmer = nltk.stem.porter.PorterStemmer()
        # group frequencies and check nmumber of words
        grouped_frequencies = []
        for doc in self.freq_words:
            v = collections.defaultdict(list)
            for key, value in sorted(doc.items()):
                v[value].append(stemmer.stem(key))
                #v[value].append(key)
            grouped_frequencies.append(v)
        m2 = np.array(
            [sum([freq ** 2 * len(tokens) for freq, tokens in doc.items()]) for doc in grouped_frequencies])
        # TODO: check zero division
        yules_k = 0
        yules_k = 10000 * ((m2 - m1) / (m1 * m1))
        # Yule's I
        # yules_k = 1 / yules_k
        return yules_k

    def calculate_d_estimate(self):
        """
        Calculate the D estimate.
        (http://dl.ndl.go.jp/view/download/digidepo_3500542_po_01.pdf?contentNo=1; some math required to get final formula)
        :return: D estimate for each essay.
        """
        # TODO: https://www.oit.ac.jp/japanese/toshokan/tosho/kiyou/jinshahen/55-2/01.pdf.
        # TODO: http://dl.ndl.go.jp/view/download/digidepo_3500542_po_01.pdf?contentNo=1&alternativeNo=
        # D = - (N * TTR**2) / (2 * (TTR - 1))
        # TODO: division by zero
        ttr = self.calculate_type_token_ratio()
        n = np.array([len(tokens) for tokens in self.corpus.tokens])
        d_estimate = - (n * ttr ** 2) / (2 * (ttr - 1))
        d_estimate = np.array([0 if np.isinf(val) else val for val in d_estimate]) # TODO: verjentno nepotrebno
        return d_estimate

    def calculate_hapax_legomena(self):
        """
        Calculates hapax legomena. (Number of words that appear only once in an essay.)
        :return: Hapax legomena for each essay.
        """
        return [len([word for word, freq in doc.items() if freq == 1]) for doc in self.freq_words]

    def calculate_advanced_guirauds_index(self):
        """
        Calculates Advanced Guiraud's index. (https://sci-hub.tw/10.1093/applin/24.2.197) TODO: tole je scihub :D
        :return: Advanced Guiraud's index for each essay.
        """
        # LINK: https://sci-hub.tw/10.1093/applin/24.2.197
        # advanced_tokens / sqrt(all_tokens)
        # https://pdfs.semanticscholar.org/a93b/a9b8f746800dc06ebeda02284cd8148d238a.pdf
        num_difficult_words = []
        file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/dale_chall_word_list.txt"
        # "C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/dale_chall_word_list.txt"
        with open(file_path, "r") as word_list_file:
            word_list = [word.replace("\n", "").lower() for word in word_list_file]
            for doc in self.filtered_tokens:
                difficult_words = []
                for token in doc:
                    if token not in word_list and token not in difficult_words:
                        difficult_words.append(token)
                num_difficult_words.append(len(difficult_words))
        num_difficult_words = np.array(num_difficult_words)
        advanced_guiraud = num_difficult_words / self.num_of_words
        return advanced_guiraud
