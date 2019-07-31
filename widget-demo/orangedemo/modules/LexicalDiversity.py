import numpy as np
import nltk
import collections
from orangedemo.modules.BaseModule import BaseModule


class LexicalDiversity(BaseModule):

    def __init__(self, corpus, corpus_sentences):
        super().__init__(corpus, corpus_sentences)

        self.attributes = []
        #TODO: ?

    def calculate_all(self, selected_attributes, attribute_dictionary, callback=None, proportions=None, i=None):
        
        if selected_attributes.cbTypeTokenRatio:
            type_token_ratio = self.calculate_type_token_ratio()
            print("Type Token Ratio: ", type_token_ratio)
            attribute_dictionary["typeTokenRatio"] = type_token_ratio

        i = self._update_progressbar(callback, proportions, i)

        if selected_attributes.cbGuiraudsIndex:
            guirauds_index = self.calculate_guirauds_index()
            print("Guirauds Index: ", guirauds_index)
            attribute_dictionary["guiraudsIndex"] = guirauds_index

        i = self._update_progressbar(callback, proportions, i)

        if selected_attributes.cbYulesK:
            yules_k = self.calculate_yules_k()
            print("Yule's K: ", yules_k)
            attribute_dictionary["yulesK"] = yules_k

        i = self._update_progressbar(callback, proportions, i)

        if selected_attributes.cbTheDEstimate:
            d_estimate = self.calculate_d_estimate()
            print("D Estimate: ", d_estimate)
            attribute_dictionary["theDEstimate"] = d_estimate

        i = self._update_progressbar(callback, proportions, i)

        if selected_attributes.cbHapaxLegomena:
            num_of_words_once = self.calculate_hapax_legomena()
            print("Hapax Legomena: ", num_of_words_once)
            attribute_dictionary["hapaxLegomena"] = num_of_words_once

        i = self._update_progressbar(callback, proportions, i)

        if selected_attributes.cbAdvancedGuirardIndex:
            advanced_guiraud = self.calculate_advanced_guirauds_index()
            print("Advanced Guirauds index: ", advanced_guiraud)
            attribute_dictionary["advancedGuiraudIndex"] = advanced_guiraud

        i = self._update_progressbar(callback, proportions, i)

        return i

    def calculate_type_token_ratio(self):
        return self.num_of_different_words / self.num_of_words

    def calculate_guirauds_index(self):
        return self.num_of_different_words / np.sqrt(self.num_of_words)

    def calculate_yules_k(self):
        m1 = np.array([len(fw) for fw in self.freq_words])
        stemmer = nltk.stem.porter.PorterStemmer()
        # group frequencies and check nmumber of words
        grouped_frequencies = []
        for doc in self.freq_words:
            v = collections.defaultdict(list)
            for key, value in sorted(doc.items()):
                v[value].append(stemmer.stem(key))
            grouped_frequencies.append(v)
        m2 = np.array(
            [sum([freq ** 2 * len(tokens) for freq, tokens in doc.items()]) for doc in grouped_frequencies])
        # TODO: check zero division
        yules_k = 0
        yules_k = 1000 * ((m2 - m1) / (m1 * m1))
        # Yule's I
        # yules_k = 1 / yules_k
        return yules_k

    def calculate_d_estimate(self):
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
        return [len([word for word, freq in doc.items() if freq == 1]) for doc in self.freq_words]

    def calculate_advanced_guirauds_index(self):
        # LINK: https://sci-hub.tw/10.1093/applin/24.2.197
        # advanced_tokens / sqrt(all_tokens)
        num_difficult_words = []
        with open("C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangedemo/dale_chall_word_list.txt",
                  "r") as word_list_file:
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
