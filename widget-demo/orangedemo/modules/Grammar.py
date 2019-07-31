import numpy as np
import string
import spacy
import language_check
import collections
from orangedemo.modules.BaseModule import BaseModule
from orangedemo.modules.parse_tree_util import get_parse_tree_height


class Grammar(BaseModule):

    def __init__(self, corpus, corpus_sentences):
        super().__init__(corpus, corpus_sentences)

        self.attributes = []

        self.pos_tag_counter = [collections.Counter([x for x in doc]) for doc in self.corpus.pos_tags]
        # clean stopwords
        self.pos_tag_counter = [{key: value for key, value in doc.items() if key not in string.punctuation and key != "''"}
                           for doc in self.pos_tag_counter]

    def calculate_all(self, selected_attributes, attribute_dictionary, callback=None, proportions=None, i=None):
        if selected_attributes.cbNumberOfDifferentPosTags:
            num_of_different_pos_tags = self.calculate_num_different_pos_tags()
            print("Num of different POS tags: ", num_of_different_pos_tags)
            attribute_dictionary["numberOfDifferentPosTags"] = num_of_different_pos_tags

        i = self._update_progressbar(callback, proportions, i)

        if selected_attributes.cbSentenceStructureTreeHeight:
            average_tree_height = self.calculate_sentence_structure_tree_height()
            print("Average sentence structure tree height: ", average_tree_height)
            attribute_dictionary["sentenceStructureTreeHeight"] = average_tree_height

        i = self._update_progressbar(callback, proportions, i)

        if selected_attributes.cbCorrectVerbForm:
            num_verb_forms = self.calculate_correct_verb_form()
            print("Number of correct verb forms: ", num_verb_forms)
            attribute_dictionary["correctVerbForm"] = num_verb_forms

        i = self._update_progressbar(callback, proportions, i)

        #if selected_attributes.cbNumberOfGrammarErrors:
        #    grammar_errors = self.calculate_num_grammar_errors()
         #   print("Grammaer errors: ", grammar_errors)
        #    attribute_dictionary["numberOfGrammarErrors"] = grammar_errors

        #i = self._update_progressbar(callback, proportions, i)

        i = self._calculate_num_each_pos_tag(selected_attributes, attribute_dictionary, callback, proportions, i)

        return i

    def calculate_num_different_pos_tags(self):
        # TODO: med POS tagi so pike, vejce, oklepaji... treba bi blo removat stop worde
        num_of_different_pos_tags = [len(doc) for doc in self.pos_tag_counter]
        return num_of_different_pos_tags

    def calculate_sentence_structure_tree_height(self):
        # https://www.oit.ac.jp/japanese/toshokan/tosho/kiyou/jinshahen/55-2/01.pdf
        print("TREE PARSING")
        tree_parser = spacy.load("en")
        docs = [tree_parser(doc) for doc in self.corpus.documents]
        # get average sentence tree height for each doc
        average_tree_height = [[get_parse_tree_height(sent.root) for sent in doc.sents] for doc in docs]
        average_tree_height = [float(sum(doc)) / len(doc) for doc in average_tree_height]
        return average_tree_height

    def calculate_correct_verb_form(self):
        # TODO: najdi knjiznico za to ali pa izracunaj delez napak ki se nanasajo na glagole
        # torej: st_napak_na_glagolih / st_verb_pos_tagov
        num_verb_forms = np.array([tags.get("VB", 0) + tags.get("VBD", 0) + tags.get("VBG", 0) +
                                   tags.get("VBN", 0) + tags.get("VBP", 0) + tags.get("VBZ", 0)
                                   for tags in self.pos_tag_counter])
        return num_verb_forms

    def _calculate_num_each_pos_tag(self, selected_attributes, attribute_dictionary, callback, proportions, i):
        if selected_attributes.cbPosCoordinatingConjunction:
            pos_count = np.array([tags.get("CC", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_CC"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosCardinalNumber:
            pos_count = np.array([tags.get("CD", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_CD"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosDeterminer:
            pos_count = np.array([tags.get("DT", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_DT"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosExistentialThere:
            pos_count = np.array([tags.get("EX", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_EX"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosForeignWord:
            pos_count = np.array([tags.get("FW", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_FW"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosPrepositionSubordinatingConjunction:
            pos_count = np.array([tags.get("IN", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_IN"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosAdjective:
            pos_count = np.array([tags.get("JJ", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_JJ"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosComparativeAdjective:
            pos_count = np.array([tags.get("JJR", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_JJR"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosSuperlativeAdjective:
            pos_count = np.array([tags.get("JJS", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_JJS"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosListItemMarker:
            pos_count = np.array([tags.get("LS", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_LS"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosModal:
            pos_count = np.array([tags.get("MD", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_MD"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosSingularMassCommonNoun:
            pos_count = np.array([tags.get("NN", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_NN"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosPluralCommonNoun:
            pos_count = np.array([tags.get("NNS", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_NNS"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosSingularProperNoun:
            pos_count = np.array([tags.get("NNP", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_NNP"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosPluralProperNoun:
            pos_count = np.array([tags.get("NNPS", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_NNPS"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosPredeterminer:
            pos_count = np.array([tags.get("PDT", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_PDT"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosPossessiveEnding:
            pos_count = np.array([tags.get("POS", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_POS"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosPersonalPronoun:
            pos_count = np.array([tags.get("PRP", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_PRP"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosPossessivePronoun:
            pos_count = np.array([tags.get("PRP$", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_PRP$"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosAdverb:
            pos_count = np.array([tags.get("RB", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_RB"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosComparativeAdverb:
            pos_count = np.array([tags.get("RBR", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_RBR"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosSuperlativeAdverb:
            pos_count = np.array([tags.get("RBS", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_RBS"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosParticle:
            pos_count = np.array([tags.get("RP", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_RP"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosSymbol:
            pos_count = np.array([tags.get("SYM", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_SYM"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosTo:
            pos_count = np.array([tags.get("TO", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_TO"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosInterjection:
            pos_count = np.array([tags.get("UH", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_UH"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosVerbBaseForm:
            pos_count = np.array([tags.get("VB", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_VB"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosVerbPastTense:
            pos_count = np.array([tags.get("VBD", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_VBD"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosVerbPresentParticiple:
            pos_count = np.array([tags.get("VBG", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_VBG"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosVerbPastParticiple:
            pos_count = np.array([tags.get("VBN", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_VBN"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosVerbNonThirdPersonSingularPresent:
            pos_count = np.array([tags.get("VBP", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_VBP"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosVerbThirdPersonSingularPresent:
            pos_count = np.array([tags.get("VBZ", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_VBZ"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosWhDeterminer:
            pos_count = np.array([tags.get("WDT", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_WDT"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosWhPronoun:
            pos_count = np.array([tags.get("WP", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_WP"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosPossessiveWhPronoun:
            pos_count = np.array([tags.get("WP$", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_WP"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        if selected_attributes.cbPosWhAdverb:
            pos_count = np.array([tags.get("WRB", 0) for tags in self.pos_tag_counter])
            attribute_dictionary["pos_WRB"] = pos_count
        i = self._update_progressbar(callback, proportions, i)
        return i
