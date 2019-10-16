import numpy as np
import string
import language_check
import collections
from orangecontrib.essaygrading.modules.BaseModule import BaseModule
from orangecontrib.essaygrading.utils.lemmatizer import lemmatizeTokens
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

name = "Content"


class Content(BaseModule):

    name = "Content"

    def __init__(self, corpus, corpus_sentences, grades, source_texts=None, graded_corpus=None):
        self._load(corpus, corpus_sentences, grades, source_texts=source_texts, graded_corpus=graded_corpus)

    # graded corpus is used when comparing "Score point level for maximum cosine similarity over all score points"
    # and "max cosine sim"
    # if empty, just compare with "leave one out" method
    def _load(self, corpus, corpus_sentences, grades, source_texts=None, graded_corpus=None):
        if corpus is not None and corpus_sentences is not None:
            super()._load(corpus, corpus_sentences)
            self.source_texts = source_texts

            self.attributes = []

            self.pos_tag_counter = [collections.Counter([x for x in doc]) for doc in self.corpus.pos_tags]
            # clean stopwords
            self.pos_tag_counter = [{key: value for key, value in doc.items() if key not in string.punctuation and key != "''"}
                               for doc in self.pos_tag_counter]

            self.spellchecker = SpellChecker()
            self.lang_check = None
            self.lang_check_errors = None

            self.tfidf_matrix = None
            self.essay_scores = np.array(grades)
            self.cosine = None

            self.graded_corpus = graded_corpus


    def _cosine_preparation(self):
        tfidf_vectorizer = TfidfVectorizer(max_features=200000, stop_words="english",
                                           use_idf=True)

        docs = lemmatizeTokens(self.corpus, join=True)
        # append source/prompt text
        if self.source_texts is not None:
            docs.append((lemmatizeTokens(self.source_texts, join=True)[0]))

        #print(docs)
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
        #self.tfidf_matrix = self.tfidf_matrix * self.tfidf_matrix.T
        print(self.tfidf_matrix)

        self.cosine = cosine_similarity(self.tfidf_matrix)
        print(self.cosine)
        #print(self.cosine[-1][:-1])

        # Now calculate cosine for ungraded vs graded essays
        if self.graded_corpus:
            docs = lemmatizeTokens(self.corpus, join=True)
            docs += lemmatizeTokens(self.graded_corpus, join=True)
            tfidf = tfidf_vectorizer.fit_transform(docs)
            # Only take upper right quarter of the similarity matrix, because we are interested in graded vs. ungraded
            # rows are ungraded, cols are graded
            print("ALL")
            print(cosine_similarity(tfidf))
            self.cosine_graded = cosine_similarity(tfidf)[:len(self.corpus), len(self.corpus):]
            #self.essay_scores = list(np.floor(self.graded_corpus.X[:, 5] / 2))
            self.essay_scores = list(np.floor(self.essay_scores / 2))
            print("COSINE GRADED")
            print(self.cosine_graded)
        else:
            # TODO: kaj je to???
            self.essay_scores = list(np.floor(self.essay_scores / 2))

        print("Cosine preparation finished")

    def calculate_all(self, selected_attributes, attribute_dictionary, callback=None, proportions=None, i=None):
        # Useful: https://community.languagetool.org/rule/list?lang=en

        # Load language-check library if necessary (takes a while)
        if selected_attributes is None or selected_attributes.cbNumberOfPunctuationErrors or selected_attributes.cbNumberOfCapitalizationErrors \
                or selected_attributes.cbNumberOfGrammarErrors:
            print("Language Check loading")
            self.lang_check = language_check.LanguageTool("en-US")
            print("Language Check LOADED")
            self.lang_check_errors = [self.lang_check.check(doc) for doc in self.corpus.documents]
            print("Language Check END")

        if selected_attributes is None or selected_attributes.cbNumberOfSpellcheckingErrors:
            errors = self.calculate_num_spellcheck_errors()
            print("Number of spellcecking errors: ", errors)
            attribute_dictionary["numberOfSpellcheckingErrors"] = errors

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfCapitalizationErrors:
            capitalization_errors = self.calculate_num_capitalization_errors()
            print("Number of capitalization errors: ", capitalization_errors)
            attribute_dictionary["numberOfCapitalizationErrors"] = capitalization_errors

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbNumberOfPunctuationErrors:
            punctuation_errors = self.calculate_num_punctuation_errors()
            print("Number of punctuation errors: ", punctuation_errors)
            attribute_dictionary["numberOfPunctuationErrors"] = punctuation_errors

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbCosineSumOfCorrelationValues or selected_attributes.cbCosinePattern \
                or selected_attributes.cbCosineSimilarityBestEssays or selected_attributes.cbCosineSimilarityMax \
                or selected_attributes.cbCosineSimilaritySourceText:
            self._cosine_preparation()

        # TODO: https://onlinelibrary.wiley.com/doi/epdf/10.1002/j.2333-8504.2011.tb02272.x

        if (selected_attributes is None or selected_attributes.cbCosineSimilaritySourceText) and self.source_texts is not None:
            cosine_source_text = self.calculate_cosine_source_text()
            print("Cosine similarity with source text: ", cosine_source_text)
            attribute_dictionary["cosineWithSourceText"] = cosine_source_text

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbCosineSimilarityMax:
            max_similarity_scores = self.calculate_cosine_max()
            print("Cosine similarity Max: ", max_similarity_scores)
            attribute_dictionary["scoreCosineSimilarityMax"] = max_similarity_scores

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbCosineSimilarityBestEssays:
            top_essay_similarities = self.calculate_cosine_best_essays()
            print("Cosine similarity with best essay: ", top_essay_similarities)
            attribute_dictionary["cosineTopEssaySimilarityAverage"] = top_essay_similarities

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbCosinePattern:
            cos_patterns = self.calculate_cosine_pattern()
            print("Cosine Patterns: ", cos_patterns)
            attribute_dictionary["cosinePattern"] = cos_patterns

        #i = self._update_progressbar(callback, proportions, i)

        if selected_attributes is None or selected_attributes.cbCosineSumOfCorrelationValues:
            cos_weighted_sum = self.calculate_cosine_correlation_values()
            print("cos_weighted_sum: ", cos_weighted_sum)
            attribute_dictionary["cosineSumOfCorrelationValues"] = cos_weighted_sum

        #i = self._update_progressbar(callback, proportions, i)

        return i

    def calculate_num_spellcheck_errors(self):
        errors = np.array([len(self.spellchecker.unknown(tokens)) for tokens in self.corpus.tokens])
        return errors

    def calculate_num_capitalization_errors(self):
        capitalization_errors = [sum([1 for e in doc_errors if e.category == "Capitalization"])
                                 for doc_errors in self.lang_check_errors]
        return capitalization_errors

    def calculate_num_punctuation_errors(self):
        punctuation_errors = [sum([1 for e in doc_errors if e.category == "Punctuation" or e.category == "Miscellaneous"])
                              for doc_errors in self.lang_check_errors]
        return punctuation_errors

    def calculate_cosine_source_text(self):
        return self.cosine[-1][:-1]

    def calculate_cosine_max(self):
        # TODO: nisem ziher: score point dokumenta kateremu je najbolj podoben?
        max_similarity_scores = []

        # Two cases: 1. if we are calculating attributes for an already graded essay
        #            then we proceed normally via leave one out strategy
        #            2. if we are calculating attributes for ungraded essays
        #            then we have to compare ungraded essays to graded essays (self.cosine_graded)

        if self.graded_corpus is None:
            # Attributes for graded essay
            for ii in range(len(self.corpus.documents)):
                row = self.cosine[ii][:-1]
                most_similar_doc_index = list(row).index(sorted(row, reverse=True)[1])
                max_similarity_scores.append(self.essay_scores[most_similar_doc_index])
            return max_similarity_scores

        else:
            # Attributes for ungraded essay
            for ii in range(len(self.corpus.documents)):
                row = self.cosine_graded[ii][:]
                most_similar_doc_index = list(row).index(sorted(row, reverse=True)[0])
                #print(ii, most_similar_doc_index)
                max_similarity_scores.append(self.essay_scores[most_similar_doc_index])
            return max_similarity_scores

    def calculate_cosine_best_essays(self):
        # average cosine similarity with top x% score essays
        # TODO: now take 5 essay, change to relative (eg 5%)

        top_essay_indexes = [self.essay_scores.index(i) for i in sorted(self.essay_scores, reverse=True)[0:5]]
        top_essay_similarities = []
        # Two cases again
        if self.graded_corpus is None:
            for ii in range(len(self.corpus.documents)):
                c = [self.cosine[ii][x] for x in top_essay_indexes if x != ii]
                #if len(c) == 0:
                #    c = 0
                #else:
                #    c = sum(c) / len(c)
                c = sum(c) / max(1, len(c))
                top_essay_similarities.append(c)

        else:
            for ii in range(len(self.corpus.documents)):
                c = [self.cosine_graded[ii][x] for x in top_essay_indexes]
                #if len(c) == 0:
                #    c = 0
                #else:
                #    c = sum(c) / len(c)
                c = sum(c) / max(1, len(c))
                top_essay_similarities.append(c)

        return top_essay_similarities

    def calculate_cosine_pattern(self):
        # sum (score*ranking) ; ranking je lahko 1. vrstni red po podobnosti; ali 2. podobnosti po vrsti
        cos_patterns = []

        # Two cases again
        for ii in range(len(self.corpus.documents)):
            # od 1 do 12 so score pointi
            cos_correlation_values = []
            for sp in range(1, int(max(self.essay_scores))):
                sp_essays = [index for index, value in enumerate(self.essay_scores) if value == sp and index != ii]
                if self.graded_corpus is None:
                    sp_cos = [self.cosine[ii][x] for x in sp_essays]
                else:
                    sp_cos = [self.cosine_graded[ii][x] for x in sp_essays]
                if len(sp_cos) == 0:
                    sp_cos = 0
                else:
                    sp_cos = sum(sp_cos) / len(sp_cos)
                cos_correlation_values.append(sp_cos)
            cos_ratings = np.argsort(cos_correlation_values)
            cos_score_points = np.array(range(1, int(max(self.essay_scores))))
            cos_pattern = np.sum(cos_score_points * cos_ratings)
            cos_patterns.append(cos_pattern)
            # TODO: normalize with (12**2 + 11**2 ...) ??
        return cos_patterns

    def calculate_cosine_correlation_values(self):
        # sesstejs cosine zgornje polovice, odstejes cosine spodne polovice
        cos_weighted_sum = []

        # Two cases again
        for ii in range(len(self.corpus.documents)):
            cos_correlation_values = []
            for sp in range(1, int(max(self.essay_scores))):
                sp_essays = [index for index, value in enumerate(self.essay_scores) if value == sp and index != ii]
                if self.graded_corpus is None:
                    sp_cos = [self.cosine[ii][x] for x in sp_essays]
                else:
                    sp_cos = [self.cosine_graded[ii][x] for x in sp_essays]
                if len(sp_cos) == 0:
                    sp_cos = 0
                else:
                    sp_cos = sum(sp_cos) / len(sp_cos)
                cos_correlation_values.append(sp_cos)
            cos_weighted_sum.append(sum(cos_correlation_values[3:]) - sum(cos_correlation_values[:3]))
        return cos_weighted_sum

    # For optimization resaons, we check for grammar errors in Content.py module instead of Grammar.py module.
    def calculate_num_grammar_errors(self):
        # TODO: https: // www.tutorialspoint.com / python / python_spelling_check.htm
        grammar_errors = [sum([1 for e in doc_errors if e.category == "Grammar"])
                          for doc_errors in self.lang_check_errors]
        return grammar_errors
