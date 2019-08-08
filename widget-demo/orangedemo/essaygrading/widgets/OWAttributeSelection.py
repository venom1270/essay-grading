# PACKAGE INSTALLATIONS
# conda install spacy
# python -m spacy download en

#PROBLEMATINO:
# height of sentence parse tree (VERY HIGH!!!!)
# grammar errors (VERRRRYYY HIGH!!!!!)

from orangedemo.essaygrading.modules import BasicMeasures, ReadabilityMeasures, LexicalDiversity, Grammar, Content, Coherence

import numpy as np

import Orange.data
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.text import Corpus
from orangecontrib.text import preprocess
from orangecontrib.text.tag import pos

from functools import partial


import copy
import string
from nltk.stem import WordNetLemmatizer

import concurrent.futures
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)

from AnyQt.QtCore import QThread, pyqtSlot

from orangedemo.essaygrading.Attributes import Attributes

class Task:
    """
    A class that will hold the state for an learner evaluation.
    """
    #: A concurrent.futures.Future with our (eventual) results.
    #: The OWLearningCurveC class must fill this field
    future = ...       # type: concurrent.futures.Future

    #: FutureWatcher. Likewise this will be filled by OWLearningCurveC
    watcher = ...      # type: FutureWatcher

    #: True if this evaluation has been cancelled. The OWLearningCurveC
    #: will setup the task execution environment in such a way that this
    #: field will be checked periodically in the worker thread and cancel
    #: the computation if so required. In a sense this is the only
    #: communication channel in the direction from the OWLearningCurve to the
    #: worker thread
    cancelled = False  # type: bool

    def cancel(self):
        """
        Cancel the task.

        Set the `cancelled` field to True and block until the future is done.
        """
        # set cancelled state
        self.cancelled = True
        # cancel the future. Note this succeeds only if the execution has
        # not yet started (see `concurrent.futures.Future.cancel`) ..
        self.future.cancel()
        # ... and wait until computation finishes
        concurrent.futures.wait([self.future])


class OWDataSamplerA(OWWidget):
    name = "Attribute selection"
    description = "Select desired attributes and preprocessing steps to be used in the model."
    icon = "../icons/DataSamplerA.svg"
    priority = 10

    class Inputs:
        data = Input("Corpus", Corpus)
        source_texts = Input("Source texts", Corpus)

    class Outputs:
        sample = Output("Attributes", Orange.data.Table)

    proportion = settings.Setting(50)
    commitOnChange = settings.Setting(0)

    attributeDictionary = {}

    cbNumberOfCharacters = settings.Setting(1)
    cbNumberOfWords = settings.Setting(1)
    cbNumberOfLongWords = settings.Setting(1)
    cbNumberOfShortWords = settings.Setting(1)
    cbMostFrequentWordLength = settings.Setting(1)
    cbAverageWordLength = settings.Setting(1)
    cbNumberOfSentences = settings.Setting(1)
    cbNumberOfLongSentences = settings.Setting(1)
    cbNumberOfShortSentences = settings.Setting(1)
    cbMostFrequentSentenceLength = settings.Setting(1)
    cbAverageSentenceLength = settings.Setting(1)
    cbNumberOfDifferentWords = settings.Setting(1)
    cbNumberOfStopwords = settings.Setting(1)

    cbGunningFogIndex = settings.Setting(1)
    cbFleschReadingEase = settings.Setting(1)
    cbFleschKincaidGradeLevel = settings.Setting(1)
    cbDaleChallReadabilityFormula = settings.Setting(1)
    cbAutomatedReadabilityIndex = settings.Setting(1)
    cbSimpleMeasureOfGobbledygook = settings.Setting(1)
    cbLix = settings.Setting(1)
    cbWordVariationIndex = settings.Setting(1)
    cbNominalRatio = settings.Setting(1)

    cbTypeTokenRatio = settings.Setting(1)
    cbGuiraudsIndex = settings.Setting(1)
    cbYulesK = settings.Setting(1)
    cbTheDEstimate = settings.Setting(1)
    cbHapaxLegomena = settings.Setting(1)
    cbAdvancedGuirardIndex = settings.Setting(1)
    cbNumberOfDifferentPosTags = settings.Setting(1)
    cbSentenceStructureTreeHeight = settings.Setting(1)
    cbCorrectVerbForm = settings.Setting(1)
    cbNumberOfGrammarErrors = settings.Setting(1)

    cbPosCoordinatingConjunction = settings.Setting(1)
    cbPosCardinalNumber = settings.Setting(1)
    cbPosDeterminer = settings.Setting(1)
    cbPosExistentialThere = settings.Setting(1)
    cbPosForeignWord = settings.Setting(1)
    cbPosPrepositionSubordinatingConjunction = settings.Setting(1)
    cbPosAdjective = settings.Setting(1)
    cbPosComparativeAdjective = settings.Setting(1)
    cbPosSuperlativeAdjective = settings.Setting(1)
    cbPosListItemMarker = settings.Setting(1)
    cbPosModal = settings.Setting(1)
    cbPosSingularMassCommonNoun = settings.Setting(1)
    cbPosPluralCommonNoun = settings.Setting(1)
    cbPosSingularProperNoun = settings.Setting(1)
    cbPosPluralProperNoun = settings.Setting(1)
    cbPosPredeterminer = settings.Setting(1)
    cbPosPossessiveEnding = settings.Setting(1)
    cbPosPersonalPronoun = settings.Setting(1)
    cbPosPossessivePronoun = settings.Setting(1)
    cbPosAdverb = settings.Setting(1)
    cbPosComparativeAdverb = settings.Setting(1)
    cbPosSuperlativeAdverb = settings.Setting(1)
    cbPosParticle = settings.Setting(1)
    cbPosSymbol = settings.Setting(1)
    cbPosTo = settings.Setting(1)
    cbPosInterjection = settings.Setting(1)
    cbPosVerbBaseForm = settings.Setting(1)
    cbPosVerbPastTense = settings.Setting(1)
    cbPosVerbPresentParticiple = settings.Setting(1)
    cbPosVerbPastParticiple = settings.Setting(1)
    cbPosVerbNonThirdPersonSingularPresent = settings.Setting(1)
    cbPosVerbThirdPersonSingularPresent = settings.Setting(1)
    cbPosWhDeterminer = settings.Setting(1)
    cbPosWhPronoun = settings.Setting(1)
    cbPosPossessiveWhPronoun = settings.Setting(1)
    cbPosWhAdverb = settings.Setting(1)

    cbNumberOfSpellcheckingErrors = settings.Setting(1)
    cbNumberOfCapitalizationErrors = settings.Setting(1)
    cbNumberOfPunctuationErrors = settings.Setting(1)

    cbCosineSimilaritySourceText = settings.Setting(1)
    cbCosineSimilarityMax = settings.Setting(1)
    cbCosineSimilarityBestEssays = settings.Setting(1)
    cbCosinePattern = settings.Setting(1)
    cbCosineSumOfCorrelationValues = settings.Setting(1)

    cbBasicCoherenceMeasures = settings.Setting(1)
    cbSpatialDataAnalysis = settings.Setting(1)
    cbSpatialAutocorrelation = settings.Setting(1)

    want_main_area = False

    def __init__(self):
        super().__init__()

        #: The current evaluating task (if any)
        self._task = None  # type: Optional[Task]
        #: An executor we use to submit learner evaluations into a thread pool
        self._executor = ThreadExecutor()

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No data on input yet, waiting to get something.')

        gui.separator(self.controlArea)

        self.attributeSelectionBox = gui.widgetBox(self.controlArea, "Attribute selection")

        self.tabWidget = gui.tabWidget(self.attributeSelectionBox)
        self.tabLexicalSophistication = gui.createTabPage(self.tabWidget, "Lexical sophistication", canScroll=True)
        self.tabGrammar = gui.createTabPage(self.tabWidget, "Grammar", canScroll=True)
        self.tabMechanics = gui.createTabPage(self.tabWidget, "Mechanics", canScroll=True)
        self.tabCoherence = gui.createTabPage(self.tabWidget, "Coherence and Spatial analysis", canScroll=True)

        self.lexicalSophisticationBox = gui.widgetBox(self.tabLexicalSophistication, orientation=0)

        self.basicBox = gui.widgetBox(self.lexicalSophisticationBox, "Basic measures")
        gui.checkBox(self.basicBox, self, "cbNumberOfCharacters", "Number of characters")
        gui.checkBox(self.basicBox, self, "cbNumberOfWords", "Number of words")
        gui.checkBox(self.basicBox, self, "cbNumberOfLongWords", "Number of long words")
        gui.checkBox(self.basicBox, self, "cbNumberOfShortWords", "Number of short words")
        gui.checkBox(self.basicBox, self, "cbMostFrequentWordLength", "Most frequent word length")
        gui.checkBox(self.basicBox, self, "cbAverageWordLength", "Average word length")
        gui.checkBox(self.basicBox, self, "cbNumberOfSentences", "Number of sentences")
        gui.checkBox(self.basicBox, self, "cbNumberOfLongSentences", "Number of long sentences")
        gui.checkBox(self.basicBox, self, "cbNumberOfShortSentences", "Number of short sentences")
        gui.checkBox(self.basicBox, self, "cbMostFrequentSentenceLength",
                     "Most frequent sentence length")
        gui.checkBox(self.basicBox, self, "cbAverageSentenceLength", "Average sentence length")
        gui.checkBox(self.basicBox, self, "cbNumberOfDifferentWords", "Number of different words")
        gui.checkBox(self.basicBox, self, "cbNumberOfStopwords", "Number of stopwords")

        self.readabilityMeasuresBox = gui.widgetBox(self.lexicalSophisticationBox, "Readability measures")
        gui.checkBox(self.readabilityMeasuresBox, self, "cbGunningFogIndex", "Gunning Fog index")
        gui.checkBox(self.readabilityMeasuresBox, self, "cbFleschReadingEase", "Flesch reading ease")
        gui.checkBox(self.readabilityMeasuresBox, self, "cbFleschKincaidGradeLevel", "Flesch Kincaid grade level")
        gui.checkBox(self.readabilityMeasuresBox, self,
                     "cbDaleChallReadabilityFormula", "Dale-Chall readability formula")
        gui.checkBox(self.readabilityMeasuresBox, self, "cbAutomatedReadabilityIndex", "Automated readability index")
        gui.checkBox(self.readabilityMeasuresBox, self,
                     "cbSimpleMeasureOfGobbledygook", "Simple measure of Gobbledygook")
        gui.checkBox(self.readabilityMeasuresBox, self, "cbLix", "LIX")
        gui.checkBox(self.readabilityMeasuresBox, self, "cbWordVariationIndex", "Word variation index")
        gui.checkBox(self.readabilityMeasuresBox, self, "cbNominalRatio", "Nominal ratio")

        self.lexicalDiversityBox = gui.widgetBox(self.lexicalSophisticationBox, "Lexical diversity")
        gui.checkBox(self.lexicalDiversityBox, self, "cbTypeTokenRatio", "Type-token-ratio")
        gui.checkBox(self.lexicalDiversityBox, self, "cbGuiraudsIndex", "Guiraud's index")
        gui.checkBox(self.lexicalDiversityBox, self, "cbYulesK", "Yule's K")
        gui.checkBox(self.lexicalDiversityBox, self, "cbTheDEstimate", "The D estimate TODO")
        gui.checkBox(self.lexicalDiversityBox, self, "cbHapaxLegomena", "Hapax legomena - words occuring only once")
        gui.checkBox(self.lexicalDiversityBox, self, "cbAdvancedGuirardIndex", "Advanced Guiraud's index")

        self.grammarBox = gui.widgetBox(self.tabGrammar)

        self.grammarGeneralBox = gui.widgetBox(self.grammarBox, "General", orientation=0)
        gui.checkBox(self.grammarGeneralBox, self, "cbNumberOfDifferentPosTags", "Number of different POS tags")
        gui.checkBox(self.grammarGeneralBox, self, "cbSentenceStructureTreeHeight", "Height of sentence parse tree")
        gui.checkBox(self.grammarGeneralBox, self, "cbCorrectVerbForm", "Correct verb form")
        gui.checkBox(self.grammarGeneralBox, self, "cbNumberOfGrammarErrors", "Number of grammar errors")

        self.numberOfEachPosTagBox = gui.widgetBox(self.grammarBox, "Number of each PoS tag", orientation=0)
        self.posGroup1 = gui.widgetBox(self.numberOfEachPosTagBox)
        gui.checkBox(self.posGroup1, self, "cbPosCoordinatingConjunction", "CC - Coordinating conjunction")
        gui.checkBox(self.posGroup1, self, "cbPosCardinalNumber", "CD - Cardinal number")
        gui.checkBox(self.posGroup1, self, "cbPosDeterminer", "DT - Determiner")
        gui.checkBox(self.posGroup1, self, "cbPosExistentialThere", "EX - Existential there")
        gui.checkBox(self.posGroup1, self, "cbPosForeignWord", "FW - Foreign word")
        gui.checkBox(self.posGroup1, self, "cbPosPrepositionSubordinatingConjunction",
                     "IN - Preposition or subordinating conjunction")
        gui.checkBox(self.posGroup1, self, "cbPosAdjective", "JJ - Adjective")
        gui.checkBox(self.posGroup1, self, "cbPosComparativeAdjective", "JJR - Adjective, comparative")
        gui.checkBox(self.posGroup1, self, "cbPosSuperlativeAdjective", "JJS - Adjective, superlative")
        gui.checkBox(self.posGroup1, self, "cbPosListItemMarker", "LS - List item marker")
        gui.checkBox(self.posGroup1, self, "cbPosModal", "MD - Modal")
        gui.checkBox(self.posGroup1, self, "cbPosSingularMassCommonNoun", "NN - Noun, singular or mass")
        gui.checkBox(self.posGroup1, self, "cbPosPluralCommonNoun", "NNS - Noun, plural")
        gui.checkBox(self.posGroup1, self, "cbPosSingularProperNoun", "NNP - Proper noun, singular")
        gui.checkBox(self.posGroup1, self, "cbPosPluralProperNoun", "NNPS - Proper noun, plural")
        gui.checkBox(self.posGroup1, self, "cbPosPredeterminer", "PDT - Predeterminer")
        gui.checkBox(self.posGroup1, self, "cbPosPossessiveEnding", "POS - Possesive ending")
        gui.checkBox(self.posGroup1, self, "cbPosPersonalPronoun", "PRP - Personal pronoun")
        self.posGroup2 = gui.widgetBox(self.numberOfEachPosTagBox)
        gui.checkBox(self.posGroup2, self, "cbPosPossessivePronoun", "PRP$ - Possessive pronoun")
        gui.checkBox(self.posGroup2, self, "cbPosAdverb", "RB - Adverb")
        gui.checkBox(self.posGroup2, self, "cbPosComparativeAdverb", "RBR - Adverb, comparative")
        gui.checkBox(self.posGroup2, self, "cbPosSuperlativeAdverb", "RBS - Adverb, superlative")
        gui.checkBox(self.posGroup2, self, "cbPosParticle", "RP - Particle")
        gui.checkBox(self.posGroup2, self, "cbPosSymbol", "SYM - Symbol")
        gui.checkBox(self.posGroup2, self, "cbPosTo", "TO - to")
        gui.checkBox(self.posGroup2, self, "cbPosInterjection", "UH - Interjection")
        gui.checkBox(self.posGroup2, self, "cbPosVerbBaseForm", "VB - Verb, base form")
        gui.checkBox(self.posGroup2, self, "cbPosVerbPastTense", "VBD - Verb, past tense")
        gui.checkBox(self.posGroup2, self, "cbPosVerbPresentParticiple", "VBG - Verb, present participle")
        gui.checkBox(self.posGroup2, self, "cbPosVerbPastParticiple", "VBN - Verb, past participle")
        gui.checkBox(self.posGroup2, self, "cbPosVerbNonThirdPersonSingularPresent",
                     "VBP - Verb, non-3rd person singular present")
        gui.checkBox(self.posGroup2, self, "cbPosVerbThirdPersonSingularPresent",
                     "VBP - Verb, 3rd person singular present")
        gui.checkBox(self.posGroup2, self, "cbPosWhDeterminer", "WDT - Wh-determiner")
        gui.checkBox(self.posGroup2, self, "cbPosWhPronoun", "WP - Wh-pronoun")
        gui.checkBox(self.posGroup2, self, "cbPosPossessiveWhPronoun", "WP$ - Possessive wh-pronoun")
        gui.checkBox(self.posGroup2, self, "cbPosWhAdverb", "WRB - Wh-adverb")

        self.mechanicsBox = gui.widgetBox(self.tabMechanics)
        self.writingErrorsBox = gui.widgetBox(self.mechanicsBox, "Writing errors")
        gui.checkBox(self.writingErrorsBox, self, "cbNumberOfSpellcheckingErrors", "Number of spellchecking errors")
        gui.checkBox(self.writingErrorsBox, self, "cbNumberOfCapitalizationErrors", "Number of capitalization errors")
        gui.checkBox(self.writingErrorsBox, self, "cbNumberOfPunctuationErrors", "Number of punctuation errors")

        self.cosineBox = gui.widgetBox(self.mechanicsBox, "Content similarity")
        gui.checkBox(self.cosineBox, self, "cbCosineSimilaritySourceText", "Cosine similarity with source text")
        gui.checkBox(self.cosineBox, self, "cbCosineSimilarityMax", "Score point level for maximum cosine similarity "
                                                                    "over all score points")
        gui.checkBox(self.cosineBox, self, "cbCosineSimilarityBestEssays", "Cosine similarity with essays that have "
                                                                           "highest score point level")
        gui.checkBox(self.cosineBox, self, "cbCosinePattern", "Pattern cosine")
        gui.checkBox(self.cosineBox, self, "cbCosineSumOfCorrelationValues", "Weighted sum of all cosine correlation "
                                                                             "values")

        #self.cosineBox = gui.widgetBox(self.tabCoherence, "Basic coherence measures")
        gui.checkBox(self.tabCoherence, self, "cbBasicCoherenceMeasures", "Basic coherence measures")
        gui.checkBox(self.tabCoherence, self, "cbSpatialDataAnalysis", "Spatial data analysis")
        #self.spatialAutocorrelationBox = gui.widgetBox(self.tabCoherence, "Spatial autocorrelation")
        gui.checkBox(self.tabCoherence, self, "cbSpatialAutocorrelation", "Spatial autocorrelation")


        self.optionsBox = gui.widgetBox(self.controlArea, "Tutorial...")
        gui.checkBox(self.optionsBox, self, "commitOnChange", "Commit data on selection change")
        gui.button(self.optionsBox, self, "Apply", callback=self._invalidate_results)
        self.optionsBox.setDisabled(True)

    # dva channela: corpus in source_texts
    # https://orange-development.readthedocs.io/tutorial-channels.html

    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.dataset = dataset
            self.infoa.setText('%d instances in input dataset' % len(dataset))
            p = preprocess.Preprocessor(tokenizer=preprocess.WordPunctTokenizer(),
                                        transformers=[preprocess.LowercaseTransformer()],
                                        pos_tagger=pos.AveragedPerceptronTagger())
                                        #normalizer=preprocess.WordNetLemmatizer())

            p_sentences = preprocess.Preprocessor(tokenizer=preprocess.PunktSentenceTokenizer())

            #c = Corpus(dataset)
            new_corpus = p(dataset)

            # domain 1 scores
            print(new_corpus.X[:,5])

            #new_corpus.tokens = [[token for token in doc if token not in string.punctuation] for doc in new_corpus.tokens]
            #TODO: Maybe find a better solution?
            new_corpus = copy.deepcopy(new_corpus)
            #self.Outputs.sample.send(new_corpus)
            self.optionsBox.setDisabled(False)
            self.selection()

            self.new_corpus = new_corpus

            self.new_corpus_sentences = p_sentences(dataset)
            new_corpus_sentences = self.new_corpus_sentences

            print(dataset)

            #self.infob.setText("Some document has so many tokens: " + str(len(new_corpus)))

            print(new_corpus.tokens[0])
            print(new_corpus.pos_tags[0])
            print(new_corpus.documents[0])

            print(new_corpus.attributes)

            #self.source_texts = None
            #if source_texts is not None:
            #    self.source_texts = p(source_texts)

            #self.calculateAttributes()
            #self.commit()
        else:
            self.dataset = None
            self.infoa.setText('No data on input yet, waiting to get something.')
            #self.infob.setText('')
            self.Outputs.sample.send("Attributes")
            self.optionsBox.setDisabled(False)

    @Inputs.source_texts
    def set_source_texts(self, source_texts):
        print("DELA")
        if source_texts is not None:
            p = preprocess.Preprocessor(tokenizer=preprocess.WordPunctTokenizer(),
                                        transformers=[preprocess.LowercaseTransformer()],
                                        pos_tagger=pos.AveragedPerceptronTagger())
            self.source_texts = p(source_texts)
        else:
            self.source_texts = None

    def selection(self):
        if self.dataset is None:
            return

    def checkCommit(self):
        if self.commitOnChange:
            self.commit()

    def handleNewSignals(self):
        if self.commitOnChange:
            self._update()

    def _update(self):
        if self._task is not None:
            # First make sure any pending tasks are cancelled.
            self.cancel()
        assert self._task is None

        if self.new_corpus is None or self.new_corpus_sentences is None:
            return

        a = Attributes()
        a.cbNumberOfCharacters = self.cbNumberOfCharacters
        a.cbNumberOfWords = self.cbNumberOfWords
        a.cbNumberOfLongWords = self.cbNumberOfLongWords
        a.cbNumberOfShortWords = self.cbNumberOfShortWords
        a.cbMostFrequentWordLength = self.cbMostFrequentWordLength
        a.cbAverageWordLength = self.cbAverageWordLength
        a.cbNumberOfSentences = self.cbNumberOfSentences
        a.cbNumberOfLongSentences = self.cbNumberOfLongSentences
        a.cbNumberOfShortSentences = self.cbNumberOfShortSentences
        a.cbMostFrequentSentenceLength = self.cbMostFrequentSentenceLength
        a.cbAverageSentenceLength = self.cbAverageSentenceLength
        a.cbNumberOfDifferentWords = self.cbNumberOfDifferentWords
        a.cbNumberOfStopwords = self.cbNumberOfStopwords

        a.cbGunningFogIndex = self.cbGunningFogIndex
        a.cbFleschReadingEase = self.cbFleschReadingEase
        a.cbFleschKincaidGradeLevel = self.cbFleschKincaidGradeLevel
        a.cbDaleChallReadabilityFormula = self.cbDaleChallReadabilityFormula
        a.cbAutomatedReadabilityIndex = self.cbAutomatedReadabilityIndex
        a.cbSimpleMeasureOfGobbledygook = self.cbSimpleMeasureOfGobbledygook
        a.cbLix = self.cbLix
        a.cbWordVariationIndex = self.cbWordVariationIndex
        a.cbNominalRatio = self.cbNominalRatio

        a.cbTypeTokenRatio = self.cbTypeTokenRatio
        a.cbGuiraudsIndex = self.cbGuiraudsIndex
        a.cbYulesK = self.cbYulesK
        a.cbTheDEstimate = self.cbTheDEstimate
        a.cbHapaxLegomena = self.cbHapaxLegomena
        a.cbAdvancedGuirardIndex = self.cbAdvancedGuirardIndex
        a.cbNumberOfDifferentPosTags = self.cbNumberOfDifferentPosTags
        a.cbSentenceStructureTreeHeight = self.cbSentenceStructureTreeHeight
        a.cbCorrectVerbForm = self.cbCorrectVerbForm
        a.cbNumberOfGrammarErrors = self.cbNumberOfGrammarErrors

        a.cbPosCoordinatingConjunction = self.cbPosCoordinatingConjunction
        a.cbPosCardinalNumber = self.cbPosCardinalNumber
        a.cbPosDeterminer = self.cbPosDeterminer
        a.cbPosExistentialThere = self.cbPosExistentialThere
        a.cbPosForeignWord = self.cbPosForeignWord
        a.cbPosPrepositionSubordinatingConjunction = self.cbPosPrepositionSubordinatingConjunction
        a.cbPosAdjective = self.cbPosAdjective
        a.cbPosComparativeAdjective = self.cbPosComparativeAdjective
        a.cbPosSuperlativeAdjective = self.cbPosSuperlativeAdjective
        a.cbPosListItemMarker = self.cbPosListItemMarker
        a.cbPosModal = self.cbPosModal
        a.cbPosSingularMassCommonNoun = self.cbPosSingularMassCommonNoun
        a.cbPosPluralCommonNoun = self.cbPosPluralCommonNoun
        a.cbPosSingularProperNoun = self.cbPosSingularProperNoun
        a.cbPosPluralProperNoun = self.cbPosPluralProperNoun
        a.cbPosPredeterminer = self.cbPosPredeterminer
        a.cbPosPossessiveEnding = self.cbPosPossessiveEnding
        a.cbPosPersonalPronoun = self.cbPosPersonalPronoun
        a.cbPosPossessivePronoun = self.cbPosPossessivePronoun
        a.cbPosAdverb = self.cbPosAdverb
        a.cbPosComparativeAdverb = self.cbPosComparativeAdverb
        a.cbPosSuperlativeAdverb = self.cbPosSuperlativeAdverb
        a.cbPosParticle = self.cbPosParticle
        a.cbPosSymbol = self.cbPosSymbol
        a.cbPosTo = self.cbPosTo
        a.cbPosInterjection = self.cbPosInterjection
        a.cbPosVerbBaseForm = self.cbPosVerbBaseForm
        a.cbPosVerbPastTense = self.cbPosVerbPastTense
        a.cbPosVerbPresentParticiple = self.cbPosVerbPresentParticiple
        a.cbPosVerbPastParticiple = self.cbPosVerbPastParticiple
        a.cbPosVerbNonThirdPersonSingularPresent = self.cbPosVerbNonThirdPersonSingularPresent
        a.cbPosVerbThirdPersonSingularPresent = self.cbPosVerbThirdPersonSingularPresent
        a.cbPosWhDeterminer = self.cbPosWhDeterminer
        a.cbPosWhPronoun = self.cbPosWhPronoun
        a.cbPosPossessiveWhPronoun = self.cbPosPossessiveWhPronoun
        a.cbPosWhAdverb = self.cbPosWhAdverb

        a.cbNumberOfSpellcheckingErrors = self.cbNumberOfSpellcheckingErrors
        a.cbNumberOfCapitalizationErrors = self.cbNumberOfCapitalizationErrors
        a.cbNumberOfPunctuationErrors = self.cbNumberOfPunctuationErrors

        a.cbCosineSimilaritySourceText = self.cbCosineSimilaritySourceText
        a.cbCosineSimilarityMax = self.cbCosineSimilarityMax
        a.cbCosineSimilarityBestEssays = self.cbCosineSimilarityBestEssays
        a.cbCosinePattern = self.cbCosinePattern
        a.cbCosineSumOfCorrelationValues = self.cbCosineSumOfCorrelationValues

        a.cbBasicCoherenceMeasures = self.cbBasicCoherenceMeasures
        a.cbSpatialDataAnalysis = self.cbSpatialDataAnalysis
        a.cbSpatialAutocorrelation = self.cbSpatialAutocorrelation

        calculate_attributes_func = partial(
            calculateAttributes,
            new_corpus=self.new_corpus,
            new_corpus_sentences=self.new_corpus_sentences,
            source_texts=self.source_texts,
            attr=a
        )

        # setup the task state
        self._task = task = Task()
        # The learning_curve[_with_test_data] also takes a callback function
        # to report the progress. We instrument this callback to both invoke
        # the appropriate slots on this widget for reporting the progress
        # (in a thread safe manner) and to implement cooperative cancellation.
        set_progress = methodinvoke(self, "setProgressValue", (float,))

        def callback(finished):
            # check if the task has been cancelled and raise an exception
            # from within. This 'strategy' can only be used with code that
            # properly cleans up after itself in the case of an exception
            # (does not leave any global locks, opened file descriptors, ...)
            if task.cancelled:
                raise KeyboardInterrupt()
            set_progress(finished * 100)

        # capture the callback in the partial function
        calculate_attributes_func = partial(calculate_attributes_func, callback=callback)

        self.progressBarInit()
        # Submit the evaluation function to the executor and fill in the
        # task with the resultant Future.
        task.future = self._executor.submit(calculate_attributes_func)
        # Setup the FutureWatcher to notify us of completion
        task.watcher = FutureWatcher(task.future)
        # by using FutureWatcher we ensure `_task_finished` slot will be
        # called from the main GUI thread by the Qt's event loop
        task.watcher.done.connect(self._task_finished)

    def cancel(self):
        """
        Cancel the current task (if any).
        """
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            # disconnect the `_task_finished` slot
            self._task.watcher.done.disconnect(self._task_finished)
            self._task = None

    @pyqtSlot(float)
    def setProgressValue(self, value):
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        """
        Parameters
        ----------
        f : Future
            The future instance holding the result of learner evaluation.
        """
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None
        self.progressBarFinished()

        try:
            results = f.result()  # type: List[Results]
        except Exception as ex:
            import logging
            # Log the exception with a traceback
            log = logging.getLogger()
            log.exception(__name__, exc_info=True)
            self.error("Exception occurred during evaluation: {!r}"
                       .format(ex))
        else:
            # split the combined result into per learner/model results ...

            self.attributeDictionary = results
            print("FINISHED")
            print(self.attributeDictionary)
            domain = Orange.data.Domain([Orange.data.ContinuousVariable.make(key) for key in self.attributeDictionary],
                                        Orange.data.ContinuousVariable.make("score"))

            arr = np.array([value for _, value in self.attributeDictionary.items()])

            print("INF CHECK")
            for k,i in self.attributeDictionary.items():
                if np.isinf(i).any():
                  print(k)

            out = Orange.data.Table.from_numpy(domain, np.array(arr).transpose(), self.new_corpus.X[:, 5])
           #print(out)
            self.Outputs.sample.send(out)

    # "reset"
    def _invalidate_results(self):
        self.attributeDictionary = {}
        self._update()

def calculateAttributes(new_corpus, new_corpus_sentences, source_texts, attr, callback):
    word_length_threshold = 7
    sentence_length_threshold = 40
    lemmatizer = WordNetLemmatizer()
    #new_corpus = self.new_corpus

    print(source_texts)

    proportions = np.linspace(0.0, 1.0, 80, endpoint=True)[1:]

    #for i in range(len(new_corpus)):
    #    for j in range(len(new_corpus.tokens[i])):
    #        new_corpus.tokens[i][j] = lemmatizer.lemmatize(new_corpus.tokens[i][j],
    #                                                       pos=get_wordnet_pos(new_corpus.pos_tags[i][j]))

    # TODO: ne odstrani vseh utf-8 punctuationov
    #new_corpus_sentences = self.new_corpus_sentences
    filtered_tokens = [[token.lower() for token in doc if token not in string.punctuation] for doc in new_corpus.tokens]
    print(new_corpus)

    attributeDictionary = {}

    i = 0

    basicMeasures = BasicMeasures.BasicMeasures(new_corpus, new_corpus_sentences)
    readabilityMeasures = ReadabilityMeasures.ReadabilityMeasures(new_corpus, new_corpus_sentences)
    lexicalDiversity = LexicalDiversity.LexicalDiversity(new_corpus, new_corpus_sentences)
    grammar = Grammar.Grammar(new_corpus, new_corpus_sentences)
    content = Content.Content(new_corpus, new_corpus_sentences, source_texts)
    coherence = Coherence.Coherence(new_corpus, new_corpus_sentences, source_texts)

    i = basicMeasures.calculate_all(attr, attributeDictionary, callback, proportions, i)
    i = readabilityMeasures.calculate_all(attr, attributeDictionary, callback, proportions, i)
    i = lexicalDiversity.calculate_all(attr, attributeDictionary, callback, proportions, i)
    i = grammar.calculate_all(attr, attributeDictionary, callback, proportions, i)
    i = content.calculate_all(attr, attributeDictionary, callback, proportions, i)
    i = coherence.calculate_all(attr, attributeDictionary, callback, proportions, i)

    '''print(flesch_reading_ease)
    for d in new_corpus.documents:
        print(textstat.flesch_reading_ease(d))'''

    print(attributeDictionary)
    return attributeDictionary



if __name__ == "__main__":
    #print(nsyl("the"))

    #nltk.download()

    #WidgetPreview(OWDataSamplerA).run(Corpus.from_file("set1_train.tsv"))

    #lang_check = language_check.LanguageTool("en-US")
    #grammar_errors = lang_check.check("Wheres my lunch? Im hungry. I found you're wallet. hey how nice to see you (not really).")
    #print(grammar_errors)
    #print(len(grammar_errors))

    #WidgetPreview(OWDataSamplerA).run(set_data=Corpus.from_file("set1_train.tsv"),
    #                                   set_source_texts=Corpus.from_file("source_texts.tsv"))

    WidgetPreview(OWDataSamplerA).run(set_data=Corpus.from_file("small_set.tsv"),
                                      set_source_texts=Corpus.from_file("source_texts.tsv"))

