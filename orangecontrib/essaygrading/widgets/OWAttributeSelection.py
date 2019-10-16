# PACKAGE INSTALLATIONS
# conda install spacy
# python -m spacy download en
import math
import copy
import string
import numpy as np
import concurrent.futures
from functools import partial
from nltk.stem import WordNetLemmatizer

from AnyQt.QtCore import QThread, pyqtSlot
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)

import Orange.data
from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from orangecontrib.text import Corpus
from orangecontrib.text import preprocess
from orangecontrib.text.tag import pos
from orangecontrib.text.widgets.utils import CheckListLayout
from orangecontrib.essaygrading.utils.task import Task
from orangecontrib.essaygrading.modules import BasicMeasures, ReadabilityMeasures, LexicalDiversity, Grammar, Content, \
    Coherence



class OWAttributeSelection(OWWidget):
    name = "Attribute selection"
    description = "Select desired attributes and preprocessing steps to be used in the model."
    icon = "../icons/DataSamplerA.svg"
    priority = 10

    class Inputs:
        data = Input("Graded Essays", Corpus)
        source_texts = Input("Source texts", Corpus)
        ungraded_essays = Input("Ungraded essays", Corpus)

    class Outputs:
        attributes_graded = Output("Attributes Graded", Orange.data.Table)
        attributes_ungraded = Output("Attributes Ungraded", Orange.data.Table)

    class Error(OWWidget.Error):
        pass
        #need_discrete_data = Msg("Need some discrete data to work with.")
        #no_disc_features = Msg("Discrete features required but data has none.")

    class Warning(OWWidget.Warning):
        no_test_data = Msg("No test data (ungraded essays) present.")
        no_grades = Msg("Could not find grades in graded corpus.")

    proportion = settings.Setting(50)
    commitOnChange = settings.Setting(0)

    attributeDictionary = {}

    selected_attributes = []
    selected_attributes_names = []

    coherence_word_embeddings = "TF-IDF"

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.METHODS = (BasicMeasures.BasicMeasures, ReadabilityMeasures.ReadabilityMeasures,
                        LexicalDiversity.LexicalDiversity, Grammar.Grammar, Content.Content, Coherence.Coherence)
        self.selected_attributes_names = [m.name for m in self.METHODS]

        self.corpus = None
        self.corpus_sentences = None
        self.source_texts = None
        self.ungraded_corpus = None
        self.ungraded_corpus_sentences = None
        self.dataset = None # I don't really need this
        self.corpus_grades = None

        #: The current evaluating task (if any)
        self._task = None  # type: Optional[Task]
        #: An executor we use to submit learner evaluations into a thread pool
        self._executor = ThreadExecutor()

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No graded data on input yet, waiting to get something.')
        self.infob = gui.widgetLabel(box, 'No ungraded data on input yet, waiting to get something.')

        self.controlArea.layout().addWidget(
            CheckListLayout("Attribute selection", self, "selected_attributes", self.selected_attributes_names, cols=2)
        )

        parametersBox = gui.widgetBox(self.controlArea, "Options")

        self.cb_word_embedding_info = gui.widgetLabel(parametersBox, 'Word Embedding: ')
        self.cb_coherence_word_embeddings = gui.comboBox(widget=parametersBox, master=self, items=("TF-IDF", "GloVe"),
                                                         value="coherence_word_embeddings", sendSelectedValue=True)

        self.optionsBox = gui.widgetBox(self.controlArea, "Controls")
        gui.checkBox(self.optionsBox, self, "commitOnChange", "Commit data on selection change")
        gui.button(self.optionsBox, self, "Apply", callback=self._invalidate_results)
        self.optionsBox.setDisabled(True)

    @Inputs.data
    def set_graded_data(self, dataset):
        if dataset is not None:

            corpus, corpus_sentences = self.prepare_data(dataset)

            essay_scores = []

            # domain 1 scores
            #print(corpus.X[:,5])
            offset = 2
            for i, value in enumerate(corpus.X[0][offset:]):
                print(str(i) + " --- " + str(value))
                if is_number(value):
                    essay_scores.append(corpus.X[:, i+offset])

            self.corpus_grades = np.array(essay_scores).transpose()




            if len(self.corpus_grades) == 0:
                self.Warning.no_grades()
                self.optionsBox.setDisabled(True)
            else:
                # Take the final grade (last column)
                self.corpus_grades = self.corpus_grades[:, -1]
                self.Warning.no_grades.clear()
                self.optionsBox.setDisabled(False)

            self.selection()

            self.corpus = corpus
            self.corpus_sentences = corpus_sentences

            print(dataset)

            self.infoa.setText('%d instances in graded input dataset' % len(dataset))

            print(corpus.tokens[0])
            print(corpus.pos_tags[0])
            print(corpus.documents[0])
            print(corpus.attributes)
            print(self.corpus_grades)

        else:
            self.dataset = None
            self.corpus = None
            self.corpus_sentences = None
            self.infoa.setText('No graded data on input yet, waiting to get something.')
            self.Outputs.attributes_graded.send(None)
            self.optionsBox.setDisabled(True)
            if self.ungraded_corpus is None:
                self.Warning.no_test_data()

    @Inputs.source_texts
    def set_source_texts(self, source_texts):
        if source_texts is not None:
            p = preprocess.Preprocessor(tokenizer=preprocess.WordPunctTokenizer(),
                                        transformers=[preprocess.LowercaseTransformer()],
                                        pos_tagger=pos.AveragedPerceptronTagger())
            self.source_texts = p(source_texts)
        else:
            self.source_texts = None

    @Inputs.ungraded_essays
    def set_ungraded_data(self, ungraded_essays):
        if ungraded_essays is not None:
            corpus, corpus_sentences = self.prepare_data(ungraded_essays)

            self.ungraded_corpus = corpus
            self.ungraded_corpus_sentences = corpus_sentences

            self.infob.setText('%d instances in ungraded input dataset' % len(ungraded_essays))

            self.Warning.no_test_data.clear()

        else:
            self.ungraded_corpus = None
            self.ungraded_corpus_sentences = None
            self.infob.setText('No ungraded data on input yet, waiting to get something.')
            self.Outputs.attributes_ungraded.send(None)
            #self.optionsBox.setDisabled(True)

    def prepare_data(self, data):
        self.dataset = data.copy()
        p = preprocess.Preprocessor(tokenizer=preprocess.WordPunctTokenizer(),
                                    transformers=[preprocess.LowercaseTransformer()],
                                    pos_tagger=pos.AveragedPerceptronTagger(),
                                    normalizer=preprocess.WordNetLemmatizer())
        p_sentences = preprocess.Preprocessor(tokenizer=preprocess.PunktSentenceTokenizer())

        corpus = p(data)
        corpus = copy.deepcopy(corpus)
        corpus_sentences = p_sentences(data)

        return corpus, corpus_sentences

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

        if self.corpus is None or self.corpus_sentences is None:
            return

        calculate_attributes_func = partial(
            calculateAttributes,
            graded_corpus=self.corpus,
            graded_corpus_sentences=self.corpus_sentences,
            ungraded_corpus=self.ungraded_corpus,
            ungraded_corpus_sentences=self.ungraded_corpus_sentences,
            source_texts=self.source_texts,
            attr=self.selected_attributes,
            word_embeddings=self.coherence_word_embeddings,
            METHODS=self.METHODS
        )

        #print(self.cb_coherence_word_embeddings)
        #exit()

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

            self.attributeDictionaryGraded = results[0]
            self.attributeDictionaryUngraded = results[1]

            outGraded = None
            # Only output if dictionary is not empty (contains attributes)
            if self.attributeDictionaryGraded:
                print("FINISHED")
                print(self.attributeDictionaryGraded)
                domain = Orange.data.Domain([Orange.data.ContinuousVariable.make(key)
                                             for key in self.attributeDictionaryGraded],
                                            Orange.data.ContinuousVariable.make("score"))

                arr = np.array([value for _, value in self.attributeDictionaryGraded.items()])

                print("INF CHECK GRADED")
                for k,i in self.attributeDictionaryGraded.items():
                    if np.isinf(i).any():
                      print(k)
                print(arr)
                #outGraded = Orange.data.Table.from_numpy(domain, np.array(arr).transpose(), self.corpus.X[:, 5])
                outGraded = Orange.data.Table.from_numpy(domain, np.array(arr).transpose(), self.corpus_grades)

            outUngraded = None
            if self.attributeDictionaryUngraded:
                domain = Orange.data.Domain(
                    [Orange.data.ContinuousVariable.make(key) for key in self.attributeDictionaryUngraded])
                arr = np.array([value for _, value in self.attributeDictionaryUngraded.items()])
                print("INF CHECK UNGRADED")
                for k,i in self.attributeDictionaryGraded.items():
                    if np.isinf(i).any():
                      print(k)
                print(arr)
                outUngraded = Orange.data.Table.from_numpy(domain, np.array(arr).transpose())

            self.Outputs.attributes_graded.send(outGraded)
            self.Outputs.attributes_ungraded.send(outUngraded)

    # "reset"
    def _invalidate_results(self):
        self.attributeDictionaryGraded = {}
        self.attributeDictionaryUngraded = {}
        self._update()

def calculateAttributes(graded_corpus, graded_corpus_sentences, source_texts, ungraded_corpus,
                        ungraded_corpus_sentences, attr, callback, word_embeddings, METHODS):
    word_length_threshold = 7
    sentence_length_threshold = 40
    lemmatizer = WordNetLemmatizer()
    #new_corpus = self.new_corpus

    print(source_texts)

    # proportions = np.linspace(0.0, 1.0, 80 * 2, endpoint=True)[1:]
    if ungraded_corpus is None:
        proportions = np.linspace(0.0, 1.0, len(attr) + 1, endpoint=True)[1:]
    else:
        proportions = np.linspace(0.0, 1.0, len(attr)*2 + 1, endpoint=True)[1:]

    #for i in range(len(new_corpus)):
    #    for j in range(len(new_corpus.tokens[i])):
    #        new_corpus.tokens[i][j] = lemmatizer.lemmatize(new_corpus.tokens[i][j],
    #                                                       pos=get_wordnet_pos(new_corpus.pos_tags[i][j]))

    # TODO: ne odstrani vseh utf-8 punctuationov
    #new_corpus_sentences = self.new_corpus_sentences
    filtered_tokens = [[token.lower() for token in doc if token not in string.punctuation]
                       for doc in graded_corpus.tokens]
    print(graded_corpus)

    attributeDictionaryGraded = {}
    attributeDictionaryUngraded = {}




    # TODO: naredi METHODS = (metoda1, metoda2, ...) ??
    names = [m.name for m in METHODS]
    modules = {}
    for m in attr:
        index = names.index(m)
        module = METHODS[index]
        if m.startswith("Coherence"):
            module = module(graded_corpus, graded_corpus_sentences, source_texts, word_embeddings)
        elif m.startswith("Content"):
            module = module(graded_corpus, graded_corpus_sentences, source_texts)
        else:
            module = module(graded_corpus, graded_corpus_sentences)
        modules[m] = module

    '''
    modules = dict()
    modules["Basic Measures"] = BasicMeasures.BasicMeasures(graded_corpus, graded_corpus_sentences)
    modules["Readability Measures"] = ReadabilityMeasures.ReadabilityMeasures(graded_corpus, graded_corpus_sentences)
    modules["Lexical Diversity"] = LexicalDiversity.LexicalDiversity(graded_corpus, graded_corpus_sentences)
    modules["Grammar"] = Grammar.Grammar(graded_corpus, graded_corpus_sentences)
    modules["Content"] = Content.Content(graded_corpus, graded_corpus_sentences, source_texts)
    modules["Coherence and Semantics"] = Coherence.Coherence(graded_corpus, graded_corpus_sentences, source_texts,
                                                             word_embeddings)
    '''
    '''
    basicMeasures = BasicMeasures.BasicMeasures(new_corpus, new_corpus_sentences)
    readabilityMeasures = ReadabilityMeasures.ReadabilityMeasures(new_corpus, new_corpus_sentences)
    lexicalDiversity = LexicalDiversity.LexicalDiversity(new_corpus, new_corpus_sentences)
    grammar = Grammar.Grammar(new_corpus, new_corpus_sentences)
    content = Content.Content(new_corpus, new_corpus_sentences, source_texts)
    coherence = Coherence.Coherence(new_corpus, new_corpus_sentences, source_texts)
    '''
    #i = basicMeasures.calculate_all(attr, attributeDictionary, callback, proportions, i)
    #i = readabilityMeasures.calculate_all(attr, attributeDictionary, callback, proportions, i)
    #i = lexicalDiversity.calculate_all(attr, attributeDictionary, callback, proportions, i)
    #i = grammar.calculate_all(attr, attributeDictionary, callback, proportions, i)
    #i = content.calculate_all(attr, attributeDictionary, callback, proportions, i)
    #i = coherence.calculate_all(attr, attributeDictionary, callback, proportions, i)

    i = 0
    for m in attr:
        i = modules[m].calculate_all(None, attributeDictionaryGraded, callback, proportions, i)
        callback(proportions[i])
        i += 1

    if ungraded_corpus and ungraded_corpus_sentences:
        '''
        modules = {}
        modules["Basic Measures"] = BasicMeasures.BasicMeasures(ungraded_corpus, ungraded_corpus_sentences)
        modules["Readability Measures"] = ReadabilityMeasures.ReadabilityMeasures(ungraded_corpus,
                                                                                  ungraded_corpus_sentences)
        modules["Lexical Diversity"] = LexicalDiversity.LexicalDiversity(ungraded_corpus, ungraded_corpus_sentences)
        modules["Grammar"] = Grammar.Grammar(ungraded_corpus, ungraded_corpus_sentences)
        modules["Content"] = Content.Content(ungraded_corpus, ungraded_corpus_sentences, source_texts,
                                             graded_corpus=graded_corpus)
        modules["Coherence and Semantics"] = Coherence.Coherence(ungraded_corpus, ungraded_corpus_sentences,
                                                                 source_texts, word_embeddings)

        '''
        modules = {}
        for m in attr:
            index = names.index(m)
            module = METHODS[index]
            if m.startswith("Coherence"):
                module = module(ungraded_corpus, ungraded_corpus_sentences, source_texts, word_embeddings)
            elif m.startswith("Content"):
                module = module(ungraded_corpus, ungraded_corpus_sentences, source_texts, graded_corpus=graded_corpus)
            else:
                module = module(ungraded_corpus, ungraded_corpus_sentences)
            modules[m] = module

        for m in attr:
            i = modules[m].calculate_all(None, attributeDictionaryUngraded, callback, proportions, i)
            callback(proportions[i])
            i += 1

    '''print(flesch_reading_ease)
    for d in new_corpus.documents:
        print(textstat.flesch_reading_ease(d))
    '''

    print(attributeDictionaryGraded)
    return attributeDictionaryGraded, attributeDictionaryUngraded


def is_number(n):
    try:
        if math.isnan(float(n)):
            return False
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    #print(nsyl("the"))

    #nltk.download()

    #WidgetPreview(OWAttributeSelection).run(Corpus.from_file("set1_train.tsv"))

    #lang_check = language_check.LanguageTool("en-US")
    #grammar_errors =
    # lang_check.check("Wheres my lunch? Im hungry. I found you're wallet. hey how nice to see you (not really).")
    #print(grammar_errors)
    #print(len(grammar_errors))

    #WidgetPreview(OWAttributeSelection).run(set_data=Corpus.from_file("../set1_train.tsv"),
    #                                   set_source_texts=Corpus.from_file("../source_texts.tsv"))

    WidgetPreview(OWAttributeSelection).run(set_graded_data=Corpus.from_file("../datasets/small_set_graded.tsv"),
                                      set_ungraded_data=Corpus.from_file("../datasets/small_set.tsv"),
                                      set_source_texts=Corpus.from_file("../datasets/source_texts.tsv"))

