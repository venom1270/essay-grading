import math
import copy
import numpy as np
import concurrent.futures
from functools import partial

from AnyQt.QtCore import QThread, pyqtSlot
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)

import Orange.data
from Orange.widgets import gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from orangecontrib.text import Corpus, preprocess
from orangecontrib.text.tag import pos
from orangecontrib.text.widgets.utils import CheckListLayout
from orangecontrib.essaygrading.utils import globals
from orangecontrib.essaygrading.utils.task import Task
from orangecontrib.essaygrading.modules import BasicMeasures, ReadabilityMeasures, LexicalDiversity, Grammar, Content, \
    Coherence


class OWAttributeSelection(OWWidget):
    name = "Attribute selection"
    description = "Select attributes to be used in the model."
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

    class Warning(OWWidget.Warning):
        no_test_data = Msg("No test data (ungraded essays) present.")
        no_grades = Msg("Could not find grades in graded corpus.")

    commitOnChange = settings.Setting(0)

    attributeDictionary = {}

    selected_attributes = [BasicMeasures.BasicMeasures.name]
    selected_attributes_names = []

    coherence_word_embeddings = globals.EMBEDDING_TFIDF

    METHODS = (BasicMeasures.BasicMeasures, ReadabilityMeasures.ReadabilityMeasures, LexicalDiversity.LexicalDiversity,
               Grammar.Grammar, Content.Content, Coherence.Coherence)

    want_main_area = False
    resizing_enabled = False

    auto_commit = settings.Setting(True)

    def __init__(self):
        super().__init__()

        self.selected_attributes_names = [m.name for m in self.METHODS]

        self.corpus = None
        self.corpus_sentences = None
        self.source_texts = None
        self.ungraded_corpus = None
        self.ungraded_corpus_sentences = None
        self.corpus_grades = None

        self._task = None  # type: Optional[Task]
        self._executor = ThreadExecutor()

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No graded data on input yet, waiting to get something.')
        self.infob = gui.widgetLabel(box, 'No ungraded data on input yet, waiting to get something.')

        self.controlArea.layout().addWidget(
            CheckListLayout("Attribute selection", self, "selected_attributes", self.selected_attributes_names, cols=2,
                            callback=self.checkCommit)
        )

        parametersBox = gui.widgetBox(self.controlArea, "Options")

        self.cb_word_embedding_info = gui.widgetLabel(parametersBox, 'Word Embedding: ')
        self.cb_coherence_word_embeddings = gui.comboBox(widget=parametersBox, master=self,
                                                         items=(globals.EMBEDDING_TFIDF, globals.EMBEDDING_GLOVE_SPACY,
                                                                globals.EMBEDDING_GLOVE_FLAIR),
                                                         value="coherence_word_embeddings", sendSelectedValue=True,
                                                         callback=self.checkCommit)

        gui.auto_apply(self.controlArea, self, "auto_commit", commit=self.commit)

    @Inputs.data
    def set_graded_data(self, dataset):
        if dataset is not None:

            essay_scores = []
            offset = 1
            for i, value in enumerate(dataset.X[0][offset:]):
                print(str(i) + " --- " + str(value))
                if is_number(value):
                    essay_scores.append(dataset.X[:, i + offset])

            self.corpus_grades = np.array(essay_scores).transpose()

            if len(self.corpus_grades) == 0:
                self.Warning.no_grades()
                # self.optionsBox.setDisabled(True) TODO: ce bo auto_apply, se tega ne rabi!!
            else:
                # Take the final grade (last column)
                self.corpus_grades = self.corpus_grades[:, -1]
                self.Warning.no_grades.clear()
                # self.optionsBox.setDisabled(False) TODO: ce bo auto_apply, se tega ne rabi!!

            self.corpus = dataset
            self.infoa.setText('%d instances in graded input dataset' % len(dataset))
            print(self.corpus_grades)

        else:
            self.corpus = None
            self.corpus_sentences = None
            self.infoa.setText('No graded data on input yet, waiting to get something.')
            self.Outputs.attributes_graded.send(None)
            # self.optionsBox.setDisabled(True) TODO: ce bo auto_apply, se tega ne rabi!!
            if self.ungraded_corpus is None:
                self.Warning.no_test_data()

    @Inputs.source_texts
    def set_source_texts(self, source_texts):
        if source_texts is not None:
            p = preprocess.Preprocessor(tokenizer=preprocess.WordPunctTokenizer(),
                                        transformers=[preprocess.LowercaseTransformer()],
                                        pos_tagger=pos.AveragedPerceptronTagger())
            self.source_texts = source_texts  # p(source_texts)
        else:
            self.source_texts = None

    @Inputs.ungraded_essays
    def set_ungraded_data(self, ungraded_essays):
        if ungraded_essays is not None:
            self.ungraded_corpus = ungraded_essays  # corpus
            self.infob.setText('%d instances in ungraded input dataset' % len(ungraded_essays))
            self.Warning.no_test_data.clear()

        else:
            self.ungraded_corpus = None
            self.ungraded_corpus_sentences = None
            self.infob.setText('No ungraded data on input yet, waiting to get something.')
            self.Outputs.attributes_ungraded.send(None)

    def checkCommit(self):
        if self.auto_commit:
            self.commit()

    def handleNewSignals(self):
        self.checkCommit()

    def commit(self):
        self._update()

    def _update(self):
        if self._task is not None:
            # First make sure any pending tasks are cancelled.
            self.cancel()
        assert self._task is None

        if self.corpus is None:  # or self.corpus_sentences is None:
            return

        calculate_attributes_func = partial(
            calculateAttributes,
            graded_corpus=self.corpus,
            ungraded_corpus=self.ungraded_corpus,
            grades=self.corpus_grades,
            source_texts=self.source_texts,
            attr=self.selected_attributes,
            word_embeddings=self.coherence_word_embeddings,
            METHODS=self.METHODS
        )

        self._task = task = Task()
        set_progress = methodinvoke(self, "setProgressValue", (float,))

        def callback(finished):
            if task.cancelled:
                raise KeyboardInterrupt()
            set_progress(finished * 100)

        calculate_attributes_func = partial(calculate_attributes_func, callback=callback)

        self.progressBarInit()
        task.future = self._executor.submit(calculate_attributes_func)
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._task_finished)

    def cancel(self):
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
            self.error("Exception occurred during evaluation: {!r}".format(ex))
        else:

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

    def _invalidate_results(self):
        self.attributeDictionaryGraded = {}
        self.attributeDictionaryUngraded = {}
        self._update()


def calculateAttributes(graded_corpus, source_texts, ungraded_corpus, grades, attr, callback, word_embeddings, METHODS):

    if METHODS is None:
        METHODS = [BasicMeasures.BasicMeasures, ReadabilityMeasures.ReadabilityMeasures,
                   LexicalDiversity.LexicalDiversity, Grammar.Grammar, Content.Content, Coherence.Coherence]

    callback(0.01)

    # Prepare data
    graded_corpus, graded_corpus_sentences = prepare_data(graded_corpus)
    source_texts = prepare_source_texts(source_texts)
    if ungraded_corpus is not None:
        ungraded_corpus, ungraded_corpus_sentences = prepare_data(ungraded_corpus)

    print(source_texts)

    if ungraded_corpus is None:
        proportions = np.linspace(0.0, 1.0, len(attr) + 1, endpoint=True)[1:]
    else:
        proportions = np.linspace(0.0, 1.0, len(attr)*2 + 1, endpoint=True)[1:]

    print(graded_corpus)

    attributeDictionaryGraded = {}
    attributeDictionaryUngraded = {}

    names = [m.name for m in METHODS]
    modules = {}
    for m in attr:
        index = names.index(m)
        module = METHODS[index]
        if m.startswith("Coherence"):
            module = module(graded_corpus, graded_corpus_sentences, grades, source_texts, word_embeddings)
        elif m.startswith("Content"):
            module = module(graded_corpus, graded_corpus_sentences, grades, source_texts, word_embeddings=word_embeddings)
        else:
            module = module(graded_corpus, graded_corpus_sentences)
        modules[m] = module

    i = 0
    for m in attr:
        i = modules[m].calculate_all(None, attributeDictionaryGraded, callback, proportions, i)
        callback(proportions[i])
        i += 1

    if ungraded_corpus and ungraded_corpus_sentences:
        modules = {}
        for m in attr:
            index = names.index(m)
            module = METHODS[index]
            if m.startswith("Coherence"):
                module = module(ungraded_corpus, ungraded_corpus_sentences, grades, source_texts, word_embeddings)
            elif m.startswith("Content"):
                module = module(ungraded_corpus, ungraded_corpus_sentences, grades, source_texts,
                                graded_corpus=graded_corpus, word_embeddings=word_embeddings)
            else:
                module = module(ungraded_corpus, ungraded_corpus_sentences)
            modules[m] = module

        for m in attr:
            i = modules[m].calculate_all(None, attributeDictionaryUngraded, callback, proportions, i)
            callback(proportions[i])
            i += 1

    print(attributeDictionaryGraded)
    return attributeDictionaryGraded, attributeDictionaryUngraded


def prepare_data(data):

    p = preprocess.Preprocessor(tokenizer=preprocess.WordPunctTokenizer(),
                                transformers=[preprocess.LowercaseTransformer()],
                                pos_tagger=pos.AveragedPerceptronTagger(),
                                normalizer=preprocess.WordNetLemmatizer(),
                                # filters=preprocess.StopwordsFilter())
                                )
    p_sentences = preprocess.Preprocessor(tokenizer=preprocess.PunktSentenceTokenizer())

    corpus = p(data)
    corpus = copy.deepcopy(corpus)
    corpus_sentences = p_sentences(data)

    return corpus, corpus_sentences


def prepare_source_texts(source_texts):
    if source_texts is not None:
        p = preprocess.Preprocessor(tokenizer=preprocess.WordPunctTokenizer(),
                                    transformers=[preprocess.LowercaseTransformer()],
                                    pos_tagger=pos.AveragedPerceptronTagger())
        st = p(source_texts)
    else:
        st = None

    return st


def is_number(n):
    try:
        if math.isnan(float(n)):
            return False
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    WidgetPreview(OWAttributeSelection).run(set_graded_data=Corpus.from_file("../datasets/small_set.tsv"),
                                      set_ungraded_data=Corpus.from_file("../datasets/small_set.tsv"),
                                      set_source_texts=Corpus.from_file("../datasets/source_texts.tsv"))

