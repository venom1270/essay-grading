# PACKAGE INSTALLATIONS
# conda install spacy
# python -m spacy download en
# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_sm
# python -m spacy download en_vectors_web_lg


import copy
import concurrent.futures
from functools import partial


import Orange.data
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)

from AnyQt.QtCore import QThread, pyqtSlot

from orangecontrib.text import Corpus
from orangecontrib.text import preprocess
from orangecontrib.text.tag import pos
from orangecontrib.essaygrading.utils.task import Task
from orangecontrib.essaygrading.utils import OntologyUtils
from orangecontrib.essaygrading.utils.widgets import FileWidget

from nltk import sent_tokenize


class OWSemanticConsistency(OWWidget):
    name = "Semantic Consistency"
    description = "Checks semantic consistency of the essay and reports on errors."
    icon = "../icons/DataSamplerA.svg"
    priority = 10

    class Inputs:
        essays = Input("Essays", Corpus)

    class Outputs:
        feedback = Output("Feedback", Orange.data.Table)

    class Error(OWWidget.Error):
        read_file = Msg("Can't read file {} ({})")
        # need_discrete_data = Msg("Need some discrete data to work with.")
        # no_disc_features = Msg("Discrete features required but data has none.")

    class Warning(OWWidget.Warning):
        invalid_source_file = Msg("Invalid source file. Only files with one line are valid.")
        # no_test_data = Msg("No test data (ungraded essays) present.")

    openie_system = "ClausIE"
    use_coreference = False

    dlgFormats = (
        "All readable files ({});;".format(
            '*' + ' *'.join(Orange.data.io.FileFormat.readers.keys())) +
        ";;".join("{} (*{})".format(f.DESCRIPTION, ' *'.join(f.EXTENSIONS))
                  for f in sorted(set(Orange.data.io.FileFormat.readers.values()),
                                  key=list(Orange.data.io.FileFormat.readers.values()).index)))

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.corpus = None
        self.corpus_sentences = None
        self.source_texts = None
        self.ungraded_corpus = None
        self.ungraded_corpus_sentences = None
        self.dataset = None # I don't really need this
        self.source_text_file = None

        #: The current evaluating task (if any)
        self._task = None  # type: Optional[Task]
        #: An executor we use to submit learner evaluations into a thread pool
        self._executor = ThreadExecutor()

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No essays on input. Waiting to get something.')
        self.source_text_info = gui.widgetLabel(box, "")
        self.update_file_info()

        parametersBox = gui.widgetBox(self.controlArea, "Options")

        self.cb_use_coreference = gui.checkBox(parametersBox, self, "use_coreference", "Use coreference")

        self.label_openie_system = gui.widgetLabel(parametersBox, 'OpenIE system: ')
        self.cb_openie_system = gui.comboBox(widget=parametersBox, master=self, items=("ClausIE", "OpenIE-5.0"),
                                                         value="openie_system", sendSelectedValue=True)

        fbox = gui.widgetBox(self.controlArea, "Optional source text file", orientation=0)
        self.file_widget = FileWidget(
            recent_files=None,
            icon_size=(16, 16),
            on_open=self.open_file,
            dialog_format=self.dlgFormats,
            dialog_title='Open Source text file',
            reload_label='Reload',
            browse_label='Browse',
            allow_empty=False,
            minimal_width=250,
        )
        fbox.layout().addWidget(self.file_widget)

        self.optionsBox = gui.widgetBox(self.controlArea, "Controls")
        gui.button(self.optionsBox, self, "Apply", callback=self._invalidate_results)
        self.optionsBox.setDisabled(True)

    @Inputs.essays
    def set_essays(self, dataset):
        if dataset is not None:

            corpus, corpus_sentences = self.prepare_data(dataset)

            self.optionsBox.setDisabled(False)

            self.corpus = corpus
            self.corpus_sentences = corpus_sentences

            self.infoa.setText('%d essays in input dataset' % len(dataset))

        else:
            self.corpus = None
            self.corpus_sentences = None
            self.infoa.setText('No graded data on input yet, waiting to get something.')
            self.Outputs.errors.send(None)
            self.optionsBox.setDisabled(True)

    def prepare_data(self, data):
        self.dataset = data.copy()
        p = preprocess.Preprocessor(tokenizer=preprocess.WordPunctTokenizer(),
                                    transformers=[preprocess.LowercaseTransformer()],
                                    pos_tagger=pos.AveragedPerceptronTagger(),
                                    normalizer=preprocess.WordNetLemmatizer())
        p_sentences = preprocess.Preprocessor(tokenizer=preprocess.PunktSentenceTokenizer(),
                                              # transformers=[preprocess.LowercaseTransformer()],
                                              # ce je to vklopljeno, pol neki nedela cist prov.
                                              pos_tagger=pos.AveragedPerceptronTagger(),
                                              )

        corpus = p(data)
        corpus = copy.deepcopy(corpus)
        corpus_sentences = p_sentences(data)

        print(corpus_sentences.tokens)

        return corpus, corpus_sentences

    def open_file(self, path=None, data=None):
        self.Error.clear()
        if data:
            self.source_texts = data
        elif path:
            try:
                file = open(path, "r", encoding="utf8")
                self.source_text_file = file.name.split("/")[-1]
                self.source_texts = file.readlines()
                file.close()
            except BaseException as err:
                self.Error.read_file(path, str(err))
        else:
            self.source_texts = None
            self.source_text_file = None

        self.update_file_info()

    def update_file_info(self):
        self.Warning.invalid_source_file.clear()
        if self.source_texts is not None:
            self.source_text_info.setText("Source text file present: " + self.source_text_file)

            if len(self.source_texts) != 1:
                self.source_text_info.setText("No source text file present.")
                self.Warning.invalid_source_file()
            else:
                self.source_texts = sent_tokenize(self.source_texts[0])
                print(self.source_texts)
        else:
            self.source_text_info.setText("No source text file present.")

    def selection(self):
        if self.dataset is None:
            return

    def checkCommit(self):
        if self.commitOnChange:
            self.commit()

    def handleNewSignals(self):
        pass  # to je ce bi meu commitOnChange

    def _update(self):
        if self._task is not None:
            # First make sure any pending tasks are cancelled.
            self.cancel()
        assert self._task is None

        if self.corpus is None or self.corpus_sentences is None:
            return

        print(self.corpus_sentences.tokens)

        check_semantic_errors_func = partial(
            checkSemanticErrors,
            sentences=self.corpus_sentences.tokens,
            openie_system=self.openie_system,
            use_coreference=self.use_coreference,
            source_text=self.source_texts  # [3] # TODO: zaenkrat se tukaj izbere keri source text bi radi
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
        calculate_attributes_func = partial(check_semantic_errors_func, callback=callback)

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
            '''domain = Orange.data.Domain([Orange.data.StringVariable.make("feedback")])

            arr = []
            for essay_feedback in results:
                for f in essay_feedback:
                    arr.append(f)

            out = Orange.data.Table.from_numpy(domain, np.array(arr).transpose())
            self.Outputs.feedback.send(out)'''

    # "reset"
    def _invalidate_results(self):
        self._update()

def checkSemanticErrors(sentences, openie_system="ClausIE", use_coreference=False, callback=None, source_text=None):

    print(sentences)

    # TODO izboljšaj if... preveri če je že ontolgija itd.

    if source_text is not None:
        print("************ SOURCE TEXT ONTOLOGY PRERPARATION *************")

        f = OntologyUtils.run_semantic_consistency_check(None, use_coref=use_coreference, openie_system=openie_system,
                                                         source_text=source_text, ontology_name="DS5_ontology.owl")
        print(f)

    print("****************** ESSAY PROCESSING *******************")

    f = OntologyUtils.run_semantic_consistency_check(sentences, use_coref=use_coreference, openie_system=openie_system,
                                                     source_text=source_text, num_threads=4, orig_ontology_name="DS5_ontology.owl")

    print(f)

    return f


if __name__ == "__main__":

    #WidgetPreview(OWSemanticConsistency).run(set_essays=Corpus.from_file("../datasets/set1_train.tsv"),
    #                                  set_source_texts=Corpus.from_file("../datasets/source_texts.tsv"))

    WidgetPreview(OWSemanticConsistency).run(set_essays=Corpus.from_file("../datasets/All datasets/set5_utf8.tsv")) #set3_small_2.tsv

