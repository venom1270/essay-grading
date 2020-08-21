# PACKAGE INSTALLATIONS
# conda install spacy
# python -m spacy download en
# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_sm
# python -m spacy download en_vectors_web_lg

import copy
import numpy as np
import concurrent.futures
from functools import partial

import Orange.data
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)

from AnyQt.QtCore import QThread, pyqtSlot

from orangecontrib.text import Corpus, preprocess
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

    class Warning(OWWidget.Warning):
        invalid_source_file = Msg("Invalid source file. Only files with one line are valid.")

    openie_system = "ClausIE"
    explain = False
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
        self.source_texts = None
        self.dataset = None  # I don't really need this
        self.source_text_file = None

        self._task = None  # type: Optional[Task]
        self._executor = ThreadExecutor()

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No essays on input. Waiting to get something.')
        self.source_text_info = gui.widgetLabel(box, "")
        self.update_file_info()

        parametersBox = gui.widgetBox(self.controlArea, "Options")

        self.cb_explain = gui.checkBox(parametersBox, self, "explain", "Detailed explanations")

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

            self.optionsBox.setDisabled(False)

            self.corpus = dataset

            self.infoa.setText('%d essays in input dataset' % len(dataset))

        else:
            self.corpus = None
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.optionsBox.setDisabled(True)

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

        if self.corpus is None:
            return

        print(self.corpus.tokens)

        check_semantic_errors_func = partial(
            checkSemanticErrors,
            corpus=self.corpus,
            openie_system=self.openie_system,
            use_coreference=self.use_coreference,
            source_text=self.source_texts,
            explain=self.explain
        )

        self._task = task = Task()
        set_progress = methodinvoke(self, "setProgressValue", (float,))

        def callback(finished):
            if task.cancelled:
                raise KeyboardInterrupt()
            set_progress(finished * 100)

        calculate_attributes_func = partial(check_semantic_errors_func, callback=callback)

        self.progressBarInit()
        task.future = self._executor.submit(calculate_attributes_func)
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._task_finished)

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()

    def cancel(self):
        """
        Cancel the current task (if any).
        """
        if self._task is not None:
            OntologyUtils.terminatePool()
            self._task.cancel()
            assert self._task.future.done()
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
            print(results)

            output = {}
            output["essay_id"] = []
            output["feedback"] = []

            # If feedback else error TODO: naredi da bo izpisal tudi errorje...
            feedback_flag = True

            output_list = []
            for result in results:
                essay_id = result[0]
                essay_feedback_detail = result[1]
                essay_errors = result[2]
                essay_feedback = result[3]

                essay_feedback_detail_string = ""
                if len(essay_feedback_detail) > 0:
                    essay_feedback_detail = essay_feedback_detail
                    for ef in essay_feedback_detail:
                        essay_feedback_detail_string += " ".join(ef) + "; "

                essay_feedback_string = ""
                if len(essay_feedback) > 0:
                    for ef in essay_feedback:
                        essay_feedback_string += " ".join(ef)
                    essay_feedback_string += " "

                output_list.append([essay_id, essay_errors[0], essay_errors[1], essay_errors[2], essay_feedback_string,
                                    essay_feedback_detail_string])

            print(output["feedback"])

            domain = Orange.data.Domain([Orange.data.ContinuousVariable.make("essayId"),
                                         Orange.data.ContinuousVariable.make("consistencyErrors"),
                                         Orange.data.ContinuousVariable.make("semanticErrors"),
                                         Orange.data.ContinuousVariable.make("sum")],
                                        metas=[Orange.data.StringVariable("feedback"),
                                               Orange.data.StringVariable("feedback_detail")])

            print(np.array(output_list)[:, -1].transpose())
            out = Orange.data.Table.from_list(domain, output_list)
            print(out)

            self.Outputs.feedback.send(out)

        except Exception as ex:
            import logging
            # Log the exception with a traceback
            log = logging.getLogger()
            log.exception(__name__, exc_info=True)
            self.error("Exception occurred during evaluation: {!r}"
                       .format(ex))
        else:
            pass

    def _invalidate_results(self):
        self._update()


def prepare_data(data):
    """p = preprocess.Preprocessor(tokenizer=preprocess.WordPunctTokenizer(),
                                transformers=[preprocess.LowercaseTransformer()],
                                pos_tagger=pos.AveragedPerceptronTagger(),
                                normalizer=preprocess.WordNetLemmatizer())
    p_sentences = preprocess.Preprocessor(tokenizer=preprocess.PunktSentenceTokenizer(),
                                          # transformers=[preprocess.LowercaseTransformer()],
                                          # ce je to vklopljeno, pol neki nedela cist prov.
                                          pos_tagger=pos.AveragedPerceptronTagger())"""

    p = preprocess.PreprocessorList([preprocess.WordPunctTokenizer(),
                                     preprocess.LowercaseTransformer(),
                                     pos.AveragedPerceptronTagger(),
                                     preprocess.WordNetLemmatizer()])

    p_sentences = preprocess.PreprocessorList([preprocess.PunktSentenceTokenizer(),
                                               pos.AveragedPerceptronTagger()])

    corpus = p(data)
    corpus = copy.deepcopy(corpus)
    corpus_sentences = p_sentences(data)

    print(corpus_sentences.tokens)

    return corpus, corpus_sentences


def checkSemanticErrors(corpus, openie_system="ClausIE", use_coreference=False, callback=None, source_text=None,
                        explain=False):
    callback(0.01)

    _, sentences = prepare_data(corpus)

    sentences = sentences.tokens
    print(sentences)

    if source_text is not None:
        print("************ SOURCE TEXT ONTOLOGY PRERPARATION *************")

        f = OntologyUtils.run_semantic_consistency_check(None, use_coref=use_coreference, openie_system=openie_system,
                                                         source_text=source_text,
                                                         explain=explain, callback=callback,
                                                         ontology_name="SourceTextOntology.owl")
        print(f)

        print("****************** ESSAY PROCESSING (with source text) *******************")
        f = OntologyUtils.run_semantic_consistency_check(sentences, use_coref=use_coreference,
                                                         openie_system=openie_system, source_text=source_text,
                                                         num_threads=4, explain=explain, callback=callback,
                                                         orig_ontology_name="SourceTextOntology.owl")
    else:
        print("****************** ESSAY PROCESSING *******************")
        f = OntologyUtils.run_semantic_consistency_check(sentences, use_coref=use_coreference,
                                                         openie_system=openie_system,
                                                         source_text=source_text, num_threads=4,
                                                         explain=explain, callback=callback)

    print(f)

    return f


if __name__ == "__main__":
    # WidgetPreview(OWSemanticConsistency).run(set_essays=Corpus.from_file("../datasets/set1_train.tsv"),
    #                                  set_source_texts=Corpus.from_file("../datasets/source_texts.tsv"))

    WidgetPreview(OWSemanticConsistency).run(
        set_essays=Corpus.from_file("../datasets/All datasets/set3_utf8.tsv"))  # set5_utf8.tsv"))  # set3_small_2.tsv
    # WidgetPreview(OWSemanticConsistency).run(set_essays=Corpus.from_file("../datasets/Lisa.tsv"))
