# PACKAGE INSTALLATIONS
# conda install spacy
# python -m spacy download en


from orangecontrib.essaygrading.modules import BasicMeasures, ReadabilityMeasures, LexicalDiversity, Grammar, Content, Coherence

import numpy as np

import Orange.data
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.utils.widgetpreview import WidgetPreview

from orangecontrib.text import Corpus
from orangecontrib.text import preprocess
from orangecontrib.text.tag import pos

from orangecontrib.text.widgets.utils import CheckListLayout

from orangecontrib.essaygrading.utils.task import Task

from functools import partial


import copy
import string
from nltk.stem import WordNetLemmatizer

import concurrent.futures
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)

from AnyQt.QtCore import QThread, pyqtSlot


class OWScore(OWWidget):
    name = "Score"
    description = "Score predictions."
    icon = "../icons/DataSamplerA.svg"
    priority = 10

    class Inputs:
        predictions = Input("Predicted scores", Orange.data.Table)
        true_scores = Input("True scores", Orange.data.Table)

    class Outputs:
        scores = Output("Score", Orange.data.Table)

    class Error(OWWidget.Error):
        pass

    class Warning(OWWidget.Warning):
        pass

    commitOnChange = settings.Setting(0)

    outDictionary = {}

    want_main_area = False

    def __init__(self):
        super().__init__()

        self.predictions = None
        self.true_scores = None


        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No data on input yet, waiting to get something.')


        self.optionsBox = gui.widgetBox(self.controlArea, "Controls")
        gui.checkBox(self.optionsBox, self, "commitOnChange", "Commit data on selection change")
        gui.button(self.optionsBox, self, "Apply", callback=self._update)
        self.optionsBox.setDisabled(True)

    @Inputs.predictions
    def set_predictions(self, predictions):
        if predictions is not None:

            self.predictions = predictions
            self.infoa.setText('Predictions set.')
            self.optionsBox.setDisabled(False)

        else:

            self.infoa.setText('No data on input yet, waiting to get something.')
            self.Outputs.scores.send(None)
            self.optionsBox.setDisabled(True)
            self.predictions = None

    @Inputs.true_scores
    def set_true_scores(self, true_scores):
        if true_scores is not None:
            self.true_scores = true_scores
            self.optionsBox.setDisabled(False)
        else:
            self.true_scores = None
            self.Outputs.scores.send(None)

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



        out = None
        if self.predictions and self.true_scores:
            print("FINISHED")

            self.outDictionary = {}
            self.outDictionary["exactAgreement"] = 0
            self.outDictionary["weightedKappa"] = 0

            s = 0

            print(self.predictions)
            print(self.true_scores)

            for i in range(len(self.predictions)):
                if round(self.predictions[i][0]) == self.true_scores[i][0]:
                    s += 1

            self.outDictionary["exactAgreement"] = s / len(self.predictions)

            domain = Orange.data.Domain([Orange.data.ContinuousVariable.make(key) for key in self.outDictionary])

            arr = np.array([[value] for _, value in self.outDictionary.items()])

            print(arr)

            out = Orange.data.Table.from_numpy(domain, np.array(arr).transpose())

        self.Outputs.scores.send(out)

