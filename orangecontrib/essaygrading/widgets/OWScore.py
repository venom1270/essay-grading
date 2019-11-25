# PACKAGE INSTALLATIONS
# conda install spacy
# python -m spacy download en


from orangecontrib.essaygrading.modules import BasicMeasures, ReadabilityMeasures, LexicalDiversity, Grammar, Content, Coherence

import numpy as np

import Orange.data
from Orange.regression import neural_network
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
        true_scores = Input("Actual scores", Orange.data.Table)

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

        '''

        self.NN = neural_network.NNRegressionLearner()
        return self.LEARNER(
            hidden_layer_sizes=self.get_hidden_layers(),
            activation=self.activation[self.activation_index],
            solver=self.solver[self.solver_index],
            alpha=self.alpha,
            random_state=None,
            max_iter=10000,
            preprocessors=self.preprocessors)

        '''
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

            pred = np.array([round(x[0]) for x in self.predictions])
            true = np.array([x[0] for x in self.true_scores])

            for i in range(len(self.predictions)):
                if pred[i] == true[i]:
                    s += 1

            print("self.predictions")
            print(self.predictions)
            print("self.true_scores")
            print(self.true_scores)

            print("pred")
            print(pred)
            print("true")
            print(true)

            self.outDictionary["exactAgreement"] = s / len(self.predictions)
            self.outDictionary["weightedKappa"] = quadratic_weighted_kappa(true, pred)

            domain = Orange.data.Domain([Orange.data.ContinuousVariable.make(key) for key in self.outDictionary])

            arr = np.array([[value] for _, value in self.outDictionary.items()])

            print(arr)

            out = Orange.data.Table.from_numpy(domain, np.array(arr).transpose())

        self.Outputs.scores.send(out)



# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))

    print(rater_a)
    print(rater_b)
    print(min_rating)
    print(max_rating)
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)
