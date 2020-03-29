import numpy as np
import Orange.data
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output




class OWScore(OWWidget):
    name = "Score"
    description = "Score predictions."
    icon = "../icons/DataSamplerA.svg"
    priority = 10

    class Inputs:
        data = Input("True scores and predictions", Orange.data.Table)

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

        self.true_scores_list = []
        self.true_scores_selection = 0

        self.predicted_scores_list = []
        self.predicted_scores_selection = 0

        self.data = None

        # Info
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No data on input yet, waiting to get something.')

        # Score selection
        true_scores_box = gui.widgetBox(self.controlArea, "Select true scores")
        self.cb_true_scores = gui.comboBox(widget=true_scores_box, master=self,
                                           items=self.true_scores_list,
                                           value="true_scores_selection", sendSelectedValue=True)

        predicted_scores_box = gui.widgetBox(self.controlArea, "Select predicted scores")
        self.cb_predicted_scores = gui.comboBox(widget=predicted_scores_box, master=self,
                                                items=self.predicted_scores_list,
                                                value="predicted_scores_selection", sendSelectedValue=True)

        # Results
        results_box = gui.widgetBox(self.controlArea, "Results")
        gui.widgetLabel(results_box, "Exact agreement: ")
        self.info_ea = gui.widgetLabel(results_box, "")
        gui.widgetLabel(results_box, "Quadratic weighted kappa: ")
        self.info_qwk = gui.widgetLabel(results_box, "")

        self.optionsBox = gui.widgetBox(self.controlArea, "Controls")
        gui.checkBox(self.optionsBox, self, "commitOnChange", "Commit data on selection change")
        gui.button(self.optionsBox, self, "Apply", callback=self._update)
        self.optionsBox.setDisabled(False)

    @Inputs.data
    def set_data(self, data):
        if data is not None:
            ts = [x.name for x in data.domain.class_vars]  # list(data.domain.class_vars)
            ps = [x.name for x in data.domain.metas]  # list(data.domain.metas)
            # ua.append(data.domain.metas)
            print("--- UA ----")
            print(ts)
            print(ps)

            self.true_scores_list = ts
            self.cb_true_scores.clear()
            self.cb_true_scores.addItems(ts)
            self.predicted_scores_list = ps
            self.cb_predicted_scores.clear()
            self.cb_predicted_scores.addItems(ps)

            # Nastavi default vrednosti
            if len(self.true_scores_list) > 0:
                self.true_scores_selection = self.true_scores_list[0]
            if len(self.predicted_scores_list) > 1:
                self.predicted_scores_selection = self.predicted_scores_list[1]

            self.data = data

        else:
            self.data = None
            self.true_scores = None
            self.predictions = None
            self.true_scores_list = []
            self.true_scores_selection = 0
            self.cb_true_scores.clear()
            self.predicted_scores_list = []
            self.predicted_scores_selection = 0
            self.cb_predicted_scores.clear()
            self.info_ea.clear()
            self.info_qwk.clear()
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

    def update_feature_selection(self):
        pass

    def _update(self):

        if self.true_scores_selection != "":
            index = self.true_scores_list.index(self.true_scores_selection) \
                if self.true_scores_selection in self.true_scores_list else -1
            if index >= 0:
                if self.true_scores_selection == "score" and len(np.array(self.data.Y).shape) == 1:
                    self.true_scores = self.data.Y
                else:
                    self.true_scores = self.data.Y[:, index] # TODO: za tole nism zihr, najbrz nikol do tega sploh ne pride?
            else:
                self.true_scores = None
        else:
            self.true_scores = None

        if self.predicted_scores_selection != "":
            index = self.predicted_scores_list.index(self.predicted_scores_selection) \
                if self.predicted_scores_selection in self.predicted_scores_list else -1
            if index >= 0:
                self.predictions = self.data.metas[:, index]
            else:
                self.predictions = None
        else:
            self.predictions = None

        print(self.true_scores)
        print(self.predictions)

        out = None
        if self.predictions is not None and self.true_scores is not None:
            print("FINISHED")

            self.outDictionary = {}
            self.outDictionary["exactAgreement"] = 0
            self.outDictionary["weightedKappa"] = 0

            s = 0
            folds = self.data.metas[:, -1]  # TODO: kaj ce ni foldov???

            print(self.predictions)
            print(self.true_scores)

            # pred = np.array([round(x[0]) for x in self.predictions])
            pred = np.array([round(x) for x in self.predictions])
            # folds = []
            print(self.predictions)
            # if self.predictions.metas is not None:
            #     folds = np.array([x[0] for x in self.predictions.metas])
            # true = np.array([x[0] for x in self.true_scores])
            true = self.true_scores

            print(folds)

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

            kappa_folds = []
            exact_folds = []
            num_folds = int(max(folds)) + 1
            if len(folds) > 0:
                for i in range(num_folds): # TODO: dynamic EDIT: je že?
                    print(i+1)
                    p_scores = np.array([pred[x] for x in range(len(pred)) if folds[x] == i])
                    t_scores = np.array([true[x] for x in range(len(true)) if folds[x] == i])
                    print(p_scores)
                    print(t_scores)
                    print(folds)
                    kappa_folds.append(quadratic_weighted_kappa(t_scores, p_scores))
                    exact_folds.append((len(p_scores) - sum(np.abs(p_scores-t_scores))) / len(p_scores))

            self.outDictionary["exactAgreement"] = s / len(self.predictions)
            self.outDictionary["exactAgreementFolds"] = sum(exact_folds) / num_folds
            self.outDictionary["weightedKappa"] = quadratic_weighted_kappa(true, pred)
            self.outDictionary["weightedKappaFolds"] = sum(kappa_folds) / num_folds

            self.info_ea.setText(str(self.outDictionary["exactAgreement"]))
            self.info_qwk.setText(str(self.outDictionary["weightedKappa"]))

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

    #print(rater_a)
    #print(rater_b)
    #print(min_rating)
    #print(max_rating)
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
