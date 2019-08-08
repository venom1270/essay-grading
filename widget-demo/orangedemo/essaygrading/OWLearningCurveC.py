import logging
from collections import OrderedDict
from functools import reduce, partial

import numpy

from AnyQt.QtWidgets import QTableWidget, QTableWidgetItem
from AnyQt.QtCore import QThread, pyqtSlot

import Orange.data
import Orange.classification
import Orange.evaluation

from Orange.widgets import widget, gui, settings
from Orange.evaluation.testing import Results

# [start-snippet-1]
import concurrent.futures
from Orange.widgets.utils.concurrent import (
    ThreadExecutor, FutureWatcher, methodinvoke
)
# [end-snippet-1]


# [start-snippet-2]
from Orange.widgets.utils.widgetpreview import WidgetPreview


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
# [end-snippet-2]


class OWLearningCurveC(widget.OWWidget):
    name = "Learning Curve (C)"
    description = ("Takes a dataset and a set of learners and shows a "
                   "learning curve in a table")
    icon = "icons/LearningCurve.svg"
    priority = 1010

    inputs = [("Data", Orange.data.Table, "set_dataset", widget.Default),]

    #: cross validation folds
    folds = settings.Setting(5)
    #: points in the learning curve
    steps = settings.Setting(10)
    #: index of the selected scoring function
    scoringF = settings.Setting(0)
    #: compute curve on any change of parameters
    commitOnChange = settings.Setting(True)

    def __init__(self):
        super().__init__()

        # sets self.curvePoints, self.steps equidistant points from
        # 1/self.steps to 1

        self.scoring = [
            ("Classification Accuracy", Orange.evaluation.scoring.CA),
            ("AUC", Orange.evaluation.scoring.AUC),
            ("Precision", Orange.evaluation.scoring.Precision),
            ("Recall", Orange.evaluation.scoring.Recall)
        ]
        #: input data on which to construct the learning curve
        self.data = None
        #: optional test data
        self.testdata = None
        #: A {input_id: Learner} mapping of current learners from input channel
        self.learners = OrderedDict()
        #: A {input_id: List[Results]} mapping of input id to evaluation
        #: results list, one for each curve point
        self.results = OrderedDict()
        #: A {input_id: List[float]} mapping of input id to learning curve
        #: point scores
        self.curves = OrderedDict()

        # [start-snippet-3]
        #: The current evaluating task (if any)
        self._task = None   # type: Optional[Task]
        #: An executor we use to submit learner evaluations into a thread pool
        self._executor = ThreadExecutor()
        # [end-snippet-3]

        # GUI
        box = gui.widgetBox(self.controlArea, "Info")
        self.infoa = gui.widgetLabel(box, 'No data on input.')
        self.infob = gui.widgetLabel(box, 'No learners.')

        gui.separator(self.controlArea)

        box = gui.widgetBox(self.controlArea, "Evaluation Scores")


        gui.separator(self.controlArea)

        box = gui.widgetBox(self.controlArea, "Options")
        gui.spin(box, self, 'folds', 2, 100, step=1,
                 label='Cross validation folds:  ', keyboardTracking=False,
                 callback=lambda:
                 self._invalidate_results() if self.commitOnChange else None)

        gui.checkBox(box, self, 'commitOnChange', 'Apply setting on any change')
        self.commitBtn = gui.button(box, self, "Apply Setting",
                                    callback=self._invalidate_results,
                                    disabled=True)

        gui.rubber(self.controlArea)

        # table widget
        self.table = gui.table(self.mainArea,
                               selectionMode=QTableWidget.NoSelection)

    ##########################################################################
    # slots: handle input signals

    def set_dataset(self, data):
        """Set the input train dataset."""
        # Clear all results/scores
        self.data = data

        if data is not None:
            self.infoa.setText('%d instances in input dataset' % len(data))
        else:
            self.infoa.setText('No data on input.')

        self.commitBtn.setEnabled(self.data is not None)

# [start-snippet-4]
    def handleNewSignals(self):
        self._update()
# [end-snippet-4]

# [start-snippet-5]
    def _update(self):
        if self._task is not None:
            # First make sure any pending tasks are cancelled.
            self.cancel()
        assert self._task is None

        if self.data is None:
            return

# [end-snippet-5]
# [start-snippet-6]
        # setup the learner evaluations as partial function capturing
        # the necessary arguments.
        learning_curve_func = partial(
            learning_curve,
            data=self.data
        )

# [end-snippet-6]
# [start-snippet-7]
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
        learning_curve_func = partial(learning_curve_func, callback=callback)
# [end-snippet-7]
# [start-snippet-8]
        self.progressBarInit()
        # Submit the evaluation function to the executor and fill in the
        # task with the resultant Future.
        task.future = self._executor.submit(learning_curve_func)
        # Setup the FutureWatcher to notify us of completion
        task.watcher = FutureWatcher(task.future)
        # by using FutureWatcher we ensure `_task_finished` slot will be
        # called from the main GUI thread by the Qt's event loop
        task.watcher.done.connect(self._task_finished)
# [end-snippet-8]

# [start-snippet-progress]
    @pyqtSlot(float)
    def setProgressValue(self, value):
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)
# [end-snippet-progress]

# [start-snippet-9]
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
            # Log the exception with a traceback
            log = logging.getLogger()
            log.exception(__name__, exc_info=True)
            self.error("Exception occurred during evaluation: {!r}"
                       .format(ex))
            # clear all results
            for key in self.results.keys():
                self.results[key] = None
        else:
            # split the combined result into per learner/model results ...
            results = [p_results
                       for p_results in results]  # type: List[List[Results]]

            self.results = results
            #print(results)
# [end-snippet-9]
        # update the display
# [end-snippet-9]

# [start-snippet-10]
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
# [end-snippet-10]

# [start-snippet-11]
    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()
# [end-snippet-11]


    def test_run_signals(self):
        data = Orange.data.Table("iris")
        indices = numpy.random.permutation(len(data))

        traindata = data[indices[:-20]]
        testdata = data[indices[-20:]]

        self.set_dataset(traindata)
        self.set_testdataset(testdata)

        l1 = Orange.classification.NaiveBayesLearner()
        l1.name = 'Naive Bayes'
        self.set_learner(l1, 1)

        l2 = Orange.classification.LogisticRegressionLearner()
        l2.name = 'Logistic Regression'
        self.set_learner(l2, 2)

        l4 = Orange.classification.SklTreeLearner()
        l4.name = "Decision Tree"
        self.set_learner(l4, 3)

    def _invalidate_results(self):
        self.data = []
        d = []
        for i in range(10000):
            row = []
            for j in range(1000):
                row.append(j)
            d.append(row)
        self.data = d
        self._update()


def learning_curve(data,
                   random_state=None, callback=None):

    new_data = []
    proportions = numpy.linspace(0.0, 1.0, len(data)+1, endpoint=True)[1:]
    i = 0
    import random
    for row in data:
        r = []
        for d in row:
            r.append(d+i*random.random())
        new_data.append(row)
        callback(proportions[i])
        i += 1

    ##results = [
    #    Orange.evaluation.CrossValidation(
    #      data, learners, k=folds,
    #        preprocessor=lambda data, p=p: select_proportion_preproc(data, p),
    #        callback=callback_wrapped(i))
    #    for i, p in enumerate(proportions)
    #]
    return new_data



if __name__ == "__main__":
    WidgetPreview(OWLearningCurveC).run(set_dataset=Orange.data.Table([[1,2,3,4,5,6],[324,234,234,234,45,76]]))
