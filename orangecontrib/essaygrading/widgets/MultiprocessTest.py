from Orange.widgets import gui
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

import multiprocessing


class MultiprcoessTest(OWWidget):
    name = "MultiprcoessTest"
    description = "MultiprcoessTest."
    icon = "../icons/DataSamplerA.svg"
    priority = 10

    want_main_area = False

    def __init__(self):
        super().__init__()
        gui.button(self.controlArea, self, "Apply", callback=self.test)

    def test(self):
        testMP()
        gui.button(self.controlArea, self, "DONE", callback=None)


def testMP():
    task_list = []
    for i in range(4):
        task_list.append((i, "qwe"))

    p = multiprocessing.Pool(processes=4)

    results = p.map(process_func, task_list, chunksize=1)

    return results


def process_func(tpl):
    try:
        print("THREAD FUNC!!!")
        i = tpl[0]
        print("### I = " + str(i))
        return i
    except:
        return "ERROR"


if __name__ == "__main__":
    WidgetPreview(MultiprcoessTest).run()
