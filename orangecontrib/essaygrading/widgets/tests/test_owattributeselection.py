import os

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.essaygrading.widgets.OWAttributeSelection import OWAttributeSelection
from orangecontrib.essaygrading.widgets.OWAttributeSelection import calculateAttributes
from orangecontrib.essaygrading.utils import globals
from orangecontrib.text import Corpus


class TestOWAttributeSelection(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWAttributeSelection)

    def test_grade_detection(self):
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\..\\datasets\\custom_data.tsv"
        corpus = Corpus.from_file(path)

        self.send_signal(self.widget.Inputs.data, corpus)

        self.assertListEqual([10, 5, 9, 4], list(self.widget.corpus_grades))

    def test_attribute_extraction_basic(self):
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\..\\datasets\\custom_data.tsv"
        corpus = Corpus.from_file(path)

        def callback(x):
            pass

        attrbiuteDictionary, _ = calculateAttributes(corpus, None, self.widget.corpus_grades, None,
                                                     ["Basic measures"],
                                                     callback, globals.EMBEDDING_TFIDF, None)

        self.assertEqual(14, len(attrbiuteDictionary))

    def test_attribute_extraction_advanced(self):
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "\\..\\datasets\\custom_data.tsv"
        corpus = Corpus.from_file(path)

        self.send_signal(self.widget.Inputs.data, corpus)

        def callback(x):
            pass

        attributeDictionary, _ = calculateAttributes(corpus, None, self.widget.corpus_grades, None,
                                                     ["Basic measures", "Lexical diversity", "Readability measures",
                                                      "Grammar", "Content", "Coherence and semantics"],
                                                     callback, globals.EMBEDDING_TFIDF, None)

        self.assertEqual(104, len(attributeDictionary))



