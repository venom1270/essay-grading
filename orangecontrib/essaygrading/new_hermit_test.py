from rdflib.graph import Graph
from rdflib.term import URIRef
from rdflib.graph import Namespace
from rdflib.namespace import RDF, OWL, RDFS

from orangecontrib.essaygrading.utils.HermiT import HermiT


g = Graph()
g.parse("C:/Users/zigsi/Desktop/OIE/HermiT/ontologies/ontology_tmp.owl", format="xml")
#g.parse("data/COSMO-Serialized.owl", format="xml")

hermit = HermiT()
hermit.check_unsatisfiable_cases(g)
