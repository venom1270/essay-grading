#pip install owlready2

#from owlready2 import *
from rdflib.graph import Graph
from rdflib.term import URIRef
from rdflib.graph import Namespace
from rdflib.namespace import OWL, RDF, RDFS
import nltk
import re

from orangedemo.essaygrading.utils.HermiT import HermiT
from orangedemo.essaygrading.utils import ExtractionManager
from orangedemo.essaygrading.utils import OpenIEExtraction
from orangedemo.essaygrading.utils.lemmatizer import breakToWords


g = Graph()
g.parse("data/COSMO-Serialized.owl", format="xml")

subObjSet = []
predSet = []
count = 0
'''



exit()
'''
hermit = HermiT()
COSMO = Namespace("http://micra.com/COSMO/COSMO.owl#")
URI_male = COSMO["MalePerson"]
URI_female = COSMO["FemalePerson"]
URI_girl = COSMO["Girl"]
URI_boy = COSMO["Boy"]
URI_lisa = COSMO["Lisa"]
URI_is = COSMO["is"]
URI_Is = COSMO["Is"]
URI_be = COSMO["Be"]

print((URI_be, RDF.type, OWL.ObjectProperty) in g)
print((URI_boy, RDF.type, OWL.Class) in g)
print((URI_lisa, RDF.type, OWL.Class) in g)
print((URI_is, RDF.type, OWL.Class) in g)
print((URI_Is, RDF.type, OWL.Class) in g)
print((URI_is, RDF.type, OWL.ObjectProperty) in g)
print((URI_Is, RDF.type, OWL.ObjectProperty) in g)

print("Adding Lisa...")
#g.add((URI_lisa, RDF.type, OWL.Class))
#hermit.check_unsatisfiable_cases(g)

print("Adding 'is'...")
g.add((URI_is, RDF.type, OWL.ObjectProperty))
hermit.check_unsatisfiable_cases(g)

print("Addding male disjointWith female")
g.add((URI_male, OWL.disjointWith, URI_female))
hermit.check_unsatisfiable_cases(g)

print("Addding female disjointWith male")
g.add((URI_female, OWL.disjointWith, URI_male))
hermit.check_unsatisfiable_cases(g)

print("Addding boy subClassOf male")
g.add((URI_boy, RDFS.subClassOf, URI_male))
hermit.check_unsatisfiable_cases(g)

print("Addding girl subClassOf female")
g.add((URI_girl, RDFS.subClassOf, URI_female))
hermit.check_unsatisfiable_cases(g)

print("***** ADDING TRIPLE RELATIONS ********")
tmp = COSMO["temp"]
print("Adding Lisa is girl....")
g.add((URI_lisa, RDF.type, URI_girl))
#g.add((tmp, RDF.type, URI_female))
#g.add((tmp, RDF.type, URI_lisa))
hermit.check_unsatisfiable_cases(g)

print("Adding Lisa is boy....")
g.add((URI_lisa, RDF.type, URI_boy))
#g.add((tmp, RDF.type, URI_male))
#g.add((tmp, RDF.type, URI_lisa))
hermit.check_unsatisfiable_cases(g, remove=False)

if (URI_male, OWL.disjointWith, URI_female) in g:
    print("DISJOINT MALE FEMALE")
if (URI_lisa, OWL.disjointWith, URI_male) in g:
    print("DISJOINT LISA MALE")






for subj, pred, obj in g:

    #if str(type(subj)) == "<class 'rdflib.term.BNode'>" or str(type(pred)) == "<class 'rdflib.term.BNode'>" or str(type(obj)) == "<class 'rdflib.term.BNode'>":
    #    count += 1
    #    g.remove((subj, pred, obj))

    subObjSet.extend([subj, obj])
    predSet.append(pred)


print(count)
print(len(g))


