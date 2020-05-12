#pip install owlready2

#from owlready2 import *
from rdflib.graph import Graph
from rdflib.term import URIRef, Literal
from rdflib.graph import Namespace
from rdflib.namespace import OWL, RDF, RDFS
import nltk
import re

from nltk.corpus import wordnet as wn

from orangecontrib.essaygrading.utils.HermiT import HermiT
from orangecontrib.essaygrading.utils import ExtractionManager
from orangecontrib.essaygrading.utils import OpenIEExtraction
from orangecontrib.essaygrading.utils.lemmatizer import breakToWords



def recurse_add_remove(ONTO, root, rdfType, operation, subj, pred):
    for el in ONTO.subjects(rdfType, root):
        print(str(el))
        if operation=="add":
            if (subj, pred, el) not in ONTO:
                ONTO.add((subj, pred, el))
        else:
            if (subj, pred, el) in ONTO:
                ONTO.remove((subj, pred, el))
        recurse_add_remove(ONTO, el, rdfType, operation, subj, pred)



'''
qwe = wn.synsets("slow_sport", pos=wn.NOUN)
for s in qwe:
    print(s)
    for lemma in s.lemmas():
        print(lemma.name())
'''
g = Graph()
#g.parse("data/COSMO-Serialized.owl", format="xml")
g.parse("external/hermit/ontologies/ontology_tmp_test_4.owl", format="xml")

subObjSet = []
predSet = []
count = 0

COSMO = Namespace("http://micra.com/COSMO/COSMO.owl#")
#g.add((COSMO["SlowSport"], RDF.type, OWL.Class))
#for meaning in g.objects(COSMO["SlowSport"], COSMO.wnsense):
#    print(meaning)
#exit()

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
URI_likes = COSMO["likes"]
URI_not_likes = COSMO["notLikes"]
URI_tennis = COSMO["Tennis"]
URI_sport = COSMO["Sport"]
URI_fast = COSMO["Fast"]
URI_slow = COSMO["Slow"]



#print("Adding 'fast'...")
#g.add((URI_fast, RDF.type, OWL.Class))

# g.remove((None, RDFS.comment, None))
# g.add((URI_lisa, RDFS.comment, Literal("qwecomment")))


#print(g.value(COSMO["Georg"], RDFS.comment))

#for x in g.triples((COSMO["Georg"], RDFS.comment, None)):
#    print(x)

#exit()

print("Addding TEST...")

URI_qwe = COSMO["qwe"]

g.add((URI_not_likes, RDF.type, OWL.ObjectProperty))
g.add((URI_qwe, RDF.type, OWL.ObjectProperty))
g.add((URI_not_likes, OWL.inverseOf, URI_qwe))
g.add((URI_qwe, OWL.inverseOf, URI_not_likes))

g.add((URI_lisa, URI_not_likes, URI_tennis))

g.add((URI_lisa, URI_qwe, URI_tennis))

hermit.check_unsatisfiable_cases(g)

exit()

print("Addding 'slow'...")
g.add((URI_slow, OWL.type, OWL.Class))

print("Addding slow disjointWith fast")
g.add((URI_slow, OWL.disjointWith, URI_fast))

print("Addding fast disjointWith slow")
g.add((URI_fast, OWL.disjointWith, URI_slow))

hermit.check_unsatisfiable_cases(g)

print("Adding TENNIS IS FAST")
g.add((URI_tennis, RDF.type, URI_fast))
hermit.check_unsatisfiable_cases(g)
print("ADDING TENNIS IS SLOW")
g.add((URI_tennis, RDF.type, URI_slow))
hermit.check_unsatisfiable_cases(g)

exit()

'''

print("Adding 'fast'...")
g.add((URI_fast, RDF.type, OWL.ObjectProperty))

print("Addding 'slow'...")
g.add((URI_slow, OWL.type, OWL.ObjectProperty))

print("Addding slow disjointWith fast")
g.add((URI_slow, OWL.propertyDisjointWith, URI_fast))

print("Addding fast disjointWith slow")
g.add((URI_fast, OWL.propertyDisjointWith, URI_slow))

hermit.check_unsatisfiable_cases(g)

print("Adding TENNIS IS FAST")
g.add((URI_tennis, URI_is, URI_fast))
hermit.check_unsatisfiable_cases(g)
print("ADDING TENNIS IS SLOW")
g.add((URI_tennis, URI_is, URI_slow))
hermit.check_unsatisfiable_cases(g)

exit()

'''
'''
print("Adding 'fast'...")
g.add((URI_fast, RDF.type, OWL.ObjectProperty))

print("Addding 'slow'...")
g.add((URI_slow, OWL.type, OWL.ObjectProperty))

print("Addding slow disjointWith fast")
g.add((URI_slow, OWL.propertyDisjointWith, URI_fast))

print("Addding fast disjointWith slow")
g.add((URI_fast, OWL.propertyDisjointWith, URI_slow))

hermit.check_unsatisfiable_cases(g)

print("Adding TENNIS IS FAST")
g.add((URI_tennis, RDF.type, URI_fast))
hermit.check_unsatisfiable_cases(g)
print("ADDING TENNIS IS SLOW")
g.add((URI_tennis, RDF.type, URI_slow))
hermit.check_unsatisfiable_cases(g)

exit()

'''




print((URI_be, RDF.type, OWL.ObjectProperty) in g)
print((URI_boy, RDF.type, OWL.Class) in g)
print((URI_lisa, RDF.type, OWL.Class) in g)
print((URI_is, RDF.type, OWL.Class) in g)
print((URI_Is, RDF.type, OWL.Class) in g)
print((URI_is, RDF.type, OWL.ObjectProperty) in g)
print((URI_Is, RDF.type, OWL.ObjectProperty) in g)


# TEST NEGIRANIH RELACIJ
print("Adding Lisa...")
g.add((URI_lisa, RDF.type, OWL.Class))
hermit.check_unsatisfiable_cases(g)

#print("Adding likes")
#g.add((URI_likes, RDF.type, OWL.ObjectProperty))
#hermit.check_unsatisfiable_cases(g)

print("Adding not likes...")
g.add((URI_not_likes, RDF.type, OWL.ObjectProperty))
hermit.check_unsatisfiable_cases(g)


# VERY USEFUL:!!!!!! https://www.w3.org/TR/owl2-quick-reference/
# http://mowl-power.cs.man.ac.uk/protegeowltutorial/resources/ProtegeOWLTutorialP4_v1_1.pdf

print("Adding disjoint likes-notlikes...")
g.add((URI_not_likes, OWL.propertyDisjointWith, URI_likes))
g.add((URI_likes, OWL.propertyDisjointWith, URI_not_likes))
hermit.check_unsatisfiable_cases(g)

print("Adding lisa likes tennis....")
#for o in g.subjects(RDFS.subClassOf, URI_sport):
#    print("SUBCLASS OF SPORT: ")
#    print(o)
temp = COSMO["temp"]
g.add((URI_lisa, URI_likes, URI_tennis))
#def recurse_add_remove(ONTO, root, rdfType, operation, subj, pred);
recurse_add_remove(g, URI_tennis, RDFS.subClassOf, "add", URI_lisa, URI_likes)
hermit.check_unsatisfiable_cases(g)

#print("Adding lisa not likes tennis...")
#g.add((URI_lisa, URI_not_likes, URI_tennis))
print("Adding lisa not likes sport...")
g.add((URI_lisa, URI_not_likes, URI_sport))
#def recurse_add_remove(ONTO, root, rdfType, operation, subj, pred);
recurse_add_remove(g, URI_sport, RDFS.subClassOf, "add", URI_lisa, URI_not_likes)
hermit.check_unsatisfiable_cases(g, remove=False)

exit()

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


