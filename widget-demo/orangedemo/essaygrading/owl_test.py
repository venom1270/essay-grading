#pip install owlready2

#from owlready2 import *
from rdflib.graph import Graph
from rdflib.term import URIRef
from rdflib.graph import Namespace
import nltk
import re

from orangedemo.essaygrading.utils.HermiT import HermiT
from orangedemo.essaygrading.utils import ExtractionManager
from orangedemo.essaygrading.utils import OpenIEExtraction
from orangedemo.essaygrading.utils.lemmatizer import breakToWords

'''
TODO

Kaja je sla nekako tako:

rdflig graph.parse (COSMO)

.triples razdelis na (O1, P, O2)
grupiras [O1, O2] in [P]
VSI teli konstrukti so oblike # isSubsetOf tko da jih locis glede na veliko zacetnico 
(problem je ker so isaClass namesto isAClass) in stemmas

nato je sla neki povezovat ID-je to se nevem tocno kaj je
bistvo je to da hranis v arrayu vse triple [O1,P,O2], zraven pa vozis se ontologijo z ID-ji
ID-ji so neki stringi npr. ClassBook (nevem ce je realen ampak primer)

potem ma ona nekio funkcijo addEltToOntology, kjer najprej doda vse relacije in objekte v ontologijo in preveri ce gre skos
in potem doda se zares v ontologijo (povezave? kaj tocno nevem?) in preveri s HermiTom


'''

import spacy


#onto_path.append("data/")
#onto = get_ontology("data/COSMO.owl")
#onto = get_ontology("http://www.cs.ox.ac.uk/isg/ontologies/UID/00793.owl")
#onto = get_ontology("http://www.micra.com/COSMO/COSMO.owl")
#onto.load()

g = Graph()
g.parse("data/COSMO-Serialized.owl", format="xml")
bicycleURI = URIRef("http://micra.com/COSMO/COSMO.owl#Bicycle")
t = g.triples((None, None, None))
print("Num of triples: ", len(g))
print(t)

subObjSet = []
predSet = []
count = 0
'''
hermit = HermiT()
hermit.check_unsatisfiable_cases(g)

exit()
'''


for subj, pred, obj in g:

    #if str(type(subj)) == "<class 'rdflib.term.BNode'>" or str(type(pred)) == "<class 'rdflib.term.BNode'>" or str(type(obj)) == "<class 'rdflib.term.BNode'>":
    #    count += 1
    #    g.remove((subj, pred, obj))

    subObjSet.extend([subj, obj])
    predSet.append(pred)


print(count)
print(len(g))
'''
COSMO = Namespace("http://micra.com/COSMO/COSMO.owl#")

for meaning in g.objects(URIRef("http://micra.com/COSMO/COSMO.owl#King"), COSMO.wnsense):
    print(meaning)
    print(re.findall(r'(\w+?)(\d+)([a-z]+)', meaning))

exit()
'''
'''
print(COSMO["Person"])
print(COSMO["http://colab.cim3.net/file/work/SICoP/ontac/COSMO/COSMO.owl#Wizard"])
print(COSMO["http://colab.cim3.net/file/work/SICoP/ontac/COSMO/COSMO.owl#Dwarf"])
print(COSMO["http://colab.cim3.net/file/work/SICoP/ontac/COSMO/COSMO.owl#Elf"])
print(COSMO["http://colab.cim3.net/file/work/SICoP/ontac/COSMO/COSMO.owl#Troll"])
exit()
g.add(("http://colab.cim3.net/file/work/SICoP/ontac/COSMO/COSMO.owl#Person", "like", "candy"))
g.add(("Lisa", "not like", "candy"))
g.serialize("data/COSMO-Serialized.owl", format="pretty-xml")
exit()
'''
uniqueSubObj = set(subObjSet)
uniqueURIRefSubObj = []
for node in uniqueSubObj:
    if str(type(node)) == "<class 'rdflib.term.URIRef'>":
        uniqueURIRefSubObj.append(node)

uniquePred = set(predSet)
uniqueURIRefPred = []
for node in uniquePred:
    if str(type(node)) == "<class 'rdflib.term.URIRef'>":
        uniqueURIRefPred.append(node)

print(len(uniqueURIRefSubObj)) #COSMO=10916
print(len(uniqueURIRefPred)) #COSMO=352


print(uniqueURIRefSubObj)
print(uniqueURIRefPred)

# uniqueURIRef = set(uniqueURIRefSubObj + uniqueURIRefPred)
# print(len(uniqueURIRef)) #COSMO=10936

# for node in uniqueURIRefSubObj:
# 	if
# 	print(str(node)[str(node).index("#")+1:])
# 	print(' - - - ')

stringSubObj = [str(node)[str(node).find("#") + 1:] for node in uniqueURIRefSubObj]
for i in range(len(stringSubObj) - 1, -1, -1):
    if stringSubObj[i] == '':
        del stringSubObj[i]
stringSubObjBroken = [breakToWords(s) for s in stringSubObj]
stringPred = [str(node)[str(node).find("#") + 1:] for node in uniqueURIRefPred]
for i in range(len(stringPred) - 1, -1, -1):
    if stringPred[i] == '':
        del stringPred[i]
stringPredBroken = [breakToWords(s) for s in stringPred]

print(stringSubObj)
print(stringSubObjBroken)
print(stringPred)
print(stringPredBroken)

# tukaj imamo razclenjene objekte in predikate

porter = nltk.PorterStemmer()

uniqueURIRef = {}
uniqueURIRef['SubObj'] = [stringSubObjBroken, uniqueURIRefSubObj]
uniqueURIRef['Pred'] = [stringPredBroken, uniqueURIRefPred]

print(uniqueURIRef["SubObj"][0][0])

stemedUniqueURIRefso = [None for v in uniqueURIRef['SubObj'][0]]
for i in range(len(uniqueURIRef['SubObj'][0])):
    stemedUniqueURIRefso[i] = [porter.stem(v) for v in uniqueURIRef['SubObj'][0][i].split()]
    stemedUniqueURIRefso[i] = ' '.join(stemedUniqueURIRefso[i])
uniqueURIRef['SubObj'].append(stemedUniqueURIRefso)

stemedUniqueURIRefp = [None for v in uniqueURIRef['Pred'][0]]
for i in range(len(uniqueURIRef['Pred'][0])):
    stemedUniqueURIRefp[i] = [porter.stem(v) for v in uniqueURIRef['Pred'][0][i].split()]
    stemedUniqueURIRefp[i] = ' '.join(stemedUniqueURIRefp[i])
uniqueURIRef['Pred'].append(stemedUniqueURIRefp)


print("MY ESSAY")

#test_essay = ["Lisa is a girl.", "She likes all kinds of sports.", "Lisa likes tennis the most.", "Tennis is a fast sport."]
#test_essay = ["Tennis is a fast sport.", "Lisa doesn't like fast sports.", "Lisa likes tennis."]
test_essay = ["Lisa is a boy.", "Lisa is a girl."]
extractionManager = ExtractionManager.ExtractionManager()
chunks = extractionManager.getChunks(test_essay)
print(extractionManager.mergeEssayAndChunks(test_essay, chunks))

print("END MY ESSAY")

URIs = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['SubObj'][0])
print(URIs)
# TODO: add VPs to predicate edges and match with URIRefs
# ALA: URIs_predicates = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['Pred'])
print("UNIQUE URI REF: " + str(uniqueURIRef["SubObj"]))
# TUKAJ imamo zdej isto razclenjenoe predikate in objekte, ampak so zraven še "Ref" vozlisca

# TODO NUJNO!!! : POFIXEJ TO DA SE CUDNO APPENDA - najprej ne stematiziran, potem stematizirano v drugacni obliki arraya - problem je ker ne najde "femal" v URIRefs...


# ADD OPENIE EXTRACTIONS TO ONTOLOGY
openie = OpenIEExtraction.ClausIE()
triples = openie.extract_triples([test_essay])
print(triples)

extractionManager.addExtractionToOntology(g, triples[0], uniqueURIRef['SubObj'], uniqueURIRef['Pred'])



'''
i = 0
for triple in t:
    print(triple)
    i += 1
    #if i > 1000: break
'''