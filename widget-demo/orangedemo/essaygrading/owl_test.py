#pip install owlready2

from owlready2 import *
from rdflib.graph import Graph
from rdflib.term import URIRef
import nltk
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

def breakToWords(s):
	charIndex = 0
	sBroken = ''
	for c in s:
		if charIndex==0:
			sBroken = sBroken + c.lower()
		elif c.isupper():
			sBroken = sBroken + ' ' + c.lower()
		else:
			sBroken = sBroken + c
		charIndex = charIndex + 1
	return(sBroken)

#onto_path.append("data/")
#onto = get_ontology("data/COSMO.owl")
onto = get_ontology("http://www.cs.ox.ac.uk/isg/ontologies/UID/00793.owl")
#onto = get_ontology("http://www.micra.com/COSMO/COSMO.owl")
onto.load()

g = Graph()
g.parse("data/COSMO.owl", format="xml")
bicycleURI = URIRef("http://micra.com/COSMO/COSMO.owl#Bicycle")
t = g.triples((None, None, None))
print("Num of triples: ", len(g))
print(t)

subObjSet = []
predSet = []
for subj, pred, obj in g:
    subObjSet.extend([subj, obj])
    predSet.append(pred)

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

# print(len(uniqueURIRefSubObj)) #COSMO=10916
# print(len(uniqueURIRefPred)) #COSMO=352
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
uniqueURIRef['SubObj'] = [stringSubObjBroken, uniqueURIRefSubObj, [None for k in range(len(uniqueURIRefSubObj))]]
uniqueURIRef['Pred'] = [stringPredBroken, uniqueURIRefPred, [None for k in range(len(uniqueURIRefPred))]]

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


print(uniqueURIRef["Pred"])
# TUKAJ imamo zdej isto razclenjenoe predikate in objekte, ampak so zraven še "Ref" vozlisca

print('ID_URI')
# if (os.path.exists("ID_URI/ID_URI"+fileName+str(DS)+".txt")):
# 	print("already exists")
# 	ID_URI = json.load(open("ID_URI/ID_URI"+fileName+str(DS)+".txt"))
# else:
ID_URI = dict((key, '') for key in [x + 1 for x in range(maxID)])
ID_URI = matchIDsWithURIs(ID_URI, bagOfEntities, uniqueURIRef['SubObj'], 'SO')
# json.dump(ID_URI, open("ID_URI/ID_URI"+fileName+str(DS)+".txt",'w'))
ID_URI = {int(k): rdflib.URIRef(v) for k, v in ID_URI.items()}

# for key, value in ID_URI.items():
# 	print(str(key) + ': ' + str(value))

# printList(bagOfEntities)


print('edgeID_URI')

'''
i = 0
for triple in t:
    print(triple)
    i += 1
    #if i > 1000: break
'''