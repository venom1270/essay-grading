from rdflib.graph import Graph
from rdflib.graph import Namespace
from rdflib.namespace import RDF, OWL, RDFS
import nltk
import neuralcoref

from orangecontrib.essaygrading.utils import ExtractionManager
from orangecontrib.essaygrading.utils import OpenIEExtraction
from orangecontrib.essaygrading.utils.lemmatizer import breakToWords

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


def run_semantic_consistency_check(essays, use_coref=False, openie_system="ClausIE"):

    # essays: [[sentence1, sentence2, ...], essay2, ...]

    print("Starting...")

    g = Graph()
    #g.parse("../data/COSMO-Serialized.owl", format="xml")
    g.parse("C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/COSMO-Serialized.owl",
            format="xml")

    subObjSet = []
    predSet = []
    count = 0

    COSMO = Namespace("http://micra.com/COSMO/COSMO.owl#")

    for subj, pred, obj in g:

        if pred == RDF.type and obj == OWL.ObjectProperty:
            predSet.append(subj)
        elif pred == RDF.type and obj == OWL.Class:
            subObjSet.append(subj)
        else:
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

    # tukaj imamo razclenjene objekte in predikate
    porter = nltk.PorterStemmer()

    uniqueURIRef = {}
    uniqueURIRef['SubObj'] = [stringSubObjBroken, uniqueURIRefSubObj]
    uniqueURIRef['Pred'] = [stringPredBroken, uniqueURIRefPred]

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


    print("Ontology preparation finished")

    prepared_essays = []

    # TODO: to se razmisli kako polepsat
    if use_coref:
        print("Using coref...")
        print("Loading coref...")
        nlp = spacy.load("en_core_web_lg")
        coref = neuralcoref.NeuralCoref(nlp.vocab)
        nlp.add_pipe(coref, name="neuralcoref")
        for i in range(len(essays)):
            print("Essay " + str(i))
            essay = " ".join(essays[i])
            doc = nlp(essay)
            # TODO: punkt sentence tokenizer?
            essay = doc._.coref_resolved.split(". ")
            essay_final = []
            for e in essay[:-1]:
                essay_final.append(e + ".")
            essay_final.append(essay[-1])
            print(essay_final)
            prepared_essays.append(essay_final)
        nlp.remove_pipe("neuralcoref")
    else:
        prepared_essays = essays
    print(prepared_essays)


    print("Final processing")

    if openie_system == "ClausIE":
        openie = OpenIEExtraction.ClausIE()
    elif openie_system == "OpenIE-5.0":
        openie = OpenIEExtraction.OpenIE5()

    final_results = []
    essays_feedback = []
    essays_errors = []

    for i, essay in enumerate(prepared_essays):

        print(" ----- Processing essay " + str(i+1) + " / " + str(len(prepared_essays)) + " --------")

        extractionManager = ExtractionManager.ExtractionManager(turbo=True)
        chunks = extractionManager.getChunks(essay)
        print(extractionManager.mergeEssayAndChunks(essay, chunks["np"], "SubjectObject"))
        print(extractionManager.mergeEssayAndChunks(essay, chunks["vp"], "Predicate"))

        URIs = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['SubObj'], "SubjectObject")
        #print(URIs)
        URIs = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['Pred'], "Predicate")
        #print(URIs)

        # ALA: URIs_predicates = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['Pred'])
        #print("UNIQUE URI REF: " + str(uniqueURIRef["SubObj"]))
        # TUKAJ imamo zdej isto razclenjenoe predikate in objekte, ampak so zraven še "Ref" vozlisca


        # ADD OPENIE EXTRACTIONS TO ONTOLOGY
        print("OpenIE extraction...")
        print(essay)
        triples = openie.extract_triples([essay])
        print(triples)

        # 'be' je v URIREF['SubObj']

        print("Adding extractions to ontology...")
        feedback, errors = extractionManager.addExtractionToOntology(g, triples[0], uniqueURIRef['SubObj'], uniqueURIRef['Pred'])
        essays_feedback.append(feedback)
        essays_errors.append(errors)

    print(essays_errors)

    return essays_feedback

