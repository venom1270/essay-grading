from rdflib.graph import Graph
from rdflib.graph import Namespace
from rdflib.namespace import RDF, OWL, RDFS
import nltk
import neuralcoref
import time
import copy
import spacy
import string
import multiprocessing

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


def coreference_resolution(essays, source_text=None):
    prepared_essays = []
    print("Loading coref...")

    nlp = spacy.load("en_core_web_lg")
    coref = neuralcoref.NeuralCoref(nlp.vocab)
    nlp.add_pipe(coref, name="neuralcoref")

    # if source_text is not None:
        # source_text = [s.translate(str.maketrans('', '', string.punctuation)) for s in source_text]

    if essays is None and source_text is not None:
        # Coref on source_text only
        st = " ".join(source_text)
        print("Length before coref: " + str(len(nltk.sent_tokenize(st))))
        doc = nlp(st)
        source_text = nltk.sent_tokenize(doc._.coref_resolved)
        print("Length after coref: " + str(len(source_text)))
        st_final = []
        #for e in source_text[:-1]:
        #    st_final.append(e + ".")
        #st_final.append(source_text[-1])
        st_final = source_text
        prepared_essays = st_final

    else:
        # Coref on essays
        for i in range(len(essays)):
            print("Essay " + str(i))
            # TODO: spremljaj ce je to vredu, problem je ker se tukaj odstrani ' in ne najde nasprotja
            #essays[i] = [s.translate(str.maketrans('', '', string.punctuation)) for s in essays[i]]
            #essay = ". ".join(essays[i]) + "."
            essay = " ".join(essays[i])
            # print(essay)
            # essay = essay.replace("! ",". ").replace("? ",". ")
            print("Length before coref: " + str(len(essays[i])))

            essay_tokenized_len = len(nltk.sent_tokenize(essay)) # da pravilno odsekamo source text stran

            #print(len(source_text))
            # Ce imamo source text, ga appendamo na zacetek, da bo coref delal cez source in esej
            if source_text is not None:
                essay = ". ".join(source_text) + ". " + essay
                print(essay)
            doc = nlp(essay)
            # print("NLTK TOKENIZE LEN:")
            # print(len(nltk.sent_tokenize(doc._.coref_resolved)))
            # print(nltk.sent_tokenize(doc._.coref_resolved))

            # print("Essay tokenized len:")
            print("Length after coref: " + str(essay_tokenized_len))

            essay = nltk.sent_tokenize(doc._.coref_resolved)  # doc._.coref_resolved.split(". ")
            # Ce imamo source text, po corefu locimo source in esej... upam da dela prav
            if source_text is not None:
                # essay = essay[len(source_text):]
                essay = essay[-essay_tokenized_len:]
            essay_final = []
            # for e in essay[:-1]: # TODO: dvojne pike mi je dal...
            #     essay_final.append(e + ".")
            # essay_final.append(essay[-1])
            essay_final = essay
            print("Final length: " + str(len(essay_final)))
            print(essay_final)
            prepared_essays.append(essay_final)

    nlp.remove_pipe("neuralcoref")
    return prepared_essays

def prepare_ontology(path):
    g = Graph()
    # g.parse("../data/COSMO-Serialized.owl", format="xml")
    # g.parse("C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/COSMO-Serialized.owl",
    #         format="xml")
    # TODO: naredi izbiro base ontologije!!!
    #g.parse("C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/DS4_ontology.owl",
    #        format="xml")

    g.parse(path, format="xml")

    subObjSet = []
    predSet = []

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

    return g, uniqueURIRef

def run_semantic_consistency_check(essays, use_coref=False, openie_system="ClausIE", source_text=None, num_threads=4, explain=False, orig_ontology_name="COSMO-Serialized.owl", ontology_name="QWE.owl", callback=None):
    '''
    TODO: ce je source text:
        1. naredi coref na source textu in poženi semantic analysys
        2. shrani zadnjo ontologijo

        3. nadaljuj normalno kot je zdaj:
        3.1 vsem esejem prependi source text spredaj in naredi coref; nakonc odprependi
        3.2 semantic analysys, s tem da uporabiš ontologijo od soruce texta

        OPTIONAL: shranjuj vmesne korake, če kaj crasha da lahko resumaš????
    '''

    if callback is not None:
        callback(0)

    print("Running semantic consistency check...")

    # essays = [essays[0]]

    if use_coref:
        original_source_text = copy.deepcopy(source_text)
        # original_essays = copy.deepcopy(essays)
        if source_text is not None:
            source_text = coreference_resolution(None, source_text)
            if essays is None:
                essays = [source_text]
            else:
                essays = coreference_resolution(essays, original_source_text)
        else:
            essays = coreference_resolution(essays)


    print("Preparing ontology... " + orig_ontology_name)

    ONTO, uniqueURIRefs = prepare_ontology("C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/" + orig_ontology_name)

    print("Starting OpenIE extraction...")

    if openie_system == "ClausIE":
        openie = OpenIEExtraction.ClausIE()
    elif openie_system == "OpenIE-5.0":
        openie = OpenIEExtraction.OpenIE5()

    original_ONTO = copy.deepcopy(ONTO)

    print("Assigning threads...")

    task_list = []
    start_time = time.time()
    for i, essay in enumerate(essays):
        print(essay)
        if len(essays) > 1:  # ce je len == 1, to pomeni da je samo source text -> gradnja ontologije
            index = i + 11827
        else:
            index = 0
        task_list.append((index, essays, original_ONTO, essay, uniqueURIRefs, openie, explain))

    print("Pooling...")

    p = multiprocessing.Pool(processes=num_threads)
    # task_list = task_list[:4]

    print("Run thread map...")
    results = p.map(thread_func, task_list, chunksize=1)

    print("**** RESULTS *****")
    for r in results:
        print(r)

    print("**** FINISHED *****")

    end_time = time.time()

    print("TIME: " + str(end_time - start_time))

    with open('C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/DS6Explanations/ALL.txt',
              'w') as file:
        for i in results:
            file.write(str(i[0]) + "\t" + str(i[2][0]) + "\t" + str(i[2][1]) + "\t" + str(i[2][2]) + "\n")

    from shutil import copyfile
    copyfile("C:/Users/zigsi/Desktop/OIE/HermiT/ontologies/ontology_tmp_test_0.owl", "C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/" + ontology_name)

    return results


def thread_func(tpl):

    print("THREAD FUNC!!!")

    i = tpl[0]
    prepared_essays = tpl[1]
    original_g = tpl[2]
    essay = tpl[3]
    uniqueURIRef = tpl[4]
    openie = tpl[5]
    explain = tpl[6]

    print(" ----- Processing essay " + str(i + 1) + " / " + str(len(prepared_essays)) + " --------")

    g = copy.deepcopy(original_g)

    # i += 11827

    extractionManager = ExtractionManager.ExtractionManager(turbo=True, i=i)
    chunks = extractionManager.getChunks(essay)
    print(extractionManager.mergeEssayAndChunks(essay, chunks["np"], "SubjectObject"))
    print(extractionManager.mergeEssayAndChunks(essay, chunks["vp"], "Predicate"))

    URIs = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['SubObj'], "SubjectObject")
    # print(URIs)
    URIs = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['Pred'], "Predicate")
    # print(URIs)

    # ALA: URIs_predicates = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['Pred'])
    # print("UNIQUE URI REF: " + str(uniqueURIRef["SubObj"]))
    # TUKAJ imamo zdej isto razclenjenoe predikate in objekte, ampak so zraven še "Ref" vozlisca

    # ADD OPENIE EXTRACTIONS TO ONTOLOGY
    print("OpenIE extraction...")
    print(essay)
    triples = []
    try:
        triples = openie.extract_triples([essay])
    except:
        import sys
        print("Unexpected error: ", sys.exc_info()[0])
        feedback = []
        errors = [-1, -1, -1]
        exc = sys.exc_info()[0]

    print(triples)

    # 'be' je v URIREF['SubObj']

    print("Adding extractions to ontology...")

    exc = ""

    try:
        feedback, errors = extractionManager.addExtractionToOntology(g, triples[0], uniqueURIRef['SubObj'],
                                                                     uniqueURIRef['Pred'], explain=explain)
    except Exception as e:
        import sys
        print("Unexpected error: ", str(e))
        feedback = []
        errors = [-1, -1, -1]
        exc = str(e)

    # Temporary? result saving
    with open('C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/DS6Explanations/' + str(i) + '.txt', 'w') as file:
        file.write(str(i))
        file.write("\n")
        file.write("Consistency errors: " + str(errors[0]))
        file.write("\n")
        file.write("Semantic errors: " + str(errors[1]))
        file.write("\n")
        file.write("Sum: " + str(errors[2]))
        file.write("\n")
        # 3je arrayi?
        for ei in range(len(feedback)):
            file.write("** EXPLANATION " + str(ei) + "**\n")
            for e in feedback[ei]:
                if len(e) > 0:
                    for f in e:
                        file.write(f)
                        file.write("\n")
                    file.write("\n\n")
            file.write("****\n")
        if exc != "":
            file.write("EXCEPTION: " + str(exc))

    return [i, feedback, errors]


if __name__ == "__main__":
    essays = [["Lisa is a boy", "Lisa is a girl"],
              ["Tennis is a fast sport", "Lisa doesn't like fast sports", "Lisa likes tennis"]]

    run_semantic_consistency_check(essays, use_coref=True)
