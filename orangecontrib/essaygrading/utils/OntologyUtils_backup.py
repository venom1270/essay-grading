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


def run_semantic_consistency_check(essays, use_coref=False, openie_system="ClausIE", source_text=None):

    # essays: [[sentence1, sentence2, ...], essay2, ...]

    if source_text is not None:
        print(source_text)


    print("Starting...")

    g = Graph()
    #g.parse("../data/COSMO-Serialized.owl", format="xml")
    # g.parse("C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/COSMO-Serialized.owl",
    #         format="xml")
    # TODO: naredi izbiro base ontologije!!!
    g.parse("C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/DS4_ontology.owl",
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

    #essays = [essays[17]]

    # essays = essays[:4]

    print("Ontology preparation finished")

    prepared_essays = []

    # Za generiranje ontologije....
    #essays = [source_text]
    print(essays)
    #source_text = None


    # TODO SPODAJ!!!
    '''
    TODO: ce je source text:
        1. naredi coref na source textu in poženi semantic analysys
        2. shrani zadnjo ontologijo
        
        3. nadaljuj normalno kot je zdaj:
        3.1 vsem esejem prependi source text spredaj in naredi coref; nakonc odprependi
        3.2 semantic analysys, s tem da uporabiš ontologijo od soruce texta
        
        OPTIONAL: shranjuj vmesne korake, če kaj crasha da lahko resumaš????
    
    '''

    # TODO: to se razmisli kako polepsat
    if use_coref:
        print("Using coref...")
        print("Loading coref...")
        nlp = spacy.load("en_core_web_lg")
        coref = neuralcoref.NeuralCoref(nlp.vocab)
        nlp.add_pipe(coref, name="neuralcoref")
        if source_text is not None:
            source_text = [s.translate(str.maketrans('', '', string.punctuation)) for s in source_text]
        for i in range(len(essays)):
            print("Essay " + str(i))
            essays[i] = [s.translate(str.maketrans('', '', string.punctuation)) for s in essays[i]]
            essay = ". ".join(essays[i])
            print(essay)
            #essay = essay.replace("! ",". ").replace("? ",". ")
            print(len(essays[i]))

            essay_tokenized_len = len(nltk.sent_tokenize(essay)) # da pravilno odsekamo source text stran

            #print(len(source_text))
            # Ce imamo source text, ga appendamo na zacetek, da bo coref delal cez source in esej
            if source_text is not None: # TODO: naredi, da najprej to ignorira ko gre cez source text, nato pa uposteva
                essay = ". ".join(source_text) + ". " + essay
                print(essay)
            doc = nlp(essay)
            # TODO: punkt sentence tokenizer?
            print("NLTK TOKENIZE LEN:")
            print(len(nltk.sent_tokenize(doc._.coref_resolved)))
            print(nltk.sent_tokenize(doc._.coref_resolved))

            print(len(source_text))

            #print(doc._.coref_resolved)


            print("Essay tokenized len:")
            print(essay_tokenized_len)

            essay = nltk.sent_tokenize(doc._.coref_resolved)  # doc._.coref_resolved.split(". ")
            # Ce imamo source text, po corefu locimo source in esej... upam da dela prav
            if source_text is not None: # TODO: naredi, da najprej to ignorira ko gre cez source text, nato pa uposteva
                # essay = essay[len(source_text):]
                essay = essay[-essay_tokenized_len:]
            essay_final = []
            for e in essay[:-1]:
                essay_final.append(e + ".")
            essay_final.append(essay[-1])
            print(essay_final)
            print(len(essay_final))
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

    original_g = copy.deepcopy(g)

    threads = []


    #queue = multiprocessing.Queue()
    task_list = []
    start_time = time.time()
    for i, essay in enumerate(prepared_essays):

        task_list.append((i, prepared_essays, original_g, essay, uniqueURIRef, openie))

        #x = threading.Thread(target=thread_func, args=(i, prepared_essays, original_g, essay, uniqueURIRef, openie))
        '''x = multiprocessing.Process(target=thread_func, args=(i, prepared_essays, original_g, essay, uniqueURIRef, openie, queue))

        x.start()
        threads.append(x)
        time.sleep(300)'''

        '''print(" ----- Processing essay " + str(i+1) + " / " + str(len(prepared_essays)) + " --------")

        g = copy.deepcopy(original_g)

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
        essays_errors.append(errors)'''

    p = multiprocessing.Pool(processes=4)
    #task_list = task_list[:4]
    results = p.map(thread_func_backup, task_list, chunksize=1)

    #p.close()
    #p.join()
    print("**** RESULTS *****")
    for r in results:
        print(r)

    print("**** FINISHED *****")

    end_time = time.time()

    print("TIME: " + str(end_time-start_time))

    with open('C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/DS4Explanations/ALL.txt', 'w') as file:
        for i in results:
            file.write(str(i[0]) + "\t" + str(i[2][0]) + "\t" + str(i[2][1]) + "\t" + str(i[2][2]) + "\n")


    return essays_feedback


#def thread_func(i, prepared_essays, original_g, essay, uniqueURIRef, openie):
def thread_func_backup(tuple):

    i = tuple[0]
    prepared_essays = tuple[1]
    original_g = tuple[2]
    essay = tuple[3]
    uniqueURIRef = tuple[4]
    openie = tuple[5]


    print(" ----- Processing essay " + str(i + 1) + " / " + str(len(prepared_essays)) + " --------")

    g = copy.deepcopy(original_g)

    i += 8863

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
                                                                     uniqueURIRef['Pred'])
    except Exception as e:
        import sys
        print("Unexpected error: ", str(e))
        feedback = []
        errors = [-1, -1, -1]
        exc = str(e)

    # Temporary? result saving
    with open('C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/DS4Explanations/' + str(i) + '.txt', 'w') as file:
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
            file("** EXPLANATION " + str(ei) + "**\n")
            for e in feedback[ei]:
                if len(e) > 0:
                    file.write(e[0])
                    file.write("\n")
            file.write("****\n")
        if exc != "":
            file.write("EXCEPTION: " + str(exc))


    #queue.put([i, feedback, errors])
    return [i, feedback, errors]
