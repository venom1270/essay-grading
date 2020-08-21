from rdflib.graph import Graph
from rdflib.namespace import RDF, OWL, RDFS
import os
import nltk
import time
import copy
import spacy
import neuralcoref
import multiprocessing

from orangecontrib.essaygrading.utils import OpenIEExtraction
from orangecontrib.essaygrading.utils.OntologyUtilsProcess import thread_func
from orangecontrib.essaygrading.utils.lemmatizer import breakToWords


def coreference_resolution(essays, source_text=None):
    """
    Performs coreference resolution on provided essays. If source text present, it:
    - prepends source text to every essay
    - performs coreference resolution on prepended essays
    - removes prepended source text from essays
    That way, source text coreferences are taken into accoutn when doing corefernce resolution on essays based on
     source text.
    :param essays: list of essays.
    :param source_text: source text string.
    :return: coreference resolved essays.
    """
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
        # for e in source_text[:-1]:
        #    st_final.append(e + ".")
        # st_final.append(source_text[-1])
        st_final = source_text
        prepared_essays = st_final

    else:
        # Coref on essays
        for i in range(len(essays)):
            print("Essay " + str(i))
            # TODO: spremljaj ce je to vredu, problem je ker se tukaj odstrani ' in ne najde nasprotja
            # essays[i] = [s.translate(str.maketrans('', '', string.punctuation)) for s in essays[i]]
            # essay = ". ".join(essays[i]) + "."
            essay = " ".join(essays[i])
            # print(essay)
            # essay = essay.replace("! ",". ").replace("? ",". ")
            print("Length before coref: " + str(len(essays[i])))

            essay_tokenized_len = len(nltk.sent_tokenize(essay))  # da pravilno odsekamo source text stran

            # print(len(source_text))
            # Ce imamo source text, ga appendamo na zacetek, da bo coref delal cez source in esej
            if source_text is not None:
                tmp = ". ".join(source_text)
                if tmp.endswith("."):
                    tmp += " " + essay
                elif tmp.endswith(". "):
                    tmp += essay
                else:
                    tmp += ". " + essay
                essay = tmp
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
    """
    Prepares ontology and extracts included URIRefs.
    :param path: path to ontology.
    :return: rdflib Graph() object, URIRef list ([0] = Subject/objects, [1] = Predicates)
    """
    g = Graph()
    # g.parse("../data/COSMO-Serialized.owl", format="xml")
    # g.parse("C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/COSMO-Serialized.owl",
    #         format="xml")
    # TODO: naredi izbiro base ontologije!!!
    # g.parse("C:/Users/zigsi/Google Drive/ASAP corpus/widget-demo/orangecontrib/essaygrading/data/DS4_ontology.owl",
    #        format="xml")

    g.parse(path, format="xml")

    if path.endswith("COSMO-Serialized.owl"):
        g.remove((None, RDFS.comment, None))

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

    # We sort this set, so output is deterministic, otherwise numbers can differ between runs
    uniqueSubObj = sorted(set(subObjSet), reverse=True)
    uniqueURIRefSubObj = []
    for node in uniqueSubObj:
        if str(type(node)) == "<class 'rdflib.term.URIRef'>":
            uniqueURIRefSubObj.append(node)

    # We sort this set, so output is deterministic, otherwise numbers can differ between runs
    uniquePred = sorted(set(predSet), reverse=True)
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


p = None


def run_semantic_consistency_check(essays, use_coref=False, openie_system="ClausIE", source_text=None, num_threads=4,
                                   explain=False, orig_ontology_name="COSMO-Serialized.owl",
                                   ontology_name="SourceTextOntology.owl", callback=None):
    """
    :param essays: list of essays.
    :param use_coref: flag to use coreference resolution.
    :param openie_system: which OpenIE system to use. ("ClausIE" or "OpenIE-5.0").
    :param source_text: optional source text.
    :param num_threads: number of threads to use for multiprocessing (default 4).
    :param explain: flag to return detailed explanations.
    :param orig_ontology_name: ontology filename to use as base ontology.
    :param ontology_name: ontology filename to use fo saving temporary ontology enriched with source text extractions.
    :param callback: Orange progessbar callback method.
    :return: [basic feedback, [consistency error count, semantic error count, sum], detailed feedback]
    """

    PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if callback is not None:
        callback(0.02)

    # Filtering essayss that were already checked (if essays are not none - not source text)
    # FILTERING MOVED TO TASK CREATION AS WE NEED PROPER INDEXES!!!
    indexes_to_keep = []
    if essays is not None and len(essays) > 1:
        indexes_to_keep = \
            [i for i, _ in enumerate(essays) if not os.path.isfile(PATH + '/data/results/' + str(i + 1) + '.txt')]
        essays = essays[indexes_to_keep]
    print(indexes_to_keep)
    print(len(essays))

    print("Running semantic consistency check...")

    # essays = [essays[0]]
    original_source_text = copy.deepcopy(source_text)
    if use_coref:
        # original_essays = copy.deepcopy(essays)
        if source_text is not None:
            source_text = coreference_resolution(None, source_text)
            if essays is None:
                essays = [source_text]
            else:
                essays = coreference_resolution(essays, original_source_text)
        else:
            essays = coreference_resolution(essays)
    else:
        if essays is None:
            essays = [source_text]

    # LEMMATIZATION
    """nlp = spacy.load("en_core_web_lg")
    e_tmp = []
    for essay in essays:
        e = []
        for sent in essay:
            doc = nlp(sent)
            for token in doc:
                print(token, token.lemma_)
            s = " ".join([token.lemma_ for token in doc]).replace("-PRON-", "me")
            e.append(s)
        e_tmp.append(e)

    essays = e_tmp
    print(essays)"""
    # END LEMMATIZATION

    print("Preparing ontology... " + orig_ontology_name)

    ONTO, uniqueURIRefs = prepare_ontology(PATH + '/data/' + orig_ontology_name)

    # writeURIRefs(uniqueURIRefs, PATH)  # Workaround for pythonw.exe not running prcesses properly...

    print("Initializing OpenIE extraction...")

    if openie_system == "ClausIE":
        openie = OpenIEExtraction.ClausIE()
    elif openie_system == "OpenIE-5.0":
        openie = OpenIEExtraction.OpenIE5()

    original_ONTO = copy.deepcopy(ONTO)

    print("Assigning threads...")

    task_list = []
    start_time = time.time()

    mp_run_size = 100
    all_tasks_chunks = []
    all_tasks = []

    count = 0
    for i, essay in enumerate(essays):
        print(essay)
        if len(essays) > 1:  # ce je len == 1, to pomeni da je samo source text -> gradnja ontologije
            index = indexes_to_keep[i] + 1
        else:
            index = 0
        if index >= 0:  # and not os.path.isfile(PATH + '/data/results/' + str(i+1) + '.txt'):
            print(index)
            task_list.append((index, essays, ONTO, essay, uniqueURIRefs, openie, explain, PATH, original_source_text))
            all_tasks.append((index, essays, ONTO, essay, uniqueURIRefs, openie, explain, PATH, original_source_text))
            count += 1
            if count >= mp_run_size:
                all_tasks_chunks.append(task_list)
                task_list = []
                count = 0
    if len(task_list) > 0:
        all_tasks_chunks.append(task_list)

    print("We have " + str(len(all_tasks_chunks)) + " tasks!")

    print("Pooling...")

    global p
    p = multiprocessing.Pool(processes=num_threads)
    # task_list = task_list[:4]

    if callback is not None:
        callback(0.03)

    print("Run thread map...")

    progress_range = 0.99 - 0.03
    current_progress = 0.03
    progress_increment = progress_range / len(all_tasks)  # TODO pravilni length

    results = []

    chunksize = max(1, int(len(task_list) / num_threads / 4))

    for i, val in enumerate(p.imap_unordered(thread_func, all_tasks, chunksize=1), 1):
        results.append(val)
        current_progress += progress_increment
        if callback is not None:
            callback(current_progress)
        print("################ FINISHED " + str(i) + "/" + str(len(task_list)) + "#########################")

    results = []

    # for tasks in all_tasks:
    #    print("#*#*#*# STARTING NEW MULTIPROCESS RUN")
    #    p = multiprocessing.Pool(processes=num_threads)
    #    results_partial = p.map(thread_func, tasks, chunksize=1)
    #    print("GOT RESULTS")
    #    p.terminate()
    #    print("TERMINATE")
    #    current_progress += progress_increment
    #    callback(current_progress)
    #    results += results_partial

    # results = p.map(thread_func, all_tasks, chunksize=1)

    results = sorted(results)

    if callback is not None:
        callback(0.99)

    print("**** RESULTS *****")
    for r in results:
        print(r)

    print("**** FINISHED *****")

    end_time = time.time()

    print("TIME: " + str(end_time - start_time))

    with open(PATH + '/data/results/ALL.txt', 'w') as file:
        for i in results:
            file.write(str(i[0]) + "\t" + str(i[2][0]) + "\t" + str(i[2][1]) + "\t" + str(i[2][2]) + "\n")

    if source_text is not None:
        from shutil import copyfile
        copyfile(PATH + '/external/hermit/ontologies/ontology_tmp_test_0.owl', PATH + '/data/' + ontology_name)

    if callback is not None:
        callback(1)

    return results


def terminatePool():
    """
    Terminates multiprocess pool in case of Widget deletion or changes. Also called when multiprocessing complete to
    make sure all processes are killed.
    """
    global p
    if p is not None:
        p.terminate()


if __name__ == "__main__":
    essays = [["Lisa is a boy", "Lisa is a girl"],
              ["Tennis is a fast sport", "Lisa doesn't like fast sports", "Lisa likes tennis"]]

    run_semantic_consistency_check(essays, use_coref=True)
