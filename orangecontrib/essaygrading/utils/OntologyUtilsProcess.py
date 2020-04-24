def thread_func(tpl):

    import copy
    from orangecontrib.essaygrading.utils.ExtractionManager import ExtractionManager

    i = tpl[0]
    prepared_essays = tpl[1]
    original_g = tpl[2]
    essay = tpl[3]
    uniqueURIRef = tpl[4]
    openie = tpl[5]
    explain = tpl[6]
    PATH = str(tpl[7])

    debugPrint(i, " ----- Processing essay " + str(i) + " / " + str(len(prepared_essays)) + " --------")

    # WORKAROUND
    # debugPrint(i, "Reading URIRefs...")
    # uniqueURIRef = readURIRefs(PATH)

    # debugPrint(i, "Parsing ontology...")
    # pathToOntology = PATH + "/data/" + original_g
    # original_g = Graph()
    # original_g.parse(pathToOntology, encoding="xml")

    # END WORKAROUND


    g = copy.deepcopy(original_g)

    extractionManager = ExtractionManager(turbo=True, i=i)
    chunks = extractionManager.getChunks(essay)
    debugPrint(i, extractionManager.mergeEssayAndChunks(essay, chunks["np"], "SubjectObject"))
    debugPrint(i, extractionManager.mergeEssayAndChunks(essay, chunks["vp"], "Predicate"))

    URIs = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['SubObj'], "SubjectObject")
    # print(URIs)
    URIs = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['Pred'], "Predicate")
    # print(URIs)

    # ALA: URIs_predicates = extractionManager.matchEntitesWithURIRefs(uniqueURIRef['Pred'])
    # print("UNIQUE URI REF: " + str(uniqueURIRef["SubObj"]))
    # TUKAJ imamo zdej isto razclenjenoe predikate in objekte, ampak so zraven še "Ref" vozlisca

    # ADD OPENIE EXTRACTIONS TO ONTOLOGY
    debugPrint(i, "OpenIE extraction...")
    debugPrint(i, essay)
    triples = []
    try:
        triples = openie.extract_triples([essay])
    except:
        import sys
        debugPrint(i, "Unexpected error: ", sys.exc_info()[0])
        feedback = []
        errors = [-1, -1, -1]
        exc = sys.exc_info()[0]

    debugPrint(i, triples)

    # 'be' je v URIREF['SubObj']

    debugPrint(i, "Adding extractions to ontology...")

    exc = ""

    try:
        feedback, errors = extractionManager.addExtractionToOntology(g, triples[0], uniqueURIRef['SubObj'],
                                                                     uniqueURIRef['Pred'], explain=explain)
    except Exception as e:
        import sys
        debugPrint(i, "Unexpected error: ", str(e))
        feedback = []
        errors = [-1, -1, -1]
        exc = str(e)

    # Temporary? result saving
    with open(PATH + '/data/results/' + str(i) + '.txt', 'w') as file:
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


def debugPrint(i='[X]', *args, **kwargs, ):
    print("[" + str(i) + "] ", end="")
    print(*args, **kwargs)


if __name__ == "__main__":
    pass
