def thread_func(tpl):
    '''
    Multiprocessing function. Takes care of OpenIE extractions and semantic consistency analysis.
    :param tpl: tuple: (index, essay list, ontology, URIRefs in ontology, openie to use, explan flag, results path)
    :return: [index, basic explantions, errors, detailed explanations]
    '''

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
    source_text = tpl[8]

    debugPrint(i, " ----- Processing essay " + str(i) + " / " + str(len(prepared_essays)) + " --------")

    g = copy.deepcopy(original_g)

    extractionManager = ExtractionManager(turbo=True, i=i)
    chunks = extractionManager.getChunks(essay, URIRefs=uniqueURIRef)


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
        if len(triples) > 0:
            feedback, errors, feedback2 = extractionManager.addExtractionToOntology(g, triples[0], essay, uniqueURIRef['SubObj'],
                                                                     uniqueURIRef['Pred'], explain=explain, source_text=source_text)
        else:
            debugPrint(i, "#### ERROR: OpenIE extraction failure")
            feedback = [["ERROR: OpenIE extraction failure"]]
            errors = [-1,-1,-1]
            feedback2 = []
    except Exception as e:
        import sys
        debugPrint(i, "Unexpected error: ", str(e))
        feedback = []
        feedback2 = []
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

    return [i, feedback, errors, feedback2]


def debugPrint(i='[X]', *args, **kwargs, ):
    '''
    Logging method. Same behavior as "print()", but prints id in format "[ID] " before content (*args).
    '''
    print("[" + str(i) + "] ", end="")
    print(*args, **kwargs)


if __name__ == "__main__":
    pass
