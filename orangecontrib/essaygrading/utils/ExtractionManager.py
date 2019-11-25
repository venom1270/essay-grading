import spacy
from nltk import word_tokenize, pos_tag, PorterStemmer
import string
import rdflib
import re
import textacy
import neuralcoref
from orangecontrib.essaygrading.utils.HermiT import HermiT
from nltk.corpus import wordnet
from orangecontrib.essaygrading.utils.lemmatizer import get_pos_tags

class ExtractionManager:

    def __init__(self, turbo=False):
        self.nlp = spacy.load("en_core_web_lg")
        # bagOfAllEntities (Kaja ima tudi te pomatchane pomoje)
        self.allEntities = {"SubjectObject": [], "Predicate": []}
        # bagOfEntities (pomatchani)- samo trenuten stavek, brez duplikatov
        self.entities = {"SubjectObject": [], "Predicate": []}
        self.URIdict = {}
        self.id = 0
        self.hermit = HermiT()
        self.COSMO = rdflib.Namespace("http://micra.com/COSMO/COSMO.owl#")
        self.allAddedTriples = []
        self.turbo = turbo

        self.consistencyErrors = 0
        self.semanticErrors = 0

    def getChunks(self, essay_sentences):
        noun_chunks = []
        verb_chunks = []
        for sentence in essay_sentences:
            doc = self.nlp(sentence)
            noun_chunks.append([chunk.text for chunk in doc.noun_chunks])

            lists = textacy.extract.pos_regex_matches(doc, r'<VERB>?<ADV>*<VERB>+') #pos_regex_matches(doc, r'<VERB>?<ADV>*<VERB>+')
            verb_chunks.append([vp.text for vp in lists])
        print(noun_chunks)
        print(verb_chunks)
        return {"np": noun_chunks, "vp": verb_chunks}

    def mergeEssayAndChunks(self, essay_sentences, chunks, type):
        tokens = [word_tokenize(sentence) for sentence in essay_sentences]

        for sentence_chunk in chunks: # Over sentences
            for chunk in sentence_chunk: # Over chunks in sentence
                # TODO?: Tukaj Kaje se preveri c je "Ive" in ga pretvori v "I ve"
                # TODO? Kaja je tukaj imela if ce je v seznamu *P-jev (VP, NP...)
                # Ce je NP, ga dodamo v 'bagOfEntities' - pri nas os vsi NP
                self.allEntities[type].append({"id": self.id, "text": self.preprocessExtraction(chunk)})
                self.id += 1
                # TODO?: Tukaj Kaja zbira se entities (ne bagOFEntities) in naredi takle seznam.. sem preveril:
                #       to je za 1) Koreference pravilno pomatchat (detajlov nism gledu) in 2) da se v povezave ontologije doda VP-je
                # entities: SEZNAM  [stavek][0] = seznam tokenov?
                # 					[stavek][1] = tip fraze (NP, VP, ...)
                #					[stavek][2] = ID v bagOfEntities
        return self.allEntities

    # URIRefs = array[0] = tokeni, [1] = URIREF, [2] = None (???ocitno ID), [3] = Stemmed
    def matchEntitesWithURIRefs(self, URIRefs, type):

        URIs = {}

        for entity in self.allEntities[type]:
           # for URI_index in range(len(URIRefs[3])):
                #print(URIRefs[3][URI_index])
                #exit()
            #print(URIRefs[2])
            similarNode, _ = self.similarNode(URIRefs[2], entity["text"], indepth=True)
            if similarNode is not None:
                print("SIMILAR")
                print(URIRefs[0][similarNode], URIRefs[1][similarNode], URIRefs[2][similarNode])
                print(entity["text"])
                self.URIdict[URIRefs[0][similarNode]] = URIRefs[1][similarNode]
                self.URIdict[URIRefs[2][similarNode]] = URIRefs[1][similarNode]
                self.entities[type].append({"id": similarNode, "text": URIRefs[0][similarNode], "URI": URIRefs[1][similarNode], "original": entity["text"]})
                URIs[URIRefs[0][similarNode]] = (URIRefs[1][similarNode], URIRefs[2][similarNode])

        return URIs


    def addExtractionToOntology(self, ONTO, extraction, URIRefsObjects, URIRefsPredicates):

        '''
        POTEK:
        1. Dobimo triple
        2. Vsak element v trojici dodamo v ontologijo
        2.1. Pogledamo ce ze obstaja, ce ne pogledamo sopomenke in nadpomenke, drugace ga dodamo v ontologjo kot nov element
        2.2. Elementu dodamo povezave (sublassof, disjoint, ...) s sopomenkami, nadpomenkami in protipomenkami (tu gledamo dvoje: wordnet in rocno dodamo "not")
        3. Dodamo trojico v ontologijo (pri tem rekurzivno povezemo celo pod do nadpomenke, saj objectpropertiji niso tranzitivni, medtem ko disjointclassi so)
        4. Preverimo feedback (rocno)
        '''

        import time
        tajm = time.time()

        EXPLAIN = True

        ner_nlp = spacy.load("en_core_web_sm")
        feedback_array = []
        #over sentences
        for ex in extraction:
            #over extractions in sentence
            for t in ex:
                print(t)

                if len(t.object) == 0 or len(t.subject) == 0 or len(t.predicate) == 0:
                    continue

                #OBJ = self.preprocessExtraction(t.object)
                #SUBJ = self.preprocessExtraction(t.subject)
                #PRED = self.preprocessExtraction(t.predicate)
                OBJ = t.object
                SUBJ = t.subject
                PRED = t.predicate

                sent = ner_nlp(SUBJ + " " + PRED + " " + OBJ)
                entityTypes = [{"type": ent.label_, "word": ent.text} for ent in sent.ents]

                # Doloci POS tage objekta poglej ce je subjekt subClassOf samostalinka.
                # Če je, dodaj vse pridevnike subjektu -> zaenkrat se da kot subClassOf.
                # TODO: refactor: Če zgornje drži, pridevnike dodaj kot BURI relacijo na subjekt.. malo je ze narjeno sam je treba dodelat
                pos_tags = get_pos_tags(OBJ, simplify=True)
                print("OBJECT POS TAGS:")
                print(pos_tags)
                subclass_noun_found = False
                if len(pos_tags) > 1: # if more than one word:
                    # Find noun and adjectives
                    print("Finding nouns...")
                    noun = [p[0] for p in pos_tags if p[1] == "n"]
                    if len(noun) >= 1:
                        print("NOUNS:  (take last one)")
                        print(noun)
                        noun = noun[-1]
                        adjectives = [p[0] for p in pos_tags if p[1] == "a"]

                        CURI = self.addElementToOntology(ONTO, noun, URIRefsObjects, "SubjectObject")
                        AURI = self.addElementToOntology(ONTO, SUBJ, URIRefsObjects, "SubjectObject")

                        # Ce je AURI subClassOf CURI, potem dodaj pridevnike
                        subclasses = ONTO.transitive_subjects(rdflib.namespace.RDFS.subClassOf, CURI)
                        if AURI in subclasses:
                            subclass_noun_found = True
                            BURI = self.addElementToOntology(ONTO, PRED, URIRefsPredicates, "Predicate")
                            print("*** PRIDEVNIKI ***")
                            for adj in adjectives:
                                print("DODAJAM PRIDEVNIK: " + adj)
                                ADJURI = self.addElementToOntology(ONTO, adj, URIRefsObjects, "SubjectObject") # TODO: morda tag za pridevnik???
                                if ADJURI is not None and ADJURI != "":
                                    print("VEZEM PRIDEVNIK NA SAMSOTALNIK!!!")
                                    # Najprej dodamo legit kot je v predikatu npr. Tennis is(Type) Fast
                                    ok = self.tryAddToOntology(ONTO, AURI, BURI, ADJURI, remove=False, explain=EXPLAIN, force=True)
                                    if ok is not True:
                                        print("Relation " + str(t) + " is inconsistent with base ontology.")
                                        if EXPLAIN:
                                            print("***** Explanation begin ******")
                                            print(ok)
                                            print("***** Explanation end ******")
                                    # Tukaj dodamo kot Subclass... Tennis subClassOf FastSport, QuickSport, HardSport...
                                    AJDNOUNURI = self.addElementToOntology(ONTO, adj + ' ' + noun, URIRefsObjects, "SubjectObject")
                                    ok = self.tryAddToOntology(ONTO, AURI, rdflib.namespace.RDFS.subClassOf, AJDNOUNURI, remove=False, explain=EXPLAIN, force=True)
                                    if ok is not True:
                                        print("Relation " + str(t) + " is inconsistent with base ontology.")
                                        if EXPLAIN:
                                            print("***** Explanation begin ******")
                                            print(ok)
                                            print("***** Explanation end ******")
                                    # TODO: to je ena ideja sam pomoje ni dobra: Dodamo se equivalent class ... Sport type Fast je ekvivalent FastSport
                            print("*** KONEC PRIDEVNIKOV ***")
                        else:
                            print("Ni v SUBCLASSESOF: " + str(CURI) + " | " + str(AURI))

                if subclass_noun_found is False:
                    AURI = self.addElementToOntology(ONTO, SUBJ, URIRefsObjects, "SubjectObject")
                    CURI = self.addElementToOntology(ONTO, OBJ, URIRefsObjects, "SubjectObject")
                    BURI = self.addElementToOntology(ONTO, PRED, URIRefsPredicates, "Predicate")

                    print("***** Extracted entities: " + str(AURI) + " " + str(BURI) + " " + str(CURI) + " ***********")
                    if AURI is None or BURI is None or CURI is None:
                        print("Skipping extraction... missing element.")
                        continue
                    print("Adding extracted triple relation...")
                    # def recurse_add_remove(self, ONTO, root, rdfType, operation, subj, pred):
                    self.recurse_add_remove(ONTO, CURI, rdflib.namespace.RDFS.subClassOf, "add", AURI, BURI)
                    ok = self.tryAddToOntology(ONTO, AURI, BURI, CURI, remove=False, explain=EXPLAIN, force=True)
                    print(t)
                    print(SUBJ + PRED + OBJ)
                    print(sent)
                    print(entityTypes)
                    # TO je zato, da
                    if len(entityTypes) > 0 and entityTypes[0]["type"] != "PERSON" and BURI == rdflib.namespace.RDF.type:
                        print("SPECIAL ADD")
                        ok = self.tryAddToOntology(ONTO, AURI, rdflib.namespace.RDFS.subClassOf, CURI, remove=False, explain=EXPLAIN)

                    if ok is not True:
                        print("Relation " + str(t) + " is inconsistent with base ontology.")
                        if EXPLAIN:
                            print("***** Explanation begin ******")
                            print(ok)
                            print("***** Explanation end ******")

                '''
                ************ OLD FEEDBACK **************
                
                if not ok:
                    # iskat mormo vedno po subClassOf, ce je RDF.type ("is")
                    fBURI = BURI
                    if BURI == rdflib.namespace.RDF.type:
                        fBURI = rdflib.namespace.RDFS.subClassOf
                    feedback = self.get_feedback(ONTO, AURI, fBURI, CURI)
                    # Try to get feedback (if property TODO) from parent
                    dr = self.get_disjoint_relation(ONTO, rdflib.namespace.OWL.propertyDisjointWith, BURI)
                    #feedback = self.get_feedback_property(ONTO, AURI, dr, CURI)
                    if feedback is None:
                        for el in ONTO.objects(rdflib.namespace.RDFS.subClassOf, CURI):
                            print("FOR OUT")
                            print(str(el))
                        for el in ONTO.subjects(rdflib.namespace.RDFS.subClassOf, CURI):
                            print("FOR OUT")
                            print(str(el))
                    if feedback is None:
                        f = "Relation " + str(t) + " is inconsistent with base ontology."
                        print(f)
                    else:
                        f = "Relation " + str(t) + " is inconsistent with a relation in ontology: '" + \
                              self.uriToString(str(feedback[0])).capitalize() + " " + \
                              self.uriToString(str(feedback[1])) + " " + \
                              self.uriToString(str(feedback[2])) + "'."
                        print(f)
                    feedback_array.append(f)
                    self.recurse_add_remove(ONTO, CURI, rdflib.namespace.RDFS.subClassOf, "remove", AURI, BURI)
                    '''

        # TODO: return errors (1, 2, 1+2)
        print("------------------------------- ERROR COUNT -------------------------")
        print("----- CONSISTENCY ERRORS: " + str(self.consistencyErrors))
        print("----- SEMANTIC ERRORS: " + str(self.semanticErrors))
        print("---------------------------------------------------------------------")


        errors = [self.consistencyErrors, self.semanticErrors, self.consistencyErrors+self.semanticErrors]

        print("Elapsed time:" + str(time.time() - tajm))
        #input()

        return feedback_array, errors



    def addElementToOntology(self, ONTO, element, URIRefs, elementType):
        print("Check similar node: ", element)
        if element.split()[0] in ["a", "an"]:
            element = " ".join(element.split()[1:])
        if elementType == "Predicate":
            print(self.entities["SubjectObject"])

        # isn't => isnot    doesn't => doesnot  didn't => didnot
        element = element.replace("n't", "not")
        element = element.replace("'ll", " will")
        element = element.replace("'s", " is") # za tole nism zihr
        # workaround preprocessing
        #element = element.replace("/", "")
        #element = element.replace("\\", "")
        #if element.find("\\") > -1:
        #    print("FOUND SLASHES!!!")
        #    print(element)
        #    input()



        index, similarNode = self.similarNode([e["text"] for e in self.entities[elementType]], element, indepth=False)
        if similarNode is None:
            # to je za "fast sportS"
            index, similarNode = self.similarNode([e["text"] for e in self.entities[elementType]], element,
                                                  indepth=False, stem=True)
            # NE DELA NAJBOLJE
            #if similarNode is None:
                # ideja: v mergeessayandchunks metgamo do zadnje besede (depth=True), tako da imamo hopefully potem vse povezano ko pridemo do sem
                #index, similarNode = self.similarNode([e["original"] for e in self.entities[elementType]], element,
                #                                      indepth=False)
        # TUKAJ PRIMERJA ENTITIJE V TEM STAVKU (zaradi optimizacije?) ki jih je pridobila s shallow parsingom
        # -> DOBI ID IN POTEM TAKOJ URIRef
        URI = None
        elementURI = ""
        if similarNode is not None:
            #node = [e["URI"] for e in self.entities if e["id"] == similarNode]

            print("Found: ", self.entities[elementType][index])
            URI = self.entities[elementType][index]["URI"]
            elementURI = URI
        else:
            URI = ""
            print("None ", element)


        # START TUKAJ SE SAMO DODAJA NODE
        print("*********************DODAJANJE ENTITIJEV*****************")

        # TODO: if to sem skipal - - is there already a node with this id and URI under the ID - - check if coref and add???
        # TODO: elif # - - is there a node with same name - -
        # TODO: elif # - - is there a node with similar name - -
        #  elif synsets: sopomenke in nadpomenke

        if element == "I":
            element = "me"
        # PRVE POSKUSAJ NAJTI V URIREfs URL. Če to ne uspe, glej nadpomenke itd.
        if elementURI == "":
            if element in URIRefs[0]:
                elementURI = URIRefs[1][URIRefs[0].index(element)]
                URI = elementURI
                print("Found element in URIREFS!! " + str(elementURI))
            elif  (elementType == "SubjectObject" and len(wordnet.synsets(element, pos=wordnet.NOUN))) or (elementType == "Predicate" and len(wordnet.synsets(element, pos=wordnet.VERB))):
                print("PRVI IF - " + element)

                # HACK
                if elementType == "Predicate" and element in ["be", "is", "are"]:
                    #elementURI = COSMO[element]
                    elementURI = rdflib.namespace.RDF.type
                else:
                    addToOntology = True
                    if elementType == "SubjectObject":
                        synsets = wordnet.synsets(element, pos=wordnet.NOUN)
                    elif elementType == "Predicate":
                        synsets = wordnet.synsets(element, pos=wordnet.VERB)
                    synsetArr = []
                    # Pogledamo, če kakšna sopomenka obstaja v entitijih
                    for s in synsets:
                        name = s.lemma_names()[0]
                        name = name.replace("_", " ")
                        print(name)
                        synsetArr.append(name)
                        #if name in [e["text"] for e in self.entities]:
                        if name in URIRefs[0]:
                            #URI = self.URIdict[name]
                            elementURI = URIRefs[1][URIRefs[0].index(name)]
                            URI = URIRefs[1][URIRefs[0].index(name)]
                            addToOntology = False
                            print("Found " + name + " in entities. Not adding to ontology.")
                            break

                    # Če ni nobene so/nadpomenke v entitijih, potem dodamo v ontologijo
                    # REWRITE: ce ni nobene sopomenke, povezemo z nadpomenkami, ki jim kasneje spodaj povezeme z disjointWIth (protipomenke)
                    if addToOntology:
                        synsetArr = [self.stemSentence(s) for s in synsetArr]
                        stemmedElement = self.stemSentence(element)
                        if stemmedElement not in synsetArr:
                            stemmedElement = synsetArr[0]
                        HURI = ""
                        for h in synsets[0].hypernyms():
                            hypernim = str(h)[8:str(h).index(".")]
                            hypernim = hypernim.replace("_", " ")
                            try:
                                index = URIRefs[0].index(hypernim)
                            except:
                                index = -1
                            if index > -1:
                                HURI = URIRefs[1][index]
                                print("Found element '" + str(URIRefs[0][index]) + "' for hypernym '" + hypernim + "'.")

                                if elementType == "SubjectObject":
                                    ontologyElement = "".join([word.capitalize() for word in element.split()])
                                    owlType = rdflib.namespace.OWL.Class
                                else:
                                    ontologyElement = element.split()[0] + "".join([word.capitalize() for word in element.split()[1:]])
                                    owlType = rdflib.namespace.OWL.ObjectProperty
                                elementURI = rdflib.URIRef(self.COSMO[ontologyElement])
                                URI = elementURI
                                self.URIdict[element] = elementURI
                                self.URIdict[ontologyElement] = elementURI
                                self.entities[elementType].append({"id": elementURI, "text": element, "URI": elementURI})
                                # TODO if type == Subject/Object elif type == Predicate
                                print("Adding element (in hypernim if) '" + element + "' to ontology as '" + str(owlType) + "' '" + str(elementURI) + "'")
                                self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDF.type, owlType)
                                '''if (elementURI, rdflib.namespace.RDF.type, owlType) in ONTO:
                                    print("NOT ADDING!!!")
                                else:
                                    ONTO.add((elementURI, rdflib.namespace.RDF.type, owlType))
                                    self.hermit.check_unsatisfiable_cases(ONTO)'''

                                if HURI != "" and elementType == "SubjectObject":
                                    print("Adding HURI '" + str(HURI) + "' as suoperclass of URI: '" + str(elementURI) + "'")
                                    self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDFS.subClassOf, HURI)
                                    '''if (elementURI, rdflib.namespace.RDFS.subClassOf, HURI) in ONTO:
                                        print("NOT ADDING!!")
                                    else:
                                        ONTO.add((elementURI, rdflib.namespace.RDFS.subClassOf, HURI))
                                        if not self.hermit.check_unsatisfiable_cases(ONTO, remove=False):
                                            print("Removing...")
                                            ONTO.remove((elementURI, rdflib.namespace.RDFS.subClassOf, HURI))'''




            # ... else: Dodaj nov node
            #if URI is None: # to bo potem else
            else:
                elementURI = self.addNewNodeToOntology(ONTO, element, elementType)
                '''if elementType == "SubjectObject":
                    ontologyElement = "".join([word.capitalize() for word in element.split()])
                else:
                    ontologyElement = element.split()[0] + "".join([word.capitalize() for word in element.split()[1:]])
                elementURI = rdflib.URIRef(self.COSMO[ontologyElement])
                self.URIdict[element] = elementURI
                self.URIdict[ontologyElement] = elementURI
                self.entities[elementType].append({"id": elementURI, "text": element, "URI": elementURI})
                # TODO if type == Subject/Object elif type == Predicate
                if elementType == "SubjectObject":
                    owlType = rdflib.namespace.OWL.Class
                else:
                    owlType = rdflib.namespace.OWL.ObjectProperty
                print("Adding element '" + element + "' to ontology as '" + str(owlType) + "', '" + str(elementURI) + "'")
                self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDF.type, owlType)'''
        # POMEMBNO: V VSEH ZGORNJIH PRIMERIH DODAS NODE IN SHRANIS URIRef, saj je to le "predpriprava" - dodali smo entitiy, potem pa to primerjamo se z triple extractionom
        if elementURI is None or elementURI == '':
            elementURI = self.addNewNodeToOntology(ONTO, element, elementType)


        # STOP TUKAJ SE SAMO DODAJA NODE

        # elementURI je treba returnat

        print("*********************DODAJANJE RELATIONOV*****************")

        # 1338+
        # TODO: tukaj zdaj pogledamo wnsense   <--- ZDAJ SEM TUKAJ
        # for meaning in O.objects(AURI, COSMO.wnsense):
        use_wordnet = True
        for meaning in ONTO.objects(URI, self.COSMO.wnsense):
            use_wordnet = False
            try:
                match = re.findall(r'(\w+?)(\d+)([a-z]+)', meaning)[0]
            except:
                print("No match for meaning: " + str(meaning))
                continue
            if match[2] == 'adj':
                match = match[0] + '.' + match[2][0] + '.0' + match[1]
            else:
                match = match[0] + '.' + match[2] + '.0' + match[1]

            try:
                s = wordnet.synset(match)
            except:
                print("No matching synsets for " + str(match))
                continue

            # Povezemo nadpomenke...
            hypernyms = s.hypernyms()
            for h in hypernyms:
                print("Nadpomenke " + str(URI))
                #print(h)
                #print(h.lemma_names)
                #print(h.lemmas)
                for lemma_name in h.lemma_names():
                    print("LN: " + lemma_name)
                    lemma_name = lemma_name.replace("_", " ")
                    lemma_name_stemmed = self.stemSentence(lemma_name)
                    HURI = ""
                    try:
                        if lemma_name in URIRefs[0]:
                            index = URIRefs[0].index(lemma_name)
                        else:
                            index = URIRefs[2].index(lemma_name_stemmed)
                    except:
                        print("'" + lemma_name + "' / '" + lemma_name_stemmed + "' not in list?????")
                        index = -1
                    if index > -1:
                        HURI = URIRefs[1][index]
                    elif lemma_name in [e["text"] for e in self.entities[elementType]]:
                        HURI = [e["URI"] for e in self.entities[elementType] if e["text"] == lemma_name][0]
                    else:
                        print("NOT FOUND!!!! LEMMA: " + lemma_name)
                    print("HURI = '" + str(HURI) + "'")
                    if HURI != "":
                        if elementType == "SubjectObject":
                            rdfType = rdflib.namespace.RDFS.subClassOf
                        else:
                            rdfType = rdflib.namespace.RDFS.subPropertyOf
                        self.tryAddToOntology(ONTO, URI, rdfType, HURI)

            # Povezemo protipomenke...
            for lemma in s.lemmas():
                for antonym in lemma.antonyms():
                    print("ANTONYM: " + antonym.name())
                    lemma_name = antonym.name().replace("_", " ")
                    lemma_name_stemmed = self.stemSentence(lemma_name)
                    ANTO_URI = ""
                    try:
                        if lemma_name in URIRefs[0]:
                            index = URIRefs[0].index(lemma_name)
                        else:
                            index = URIRefs[2].index(lemma_name_stemmed)
                    except:
                        print("'" + lemma_name + "' / '" + lemma_name_stemmed + "' not in list?????")
                        index = -1
                    if index > -1:
                        ANTO_URI = URIRefs[1][index]
                    elif lemma_name in [e["text"] for e in self.entities[elementType]]:
                        ANTO_URI = [e["URI"] for e in self.entities[elementType] if e["text"] == lemma_name][0]
                    else:
                        print("NOT FOUND!!!! LEMMA: " + lemma_name)
                    print("ANTO_URI = '" + str(ANTO_URI) + "'")
                    if ANTO_URI != "":
                        rdfType = rdflib.namespace.OWL.disjointWith
                        self.tryAddToOntology(ONTO, URI, rdfType, ANTO_URI, symetric=True)

        # Zdaj pe še pogledamo če gre za Predikat z "not" -> v tem primeru najdem najbližji pozitiven predikat in povežemo
        if elementType == "Predicate" and "not " in element:
            print("**** ISCEM NEGACIJO Z NOT..... ****")
            i, similarNode = self.similarNode(URIRefs[2], element[element.index("not ")+4:], indepth=False)
            print(element[element.index("not ")+4:])
            if similarNode is not None:
                print("Without not+: " + similarNode)
                elementAntonymURI = URIRefs[1][i]
                print("Adding _not_ predicate antonym to ontology: " + str(elementURI) + " OWL.propertyDisjointWith " + str(elementAntonymURI))
                self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.OWL.propertyDisjointWith, elementAntonymURI, symetric=True)
            else:
                print("None")
            # TODO?: iz wordneta, ampak to mislim da ima ponesreci podvojeno..



        # ---> pogledamo nadpomenke in jih dodajamo/preverjamo
        # ---> pogledamo protipomenke in jih dodajamo/preverjamo (kot DisjointWith)
        # ---> poskuša tudi s korenjenimi protipomenkami

        # TODO: uporabimo wordnet (meanTF) - samo ce COSMO.wnsesne ni nasel nobenega meaninga
        # naredimo enako kot zgoraj (tiste 3 ---->)
        if use_wordnet:
            pass


        # TO JE TO!!!!! NOW GTW
        return elementURI

    def tryAddToOntology(self, ONTO, subj_URI, type, obj_URI, symetric=False, remove=True, explain=False, force=False):

        if (subj_URI, type, obj_URI) in self.allAddedTriples:
            print("Already tried adding triple! " + str(subj_URI) + " " + str(type) + " " + str(obj_URI))
            return True

        if (subj_URI, type, obj_URI) not in ONTO:
            print("elementURI: " + str(subj_URI))
            print("Adding URI '" + str(subj_URI) + "' to ontology as " + str(type) + " of '" + str(obj_URI) + "'")
            self.allAddedTriples.append((subj_URI, type, obj_URI))
            ONTO.add((subj_URI, type, obj_URI))
            if symetric: # if we want a symetric relation (e.g. disjointWith)
                ONTO.add((obj_URI, type, subj_URI))
                self.allAddedTriples.append((obj_URI, type, subj_URI))

            # If TURBO then we don't check for inconsistency, EXCEPT if it's forced -- usually when adding whole extraction
            if not self.turbo or force:
                # If explain==True, then we will return explanations IF ontology is inconsistent, otherwise True or False
                check = self.hermit.check_unsatisfiable_cases(ONTO, remove=remove, explain=explain)
                # We just check for True -> if consistent then we return True, else explanations or False
                if not isinstance(check, bool) or check is False:
                    if isinstance(check, bool):
                        # This means the ontology is not consistent (boolean was returned)
                        self.consistencyErrors += 1
                    else:
                        # A non-boolean value was returner - explanations; this means it's a semantic error
                        self.semanticErrors += 1
                    # TODO: improve this... because we have turbo mode now
                    print("Removing...")
                    print(str(subj_URI))
                    print(str(type))
                    print(str(obj_URI))
                    if not self.turbo:
                        ONTO.remove((subj_URI, type, obj_URI))
                        ONTO.remove((obj_URI, type, subj_URI))
                return check
            else:
                True
        else:
            print("Already in ONTO! " + str(subj_URI) + " " + str(type) + " " + str(obj_URI))
        return True

    def addNewNodeToOntology(self, ONTO, element, elementType):
        if elementType == "SubjectObject":
            ontologyElement = "".join([word.capitalize() for word in element.split()])
        else:
            ontologyElement = element.split()[0] + "".join([word.capitalize() for word in element.split()[1:]])
        elementURI = rdflib.URIRef(self.COSMO[ontologyElement])
        self.URIdict[element] = elementURI
        self.URIdict[ontologyElement] = elementURI
        self.entities[elementType].append({"id": elementURI, "text": element, "URI": elementURI})
        # TODO if type == Subject/Object elif type == Predicate
        if elementType == "SubjectObject":
            owlType = rdflib.namespace.OWL.Class
        else:
            owlType = rdflib.namespace.OWL.ObjectProperty
        print("Adding element '" + element + "' to ontology as '" + str(owlType) + "', '" + str(elementURI) + "'")
        self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDF.type, owlType)
        return elementURI

    def similarNode(self, nodes, newNode, indepth=True, stem=False):
        # indepth: to pomen da zacnes sekat besede iz newNode od zacetka do konca in upas da se kej ujame
        for n in range(len(nodes)):
            node1 = nodes[n]
            node2 = newNode
            if stem:
                node1 = self.stemSentence(node1)
                node2 = self.stemSentence(node2)
            if self.sentenceSimilarity(node1, node2) >= 0.7:
                return n, nodes[n]
            if indepth:
                splitEntity = node2
                while splitEntity.find(" ") > 0:
                    splitEntity = " ".join(splitEntity.split(" ")[1:])
                    if self.sentenceSimilarity(node1, splitEntity) >= 0.7:
                        return n, nodes[n]
        return None, None

    # To zato ker ObjectPropertiji niso tranizitivni
    # Lahko se jih nastavi  da so samo potem nemors disjunktnosti delat
    # Torej bomo rekurzivno sli do dna drevesa in nastavli relacijo
    # Depth = depth limit ce gre kaj narobe
    def recurse_add_remove(self, ONTO, root, rdfType, operation, subj, pred, depth=20):
        if depth <= 0:
            print("DEPTH WARNING: root: " + str(root) + ", rdfType: " + str(rdfType) + ", subj: " + str(subj) +
                  ", pred: " + str(pred))
            return
        for el in ONTO.subjects(rdfType, root):
            #print(str(el))
            if operation == "add":
                if (subj, pred, el) not in ONTO:
                    ONTO.add((subj, pred, el))
            else:
                if (subj, pred, el) in ONTO:
                    ONTO.remove((subj, pred, el))
            self.recurse_add_remove(ONTO, el, rdfType, operation, subj, pred, depth-1)

    def get_disjoint_relation(self, ONTO, relation, pred):
        drg = ONTO.subjects(relation, pred)
        disjointRelation = None
        print("Looking for disjoint relation")
        for dr in drg:
            print("Found " + str(dr))
            disjointRelation = dr
            break
        return disjointRelation

    def get_feedback(self, ONTO, subj, pred, obj):
        print("FEEDBACK")
        disjointRelation = self.get_disjoint_relation(ONTO, rdflib.namespace.OWL.propertyDisjointWith, pred)
        if disjointRelation is None:
            print("Couldn't find disjoint relation")
            # Check if obj disjointWith something: example: "Lisa is a boy. Lisa is a girl."
            print("Checking " + str(obj) + " disjointWith something...")
            for o in ONTO.objects(obj, rdflib.namespace.OWL.disjointWith):
                print("Found " + str(o) + " as disjointWith " + str(obj) + ". Returning...")
                #return (subj, pred, o)
                return (subj, "#is", o)
            return None
        else:
            parent_check = self.get_feedback_r(ONTO, subj, disjointRelation, obj, direction="parent")
            if parent_check is not None:
                return parent_check
                #return (subj, disjointRelation, obj)
            children_check = self.get_feedback_r(ONTO, subj, disjointRelation, obj, direction="children")
            return children_check
            #if children_check:
                #return (subj, disjointRelation, obj)

    def get_feedback_r(self, ONTO, subj, pred, obj, direction):
        print("Checking " + str(obj) + " --- " + direction)
        # Find first match, and then keep checking parent until no matches...
        r = None
        if (subj, pred, obj) in ONTO:
            print("FOUND!!!!")
            #return (subj, pred, obj)
            r = (subj, pred, obj)
        if direction == "parent":
            for el in ONTO.objects(obj, rdflib.namespace.RDFS.subClassOf):
                ret = self.get_feedback_r(ONTO, subj, pred, el, direction="parent")
                if ret is not None:
                    r = ret
                    #return ret
        elif direction == "children":
            for el in ONTO.subjects(rdflib.namespace.RDFS.subClassOf, obj):
                ret = self.get_feedback_r(ONTO, subj, pred, el, direction="children")
                if ret is not None:
                    r = ret
                    #return ret
        return r
        #return None

    def get_feedback_property(self, ONTO, subj, pred, obj, direction="parent"):
        print("Checking " + str(obj) + " --- " + direction + " " + str(pred) + " " + str(subj))
        # Find first match, and then keep checking parent until no matches...
        r = None
        if (subj, pred, obj) in ONTO:
            print("FOUND!!!!")
            r = (subj, pred, obj)
        if direction == "parent":
            for el in ONTO.objects(obj, rdflib.namespace.RDFS.subClassOf):
                print("FOR: " + str(el))
                ret = self.get_feedback_r(ONTO, subj, pred, el, direction="parent")
                if ret is not None:
                    return ret
                elif r is not None:
                    return r
        return None

    def uriToString(self, URI):
        s = URI[URI.index("#")+1:]
        fullString = s[0].lower()
        for c in s[1:]:
            if c.isupper():
                fullString += " "
            fullString += c
        return fullString

    def sentenceSimilarity(self, s1, s2):
        count = 0
        #print("S1: ", s1, "  ---- S2: ", s2)
        s1 = word_tokenize(s1)
        s2 = word_tokenize(s2)
        for word in s1:
            if (word in s2):
                count = count + 1
        return (count * 2 / (len(s1) + len(s2)))

    def preprocessExtraction(self, extraction):
        if extraction.startswith("T:"):
            extraction = extraction[2:]
        # Remove punctuation
        extraction = extraction.translate(extraction.maketrans("", "", string.punctuation))
        # TODO: remove determiners
        words = [i for i in extraction.split() if i not in ["a", "an", "the"]] # to je blo zakomentirano
        extraction = ' '.join(words) # to je blo zakomentirano
        # Odstrani /, ker drugace pride do napake...
        if "/" in extraction:
            print("PREPROCESS COLLAGE")
            print(extraction)
        extraction = extraction.replace("/", " ")
        # Remove prepositions
        tmp = word_tokenize(extraction)
        pos = pos_tag(tmp)
        words_pos = [i[0] for i in pos if (i[1] != 'IN' or i[0] == 'adore')]
        extraction = ' '.join(words_pos)
        # Stemming
        return self.stemSentence(extraction)

    def stemSentence(self, s):
        porter = PorterStemmer()
        extraction = [porter.stem(i) for i in s.split()]
        return ' '.join(extraction)

