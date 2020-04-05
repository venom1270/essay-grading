import spacy
from nltk import word_tokenize, pos_tag, PorterStemmer
import string
import rdflib
import re
import textacy
import copy
from orangecontrib.essaygrading.utils.HermiT import HermiT
from nltk.corpus import wordnet
from orangecontrib.essaygrading.utils.lemmatizer import get_pos_tags


class ExtractionManager:

    def __init__(self, turbo=False, i=0):
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


        '''
        turbo: add all triple entities, then add triple and check for errors (only after adding triple, so once in total)
        EXPLAIN: return HermiT explanations if true (may take a long time)
        REITERATION: applicable if turbo == true; if there is an error after adding triple, we revert changes, and turn 
                    off turbo for that one triple; we check each addition to ontology with hermit
        EXPLAIN_ON_REITERATION: on reiteration, do we want explanations or not?
    
        
        '''

        self.turbo = turbo
        self.EXPLAIN = True
        self.EXPLAIN_ON_REITERATION = True
        self.REITERATION = True

        self.consistencyErrors = 0
        self.semanticErrors = 0
        self.i = i

        self.depth_warning = False

    def getChunks(self, essay_sentences):
        '''
        This function tries to extract NP (noun prhases) and VP (verb phrases) from essay sentences.
        Those phrases are then added to self.allEntities
        :param essay_sentences:
        :return:
        '''
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
        '''
        Add NP and VP phrases to self.allEntites// self.allEntites["SubObj"/"Pred"].
        The intention is to save the original sentence/phrase, when returning feedback.
        :param essay_sentences:
        :param chunks:
        :param type:
        :return:
        '''
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
        '''
        Match gathered phrases in self.allEntities with URIRefs: we check by using similarNode function:
        it checks URIdict (dicitionary for fast lookup), else checks string character matching (>70%).
        self.allEntities is used for the last time here: expanded info is stored in self.entities, which is used
        if URIdict matching fails, before checking string character matching.
        :param URIRefs:
        :param type:
        :return:
        '''

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
                self.URIdict[URIRefs[0][similarNode]] = URIRefs[1][similarNode]  # Add "text" (original) to URIdict
                self.URIdict[URIRefs[2][similarNode]] = URIRefs[1][similarNode]  # Add stemmed version to URIdict
                self.entities[type].append({"id": similarNode, "text": URIRefs[0][similarNode], "URI": URIRefs[1][similarNode], "stemmed": URIRefs[2][similarNode], "original": entity["text"]})
                URIs[URIRefs[0][similarNode]] = (URIRefs[1][similarNode], URIRefs[2][similarNode])

        return URIs


    def addExtractionToOntology(self, ONTO, extraction, URIRefsObjects, URIRefsPredicates, explain=False):

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

        self.EXPLAIN = False
        self.EXPLAIN_ON_REITERATION = explain
        self.REITERATION = True

        ner_nlp = spacy.load("en_core_web_sm")
        feedback_array = []

        ok = None

        #over sentences
        for ex in extraction:
            #over extractions in sentence
            for t in ex:
                print(t)

                old_ONTO = copy.deepcopy(ONTO)
                old_turbo = self.turbo
                old_explanations = self.EXPLAIN
                old_added_triples = copy.deepcopy(self.allAddedTriples)
                old_URIRefsObjects = copy.deepcopy(URIRefsObjects)
                old_URIRefsPredicates = copy.deepcopy(URIRefsPredicates)
                old_entites = copy.deepcopy(self.entities)
                old_URIdict = copy.deepcopy(self.URIdict)
                repeatIteration = True

                if len(t.object) == 0 or len(t.subject) == 0 or len(t.predicate) == 0:
                    continue

                OBJ = self.preprocessExtraction(t.object, stem=True)
                SUBJ = self.preprocessExtraction(t.subject, stem=True)
                #PRED = self.preprocessExtraction(t.predicate, stem=True)
                #OBJ = t.object
                #SUBJ = t.subject
                PRED = t.predicate.replace("/", " ").replace("\\", "")

                if len(OBJ) == 0 or len(SUBJ) == 0 or len(PRED) == 0:
                    continue

                while repeatIteration:
                    repeatIteration = False

                    sent = ner_nlp(SUBJ + " " + PRED + " " + OBJ)
                    entityTypes = [{"type": ent.label_, "word": ent.text} for ent in sent.ents]

                    # Doloci POS tage objekta poglej ce je subjekt subClassOf samostalinka.
                    # Če je, dodaj vse pridevnike subjektu -> zaenkrat se da kot subClassOf.
                    # TODO: refactor: Če zgornje drži, pridevnike dodaj kot BURI relacijo na subjekt.. malo je ze narjeno sam je treba dodelat
                    pos_tags = get_pos_tags(t.object, simplify=True)
                    # t.object je prav - ce dam preprocesirano notr, zgubim informacije o pridevnikih
                    print("OBJECT POS TAGS:")
                    print(pos_tags)
                    subclass_noun_found = False
                    if len(pos_tags) > 1: # if more than one word:
                        # Find noun and adjectives
                        print("Finding nouns...")
                        noun = [p[0] for p in pos_tags if p[1] == "n"]
                        adjectives = [p[0] for p in pos_tags if p[1] == "a"]
                        print(pos_tags)
                        if len(noun) >= 1 and len(adjectives) >= 1:
                            print("NOUNS:  (take last one)")
                            print(noun)
                            noun = noun[-1]

                            # "problem", ce je (narobe) najden samo noun "a" ali "an"
                            if noun not in ["a", "an"]:
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
                                            ok1 = self.tryAddToOntology(ONTO, AURI, BURI, ADJURI, remove=False, explain=self.EXPLAIN, force=True, is_triple=True)
                                            if ok1 is not True:
                                                print("Relation " + str(t) + " is inconsistent with base ontology.")
                                                if self.EXPLAIN:
                                                    print("***** Explanation begin ******")
                                                    print(ok1)
                                                    feedback_array.append(ok1)
                                                    print("***** Explanation end ******")
                                            # Tukaj dodamo kot Subclass... Tennis subClassOf FastSport, QuickSport, HardSport...
                                            AJDNOUNURI = self.addElementToOntology(ONTO, adj + ' ' + noun, URIRefsObjects, "SubjectObject")
                                            ok2 = self.tryAddToOntology(ONTO, AURI, rdflib.namespace.RDFS.subClassOf, AJDNOUNURI, remove=False, explain=self.EXPLAIN, force=True)
                                            if ok2 is not True:
                                                print("Relation " + str(t) + " is inconsistent with base ontology.")
                                                if self.EXPLAIN:
                                                    print("***** Explanation begin ******")
                                                    print(ok2)
                                                    feedback_array.append(ok2)
                                                    print("***** Explanation end ******")
                                            # TODO: to je ena ideja sam pomoje ni dobra: Dodamo se equivalent class ... Sport type Fast je ekvivalent FastSport

                                            # Za na koncu - če je ontologija borked, pol moramo vzeti staro
                                            if ok1 and ok2:
                                                ok = True
                                            else:
                                                ok = False

                                            # To je za ponovitev iterationa s Turbo=False
                                            if (not ok1 or not ok2) and self.turbo and self.REITERATION:
                                                print("SWITCHING OFF TURBO FOR ONE ITERATION")
                                                self.turbo = False
                                                self.EXPLAIN = self.EXPLAIN_ON_REITERATION
                                                repeatIteration = True
                                                ONTO = copy.deepcopy(old_ONTO)
                                                self.allAddedTriples = copy.deepcopy(old_added_triples)
                                                URIRefsObjects = copy.deepcopy(old_URIRefsObjects)
                                                URIRefsPredicates = copy.deepcopy(old_URIRefsPredicates)
                                                self.entities = copy.deepcopy(old_entites)
                                                self.URIdict = copy.deepcopy(old_URIdict)

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
                        self.depth_warning = False
                        self.recurse_add_remove(ONTO, CURI, rdflib.namespace.RDFS.subClassOf, "add", AURI, BURI)
                        # If add/remove is stuck in infinite cycle, remove added things and skip this triple
                        if self.depth_warning:
                            self.recurse_add_remove(ONTO, CURI, rdflib.namespace.RDFS.subClassOf, "remove", AURI, BURI)
                            print("CONTINUING...")
                            ONTO = copy.deepcopy(old_ONTO)
                            self.allAddedTriples = copy.deepcopy(old_added_triples)
                            URIRefsObjects = copy.deepcopy(old_URIRefsObjects)
                            URIRefsPredicates = copy.deepcopy(old_URIRefsPredicates)
                            self.entities = copy.deepcopy(old_entites)
                            self.URIdict = copy.deepcopy(old_URIdict)
                            self.depth_warning = False
                            continue
                        ok = self.tryAddToOntology(ONTO, AURI, BURI, CURI, remove=False, explain=self.EXPLAIN, force=True, is_triple=True)
                        print(t)
                        print(SUBJ + PRED + OBJ)
                        print(sent)
                        print(entityTypes)
                        # TO je zato, da
                        if len(entityTypes) > 0 and entityTypes[0]["type"] != "PERSON" and BURI == rdflib.namespace.RDF.type:
                            print("SPECIAL ADD")
                            ok = self.tryAddToOntology(ONTO, AURI, rdflib.namespace.RDFS.subClassOf, CURI, remove=False, explain=self.EXPLAIN, force=True, is_triple=True)

                        if ok is not True:
                            print("Relation " + str(t) + " is inconsistent with base ontology.")
                            self.recurse_add_remove(ONTO, CURI, rdflib.namespace.RDFS.subClassOf, "remove", AURI, BURI)
                            if self.EXPLAIN:
                                print("***** Explanation begin ******")
                                print(ok)
                                feedback_array.append(ok)
                                print("***** Explanation end ******")
                            if self.turbo is True and self.REITERATION:
                                print("SWITCHING OFF TURBO FOR ONE ITERATION")
                                self.turbo = False
                                self.EXPLAIN = self.EXPLAIN_ON_REITERATION
                                repeatIteration = True
                                ONTO = copy.deepcopy(old_ONTO)
                                self.allAddedTriples = copy.deepcopy(old_added_triples)
                                URIRefsObjects = copy.deepcopy(old_URIRefsObjects)
                                URIRefsPredicates = copy.deepcopy(old_URIRefsPredicates)
                                self.entities = copy.deepcopy(old_entites)
                                self.URIdict = copy.deepcopy(old_URIdict)


                if not self.turbo and old_turbo is True and self.REITERATION:
                    print("SWTCHING ON TURBO... RESUMING NORMAL OPERATIONS")
                    self.turbo = True
                    self.EXPLAIN = old_explanations

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
                #input()  # PER TRIPLE

        # TODO: return errors (1, 2, 1+2)
        print("------------------------------- ERROR COUNT -------------------------")
        print("----- CONSISTENCY ERRORS: " + str(self.consistencyErrors))
        print("----- SEMANTIC ERRORS: " + str(self.semanticErrors))
        print("---------------------------------------------------------------------")

        if ok is not True:
            print("### FINISHED BUT OLD ONTOLOGY IS INCONSISTENT... WRITING OLD ONTOLOGY ###")
            self.hermit.check_unsatisfiable_cases(old_ONTO, remove=False, explain=explain, i=self.i)
        else:
            print("### FINISHED, ONTOLOGY CONSISTENT ###")

        errors = [self.consistencyErrors, self.semanticErrors, self.consistencyErrors+self.semanticErrors]

        print("Elapsed time:" + str(time.time() - tajm))
        #input()  # PER ESSAY

        return feedback_array, errors


    def checkSimilarNode(self, ONTO, element, URIRefs, elementType):
        # Check if similar node exists (nodes that were processed before starting this step)

        # Najprej preveri, ce je podobna stvar v URIRefs... ce ni, spodaj gledamo self.entites (to so entiteta,
        # ki se prej pomatachajo glede na NP in VP v eseju s 70%(?) ujemanjem

        if element in self.URIdict:
            print("URIDICT WORKS!!!")
            return self.URIdict[element], self.URIdict[element]

        index, similarNode = self.similarNode(URIRefs[0], element, indepth=False, stem=False)
        if similarNode is None:
            index, similarNode = self.similarNode(URIRefs[0], element, indepth=False, stem=True)
        '''if element == "like":
            print(URIRefs)
            print(element)
            print(index)
            print(similarNode)
            exit()'''
        if similarNode is not None:
            URI = URIRefs[1][index]
            return URI, URI

        # Pogledamo tudi stemmane matche (self.entities)
        stemmedEntites = [e["stemmed"] for e in self.entities[elementType]]
        index, similarNode = self.similarNode(stemmedEntites, element, indepth=False, stem=False)
        if similarNode is not None:
            URI = self.entities[elementType][index]["URI"]
            return URI, URI

        index, similarNode = self.similarNode([e["text"] for e in self.entities[elementType]], element, indepth=False)
        if similarNode is None:
            # to je za "fast sportS"
            # Check again, but this time WITH STEMMING
            index, similarNode = self.similarNode([e["text"] for e in self.entities[elementType]], element,
                                                  indepth=False, stem=True)
            ''' NE DELA NAJBOLJE
            #if similarNode is None:
                # ideja: v mergeessayandchunks mergamo do zadnje besede (depth=True), tako da imamo hopefully potem vse 
                # povezano ko pridemo do sem
                #index, similarNode = self.similarNode([e["original"] for e in self.entities[elementType]], element,
                #                                      indepth=False)
        # TUKAJ PRIMERJA ENTITIJE V TEM STAVKU (zaradi optimizacije?) ki jih je pridobila s shallow parsingom '''
        # -> DOBI ID IN POTEM TAKOJ URIRef

        URI = None
        elementURI = ""
        if similarNode is not None:
            # node = [e["URI"] for e in self.entities if e["id"] == similarNode]

            # If we found a similar node, we use it and skip other steps
            print("Found: ", self.entities[elementType][index])
            URI = self.entities[elementType][index]["URI"]
            elementURI = URI
        else:
            # If similar node not found, we continue checking synonyms and hypernyms
            URI = ""
            print("None ", element)

        return elementURI, URI


    def checkNodeSynonymsHypernyms(self, ONTO, element, URIRefs, elementType, URI):
        elementURI = ""

        print("PRVI IF - " + element)

        # If 'be' predicate, we already have a type for it
        if elementType == "Predicate" and element in ["be", "is", "are"]:
            # elementURI = COSMO[element]
            elementURI = rdflib.namespace.RDF.type
        # Else add synonyms to ontology
        else:
            addToOntology = True
            if elementType == "SubjectObject":
                synsets = wordnet.synsets(element, pos=wordnet.NOUN)
            elif elementType == "Predicate":
                synsets = wordnet.synsets(element, pos=wordnet.VERB)
            synsetArr = []
            # Find all synonyms; if any of them are already in ontology, use that and skip adding new ones
            for s in synsets:
                if len(s.lemma_names()) == 0:
                    continue
                name = s.lemma_names()[0]
                name = name.replace("_", " ")
                print(name)
                synsetArr.append(name)
                # if name in [e["text"] for e in self.entities]:
                if name in URIRefs[0]:
                    # URI = self.URIdict[name]
                    elementURI = URIRefs[1][URIRefs[0].index(name)]
                    URI = URIRefs[1][URIRefs[0].index(name)]
                    addToOntology = False
                    print("Found " + name + " in entities. Not adding to ontology.")
                    break

            # Če ni nobene so/nadpomenke v entitijih, potem dodamo v ontologijo
            # REWRITE: ce ni nobene sopomenke, povezemo z nadpomenkami, ki jim kasneje spodaj povezeme z disjointWIth (protipomenke)
            # If no synonym was found in ontology, check their hypernyms and try to use that
            if addToOntology:
                synsetArr = [self.stemSentence(s) for s in synsetArr]
                stemmedElement = self.stemSentence(element)
                if stemmedElement not in synsetArr and len(synsetArr) > 0:
                    stemmedElement = synsetArr[0]
                HURI = ""
                for h in synsets[0].hypernyms():
                    hypernim = str(h)[8:str(h).index(".")]
                    hypernim = hypernim.replace("_", " ")
                    # Find out if hypernym exists in our ontology
                    try:
                        index = URIRefs[0].index(hypernim)
                    except:
                        index = -1
                    if index > -1:
                        # If it does, we add the element and the connection to the hypernym (subclassOf)
                        HURI = URIRefs[1][index]
                        print("Found element '" + str(URIRefs[0][index]) + "' for hypernym '" + hypernim + "'.")

                        if elementType == "SubjectObject":
                            ontologyElement = "".join([word.capitalize() for word in element.split()])
                            owlType = rdflib.namespace.OWL.Class
                        else:
                            ontologyElement = element.split()[0] + "".join(
                                [word.capitalize() for word in element.split()[1:]])
                            owlType = rdflib.namespace.OWL.ObjectProperty
                        elementURI = rdflib.URIRef(self.COSMO[ontologyElement])
                        URI = elementURI
                        self.URIdict[element] = elementURI
                        self.URIdict[ontologyElement] = elementURI
                        self.entities[elementType].append({"id": elementURI, "text": element, "URI": elementURI, "stemmed": element}) # TODO?: "stemmed" is not really stemmed here
                        print("Adding element (in hypernim if) '" + element + "' to ontology as '" + str(
                            owlType) + "' '" + str(elementURI) + "'")
                        self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDF.type, owlType)
                        if HURI != "" and elementType == "SubjectObject":
                            print("Adding HURI '" + str(HURI) + "' as suoperclass of URI: '" + str(elementURI) + "'")
                            self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDFS.subClassOf, HURI)
        return elementURI, URI

    def addRelationHypernymsAntonyms(self, ONTO, element, URIRefs, elementType, URI):
        use_wordnet = True
        for meaning in ONTO.objects(URI, self.COSMO.wnsense):
            use_wordnet = False
            # We find approptiate synsets
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

            # 1. Add relations to hypernyms
            hypernyms = s.hypernyms()
            for h in hypernyms:
                print("Nadpomenke " + str(URI))
                # print(h)
                # print(h.lemma_names)
                # print(h.lemmas)
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
                    # If hypernym was found in ontology, we add subClassOf or subPropertyOf relation
                    if HURI != "":
                        if elementType == "SubjectObject":
                            rdfType = rdflib.namespace.RDFS.subClassOf
                        else:
                            rdfType = rdflib.namespace.RDFS.subPropertyOf
                        self.tryAddToOntology(ONTO, URI, rdfType, HURI)

            # 2. Add relations to antonyms
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

    def addElementToOntology(self, ONTO, element, URIRefs, elementType):
        print("Check similar node: ", element)
        if element.split()[0] in ["a", "an"]:
            element = " ".join(element.split()[1:])
        #if elementType == "Predicate": ZAKVA JE TO SPLOH???
            #print(self.entities["SubjectObject"])

        # isn't => isnot    doesn't => doesnot  didn't => didnot
        element = element.replace("n't", "not")
        element = element.replace("'ll", " will")
        element = element.replace("'s", " is")  # za tole nism zihr
        # workaround preprocessing
        # element = element.replace("/", "")
        # element = element.replace("\\", "")
        # if element.find("\\") > -1:
        #    print("FOUND SLASHES!!!")
        #    print(element)
        #    input()

        elementURI, URI = self.checkSimilarNode(ONTO, element, URIRefs, elementType)


        # START TUKAJ SE SAMO DODAJA NODE
        print("*********************DODAJANJE ENTITIJEV*****************")

        # TODO: if to sem skipal - - is there already a node with this id and URI under the ID - - check if coref and add???
        # TODO: elif # - - is there a node with same name - -
        # TODO: elif # - - is there a node with similar name - -
        #  elif synsets: sopomenke in nadpomenke

        '''
        This part is just for adding the current element in the ontology. Later, we add relations.
        STEPS:
        1. Check if element in URIRefs
        2. Else check if has synonyms
        3. Else check if has hypernyms
        4. Add element
        '''

        if element == "I":
            element = "me"
        # PRVE POSKUSAJ NAJTI V URIREfs URL. Če to ne uspe, glej nadpomenke itd.
        # If element not found above, go through this procedure
        if elementURI == "":
            # If element in URIRefs, use that
            if element in URIRefs[0]:
                elementURI = URIRefs[1][URIRefs[0].index(element)]
                URI = elementURI
                print("Found element in URIREFS!! " + str(elementURI))
            # Else check if element has any synonyms/hypernyms
            elif (elementType == "SubjectObject" and len(wordnet.synsets(element, pos=wordnet.NOUN))) or (
                    elementType == "Predicate" and len(wordnet.synsets(element, pos=wordnet.VERB))):
                elementURI, URI = self.checkNodeSynonymsHypernyms(ONTO, element, URIRefs, elementType, URI)
            # ... else: Dodaj nov node
            # If no synonyms/hypernyms then add to ontology as new independent node
            else:
                elementURI = self.addNewNodeToOntology(ONTO, element, elementType)
        # EDIT: nevem kaj je to... POMEMBNO: V VSEH ZGORNJIH PRIMERIH DODAS NODE IN SHRANIS URIRef, saj je to le
        # "predpriprava" - dodali smo entitiy, potem pa to primerjamo se z triple extractionom
        if elementURI is None or elementURI == '':
            elementURI = self.addNewNodeToOntology(ONTO, element, elementType)



        print("*********************DODAJANJE RELATIONOV*****************")

        # 1338+
        '''
        Here we add relations (disjointWith etc.)
        STEPS:
        1. Add relations to hypernyms
        2. Add relations to antonyms
        3. Check if it's a 'not' predicate and add appropriate disjointWith relations
        '''

        # 1. and 2.
        self.addRelationHypernymsAntonyms(ONTO, element, URIRefs, elementType, URI)

        # 3. Check if it's a 'not' predicate and add appropriate disjointWith relations
        # If 'not', we find the closest positive predicate and add disjointWith
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


        # TODO: uporabimo wordnet (meanTF) - samo ce COSMO.wnsesne ni nasel nobenega meaninga
        # naredimo enako kot zgoraj (tiste 3 ---->)
        # EDIT: kaj je zdej s tem... tega pomoje ne rabim
        #if use_wordnet:
        #    pass

        # Return the new added element
        return elementURI

    def tryAddToOntology(self, ONTO, subj_URI, type, obj_URI, symetric=False, remove=True, explain=False, force=False, is_triple=False):

        if (subj_URI, type, obj_URI) in self.allAddedTriples:
            print("Already tried adding triple! " + str(subj_URI) + " " + str(type) + " " + str(obj_URI))
            # return True # TODO zdej je pocasnejs ampak bolj ziher: problem ce dvakrat dodajamo skor isto in se ujame
            #                       tukaj, vbistvu so se pa zaradi turbota vmes dodajali razlicni classi
            if is_triple and self.turbo:
                return self.hermit.check_unsatisfiable_cases(ONTO, remove=remove, explain=explain, i=self.i)
            else:
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
                check = self.hermit.check_unsatisfiable_cases(ONTO, remove=remove, explain=explain, i=self.i)
                # We just check for True -> if consistent then we return True, else explanations or False
                if not isinstance(check, bool) or check is False:
                    # There was an error - if we are adding a triple (is_triple), we count it as semantic error, else it's a consistency error
                    '''if isinstance(check, bool):
                        # This means the ontology is not consistent (boolean was returned)
                        if self.turbo and self.REITERATION:
                            pass # If we are in turbo mode with reiteration, then don't count error - it will be counterd in reiteration
                        else:
                            self.consistencyErrors += 1
                    else:
                        # A non-boolean value was returned - explanations; this means it's a semantic error EDIT TODO: not entirely true...
                        if self.turbo and self.REITERATION:
                            pass # If we are in turbo mode with reiteration, then don't count error - it will be counterd in reiteration
                        else:
                            self.semanticErrors += 1'''
                    if not is_triple:
                        # Not adding a triple; consistency error
                        if self.turbo and self.REITERATION:
                            pass  # If we are in turbo mode with reiteration, then don't count error - it will be counterd in reiteration
                        else:
                            self.consistencyErrors += 1
                    else:
                        # Adding a triple; semantic error
                        if self.turbo and self.REITERATION:
                            pass  # If we are in turbo mode with reiteration, then don't count error - it will be counted in reiteration
                        else:
                            self.semanticErrors += 1
                    # TODO: improve this... because we have turbo mode now -EDIT- it's OK now i think?
                    print("Removing...")
                    print(str(subj_URI))
                    print(str(type))
                    print(str(obj_URI))
                    #if not self.turbo:
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
        self.entities[elementType].append({"id": elementURI, "text": element, "URI": elementURI, "stemmed": element}) # TODO?: "stemmed" is no really stemmed here
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
    def recurse_add_remove(self, ONTO, root, rdfType, operation, subj, pred, depth=200):
        if depth <= 0:
            print("DEPTH WARNING: root: " + str(root) + ", rdfType: " + str(rdfType) + ", subj: " + str(subj) +
                  ", pred: " + str(pred))
            self.depth_warning = True
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
            if self.depth_warning:
                break

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

    def preprocessExtraction(self, extraction, stem=True):
        if extraction.startswith("T:"):
            extraction = extraction[2:]
        # Make sure to properly remove apostophe TODO: check if this is OK
        extraction = extraction.replace("n't", "not")
        extraction = extraction.replace("'ll", " will")
        extraction = extraction.replace("'s", " is")  # za tole nism zihr
        # Remove punctuation
        extraction = extraction.translate(extraction.maketrans("", "", string.punctuation))
        # TODO: remove determiners
        words = [i for i in extraction.split() if i not in ["a", "an", "the"]]  # to je blo zakomentirano
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
        # Stemming - don't stem when processing extraction with ontology
        if stem:
            return self.stemSentence(extraction)
        else:
            return extraction

    def stemSentence(self, s):
        porter = PorterStemmer()
        extraction = [porter.stem(i) for i in s.split()]
        return ' '.join(extraction)

