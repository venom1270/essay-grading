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
from orangecontrib.essaygrading.utils.URIRefEntity import EntityStore, URIRefEntity


class ExtractionManager:

    def __init__(self, turbo=False, i=0):
        '''
        Init method. Initilizes all impotant internal variables.
        :param turbo: boolean, enable "turbo" mode: check consistency only after adding whole triple.
        :param i: int, id of this instance, used for logging during multiprocessing.
        '''
        self.nlp = spacy.load("en_core_web_lg")
        self.hermit = HermiT()
        self.COSMO = rdflib.Namespace("http://micra.com/COSMO/COSMO.owl#")
        self.allAddedTriples = []
        self.entityStore = EntityStore()


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

        self.enable_debug_print = True

        self.depth_warning = False

    def debugPrint(self, *args, **kwargs):
        '''
        Logging method. Same behavior as "print()", but prints id in format "[ID] " before content (*args).
        '''
        if self.enable_debug_print:
            print("[" + str(self.i) + "] ", end="", flush=True)
            print(*args, **kwargs, flush=True)

    def getChunks(self, essay_sentences, URIRefs):
        '''
        This function tries to extract NP (noun prhases) and VP (verb phrases) from essay sentences.
        These are then matched with URIRefs. In case of a match, a URIRefEntity gets added to EntityStore.
        :param essay_sentences: array of essay sentences.
        :return: list of stored entites in EntityStore.
        '''
        noun_chunks = []
        verb_chunks = []
        for sentence in essay_sentences:
            doc = self.nlp(sentence)
            noun_chunks.append([chunk.text for chunk in doc.noun_chunks])

            lists = textacy.extract.pos_regex_matches(doc, r'<VERB>?<ADV>*<VERB>+')
            # pos_regex_matches(doc, r'<VERB>?<ADV>*<VERB>+')
            verb_chunks.append([vp.text for vp in lists])

        self.debugPrint(noun_chunks)
        self.debugPrint(verb_chunks)

        URISubObj = URIRefs["SubObj"]
        URIPred = URIRefs["Pred"]

        for sentence_chunk in noun_chunks:  # Over sentences
            for chunk in sentence_chunk:  # Over chunks in sentence
                entityText = self.preprocessExtraction(chunk)
                similarNodeIndex, _ = self.similarNode(URISubObj[2], entityText, indepth=False) # Originalno je biu indepth=True
                if similarNodeIndex is not None:
                    self.debugPrint("SIMILAR")
                    self.debugPrint(URISubObj[0][similarNodeIndex], URISubObj[1][similarNodeIndex], URISubObj[2][similarNodeIndex])
                    self.debugPrint(entityText)
                    self.entityStore.add(text=entityText, stemmed=self.stem(entityText), URIRef=URISubObj[1][similarNodeIndex], type="SubjectObject", original=chunk)
                else:
                    pass
                    # TODO: se tukaj kej?

        for sentence_chunk in verb_chunks:  # Over sentences
            for chunk in sentence_chunk:  # Over chunks in sentence
                entityText = self.preprocessExtraction(chunk)
                similarNodeIndex, _ = self.similarNode(URIPred[2], entityText, indepth=False)
                if similarNodeIndex is not None:
                    self.debugPrint("SIMILAR")
                    self.debugPrint(URIPred[0][similarNodeIndex], URIPred[1][similarNodeIndex], URIPred[2][similarNodeIndex])
                    self.debugPrint(entityText)
                    self.entityStore.add(text=entityText, stemmed=self.stem(entityText), URIRef=URIPred[1][similarNodeIndex], type="Predicate", original=chunk)
                else:
                    pass
                    # TODO: se tukaj kej?

        # Add URIRefs to entityStore if not present yet
        # TODO: TEST THIS, may not work
        for i in range(len(URISubObj[0])):
            if self.entityStore.find(URIRef=URISubObj[1][i], type="SubjectObject") is None:
                self.entityStore.add(text=URISubObj[0][i], stemmed=URISubObj[2][i], URIRef=URISubObj[1][i],
                                     type="SubjectObject", original=None)
        for i in range(len(URIPred[0])):
            if self.entityStore.find(URIRef=URIPred[1][i], type="Predicate") is None:
                self.entityStore.add(text=URIPred[0][i], stemmed=URIPred[2][i], URIRef=URIPred[1][i],
                                     type="Predicate", original=None)

        self.entityStore.snapshot()
        return self.entityStore.get_list()


    def addExtractionToOntology(self, ONTO, extraction, essay, URIRefsObjects, URIRefsPredicates, explain=False, source_text=None):
        '''
        Adds OpenIE extractions to ontology and checks for semantic consistency. Core of this class.
        Basic explanations are always returned (which sentences are "clashing").
        Detailed explanations (explain flag) include information about opposite relations and other inconsistencies.

        POTEK:
        1. Dobimo triple
        2. Vsak element v trojici dodamo v ontologijo
        2.1. Pogledamo ce ze obstaja, ce ne pogledamo sopomenke in nadpomenke, drugace ga dodamo v ontologjo kot nov element
        2.2. Elementu dodamo povezave (sublassof, disjoint, ...) s sopomenkami, nadpomenkami in protipomenkami (tu gledamo dvoje: wordnet in rocno dodamo "not")
        3. Dodamo trojico v ontologijo (pri tem rekurzivno povezemo celo pod do nadpomenke, saj objectpropertiji niso tranzitivni, medtem ko disjointclassi so)
        4. Preverimo feedback (rocno)

        :param ONTO: rdflib Graph() object, represents ontology.
        :param extraction: list of extractions (Triple() objects).
        :param essay: list of essay sentences before extraction, used for more readable feedback explanations.
        :param URIRefsObjects: rdflib URIRefs objects from ontology (ONTO) which represent OpenIE subjects and objects.
        :param URIRefsPredicates: rdflib URIRefs objects from ontology (ONTO) which represent OpenIE predicates.
        :param explain: boolen, return detailed explantaions if True, otherwise only basic explanations.
        :param removeComments: boolen, removes ALL comments from ontology before adding any triples. Used for basic feedback.
        :return: list in following format: [basic_feedback, [consistency errors, semantic errors, sum], detailed feedback]
        '''

        import time
        tajm = time.time()

        self.EXPLAIN = False
        self.EXPLAIN_ON_REITERATION = explain
        self.REITERATION = True

        ner_nlp = spacy.load("en_core_web_sm")
        feedback_array = []
        feedback_array_custom = []

        ok = None

        #ONTO.remove((None, rdflib.namespace.RDFS.comment, None))

        #over sentences
        for ex in extraction:
            #over extractions in sentence
            for t in ex:
                self.debugPrint(t)

                self.currentId = t.index

                old_ONTO = copy.deepcopy(ONTO)
                old_turbo = self.turbo
                old_explanations = self.EXPLAIN
                old_added_triples = copy.deepcopy(self.allAddedTriples)
                old_URIRefsObjects = copy.deepcopy(URIRefsObjects)
                old_URIRefsPredicates = copy.deepcopy(URIRefsPredicates)
                repeatIteration = True

                if len(t.object) == 0 or len(t.subject) == 0 or len(t.predicate) == 0:
                    continue

                OBJ = self.preprocessExtraction(t.object, stem=True)
                SUBJ = self.preprocessExtraction(t.subject, stem=True)
                PRED = t.predicate.replace("/", " ").replace("\\", "").replace("-", " ")

                if len(OBJ) == 0 or len(SUBJ) == 0 or len(PRED) == 0 or OBJ == " " or PRED == " " or SUBJ == " ":
                    continue

                while repeatIteration:
                    repeatIteration = False

                    sent = ner_nlp(SUBJ + " " + PRED + " " + OBJ)
                    entityTypes = [{"type": ent.label_, "word": ent.text} for ent in sent.ents]

                    # Doloci POS tage objekta poglej ce je subjekt subClassOf samostalnika.
                    # Če je, dodaj vse pridevnike subjektu -> zaenkrat se da kot subClassOf.
                    # TODO: refactor: Če zgornje drži, pridevnike dodaj kot BURI relacijo na subjekt.. malo je ze narjeno sam je treba dodelat
                    pos_tags = get_pos_tags(t.object.replace("\\", ""), simplify=True)
                    # t.object je prav - ce dam preprocesirano notr, zgubim informacije o pridevnikih
                    self.debugPrint("OBJECT POS TAGS:")
                    self.debugPrint(pos_tags)
                    subclass_noun_found = False
                    if len(pos_tags) > 1: # if more than one word:
                        # Find noun and adjectives
                        self.debugPrint("Finding nouns...")
                        noun = [p[0] for p in pos_tags if p[1] == "n"]
                        adjectives = [p[0] for p in pos_tags if p[1] == "a"]
                        self.debugPrint(pos_tags)
                        if len(noun) >= 1 and len(adjectives) >= 1:
                            self.debugPrint("NOUNS:  (take last one)")
                            self.debugPrint(noun)
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
                                    self.debugPrint("*** PRIDEVNIKI ***")
                                    for adj in adjectives:
                                        self.debugPrint("DODAJAM PRIDEVNIK: " + adj)
                                        ADJURI = self.addElementToOntology(ONTO, adj, URIRefsObjects, "SubjectObject") # TODO: morda tag za pridevnik???
                                        if ADJURI is not None and ADJURI != "":
                                            self.debugPrint("VEZEM PRIDEVNIK NA SAMSOTALNIK!!!")
                                            # Najprej dodamo legit kot je v predikatu npr. Tennis is(Type) Fast
                                            ok1 = self.tryAddToOntology(ONTO, AURI, BURI, ADJURI, remove=False, explain=self.EXPLAIN, force=True, is_triple=True)
                                            if ok1 is not True:
                                                self.debugPrint("Relation " + str(t) + " is inconsistent with base ontology.")
                                                if self.EXPLAIN:
                                                    self.debugPrint("***** Explanation begin ******")
                                                    self.debugPrint(ok1)
                                                    # feedback_array.append(ok1)
                                                    feedback_array += ok1
                                                    self.debugPrint("***** Explanation end ******")
                                            # Tukaj dodamo kot Subclass... Tennis subClassOf FastSport, QuickSport, HardSport...
                                            AJDNOUNURI = self.addElementToOntology(ONTO, adj + ' ' + noun, URIRefsObjects, "SubjectObject")
                                            ok2 = self.tryAddToOntology(ONTO, AURI, rdflib.namespace.RDFS.subClassOf, AJDNOUNURI, remove=False, explain=self.EXPLAIN, force=True)
                                            if ok2 is not True:
                                                self.debugPrint("Relation " + str(t) + " is inconsistent with base ontology.")
                                                if self.EXPLAIN:
                                                    self.debugPrint("***** Explanation begin ******")
                                                    self.debugPrint(ok2)
                                                    # feedback_array.append(ok2)
                                                    feedback_array += ok2
                                                    self.debugPrint("***** Explanation end ******")
                                            # TODO: to je ena ideja sam pomoje ni dobra: Dodamo se equivalent class ... Sport type Fast je ekvivalent FastSport

                                            # Za na koncu - če je ontologija borked, pol moramo vzeti staro
                                            if ok1 and ok2:
                                                ok = True
                                            else:
                                                ok = False

                                            # To je za ponovitev iterationa s Turbo=False
                                            if (not ok1 or not ok2) and self.turbo and self.REITERATION:
                                                self.debugPrint("SWITCHING OFF TURBO FOR ONE ITERATION - ADJECTIVES")
                                                self.turbo = False
                                                self.EXPLAIN = self.EXPLAIN_ON_REITERATION
                                                repeatIteration = True
                                                ONTO = copy.deepcopy(old_ONTO)
                                                self.allAddedTriples = copy.deepcopy(old_added_triples)
                                                URIRefsObjects = copy.deepcopy(old_URIRefsObjects)
                                                URIRefsPredicates = copy.deepcopy(old_URIRefsPredicates)
                                                self.entityStore.restore_snapshot()

                                                self.debugPrint("*** KONEC PRIDEVNIKOV ***")
                                else:
                                    self.debugPrint("Ni v SUBCLASSESOF: " + str(CURI) + " | " + str(AURI))

                    if subclass_noun_found is False:
                        AURI = self.addElementToOntology(ONTO, SUBJ, URIRefsObjects, "SubjectObject")
                        CURI = self.addElementToOntology(ONTO, OBJ, URIRefsObjects, "SubjectObject")
                        BURI = self.addElementToOntology(ONTO, PRED, URIRefsPredicates, "Predicate")

                        self.debugPrint("***** Extracted entities: " + str(AURI) + " " + str(BURI) + " " + str(CURI) + " ***********")
                        if AURI is None or BURI is None or CURI is None:
                            self.debugPrint("Skipping extraction... missing element.")
                            continue
                            self.debugPrint("Adding extracted triple relation...")
                        # def recurse_add_remove(self, ONTO, root, rdfType, operation, subj, pred):
                        self.depth_warning = False
                        self.recurse_add_remove(ONTO, CURI, rdflib.namespace.RDFS.subClassOf, "add", AURI, BURI)
                        # If add/remove is stuck in infinite cycle, remove added things and skip this triple
                        if self.depth_warning:
                            self.recurse_add_remove(ONTO, CURI, rdflib.namespace.RDFS.subClassOf, "remove", AURI, BURI)
                            self.debugPrint("CONTINUING...")
                            ONTO = copy.deepcopy(old_ONTO)
                            self.allAddedTriples = copy.deepcopy(old_added_triples)
                            URIRefsObjects = copy.deepcopy(old_URIRefsObjects)
                            URIRefsPredicates = copy.deepcopy(old_URIRefsPredicates)
                            self.entityStore.restore_snapshot()
                            self.depth_warning = False
                            # continue
                        ok = self.tryAddToOntology(ONTO, AURI, BURI, CURI, remove=False, explain=self.EXPLAIN, force=True, is_triple=True)
                        self.debugPrint(t)
                        self.debugPrint(SUBJ + PRED + OBJ)
                        self.debugPrint(sent)
                        self.debugPrint(entityTypes)
                        # TO je zato, da
                        if len(entityTypes) > 0 and entityTypes[0]["type"] != "PERSON" and BURI == rdflib.namespace.RDF.type:
                            self.debugPrint("SPECIAL ADD")
                            ok = self.tryAddToOntology(ONTO, AURI, rdflib.namespace.RDFS.subClassOf, CURI, remove=False, explain=self.EXPLAIN, force=True, is_triple=True)

                        # Turbo check - cuz find explanations only when we are in a reiteration (turbo=off) - searching for error.
                        if not self.turbo and ok is not True:
                            self.debugPrint("IN CUSTOM FEEDBACK")
                            # iskat mormo vedno po subClassOf, ce je RDF.type ("is")
                            fBURI = BURI
                            if BURI == rdflib.namespace.RDF.type:
                                fBURI = rdflib.namespace.RDFS.subClassOf
                            feedback = self.get_feedback(ONTO, AURI, fBURI, CURI)
                            if feedback is None:
                                f = "Relation '" + str(t.subject) + " " + str(t.predicate) + " " + str(t.object) + \
                                    "' is inconsistent with a relation in ontology."
                            else:
                                # Get original triple text index from ontology RDFS.comment
                                index = self.get_feedback_essay_index(ONTO, feedback[0], feedback[1], feedback[2])
                                feedback_string = ""
                                if index is None:
                                    self.debugPrint("Feeback index is None")
                                    feedback_string = self.uriToString(str(feedback[0])).capitalize() + " " + \
                                                    self.uriToString(str(feedback[1])) + " " + \
                                                    self.uriToString(str(feedback[2])) + "'."
                                else:
                                    self.debugPrint("Feedback index valid!")
                                    # If index > 0: essay; else (index < 0) source text inconsistency
                                    if source_text is None or index >= 1:
                                        feedback_string = essay[index-1] + "'. "
                                    else:
                                        feedback_string = source_text[(-index)-1] + "' (from source text). "
                                #f = "Relation '" + str(t.subject) + " " + str(t.predicate) + " " + str(t.object) + \
                                f = "Relation '" + essay[int(t.index)-1] + \
                                    "' is inconsistent with a relation in ontology: '" + feedback_string

                            feedback_array_custom.append([f])


                        if ok is not True:
                            self.debugPrint("Relation " + str(t) + " is inconsistent with base ontology.")
                            self.recurse_add_remove(ONTO, CURI, rdflib.namespace.RDFS.subClassOf, "remove", AURI, BURI)
                            if self.EXPLAIN:
                                self.debugPrint("***** Explanation begin ******")
                                self.debugPrint(ok)
                                feedback_array += ok
                                self.debugPrint("***** Explanation end ******")
                            if self.turbo is True and self.REITERATION:
                                self.debugPrint("SWITCHING OFF TURBO FOR ONE ITERATION - NORMAL")
                                self.turbo = False
                                self.EXPLAIN = self.EXPLAIN_ON_REITERATION
                                repeatIteration = True
                                ONTO = copy.deepcopy(old_ONTO)
                                self.allAddedTriples = copy.deepcopy(old_added_triples)
                                URIRefsObjects = copy.deepcopy(old_URIRefsObjects)
                                URIRefsPredicates = copy.deepcopy(old_URIRefsPredicates)
                                self.entityStore.restore_snapshot()



                if not self.turbo and old_turbo is True and self.REITERATION:
                    self.debugPrint("SWTCHING ON TURBO... RESUMING NORMAL OPERATIONS")
                    self.turbo = True
                    self.EXPLAIN = old_explanations

                self.debugPrint("------------------------------- ERROR COUNT -------------------------")
                self.debugPrint("----- CONSISTENCY ERRORS: " + str(self.consistencyErrors))
                self.debugPrint("----- SEMANTIC ERRORS: " + str(self.semanticErrors))
                self.debugPrint("---------------------------------------------------------------------")

        if ok is not True:
            self.debugPrint("### FINISHED BUT OLD ONTOLOGY IS INCONSISTENT... WRITING OLD ONTOLOGY ###")
            self.hermit.check_unsatisfiable_cases(old_ONTO, remove=False, explain=explain, i=self.i)
        else:
            self.debugPrint("### FINISHED, ONTOLOGY CONSISTENT ###")

        errors = [self.consistencyErrors, self.semanticErrors, self.consistencyErrors+self.semanticErrors]

        self.debugPrint("Elapsed time:" + str(time.time() - tajm))

        return feedback_array, errors, feedback_array_custom


    def checkSimilarNode(self, ONTO, element, URIRefs, elementType):
        '''
        Checks if similar node exists. It checks in the following order:
        - EntityStore 100% match: text and stemmed
        - similarNode() in URIRefs: text and stemmed
        - EntityStore similarNode() match: text and stemmed
        :param ONTO: rdflib Graph() object, out ontology.
        :param element: string element we are searching for.
        :param URIRefs: list of URIRefs in base ontology.
        :param elementType: string, "SubjectObject" or "Predicate".
        :return: element URIRef if exists, otherwise empty string.
        '''
        # Check if similar node exists (nodes that were processed before starting this step)


        # Check for 100% match in entityStore - text and stemmed
        el = self.entityStore.find(text=element, type=elementType)
        if el is None:
            el = self.entityStore.find(stemmed=element, type=elementType)
        if el is not None:
            self.debugPrint("Found: ", el.text, str(el.URIRef))
            return el.URIRef, el.URIRef

        # Check in entitystore similarnode
        lookup = self.entityStore.get_similar_node(element)
        if lookup is not None:
            return lookup, lookup

        # Check for similarMatch in URIRefs TODO: test this; uncomment if not working
        '''index, similarNode = self.similarNode(URIRefs[0], element, indepth=False, stem=False)
        if similarNode is None:
            index, similarNode = self.similarNode(URIRefs[0], element, indepth=False, stem=True)
        if similarNode is not None:
            self.debugPrint("Found: ", URIRefs[0][index], str(URIRefs[1][index]))
            URI = URIRefs[1][index]
            return URI, URI'''

        # Check for similarMatch text
        entityList = self.entityStore.get_list("text", type=elementType)
        index, similarNode = self.similarNode(entityList, element, indepth=False, stem=False)
        if similarNode is not None:
            el = self.entityStore.get_by_index(index, type=elementType)
            self.entityStore.add_similar_node(element, el.URIRef) # This is just for lookup so we don't have to run similarNode with same element again
            self.debugPrint("Found: ", el.text, str(el.URIRef))
            return el.URIRef, el.URIRef

        # Check for similarMatch stemmed
        entityList = self.entityStore.get_list("stemmed", type=elementType)
        index, similarNode = self.similarNode(entityList, element, indepth=False)
        stm = False
        if similarNode is None:
            # to je za "fast sportS"
            # Check again, but this time WITH STEMMING --- STEM ALL
            #index, similarNode = self.similarNode([e["text"] for e in self.entities[elementType]], element,
                                                  #indepth=False, stem=True)
            index, similarNode = self.similarNode(entityList, element, indepth=False, stem=True)
            s = True

        URI = None
        elementURI = ""
        if similarNode is not None:
            el = self.entityStore.get_by_index(index, type=elementType)
            if stm:
                self.entityStore.add_similar_node(self.stem(element), el.URIRef)  # This is just for lookup so we don't have to run similarNode with same element again
            else:
                self.entityStore.add_similar_node(element, el.URIRef)  # This is just for lookup so we don't have to run similarNode with same element again
            self.debugPrint("Found: ", el.text, str(el.URIRef))
            URI = el.URIRef
            elementURI = URI
        else:
            # If similar node not found, we continue checking synonyms and hypernyms
            URI = ""
            self.debugPrint("None ", element)

        return elementURI, URI


    def checkNodeSynonymsHypernyms(self, ONTO, element, URIRefs, elementType):
        '''
        Check for element synonyms and hypernyms. Add match to ontology and return.
        If synonym found, return synonym. Else check hypernyms.
        If any found and is type SubjectObject, add element and relation to the hypernym, return element.
        Else return empty string.
        :param ONTO: rdflib Graph() object which represents our ontology.
        :param element: string element we are checking for.
        :param URIRefs: list of URIRefs in base ontology.
        :param elementType: "SubjectObject" or "Predicate".
        :return: element URIRef.
        '''
        elementURI = ""
        URI = ""

        self.debugPrint("PRVI IF - " + element)

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
            # TODO: we can add a new one and use that??? COUNTER: but then we lose all connections a hypernym or other synonym may bring.
            for s in synsets:
                if len(s.lemma_names()) == 0:
                    continue
                name = s.lemma_names()[0].replace("_", " ")
                self.debugPrint(name)
                synsetArr.append(name)
                # if name in [e["text"] for e in self.entities]:
                # TODO: test this
                '''if name in URIRefs[0]:
                    elementURI = URIRefs[1][URIRefs[0].index(name)]
                    URI = URIRefs[1][URIRefs[0].index(name)]
                    addToOntology = False
                    self.debugPrint("Found " + name + " in URIRefs. Not adding to ontology.")
                    break'''
                entity = self.entityStore.find(text=name, type=elementType)
                if entity is not None:
                    elementURI = entity.URIRef
                    URI = elementURI
                    addToOntology = False
                    self.debugPrint("Found " + name + " in entityStore (synsets). Not adding to ontology.")
                    break

            # Če ni nobene so/nadpomenke v entitijih, potem dodamo v ontologijo
            # REWRITE: ce ni nobene sopomenke, povezemo z nadpomenkami, ki jim kasneje spodaj povezeme z disjointWIth (protipomenke)
            # If no synonym was found in ontology, check their hypernyms and try to use that
            # TODO: a nebi najprej dodal morebitne sinonime, sele nato pa sel na hypernyme?
            if addToOntology:
                synsetArr = [self.stemSentence(s) for s in synsetArr]
                stemmedElement = self.stemSentence(element)
                if stemmedElement not in synsetArr and len(synsetArr) > 0:
                    stemmedElement = synsetArr[0]
                HURI = ""
                for h in synsets[0].hypernyms():
                    hypernim = str(h)[8:str(h).index(".")].replace("_", " ")
                    # Find out if hypernym exists in our ontology
                    '''try:
                        index = URIRefs[0].index(hypernim)
                        HURI = URIRefs[1][index]
                        self.debugPrint(
                            "Found element '" + str(URIRefs[0][index]) + "' for hypernym '" + hypernim + "'.")
                    except:
                        index = -1
                        # Try searching entityStore
                        if self.entityStore.find(text=hypernim, type=elementType) is not None:
                            HURI = self.entityStore.find(text=hypernim, type=elementType).URIRef
                            self.debugPrint(
                                "Found element '" + str(HURI) + "' for hypernym '" + hypernim + "' (entitystore).")'''
                    # Try searching entityStore
                    if self.entityStore.find(text=hypernim, type=elementType) is not None:
                        HURI = self.entityStore.find(text=hypernim, type=elementType).URIRef
                        self.debugPrint(
                            "Found element '" + str(HURI) + "' for hypernym '" + hypernim + "' (entitystore).")
                    ## if index > -1:
                    if HURI != "":
                        # If it does, we add the element and the connection to the hypernym (subclassOf)

                        # self.debugPrint("Found element '" + str(URIRefs[0][index]) + "' for hypernym '" + hypernim + "'.")

                        if elementType == "SubjectObject":
                            ontologyElement = "".join([word.capitalize() for word in element.split()])
                            owlType = rdflib.namespace.OWL.Class
                        else:
                            ontologyElement = element.split()[0] + "".join(
                                [word.capitalize() for word in element.split()[1:]])
                            owlType = rdflib.namespace.OWL.ObjectProperty
                        elementURI = rdflib.URIRef(self.COSMO[ontologyElement])
                        URI = elementURI
                        self.entityStore.add(elementURI, element, None, elementType, original=element) # Tukaj dodamo element, dap otem dodamo se relacijo nadpomenke...
                        self.debugPrint("Adding element (in hypernim if) '" + element + "' to ontology as '" + str(
                            owlType) + "' '" + str(elementURI) + "'")
                        self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDF.type, owlType)
                        if HURI != "" and elementType == "SubjectObject":
                            self.debugPrint("Adding HURI '" + str(HURI) + "' as suoperclass of URI: '" + str(elementURI) + "'")
                            self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDFS.subClassOf, HURI)
        return elementURI, URI

    def addRelationHypernymsAntonyms(self, ONTO, element, URIRefs, elementType, URI):
        '''
        Add subClassOf (hypernym) and disjointWith (antonym) relations to ontology for element 'element'.
        It looks through element's synsets and adds relation if synset element in ontology.
        :param ONTO: rdflib Graph() object which represents our ontology.
        :param element: string element we are checking for.
        :param URIRefs: list of URIRefs in base ontology.
        :param elementType: "SubjectObject" or "Predicate".
        :param URI: URIRef of element.
        '''


        use_wordnet = True
        for meaning in ONTO.objects(URI, self.COSMO.wnsense):
            use_wordnet = False
            # We find approptiate synsets
            try:
                match = re.findall(r'(\w+?)(\d+)([a-z]+)', meaning)[0]
            except:
                self.debugPrint("No match for meaning: " + str(meaning))
                continue
            if match[2] == 'adj':
                match = match[0] + '.' + match[2][0] + '.0' + match[1]
            else:
                match = match[0] + '.' + match[2] + '.0' + match[1]

            try:
                s = wordnet.synset(match)
            except:
                self.debugPrint("No matching synsets for " + str(match))
                continue

            # 1. Add relations to hypernyms
            hypernyms = s.hypernyms()
            for h in hypernyms:
                self.debugPrint("Nadpomenke " + str(URI))
                # print(h)
                # print(h.lemma_names)
                # print(h.lemmas)
                for lemma_name in h.lemma_names():
                    self.debugPrint("LN: " + lemma_name)
                    lemma_name = lemma_name.replace("_", " ")
                    lemma_name_stemmed = self.stemSentence(lemma_name)
                    HURI = ""
                    '''try:
                        if lemma_name in URIRefs[0]:
                            index = URIRefs[0].index(lemma_name)
                        else:
                            index = URIRefs[2].index(lemma_name_stemmed)
                    except:
                        self.debugPrint("'" + lemma_name + "' / '" + lemma_name_stemmed + "' not in list?????")
                        index = -1'''
                    index = -1 ## TODO: test
                    if index > -1:
                        HURI = URIRefs[1][index]
                    elif lemma_name in self.entityStore.get_list("text", elementType):
                        # HURI = [e["URI"] for e in self.entities[elementType] if e["text"] == lemma_name][0]
                        HURI = self.entityStore.find(text=lemma_name, type=elementType).URIRef
                    else:
                        self.debugPrint("NOT FOUND!!!! LEMMA: " + lemma_name)
                        self.debugPrint("HURI = '" + str(HURI) + "'")
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
                    self.debugPrint("ANTONYM: " + antonym.name())
                    lemma_name = antonym.name().replace("_", " ")
                    lemma_name_stemmed = self.stemSentence(lemma_name)
                    ANTO_URI = ""
                    '''try:
                        if lemma_name in URIRefs[0]:
                            index = URIRefs[0].index(lemma_name)
                        else:
                            index = URIRefs[2].index(lemma_name_stemmed)
                    except:
                        self.debugPrint("'" + lemma_name + "' / '" + lemma_name_stemmed + "' not in list?????")
                        index = -1'''
                    index = -1 ## TODO: TEST
                    if index > -1:
                        ANTO_URI = URIRefs[1][index]
                    elif lemma_name in self.entityStore.get_list("text", elementType):
                        ANTO_URI = self.entityStore.find(text=lemma_name, type=elementType).URIRef
                    else:
                        self.debugPrint("NOT FOUND!!!! LEMMA: " + lemma_name)
                    self.debugPrint("ANTO_URI = '" + str(ANTO_URI) + "'")
                    if ANTO_URI != "":
                        rdfType = rdflib.namespace.OWL.disjointWith
                        self.tryAddToOntology(ONTO, URI, rdfType, ANTO_URI, symetric=True)

    def addElementToOntology(self, ONTO, element, URIRefs, elementType):
        '''
        Add element to ontology, callin other relevant methods for synonyms, hypernyms, relations... in the process.
        :param ONTO: rdflib Graph() object which represents our ontology.
        :param element: string element we are checking for.
        :param URIRefs: list of URIRefs in base ontology.
        :param elementType: "SubjectObject" or "Predicate".
        :return: element URIRef.
        '''


        self.debugPrint("Check similar node: ", element)
        if element.split()[0] in ["a", "an"]:
            element = " ".join(element.split()[1:])
        #if elementType == "Predicate": ZAKVA JE TO SPLOH???
            #print(self.entities["SubjectObject"])

        # isn't => isnot    doesn't => doesnot  didn't => didnot
        element = element.replace("n't", "not")
        element = element.replace("'ll", " will")
        element = element.replace("'s", " is")  # za tole nism zihr

        elementURI, URI = self.checkSimilarNode(ONTO, element, URIRefs, elementType)


        # START TUKAJ SE SAMO DODAJA NODE
        self.debugPrint("*********************DODAJANJE ENTITIJEV*****************")
        '''
        This part is just for adding the current element in the ontology. Later, we add relations.
        STEPS:
        1. Check if element in URIRefs
        2. Else check if has synonyms IN ONTOLOGY. If yes, use synonym.
        3. Else check if has hypernyms IN ONTOLOGY. If yes, add as new node with subClassOf relation to hypernym.
        4. Else element to ontology AS AN INDIVIDUAL ELEMENT.
        
        We then add relations to connect it to eventual synonyms, hypernyms and antonyms.
        '''

        if element == "I":
            element = "me"
        # PRVE POSKUSAJ NAJTI V URIREfs URL. Če to ne uspe, glej nadpomenke itd.
        # If element not found above, go through this procedure
        if elementURI == "":
            # If element in URIRefs, use that TODO low priority: to se najbrz naredi ze v checkSImilarNode!
            '''if element in URIRefs[0]:
                elementURI = URIRefs[1][URIRefs[0].index(element)]
                URI = elementURI
                self.debugPrint("Found element in URIREFS!! " + str(elementURI))
            # Else check if element has any (if they exist) synonyms/hypernyms'''
            if (elementType == "SubjectObject" and len(wordnet.synsets(element, pos=wordnet.NOUN))) or (
                    elementType == "Predicate" and len(wordnet.synsets(element, pos=wordnet.VERB))):
                elementURI, URI = self.checkNodeSynonymsHypernyms(ONTO, element, URIRefs, elementType)
            # If no synonyms/hypernyms then add to ontology as new independent node
            else:
                elementURI = self.addNewNodeToOntology(ONTO, element, elementType)
        # EDIT: nevem kaj je to... POMEMBNO: V VSEH ZGORNJIH PRIMERIH DODAS NODE IN SHRANIS URIRef, saj je to le
        # "predpriprava" - dodali smo entitiy, potem pa to primerjamo se z triple extractionom
        if elementURI is None or elementURI == '':
            elementURI = self.addNewNodeToOntology(ONTO, element, elementType)

        self.debugPrint("*********************DODAJANJE RELATIONOV*****************")

        '''
        Now we have added a node. It might have already been added (in entitystore/URIRefs), 
        we might have found it as a synonym, or added a new node with hypernym (subClassOf) relation.
        Worst case we just added it because it didn't match any of the above criteria.
        Now we add relations (disjointWith etc.)
        STEPS:
        1. Add relations to hypernyms --- find all elementURI hypernyms and add subClassOf relation
        2. Add relations to antonyms --- find all elementURI antonyms and add disjointWith relation
        3. Check if it's a 'not' predicate and add appropriate disjointWith relations
        '''

        # 1. and 2.
        self.addRelationHypernymsAntonyms(ONTO, element, URIRefs, elementType, URI)

        # 3. Check if it's a 'not' predicate and add appropriate disjointWith relations
        # If 'not', we find the closest positive predicate and add disjointWith
        if elementType == "Predicate" and "not " in element:
            self.debugPrint("**** ISCEM NEGACIJO Z NOT..... ****")
            i, similarNode = self.similarNode(URIRefs[2], element[element.index("not ")+4:], indepth=False)
            # TODO: also check entityStore for "not" relation?
            self.debugPrint(element[element.index("not ")+4:])
            if similarNode is not None:
                self.debugPrint("Without not+: " + similarNode)
                elementAntonymURI = URIRefs[1][i]
                self.debugPrint("Adding _not_ predicate antonym to ontology: " + str(elementURI) + " OWL.propertyDisjointWith " + str(elementAntonymURI))
                self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.OWL.propertyDisjointWith, elementAntonymURI, symetric=True)
            else:
                self.debugPrint("None")
            # TODO?: iz wordneta, ampak to mislim da ima ponesreci podvojeno..

        # Return the new added element
        return elementURI

    def tryAddToOntology(self, ONTO, subj_URI, type, obj_URI, symetric=False, remove=True, explain=False, force=False, is_triple=False):
        '''
        Try adding triple (subject, predicate, object) to ontology. If any consistency errors arise triple is not added.
        :param ONTO: rdflib Graph() object representing our ontology.
        :param subj_URI: URIRef of subject.
        :param type: URIRef of predicate.
        :param obj_URI: URIRef of object.
        :param symetric: flag if we want to add symetric relation e.g. (subj, pred, obj) AND (obj, pred, subj).
        :param remove: flag to remove temp ontology file upon completion. Set to false if you want to take a look afterwards.
        :param explain: flag to return detailed explanations.
        :param force: flag to force adding triple, regardless of internal state.
        :param is_triple: flag True if we want to add a triple, False if want to only check ontology consistency.
        :return: True if successful, False or list of explanations if unsuccessful.
        '''

        if (subj_URI, type, obj_URI) in self.allAddedTriples:
            self.debugPrint("Already tried adding triple! " + str(subj_URI) + " " + str(type) + " " + str(obj_URI))
            # return True # TODO zdej je pocasnejs ampak bolj ziher: problem ce dvakrat dodajamo skor isto in se ujame
            #                       tukaj, vbistvu so se pa zaradi turbota vmes dodajali razlicni classi.. EDIT: wut?
            if is_triple and self.turbo:
                return self.hermit.check_unsatisfiable_cases(ONTO, remove=remove, explain=explain, i=self.i)
            else:
                return True

        if str.lower(subj_URI) == str.lower(obj_URI):
            self.debugPrint("ADD PREVENTED: reflexive realtion.")
            return True

        if (subj_URI, type, obj_URI) not in ONTO:
            self.debugPrint("elementURI: " + str(subj_URI))
            self.debugPrint("Adding URI '" + str(subj_URI) + "' to ontology as " + str(type) + " of '" + str(obj_URI) + "'")
            self.allAddedTriples.append((subj_URI, type, obj_URI))
            ONTO.add((subj_URI, type, obj_URI))
            # ONTO.remove((subj_URI, rdflib.namespace.RDFS.comment, None))  # Remove any default comments;; ce jt ole odkomentirano pol tud Lisa ne najde v FINAL COMMENT TESTU TODO
            # For feedback: we add index to each element: we then get all elements and return index that matches all of them
            # Negative number in comment indicates triple from source text
            if self.i > 0:
                commentId = str(self.currentId)
            else:
                commentId = str(-int(self.currentId))
            ONTO.add((subj_URI, rdflib.namespace.RDFS.comment, rdflib.Literal(commentId)))
            ONTO.add((obj_URI, rdflib.namespace.RDFS.comment, rdflib.Literal(commentId)))
            ONTO.add((type, rdflib.namespace.RDFS.comment, rdflib.Literal(commentId)))
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
                    self.debugPrint("Removing...")
                    self.debugPrint(str(subj_URI))
                    self.debugPrint(str(type))
                    self.debugPrint(str(obj_URI))
                    # TODO: tole bi sicer moral removeat, ampak PO FEEDBACK CHECKU!!!; zaenkrat mam sam zakomentiran pa dela...
                    # Mogoce pa nerabim tega removat... itak gledam use 3 clene, tko da ce pri enmu sfali ni tok panike najbrz?
                    # ONTO.remove((subj_URI, rdflib.namespace.RDFS.comment, rdflib.Literal(str(self.currentId))))
                    # ONTO.remove((obj_URI, rdflib.namespace.RDFS.comment, rdflib.Literal(str(self.currentId))))
                    # ONTO.remove((type, rdflib.namespace.RDFS.comment, rdflib.Literal(str(self.currentId))))
                    ONTO.remove((subj_URI, type, obj_URI))
                    ONTO.remove((obj_URI, type, subj_URI))
                return check
            else:
                True
        else:
            self.debugPrint("Already in ONTO! " + str(subj_URI) + " " + str(type) + " " + str(obj_URI))
        return True

    def addNewNodeToOntology(self, ONTO, element, elementType):
        '''
        Adds a new node to ontology. Usually called if no element, synonyms and hypernyms found.
        :param ONTO: rdflib Graph() object representing our ontology.
        :param element: string element.
        :param elementType: "SubjectObject" or "Predicate".
        :return: URIRef of added element.
        '''
        if elementType == "SubjectObject":
            ontologyElement = "".join([word.capitalize() for word in element.split()])
        else:
            ontologyElement = element.split()[0] + "".join([word.capitalize() for word in element.split()[1:]])
        elementURI = rdflib.URIRef(self.COSMO[ontologyElement])
        self.entityStore.add(elementURI, element, None, elementType, original=element)
        if elementType == "SubjectObject":
            owlType = rdflib.namespace.OWL.Class
        else:
            owlType = rdflib.namespace.OWL.ObjectProperty
        self.debugPrint("Adding element '" + element + "' to ontology as '" + str(owlType) + "', '" + str(elementURI) + "'")
        self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDF.type, owlType)
        return elementURI

    def similarNode(self, nodes, newNode, indepth=True, stem=False):
        '''
        Check if newNode is similar to and element in nodes. It's based on letter matching with a threshold of 70%.
        :param nodes: list of elements to compare to (string).
        :param newNode: string element to compare with.
        :param indepth: flag to cut sentence into words; higher match probability but prone to error.
        :param stem: flag to stem every element before comparison.
        :return: index, matching node string or None, None
        '''

        # indepth: to pomen da zacnes sekat besede iz newNode od zacetka do konca in upas da se kej ujame
        for n in range(len(nodes)):
            node1 = nodes[n]
            node2 = newNode
            if stem:
                node1 = self.stemSentence(node1)
                node2 = self.stemSentence(node2)
            if self.sentenceSimilarity(node1, node2) >= 0.7:
                # Add to dictionary lookup # TODO: to sem imel na laptopu za speedup, mogoce tud kle se splaca to met...
                '''
                if URIRefs is not None:
                    self.URIdict[newNode] = URIRefs[n]
                    if stem:
                        self.URIdict[node2] = URIRefs[n]
                '''
                return n, nodes[n]
            if indepth:
                splitEntity = node2
                while splitEntity.find(" ") > 0:
                    splitEntity = " ".join(splitEntity.split(" ")[1:])
                    if self.sentenceSimilarity(node1, splitEntity) >= 0.7:
                        # Add to dictionary lookup
                        '''
                        if URIRefs is not None:
                            self.URIdict[newNode] = URIRefs[n]
                            if stem:
                                self.URIdict[node2] = URIRefs[n]
                        '''
                        return n, nodes[n]
        return None, None

    # To zato ker ObjectPropertiji niso tranizitivni
    # Lahko se jih nastavi  da so samo potem nemors disjunktnosti delat
    # Torej bomo rekurzivno sli do dna drevesa in nastavli relacijo
    # Depth = depth limit ce gre kaj narobe
    def recurse_add_remove(self, ONTO, root, rdfType, operation, subj, pred, depth=200):
        '''
        Because OWL.ObjectProperty-s are not transitive, we add those relations to every element in the children chain.
        Also supports removing in the opposite fashion.
        :param ONTO: rdflib Graph() object.
        :param root: current element.
        :param rdfType: "SubjectObject" or "Predicate".
        :param operation: "add" or "remove".
        :param subj: subject URIRef.
        :param pred: predicate URIRef.
        :param depth: maximum recursion depth.
        '''
        if depth <= 0:
            self.debugPrint("DEPTH WARNING: root: " + str(root) + ", rdfType: " + str(rdfType) + ", subj: " + str(subj) +
                  ", pred: " + str(pred))
            self.depth_warning = True
            return
        for el in ONTO.subjects(rdfType, root):
            #self.debugPrint(str(el))
            if operation == "add":
                if (subj, pred, el) not in ONTO:
                    ONTO.add((subj, pred, el))
            else:
                if (subj, pred, el) in ONTO:
                    ONTO.remove((subj, pred, el))
            self.recurse_add_remove(ONTO, el, rdfType, operation, subj, pred, depth-1)
            if self.depth_warning:
                break

    def get_disjoint_relation(self, ONTO, relation, object):
        '''
        Finds subjects of disjoint relation. Parameters are disjoint relation (e.g. if we have "not like" relation,
        we pass "like" to this method to return all subjects with this relation on an object. Select first one and
        return.
        :param ONTO: rdflib Graph() object.
        :param relation: URIREF of relation.
        :param pred: URIREF of predicate.
        :return: found disjoint relation URIRef or None.
        '''
        drg = ONTO.subjects(relation, object)
        disjointRelation = None
        self.debugPrint("Looking for disjoint relation")
        for dr in drg:
            self.debugPrint("Found " + str(dr))
            disjointRelation = dr
            break
        return disjointRelation

    def get_feedback(self, ONTO, subj, pred, obj):
        '''
        Recursively get basic feedback.
        :param ONTO: rdflib Graph() object.
        :param subj: subject URIRef.
        :param pred: predicate URIRef.
        :param obj: object URIRef.
        :return: triple (subject, predicate, object) that is inconsistent with passed triple or None if none found.
        '''
        self.debugPrint("FEEDBACK")
        disjointRelation = self.get_disjoint_relation(ONTO, rdflib.namespace.OWL.propertyDisjointWith, pred)
        if disjointRelation is None:
            self.debugPrint("Couldn't find disjoint relation")
            # Check if obj disjointWith something: example: "Lisa is a boy. Lisa is a girl."
            for o in ONTO.objects(obj, rdflib.namespace.OWL.disjointWith):
                self.debugPrint("Found " + str(o) + " as disjointWith " + str(obj) + ". Returning...")
                #return (subj, pred, o)
                #return (subj, "#is", o)
                return (subj, pred, o)
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
        '''
        Helper recursion method for get_feedback()
        '''
        self.debugPrint("Checking " + str(obj) + " --- " + direction)
        # Find first match, and then keep checking parent until no matches...
        r = None
        if (subj, pred, obj) in ONTO:
            self.debugPrint("FOUND!!!!")
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
        '''
        Because of non-transitev ObjectProperty objects, we add  relations to every child element.
        So we have to check from inconsistent element to the uppermost parent with the same relation to get accurate
        feedback relation.
        :param ONTO: rdflib Graph() object.
        :param subj: subject URIRef.
        :param pred: predicate URIRef.
        :param obj: object URIRef.
        :param direction: recursion directuon ("parent").
        :return: triple with uppermost parent relation or None.
        '''
        self.debugPrint("Checking " + str(obj) + " --- " + direction + " " + str(pred) + " " + str(subj))
        # Find first match, and then keep checking parent until no matches...
        r = None
        if (subj, pred, obj) in ONTO:
            self.debugPrint("FOUND!!!!")
            r = (subj, pred, obj)
        if direction == "parent":
            for el in ONTO.objects(obj, rdflib.namespace.RDFS.subClassOf):
                ret = self.get_feedback_r(ONTO, subj, pred, el, direction="parent")
                if ret is not None:
                    return ret
                elif r is not None:
                    return r
        return None

    def get_feedback_essay_index(self, ONTO, subj, pred, obj):
        '''
        To provide accurate basic feedback sentences (as they were written originally in the essay),
        we find the relation index and return it.
        When adding element to ontology, a index representing essay sentence index is also added to ontology as
        RDFS.comment. This method checks all RDFS.comments of all three elements of the triple and returns the common
        one as there may be more sentences which use the same element. If there is more than one intersection,
        return the on with the lowest value, as that is the triple that was added the earliest.
        :param ONTO: rdflib Graph() object.
        :param subj: subject URIRef.
        :param pred: predicate URIRef.
        :param obj: object URIRef.
        :return: essay sentence index.
        '''
        self.debugPrint("Feedback index for " + str(subj) + " " + str(pred) + " " + str(obj))
        index_subj = ONTO.triples((subj, rdflib.namespace.RDFS.comment, None))
        index_pred = ONTO.triples((pred, rdflib.namespace.RDFS.comment, None))
        index_obj = ONTO.triples((obj, rdflib.namespace.RDFS.comment, None))
        if index_subj is not None and index_pred is not None and index_obj is not None:
            try:
                i1 = set([int(x[2]) for x in index_subj])
                i2 = set([int(x[2]) for x in index_pred])
                i3 = set([int(x[2]) for x in index_obj])
                self.debugPrint(i1)
                self.debugPrint(i2)
                self.debugPrint(i3)
                # If predicate is a predefined class type, ignore it, as it's set is empty
                if pred == rdflib.namespace.RDF.type:
                    intersection = i1.intersection(i3)
                else:
                    intersection = i1.intersection(i2).intersection(i3)
                self.debugPrint("Intersection length is " + str(len(intersection)))
                minIntersection = None
                for x in intersection:
                    self.debugPrint(x)
                    if minIntersection is None or x < minIntersection:
                        minIntersection = x
                return minIntersection
            except:
                self.debugPrint("Error finding index.")

        return None


    def uriToString(self, URI):
        '''
        Convert URIRef to element string, striping URI from the string.
        :param URI: URIRef.
        :return: URIRef in element string form.
        '''
        s = URI[URI.index("#")+1:]
        fullString = s[0].lower()
        for c in s[1:]:
            if c.isupper():
                fullString += " "
            fullString += c
        return fullString

    def sentenceSimilarity(self, s1, s2):
        '''
        Calculates similarity between two sentences by comparing word by word.
        :param s1: sentence 1.
        :param s2: sentence 2.
        :return: similarity rate.
        '''
        count = 0
        #self.debugPrint("S1: ", s1, "  ---- S2: ", s2)
        s1 = word_tokenize(s1)
        s2 = word_tokenize(s2)
        for word in s1:
            if (word in s2):
                count = count + 1
        return (count * 2 / (len(s1) + len(s2)))

    def preprocessExtraction(self, extraction, stem=True):
        '''
        Preprecessing method for extractions (triple elements).
        :param extraction: string element.
        :param stem: flag if stem.
        :return: preprocessed extraction string.
        '''
        if extraction.startswith("T:"):
            extraction = extraction[2:]
        # Remove slashes
        extraction = extraction.replace("/", " ").replace("\\", "")
        # Make sure to properly remove apostophe TODO: check if this is OK
        extraction = extraction.replace("n't", "not")
        extraction = extraction.replace("'ll", " will")
        extraction = extraction.replace("'s", " is")  # za tole nism zihr
        # Remove punctuation
        extraction = extraction.translate(extraction.maketrans("", "", string.punctuation))
        # TODO: remove determiners
        words = [i for i in extraction.split() if i not in ["a", "an", "the"]]  # to je blo zakomentirano
        extraction = ' '.join(words) # to je blo zakomentirano
        '''# Odstrani /, ker drugace pride do napake...
        if "/" in extraction:
            self.debugPrint("PREPROCESS COLLAGE")
            self.debugPrint(extraction)
        extraction = extraction.replace("/", " ")'''
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
        '''
        Helper method for stemming sentences.
        :param s: sentence string.
        :return: stemmed sentence.
        '''
        porter = PorterStemmer()
        extraction = [porter.stem(i) for i in s.split()]
        return ' '.join(extraction)

    def stem(self, s):
        '''
        Helper method for stemming.
        :param s: strign to stem.
        :return: stemmed string.
        '''
        return PorterStemmer().stem(s)

