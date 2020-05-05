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
        if self.enable_debug_print:
            print("[" + str(self.i) + "] ", end="", flush=True)
            print(*args, **kwargs, flush=True)

    def getChunks(self, essay_sentences, URIRefs):
        '''
        This function tries to extract NP (noun prhases) and VP (verb phrases) from essay sentences.
        Those phrases are then added to self.allEntities in the next method
        :param essay_sentences:
        :return:
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
                similarNodeIndex, _ = self.similarNode(URISubObj[2], entityText, indepth=True)
                if similarNodeIndex is not None:
                    self.debugPrint("SIMILAR")
                    self.debugPrint(URISubObj[0][similarNodeIndex], URISubObj[1][similarNodeIndex], URISubObj[2][similarNodeIndex])
                    self.debugPrint(entityText)
                    self.entityStore.add(text=entityText, stemmed=None, URIRef=URISubObj[1][similarNodeIndex], type="SubjectObject", original=chunk)
                else:
                    pass
                    # TODO: se tuakaj kej?

        for sentence_chunk in verb_chunks:  # Over sentences
            for chunk in sentence_chunk:  # Over chunks in sentence
                entityText = self.preprocessExtraction(chunk)
                similarNodeIndex, _ = self.similarNode(URIPred[2], entityText, indepth=True)
                if similarNodeIndex is not None:
                    self.debugPrint("SIMILAR")
                    self.debugPrint(URIPred[0][similarNodeIndex], URIPred[1][similarNodeIndex], URIPred[2][similarNodeIndex])
                    self.debugPrint(entityText)
                    self.entityStore.add(text=entityText, stemmed=None, URIRef=URIPred[1][similarNodeIndex], type="Predicate", original=chunk)
                else:
                    pass
                    # TODO: se tuakaj kej?

        self.entityStore.snapshot()
        return self.entityStore.get_list()


    def addExtractionToOntology(self, ONTO, extraction, essay, URIRefsObjects, URIRefsPredicates, explain=False):

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
        feedback_array_custom = []

        ok = None

        ONTO.remove((None, rdflib.namespace.RDFS.comment, None))

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
                                                    feedback_array.append(ok1)
                                                    self.debugPrint("***** Explanation end ******")
                                            # Tukaj dodamo kot Subclass... Tennis subClassOf FastSport, QuickSport, HardSport...
                                            AJDNOUNURI = self.addElementToOntology(ONTO, adj + ' ' + noun, URIRefsObjects, "SubjectObject")
                                            ok2 = self.tryAddToOntology(ONTO, AURI, rdflib.namespace.RDFS.subClassOf, AJDNOUNURI, remove=False, explain=self.EXPLAIN, force=True)
                                            if ok2 is not True:
                                                self.debugPrint("Relation " + str(t) + " is inconsistent with base ontology.")
                                                if self.EXPLAIN:
                                                    self.debugPrint("***** Explanation begin ******")
                                                    self.debugPrint(ok2)
                                                    feedback_array.append(ok2)
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
                            continue
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
                                    feedback_string = essay[index-1] + "'"
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
                                feedback_array.append(ok)
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
        # Check if similar node exists (nodes that were processed before starting this step)


        # Check for 100% match - text and stemmed
        el = self.entityStore.find(text=element, type=elementType)
        if el is None:
            el = self.entityStore.find(stemmed=element, type=elementType)
        if el is not None:
            self.debugPrint("Found: ", el.text, str(el.URIRef))
            return el.URIRef, el.URIRef

        # Check for similarMatch in URIRefs
        index, similarNode = self.similarNode(URIRefs[0], element, indepth=False, stem=False)
        if similarNode is None:
            index, similarNode = self.similarNode(URIRefs[0], element, indepth=False, stem=True)
        if similarNode is not None:
            self.debugPrint("Found: ", URIRefs[0][index], str(URIRefs[1][index]))
            URI = URIRefs[1][index]
            return URI, URI

        # Check for similarMatch stemmed
        stemmedEntities = self.entityStore.get_list("stemmed", type=elementType)
        index, similarNode = self.similarNode(stemmedEntities, element, indepth=False, stem=False)
        if similarNode is not None:
            el = stemmedEntities[index]
            self.debugPrint("Found: ", el.text, str(el.URIRef))
            return el.URIRef, el.URIRef

        # Check for similarMatch text
        index, similarNode = self.similarNode(self.entityStore.get_list("text", type=elementType), element, indepth=False)
        if similarNode is None:
            # to je za "fast sportS"
            # Check again, but this time WITH STEMMING --- STEM ALL
            #index, similarNode = self.similarNode([e["text"] for e in self.entities[elementType]], element,
                                                  #indepth=False, stem=True)
            index, similarNode = self.similarNode(self.entityStore.get_list("text", type=elementType), element,
                                                  indepth=False, stem=True)

        URI = None
        elementURI = ""
        if similarNode is not None:
            el = self.entityStore.get_by_index(index, type=elementType)
            self.debugPrint("Found: ", el.text, str(el.URIRef))
            URI = el.URIRef
            elementURI = URI
        else:
            # If similar node not found, we continue checking synonyms and hypernyms
            URI = ""
            self.debugPrint("None ", element)

        return elementURI, URI


    def checkNodeSynonymsHypernyms(self, ONTO, element, URIRefs, elementType, URI):
        elementURI = ""

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
            for s in synsets:
                if len(s.lemma_names()) == 0:
                    continue
                name = s.lemma_names()[0].replace("_", " ")
                self.debugPrint(name)
                synsetArr.append(name)
                # if name in [e["text"] for e in self.entities]:
                if name in URIRefs[0]:
                    elementURI = URIRefs[1][URIRefs[0].index(name)]
                    URI = URIRefs[1][URIRefs[0].index(name)]
                    addToOntology = False
                    self.debugPrint("Found " + name + " in URIRefs. Not adding to ontology.")
                    break
                if name in self.entityStore.find(text=name, type=elementType):
                    elementURI = self.entityStore.find(text=name, type=elementType).URIRef
                    URI = elementURI
                    addToOntology = False
                    self.debugPrint("Found " + name + " in entityStore (synsets). Not adding to ontology.")
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
                    hypernim = str(h)[8:str(h).index(".")].replace("_", " ")
                    # Find out if hypernym exists in our ontology
                    try:
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
                                "Found element '" + str(HURI) + "' for hypernym '" + hypernim + "' (entitystore).")
                    if index > -1:
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
                        # self.entityStore.add(elementURI, ontologyElement, None, elementType, original=element) # TODO: DELETE THIS IF NOT WORKING
                        self.entityStore.add(elementURI, element, None, elementType, original=element)
                        self.debugPrint("Adding element (in hypernim if) '" + element + "' to ontology as '" + str(
                            owlType) + "' '" + str(elementURI) + "'")
                        self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.RDF.type, owlType)
                        if HURI != "" and elementType == "SubjectObject":
                            self.debugPrint("Adding HURI '" + str(HURI) + "' as suoperclass of URI: '" + str(elementURI) + "'")
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
                    try:
                        if lemma_name in URIRefs[0]:
                            index = URIRefs[0].index(lemma_name)
                        else:
                            index = URIRefs[2].index(lemma_name_stemmed)
                    except:
                        self.debugPrint("'" + lemma_name + "' / '" + lemma_name_stemmed + "' not in list?????")
                        index = -1
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
                    try:
                        if lemma_name in URIRefs[0]:
                            index = URIRefs[0].index(lemma_name)
                        else:
                            index = URIRefs[2].index(lemma_name_stemmed)
                    except:
                        self.debugPrint("'" + lemma_name + "' / '" + lemma_name_stemmed + "' not in list?????")
                        index = -1
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
                self.debugPrint("Found element in URIREFS!! " + str(elementURI))
            # Else check if element has any synonyms/hypernyms
            elif (elementType == "SubjectObject" and len(wordnet.synsets(element, pos=wordnet.NOUN))) or (
                    elementType == "Predicate" and len(wordnet.synsets(element, pos=wordnet.VERB))):
                elementURI, URI = self.checkNodeSynonymsHypernyms(ONTO, element, URIRefs, elementType, URI)
            # If no synonyms/hypernyms then add to ontology as new independent node
            else:
                elementURI = self.addNewNodeToOntology(ONTO, element, elementType)
        # EDIT: nevem kaj je to... POMEMBNO: V VSEH ZGORNJIH PRIMERIH DODAS NODE IN SHRANIS URIRef, saj je to le
        # "predpriprava" - dodali smo entitiy, potem pa to primerjamo se z triple extractionom
        if elementURI is None or elementURI == '':
            elementURI = self.addNewNodeToOntology(ONTO, element, elementType)

        self.debugPrint("*********************DODAJANJE RELATIONOV*****************")

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
            self.debugPrint("**** ISCEM NEGACIJO Z NOT..... ****")
            i, similarNode = self.similarNode(URIRefs[2], element[element.index("not ")+4:], indepth=False)
            self.debugPrint(element[element.index("not ")+4:])
            if similarNode is not None:
                self.debugPrint("Without not+: " + similarNode)
                elementAntonymURI = URIRefs[1][i]
                self.debugPrint("Adding _not_ predicate antonym to ontology: " + str(elementURI) + " OWL.propertyDisjointWith " + str(elementAntonymURI))
                self.tryAddToOntology(ONTO, elementURI, rdflib.namespace.OWL.propertyDisjointWith, elementAntonymURI, symetric=True)
            else:
                self.debugPrint("None")
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
            self.debugPrint("Already tried adding triple! " + str(subj_URI) + " " + str(type) + " " + str(obj_URI))
            # return True # TODO zdej je pocasnejs ampak bolj ziher: problem ce dvakrat dodajamo skor isto in se ujame
            #                       tukaj, vbistvu so se pa zaradi turbota vmes dodajali razlicni classi
            if is_triple and self.turbo:
                return self.hermit.check_unsatisfiable_cases(ONTO, remove=remove, explain=explain, i=self.i)
            else:
                return True

        if (subj_URI, type, obj_URI) not in ONTO:
            self.debugPrint("elementURI: " + str(subj_URI))
            self.debugPrint("Adding URI '" + str(subj_URI) + "' to ontology as " + str(type) + " of '" + str(obj_URI) + "'")
            self.allAddedTriples.append((subj_URI, type, obj_URI))
            ONTO.add((subj_URI, type, obj_URI))
            # ONTO.remove((subj_URI, rdflib.namespace.RDFS.comment, None))  # Remove any default comments;; ce jt ole odkomentirano pol tud Lisa ne najde v FINAL COMMENT TESTU TODO
            # For feedback: we add index to each element: we then get all elements and return index that matches all of them
            ONTO.add((subj_URI, rdflib.namespace.RDFS.comment, rdflib.Literal(str(self.currentId))))
            ONTO.add((obj_URI, rdflib.namespace.RDFS.comment, rdflib.Literal(str(self.currentId))))
            ONTO.add((type, rdflib.namespace.RDFS.comment, rdflib.Literal(str(self.currentId))))
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
        if elementType == "SubjectObject":
            ontologyElement = "".join([word.capitalize() for word in element.split()])
        else:
            ontologyElement = element.split()[0] + "".join([word.capitalize() for word in element.split()[1:]])
        elementURI = rdflib.URIRef(self.COSMO[ontologyElement])
        # self.entityStore.add(elementURI, ontologyElement, None, elementType, original=element) # TODO: ORIGINAL???
        self.entityStore.add(elementURI, element, None, elementType, original=element)
        if elementType == "SubjectObject":
            owlType = rdflib.namespace.OWL.Class
        else:
            owlType = rdflib.namespace.OWL.ObjectProperty
        self.debugPrint("Adding element '" + element + "' to ontology as '" + str(owlType) + "', '" + str(elementURI) + "'")
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

    def get_disjoint_relation(self, ONTO, relation, pred):
        drg = ONTO.subjects(relation, pred)
        disjointRelation = None
        self.debugPrint("Looking for disjoint relation")
        for dr in drg:
            self.debugPrint("Found " + str(dr))
            disjointRelation = dr
            break
        return disjointRelation

    def get_feedback(self, ONTO, subj, pred, obj):
        self.debugPrint("FEEDBACK")
        disjointRelation = self.get_disjoint_relation(ONTO, rdflib.namespace.OWL.propertyDisjointWith, pred)
        if disjointRelation is None:
            self.debugPrint("Couldn't find disjoint relation")
            # Check if obj disjointWith something: example: "Lisa is a boy. Lisa is a girl."
            for o in ONTO.objects(obj, rdflib.namespace.OWL.disjointWith):
                self.debugPrint("Found " + str(o) + " as disjointWith " + str(obj) + ". Returning...")
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
                intersection = i1.intersection(i2).intersection(i3)
                self.debugPrint("Intersection length is " + str(len(intersection)))
                for x in intersection:
                    self.debugPrint(x)
                return intersection.pop()
            except:
                self.debugPrint("Error finding index.")

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
        #self.debugPrint("S1: ", s1, "  ---- S2: ", s2)
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
            self.debugPrint("PREPROCESS COLLAGE")
            self.debugPrint(extraction)
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

    def stem(self, s):
        return PorterStemmer().stem(s)

