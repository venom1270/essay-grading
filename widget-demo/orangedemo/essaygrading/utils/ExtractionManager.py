import spacy
from nltk import word_tokenize, pos_tag, PorterStemmer
import string
import rdflib
import re
from orangedemo.essaygrading.utils.HermiT import HermiT
from nltk.corpus import wordnet
from orangedemo.essaygrading.utils.lemmatizer import breakToWords

class ExtractionManager:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        # bagOfAllEntities (Kaja ima tudi te pomatchane pomoje)
        self.allEntities = []
        # bagOfEntities (pomatchani)- samo trenuten stavek, brez duplikatov
        self.entities = []
        self.URIdict = {}
        self.id = 0
        self.hermit = HermiT()

    def getChunks(self, essay_sentences):
        # TODO: this is only for NP; add VP, ...
        all_chunks = []
        for sentence in essay_sentences:
            doc = self.nlp(sentence)
            all_chunks.append([chunk.text for chunk in doc.noun_chunks])
        return all_chunks

    def mergeEssayAndChunks(self, essay_sentences, chunks):
        tokens = [word_tokenize(sentence) for sentence in essay_sentences]

        for sentence_chunk in chunks: # Over sentences
            for chunk in sentence_chunk: # Over chunks in sentence
                # TODO?: Tukaj Kaje se preveri c je "Ive" in ga pretvori v "I ve"
                # TODO? Kaja je tukaj imela if ce je v seznamu *P-jev (VP, NP...)
                # Ce je NP, ga dodamo v 'bagOfEntities' - pri nas os vsi NP
                self.allEntities.append({"id": self.id, "text": self.preprocessExtraction(chunk)})
                self.id += 1
                # TODO?: Tukaj Kaja zbira se entities (ne bagOFEntities) in naredi takle seznam.. sem preveril:
                #       to je za 1) Koreference pravilno pomatchat (detajlov nism gledu) in 2) da se v povezave ontologije doda VP-je
                # entities: SEZNAM  [stavek][0] = seznam tokenov?
                # 					[stavek][1] = tip fraze (NP, VP, ...)
                #					[stavek][2] = ID v bagOfEntities
        return self.allEntities

    # URIRefs = array[0] = tokeni, [1] = URIREF, [2] = None (???ocitno ID), [3] = Stemmed
    def matchEntitesWithURIRefs(self, URIRefs):

        URIs = {}

        for entity in self.allEntities:
           # for URI_index in range(len(URIRefs[3])):
                #print(URIRefs[3][URI_index])
                #exit()
            similarNode, _ = self.similarNode(URIRefs[2], entity["text"], indepth=True)
            if similarNode is not None:
                print("SIMILAR")
                print(URIRefs[0][similarNode], URIRefs[1][similarNode], URIRefs[2][similarNode])
                print(entity["text"])
                self.URIdict[URIRefs[0][similarNode]] = URIRefs[1][similarNode]
                self.URIdict[URIRefs[2][similarNode]] = URIRefs[1][similarNode]
                self.entities.append({"id": similarNode, "text": URIRefs[0][similarNode], "URI": URIRefs[1][similarNode]})
                URIs[URIRefs[0][similarNode]] = (URIRefs[1][similarNode], URIRefs[2][similarNode])

        return URIs


    def addExtractionToOntology(self, ONTO, extraction, URIRefsObjects, URIRefsPredicates):
        for t in extraction:
            print(t)
            self.addElementToOntology(ONTO, t[0].subject, URIRefsObjects, "SubjectObject")
            self.addElementToOntology(ONTO, t[0].object, URIRefsObjects, "SubjectObject")
            self.addElementToOntology(ONTO, t[0].predicate, URIRefsPredicates, "Predicate")

    def addElementToOntology(self, ONTO, element, URIRefs, elementType):
        COSMO = rdflib.Namespace("http://micra.com/COSMO/COSMO.owl#")  # TODO: move elsewhere
        print("Check similar node: ", element)
        if element.split()[0] in ["a", "an"]:
            element = " ".join(element.split()[1:])
        index, similarNode = self.similarNode([e["text"] for e in self.entities], element, indepth=False)
        # TUKAJ PRIMERJA ENTITIJE V TEM STAVKU (zaradi optimizacije?) ki jih je pridobila s shallow parsingom
        # -> DOBI ID IN POTEM TAKOJ URIRef
        URI = None
        if similarNode is not None:
            #node = [e["URI"] for e in self.entities if e["id"] == similarNode]

            print("Found: ", self.entities[index])
            URI = self.entities[index]["URI"]
        else:
            URI = ""
            print("None ", element)


        # START TUKAJ SE SAMO DODAJA NODE
        print("*********************DODAJANJE ENTITIJEV*****************")

        # TODO: if to sem skipal - - is there already a node with this id and URI under the ID - - check if coref and add???
        # TODO: elif # - - is there a node with same name - -
        # TODO: elif # - - is there a node with similar name - -
        #  elif synsets: sopomenke in nadpomenke
        elementURI = None
        if  (elementType == "SubjectObject" and len(wordnet.synsets(element, pos=wordnet.NOUN))) or (elementType == "Predicate" and len(wordnet.synsets(element, pos=wordnet.VERB))):
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
                synsetArr.append(name)
                #if name in [e["text"] for e in self.entities]:
                if name in URIRefs[0]:
                    #URI = self.URIdict[name]
                    elementURI = URIRefs[1][URIRefs[0].index(name)]
                    URI = URIRefs[1][URIRefs[0].index(name)]
                    addToOntology = False
                    print("Found " + name + " in entities. Not adding to ontology.")

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
                    index = URIRefs[0].index(hypernim)
                    if index > -1:
                        HURI = URIRefs[1][index]
                        print("Found element '" + URIRefs[0][index] + "' for hypernym '" + hypernim + "'.")

                        if elementType == "SubjectObject":
                            ontologyElement = "".join([word.capitalize() for word in element.split()])
                            owlType = rdflib.namespace.OWL.Class
                        else:
                            ontologyElement = element.split()[0] + "".join([word.capitalize() for word in element.split()[1:]])
                            owlType = rdflib.namespace.OWL.ObjectProperty
                        elementURI = rdflib.URIRef(COSMO[ontologyElement])
                        URI = elementURI
                        self.URIdict[element] = elementURI
                        self.URIdict[ontologyElement] = elementURI
                        self.entities.append({"id": elementURI, "text": element, "URI": elementURI})
                        # TODO if type == Subject/Object elif type == Predicate
                        print("Adding element (in hypernim if) '" + element + "' to ontology as '" + owlType + "' " + elementURI)
                        if (elementURI, rdflib.namespace.RDF.type, owlType) in ONTO:
                            print("NOT ADDING!!!")
                        else:
                            ONTO.add((elementURI, rdflib.namespace.RDF.type, owlType))
                            self.hermit.check_unsatisfiable_cases(ONTO)

                        if HURI != "" and elementType == "SubjectObject":
                            print("Adding HURI " + HURI + " as suoperclass of URI: " + elementURI)
                            if (elementURI, rdflib.namespace.RDFS.subClassOf, HURI) in ONTO:
                                print("NOT ADDING!!")
                            else:
                                ONTO.add((elementURI, rdflib.namespace.RDFS.subClassOf, HURI))
                                if not self.hermit.check_unsatisfiable_cases(ONTO, remove=False):
                                    print("Removing...")
                                    ONTO.remove((elementURI, rdflib.namespace.RDFS.subClassOf, HURI))




        # ... else: Dodaj nov node
        #if URI is None: # to bo potem else
        else:
            if elementType == "SubjectObject":
                ontologyElement = "".join([word.capitalize() for word in element.split()])
            else:
                ontologyElement = element.split()[0] + "".join([word.capitalize() for word in element.split()[1:]])
            elementURI = rdflib.URIRef(COSMO[ontologyElement])
            self.URIdict[element] = elementURI
            self.URIdict[ontologyElement] = elementURI
            self.entities.append({"id": elementURI, "text": element, "URI": elementURI})
            # TODO if type == Subject/Object elif type == Predicate
            print("Adding element '" + element + "' to ontology as OWL.Class " + elementURI)
            ONTO.add((elementURI, rdflib.namespace.RDF.type, rdflib.namespace.OWL.Class))
            self.hermit.check_unsatisfiable_cases(ONTO)
        # POMEMBNO: V VSEH ZGORNJIH PRIMERIH DODAS NODE IN SHRANIS URIRef, saj je to le "predpriprava" - dodali smo entitiy, potem pa to primerjamo se z triple extractionom


        # STOP TUKAJ SE SAMO DODAJA NODE

        print("*********************DODAJANJE RELATIONOV*****************")

        # 1338+
        # TODO: tukaj zdaj pogledamo wnsense   <--- ZDAJ SEM TUKAJ
        # for meaning in O.objects(AURI, COSMO.wnsense):
        use_wordnet = True
        for meaning in ONTO.objects(URI, COSMO.wnsense):
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
                #print(h)
                #print(h.lemma_names)
                #print(h.lemmas)
                for lemma_name in h.lemma_names():
                    print("LN: " + lemma_name)
                    lemma_name = lemma_name.replace("_", " ")
                    lemma_name = self.stemSentence(lemma_name)
                    HURI = ""
                    try:
                        index = URIRefs[2].index(lemma_name)
                    except:
                        print("'" + lemma_name + "' not in list?????")
                        index = -1
                    if index > -1:
                        HURI = URIRefs[1][index]
                    elif lemma_name in [e["text"] for e in self.entities]:
                        HURI = [e["URI"] for e in self.entities if e["text"] == lemma_name][0]
                    else:
                        print("NOT FOUND!!!! LEMMA: " + lemma_name)
                    print("HURI = " + HURI)
                    if HURI != "":
                        if elementType == "SubjectObject":
                            rdfType = rdflib.namespace.RDFS.subClassOf
                        else:
                            rdfType = rdflib.namespace.RDFS.subPropertyOf
                        if (URI, rdfType, HURI) not in ONTO:
                            print("Adding URI '" + URI + "' to ontology as " + rdfType + " of " + HURI)
                            ONTO.add((elementURI, rdfType, HURI))
                            if not self.hermit.check_unsatisfiable_cases(ONTO):
                                print("Removing...")
                                ONTO.remove((elementURI, rdfType, HURI))
                        else:
                            print("Already in ONTO!" + URI + " " + rdfType + " " + HURI)

            # Povezemo protipomenke...
            for lemma in s.lemmas():
                for antonym in lemma.antonyms():
                    print("ANTONYM: " + antonym.name())
                    lemma_name = self.stemSentence(antonym.name().replace("_", " "))
                    ANTO_URI = ""
                    try:
                        index = URIRefs[2].index(lemma_name)
                    except:
                        print("'" + lemma_name + "' not in list?????")
                        index = -1
                    if index > -1:
                        ANTO_URI = URIRefs[1][index]
                    elif lemma_name in [e["text"] for e in self.entities]:
                        ANTO_URI = [e["URI"] for e in self.entities if e["text"] == lemma_name][0]
                    else:
                        print("NOT FOUND!!!! LEMMA: " + lemma_name)
                    print("ANTO_URI = " + ANTO_URI)
                    if ANTO_URI != "":
                        rdfType = rdflib.namespace.OWL.disjointWith
                        if (URI, rdfType, ANTO_URI) not in ONTO:
                            print("elementURI: " + elementURI)
                            print("Adding URI '" + URI + "' to ontology as " + rdfType + " of " + ANTO_URI)
                            ONTO.add((elementURI, rdfType, ANTO_URI))
                            ONTO.add((ANTO_URI, rdfType, elementURI))
                            if not self.hermit.check_unsatisfiable_cases(ONTO):
                                print("Removing...")
                                ONTO.remove((elementURI, rdfType, ANTO_URI))
                                ONTO.remove((ANTO_URI, rdfType, elementURI))
                        else:
                            print("Already in ONTO!" + URI + " " + rdfType + " " + ANTO_URI)
                # TODO?: iz wordneta, ampak to mislim da ima ponesreci podvojeno..



        # ---> pogledamo nadpomenke in jih dodajamo/preverjamo
        # ---> pogledamo protipomenke in jih dodajamo/preverjamo (kot DisjointWith)
        # ---> poskuša tudi s korenjenimi protipomenkami

        # TODO: uporabimo wordnet (meanTF) - samo ce COSMO.wnsesne ni nasel nobenega meaninga
        # naredimo enako kot zgoraj (tiste 3 ---->)
        if use_wordnet:
            pass


        # TO JE TO!!!!! NOW GTW


    def similarNode(self, nodes, newNode, indepth=True):
        # indepth: to pomen da zacnes sekat besede iz newNode od zacetka do konca in upas da se kej ujame
        for n in range(len(nodes)):
            if self.sentenceSimilarity(nodes[n], newNode) >= 0.7:
                return n, nodes[n]
            if indepth:
                splitEntity = newNode
                while splitEntity.find(" ") > 0:
                    splitEntity = " ".join(splitEntity.split(" ")[1:])
                    if self.sentenceSimilarity(nodes[n], splitEntity) >= 0.7:
                        return n, nodes[n]
        return None, None

    def sentenceSimilarity(self, s1, s2):
        count = 0
        #print("S!: ", s1)
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
        # words = [i for i in extrPart.split() if i not in determiners]
        # extrPart = ' '.join(words)
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

