import spacy
from nltk import word_tokenize, pos_tag, PorterStemmer
import string

class ExtractionManager:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.entities = []
        self.id = 0

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
                self.entities.append({"id": self.id, "text": self.preprocessExtraction(chunk)})
                self.id += 1
                # TODO?: Tukaj Kaja zbira se entities (ne bagOFEntities) in naredi takle seznam.. sem preveril:
                #       to je za 1) Koreference pravilno pomatchat (detajlov nism gledu) in 2) da se v povezave ontologije doda VP-je
                # entities: SEZNAM  [stavek][0] = seznam tokenov?
                # 					[stavek][1] = tip fraze (NP, VP, ...)
                #					[stavek][2] = ID v bagOfEntities
        return self.entities

    # URIRefs = array[0] = tokeni, [1] = URIREF, [2] = None (???), [3] = Stemmed
    def matchEntitesWithURIRefs(self, URIRefs):

        URIs = {}

        for entity in self.entities:
           # for URI_index in range(len(URIRefs[3])):
                #print(URIRefs[3][URI_index])
                #exit()
            similarNode = self.similarNode(URIRefs[2], entity["text"])
            if similarNode is not None:
                print("SIMILAR")
                print(URIRefs[0][similarNode], URIRefs[1][similarNode], URIRefs[2][similarNode])
                print(entity["text"])
                URIs[URIRefs[0][similarNode]] = (URIRefs[1][similarNode], URIRefs[2][similarNode])
            # TODO: else odvzemaj characterje/besede spredaj in poskusaj ponovno... (primer: a girl se ne veze z girl, ceprav bi se skoraj moralo)

        return URIs

    def similarNode(self, nodes, newNode):
        for n in range(len(nodes)):
            if self.sentenceSimilarity(nodes[n], newNode) >= 0.7:
                return n
        return None

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
        porter = PorterStemmer()
        extraction = [porter.stem(i) for i in extraction.split()]
        return ' '.join(extraction)

