from orangecontrib.essaygrading.modules.BaseModule import BaseModule
from sklearn.feature_extraction.text import TfidfVectorizer
from orangecontrib.essaygrading.utils.lemmatizer import lemmatizeTokens, breakToWords
from orangecontrib.essaygrading.utils import globals
from scipy.spatial import distance
from scipy.sparse import issparse
import numpy as np
import spacy
import string
from nltk.corpus import stopwords
import math
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

name = "Coherence"


# GLOVE Word Embeddings
# python -m spacy download en_vectors_web_lg
class Coherence(BaseModule):

    name = "Coherence and semantics"

    def __init__(self, corpus, corpus_sentences, grades, source_texts=None, word_embeddings=globals.EMBEDDING_TFIDF):
        # TODO: ZAKAJ IMAMO TUKAJ 'grades', saj jih nikjer ne uporabljamo??!!
        """
        Overrides parent __init__ and calls _load().
        :param corpus: Tokenized essay Corpus.
        :param corpus_sentences: Tokenized (by sentence) essay Corpus.
        :param grades: Array of essay grades (ints)
        :param source_texts: Corpus of source texts (optional)
        :param word_embeddings: Word embeddings to use ('TF-IDF' or 'GloVe')
        """
        self._load(corpus, corpus_sentences, grades, source_texts, word_embeddings)

    def _load(self, corpus, corpus_sentences, grades, source_texts=None, word_embeddings=globals.EMBEDDING_TFIDF):
        """
        Calls parent _load() and sets additional parameters.
        :param corpus: Tokenized essay Corpus.
        :param corpus_sentences: Tokenized (by sentence) essay Corpus.
        :param grades: Array of essay grades
        :param source_texts: Corpus of source texts (optional)
        :param word_embeddings: Word embeddings to use ('TF-IDF' or 'GloVe')
        """
        if corpus is not None and corpus_sentences is not None:
            super()._load(corpus, corpus_sentences)

            self.source_texts = source_texts
            self.grades = np.array(grades)

            self.attributes = []
            self.corpus_parts = []
            self.tfidf_parts = []
            self.essay_scores = []
            self.distance_matrix = []

            self.word_embeddings = word_embeddings

    def calculate_all(self, selected_attributes, attribute_dictionary, callback=None, proportions=None, i=None):
        """
        Calculates all attributes in this module.
        :param selected_attributes: Object with attributes to calculate (boolean flags). If None, calculate all.
        :param attribute_dictionary: Attribute dicitionary which will be filled with calculated attributes.
        :param callback: Callback update function for progressbar.
        :param proportions: List of possible progressbar values.
        :param i: Index of current progressbar value.
        :return: i (index of progressbar value).
        """

        self.preprocess()

        D_euc = self.calculate_distance_matrix('euclidean')
        D_cos = self.calculate_distance_matrix('cosine')
        C, CD_euc, CD_cos = self.calculate_centroids()

        if selected_attributes is None or selected_attributes.cbBasicCoherenceMeasures:
            # Avg/Min/Max/Index neighbour and any points, Clark Evans, Avg to NN, Cumulative frequency
            print("ALL EUC NEIGH")
            a, mi, ma, ind = self.calculate_neighbour_distances(D_euc)
            print(a)
            print(mi)
            print(ma)
            print(ind)
            attribute_dictionary["avgDistanceNeighbouringPointsEuc"] = a
            attribute_dictionary["minDistanceNeighbouringPointsEuc"] = mi
            attribute_dictionary["maxDistanceNeighbouringPointsEuc"] = ma
            attribute_dictionary["indexDistanceNeighbouringPointsEuc"] = ind
            print("ALL COS NEIGH")
            a, mi, ma, ind = self.calculate_neighbour_distances(D_cos)
            print(a)
            print(mi)
            print(ma)
            print(ind)
            attribute_dictionary["avgDistanceNeighbouringPointsCos"] = a
            attribute_dictionary["minDistanceNeighbouringPointsCos"] = mi
            attribute_dictionary["maxDistanceNeighbouringPointsCos"] = ma
            attribute_dictionary["indexDistanceNeighbouringPointsCos"] = ind

            print("ALL EUC ANY")
            a, ma = self.calculate_any_point_distances(D_euc)
            print(a)
            print(ma)
            attribute_dictionary["avgDistanceAnyPointsEuc"] = a
            attribute_dictionary["maxDistanceAnyPointsEuc"] = ma
            print("ALL COS ANY")
            a, ma = self.calculate_any_point_distances(D_cos)
            print(a)
            print(ma)
            attribute_dictionary["avgDistanceAnyPointsCos"] = a
            attribute_dictionary["maxDistanceAnyPointsCos"] = ma

            ce, a_nn, freq = self.calculate_nn_distances(D_euc)
            print("CLARK EVANS")
            print(ce)
            print("AVG NN")
            print(a_nn)
            print("FREQ")
            print(freq)
            attribute_dictionary["clarkEvansNearestNeighbour"] = ce
            attribute_dictionary["avgDistanceNearestNeighbour"] = a_nn
            attribute_dictionary["cumulativeFrequencyDistribution"] = freq

        if selected_attributes is None or selected_attributes.cbSpatialDataAnalysis:
            print("CENTROID EUC")
            a, mi, ma, ind = self.calculate_centroid_distances(CD_euc)
            print(a)
            print(mi)
            print(ma)
            print(ind)
            attribute_dictionary["avgDistanceCentroidEuc"] = a
            attribute_dictionary["minDistanceCentroidEuc"] = mi
            attribute_dictionary["maxDistanceCentroidEuc"] = ma
            attribute_dictionary["indexDistanceCentroidEuc"] = ind

            print("CENTROID COS")
            a, mi, ma, ind = self.calculate_centroid_distances(CD_cos)
            print(a)
            print(mi)
            print(ma)
            print(ind)
            attribute_dictionary["avgDistanceCentroidCos"] = a
            attribute_dictionary["minDistanceCentroidCos"] = mi
            attribute_dictionary["maxDistanceCentroidCos"] = ma
            attribute_dictionary["indexDistanceCentroidCos"] = ind

            print("STD/REL DISTANCE")
            std, rel = self.calculate_standard_distance(C)
            print("STD")
            print(std)
            print("REL")
            print(rel)
            attribute_dictionary["standardDistance"] = std
            attribute_dictionary["relativeDistance"] = rel

            print("DETERMINANTS")
            # TODO:?? dela samo so vrenosti prakticno 0
            det = self.calculate_determinant(D_euc)
            print(det)
            attribute_dictionary["determinantDistanceMatrix"] = det

        if selected_attributes is None or selected_attributes.cbSpatialAutocorrelation:
            print("MORANS I")
            morans_i = self.calculate_morans_i(C)
            print(morans_i)
            attribute_dictionary["moransI"] = morans_i

            print("GEARYS C")
            gearys_c = self.calculate_gearys_c(C)
            print(gearys_c)
            attribute_dictionary["gearysC"] = gearys_c

            print("GETIS G")
            a, _ = self.calculate_any_point_distances(D_euc)
            g = self.calculate_getis_g(D_euc, a)
            print(g)
            attribute_dictionary["getissG"] = g

        print(attribute_dictionary)

        return i

    def calculate_neighbour_distances(self, D):
        """
        Calculates average, minimum, maximum distances between neighbours and their index.
        :param D: Distance matrix.
        :return: Average distances, minimum distances, maximum distances, indexes (4 parameters, each is an array).
        """
        averages = []
        minimums = []
        maximums = []
        for d in D:
            neighbour_distances = [d[i,i+1] for i in range(0, d.shape[0]-1)]
            # if neighbour_distances[0] == 1:
            #     print(self.tfidf_parts[-2])
            #     print(neighbour_distances)
            #     input()
            if len(neighbour_distances) == 0:
                averages.append(0)
                minimums.append(0)
                maximums.append(0)
            else:
                averages.append(sum(neighbour_distances) / len(neighbour_distances))
                minimums.append(min(neighbour_distances))
                maximums.append(max(neighbour_distances))

        indexes = np.divide(np.array(minimums), np.array(maximums))
        nans = np.isnan(indexes)
        indexes[nans] = 0
        return averages, minimums, maximums, indexes

    def calculate_any_point_distances(self, D):
        """
        Calculates average and maximum distances between any two points.
        :param D: Distance matrix.
        :return: Average distances, maximum distances (2 parameters, each is an array).
        """
        averages = []
        maximums = []
        for d in D:
            n = d.shape[0]
            distances = np.triu(d)
            averages.append(np.sum(distances) / max(1, (n*(n-1)/2)))
            maximums.append(np.amax(distances))
            #nans = np.where(averages[-1] == np.nan)[0]
            #averages[-1][nans] = 0
            #nans = np.where(maximums[-1] == np.nan)[0]
            #maximums[-1][nans] = 0
        return averages, maximums

    def calculate_nn_distances(self, D):
        """
        Calculate Clark-Evans distances, Average distances between nearest neighbours and Cumulative frequency distances.
        :param D: Distance matrix.
        :return: Clark-Evans distances, average distances, cumulative frequency distances (3 parameters, each is an array).
        """
        # TODO: tale n>1 je hack.. popravi
        clark_evans = []
        averages = []
        cumulative_freq = []
        for d in D:
            n = d.shape[0]
            # Find nearest neighbour
            s = 0
            for i in range(n):
                if n > 1:
                    s += np.amin(np.delete(d[i,:], i))
            averages.append(s/n)
            ce = 2*math.sqrt(n)*s / n
            clark_evans.append(ce)
            count = 0
            for i in range(n):
                if n > 1:
                    if np.amin(np.delete(d[i,:], i)) <= averages[-1]: #TODO: maybe use matrix to store values
                        count += 1
            cumulative_freq.append(float(count)/n)
        return clark_evans, averages, cumulative_freq # TODO: za cfreq nism zihr :/

    def calculate_centroid_distances(self, C):
        """
        Calculates average, minimum, maximum distances to centroid and their indexes.
        :param C: Distance matrix to centroid.
        :return: Average distances, minimum distances, maximum distances, indexes (4 parameters, each is an array).
        """
        averages = []
        minimums = []
        maximums = []

        for centroid_distance in C:
            averages.append(sum(centroid_distance) / len(centroid_distance))
            minimums.append(min(centroid_distance))
            maximums.append(max(centroid_distance))

        indexes = np.divide(np.array(minimums), np.array(maximums))
        nans = np.isnan(indexes)
        indexes[nans] = 0

        return averages, minimums, maximums, indexes

    def calculate_standard_distance(self, C):
        """
        Calculates standard distance and relative distance.
        :param C: Distance matrix to centroid.
        :return: Standard distances, relative distances (2 parameters, each is an array).
        """
        # C = centroids
        standard_distances = []
        relative_distances = []
        for doc_i in range(len(self.tfidf_parts)):
            doc = self.tfidf_parts[doc_i]
            # formula n=doc.shape[1], N=doc.shape[0]
            s = 0
            # tole rabi kar dolgo
            # for k in range(doc.shape[1]):
            #    for i in range(doc.shape[0]):
            #        s += (doc[i,k] - C[doc_i][k]) ** 2

            # optimizirano
            M = doc - C[doc_i]
            M = np.power(M, 2)
            s = np.sum(M)
            s /= doc.shape[0]
            standard_distances.append(math.sqrt(s))
            # relative distance
            d_max = 0
            for i in doc:
                if issparse(i):
                    i = i.todense()
                d_max = max(d_max, abs(self.euclidean_distance(np.array([i]), np.array([C[doc_i]]))))
                if d_max == 0:
                    d_max = 1
            relative_distances.append(standard_distances[-1]/d_max)

        return standard_distances, relative_distances

    def calculate_determinant(self, D):
        """
        Calculates matrix determinant.
        :param D: Input matrix. In our case it's the distance matrix.
        :return: Determinants of distance matrices.
        """
        determinants = []
        for doc in D:
            determinants.append(np.linalg.det(doc))
        return determinants

    def calculate_morans_i(self, C):
        """
        Calculates Moran's I. (https://en.wikipedia.org/wiki/Moran%27s_I)
        :param C: Distance matrix to centroid.
        :return: Moran's I for each essay.
        """
        morans_i = []
        for doc_i in range(len(self.tfidf_parts)):
            doc = self.tfidf_parts[doc_i]
            N = doc.shape[0] # st. tock
            n = doc.shape[1] # st. komponent
            S = (N - 1)*2 # vsota utezi TODO: je to res: sosedov je n-1 ?? update: JA?
            m = 0 # sprotna vsota
            D = doc - C[doc_i]
            '''
            for k in range(doc.shape[1]):
                # utez je ena smao, ce sta soseda - torej pride v postev samo i in i+1
                numerator = 0
                for i in range(doc.shape[0]):
                    for j in range(doc.shape[0]):
                        if abs(i-j) == 1:
                            numerator += D[i,k] * D[j,k]
                denominator = sum([D[i,k] ** 2 for i in range(doc.shape[0])])

                m += numerator / denominator
            '''
            # treba je narest v dveh delih, ker so notrnji >1 in <n elementi dvakrat v formuli (stevec), imenovalec
            # je isti
            D = doc - C[doc_i]
            Di = np.delete(D, -1, axis=0)
            Dj = np.delete(D, 0, axis=0)
            D_numerator = np.sum(np.multiply(Di, Dj), axis=0)
            Dj = np.delete(D, -1, axis=0)
            Di = np.delete(D, 0, axis=0)
            D_numerator += np.sum(np.multiply(Di, Dj), axis=0)
            D_denominator = np.sum(np.power(D, 2), axis=0)
            # Take care of zeros
            zeros = np.where(D_denominator == 0)[0]
            D_denominator[zeros] = 1
            D_numerator[zeros] = 0
            # Final steps
            m = np.sum(np.divide(D_numerator, D_denominator))
            if N < 2:
                m = 0
            else:
                m = (N/S) * (m/n)
            morans_i.append(m)

        return morans_i

    def calculate_gearys_c(self, C):
        """
        Calculates Geary's C. (https://en.wikipedia.org/wiki/Geary%27s_C)
        :param C: Distance matrix to centroids. |||| FIX: Centroid matrix ("mean centre matrix")
        :return: Geary's C for each essay.
        """
        gearys_c = []
        for doc_i in range(len(self.tfidf_parts)):
            doc = self.tfidf_parts[doc_i]
            if issparse(doc):
                doc = doc.todense()
            N = doc.shape[0]  # st. tock
            n = doc.shape[1]  # st. komponent
            S = (N - 1)*2  # vsota utezi
            c = 0  # sprotna vsota
            D = doc - C[doc_i]
            D_mean = doc - C[doc_i]

            '''for k in range(doc.shape[1]):
                # utez je ena smao, ce sta soseda - torej pride v postev samo i in i+1
                numerator = 0
                denominator = 0
                for i in range(doc.shape[0]):
                    for j in range(doc.shape[0]):
                        if abs(i-j) == 1:
                            numerator += ((doc[i,k] - doc[j,k])**2)
                for i in range(doc.shape[0]):
                    #for j in range(doc.shape[0]):
                        #if abs(i-j) == 1:
                    denominator += (D[i,k]**2)

                if denominator != 0:
                    c += numerator / denominator'''

            # treba je narest v dveh delih, ker so notrnji >1 in <n elementi dvakrat v formuli (stevec),
            # imenovalec je isti
            D = doc
            Di = np.delete(D, -1, axis=0)
            Dj = np.delete(D, 0, axis=0)
            D_numerator = np.sum(np.power(np.subtract(Di, Dj), 2), axis=0)
            # D_denominator = np.sum(np.power(Di - C[doc_i], 2), axis=0)
            Dj = np.delete(D, -1, axis=0)
            Di = np.delete(D, 0, axis=0)
            D_numerator += np.sum(np.power(np.subtract(Di, Dj), 2), axis=0)
            # D_denominator += np.sum(np.power(Di - C[doc_i], 2), axis=0)
            D_denominator = np.sum(np.power(D_mean, 2), axis=0)
            # Take care of zeros
            zeros = np.where(D_denominator == 0)[0]
            D_denominator[zeros] = 1
            D_numerator[zeros] = 0
            # Final steps
            c = np.sum(np.divide(D_numerator, D_denominator))

            S = (N - 1) * 2

            if N < 2:
                c = 0
            else:
                c = ((N-1)/(2*S)) * (c/n)
            gearys_c.append(c)

        return gearys_c

    # distance_threshold = array of avg distance between any two points
    # D_mat = distance matrix
    def calculate_getis_g(self, D_mat, distance_threshold):
        """
        Calculates Gettis-Ord General G.
        :param D_mat: Distance matrix.
        :param distance_threshold: Array of average distance between any two points.
        :return: Gettis' G for each essay.
        """
        getis_g = []
        for doc_i in range(len(self.tfidf_parts)):
            doc = self.tfidf_parts[doc_i]
            if issparse(doc):
                doc = doc.todense()
            N = doc.shape[0]  # st. tock
            n = doc.shape[1]  # st. komponent
            S = (N - 1) * 2  # vsota utezi TODO: je to res: sosedov je n-1 ?? JA?
            g = 0  # sprotna vsota
            D = doc
            d = distance_threshold[doc_i]
            '''
            for k in range(doc.shape[1]):
                numerator = 0
                denominator = 0
                for i in range(doc.shape[0]):
                    for j in range(doc.shape[0]):
                        if j == i:
                            continue
                        denominator += (D[i,k] * D[j,k])
                        if D_mat[doc_i][i][j] <= d:
                            numerator += (D[i,k] * D[j,k])

                if denominator != 0:
                    g += numerator / denominator
            '''
            # nimam blage veze kaj je ta black magic z broadcastanjem, ampak dela :DDDD
            D = doc
            # matrika utezi (1 ali 0) veliksoti NxN - katere vsote pridejo v postev
            W = np.where(D_mat[doc_i] <= distance_threshold[doc_i], 1, 0)
            W_ij = np.ones((len(W), len(W)))  # Weight matrix that makes sure i =/= j
            np.fill_diagonal(W_ij, 0)  # Fill diagonals with zeros
            D = np.array(D)
            # broadcastanje - iz tega rata pol 3D array, vsak element (vrstica) z vsakim - 3D zato, ker je vec komponent
            multiplied = np.multiply(D[None, :, :], D[:, None, :])
            weighted = multiplied * W[:, :, None]  # braodcastamo in mnozimo se z utezmi
            weighted = weighted * W_ij[:, :, None]
            D_numerator = np.sum(weighted, axis=(1, 0))
            D_denominator = np.sum(multiplied * W_ij[:, :, None], axis=(1, 0))
            # Now remove zeros from denominator and numerator
            zeros_locations = np.where(D_denominator == 0)[0]
            D_denominator[zeros_locations] = 1.0
            D_numerator[zeros_locations] = 0.0
            # Final steps
            g = np.sum(np.divide(D_numerator, D_denominator))

            if N < 2:
                g = 0
            else:
                g = g/n
            getis_g.append(g)

        return getis_g

    def calculate_centroids(self):
        """
        Calculates centroids and distances using euclidean and cosine metrics
        :return: Centroids, Centroid distance matrix (euclidean), Centroid distance matrix (cosine) (3 parameters)
        """
        C = []
        CD_euc = [] # centroid distance matrix
        CD_cos = []

        for doc in self.tfidf_parts:
            cd_euc = []
            cd_cos = []

            if issparse(doc):
                centroid = np.sum(doc, axis=0).A[0] / doc.shape[0]
                d = np.append(doc.todense(), [centroid], axis=0)
            else:
                centroid = np.sum(doc, axis=0) / doc.shape[0]
                d = np.append(doc, [centroid], axis=0)

            # for i in range(doc.shape[0]):
            #    cd_euc.append(self.euclidean_distance(doc[i].toarray(), centroid))
            #    cd_cos.append(cosine_similarity(doc[i].toarray(), [centroid]).tolist()[0][0])

            # c_mat = np.tile(centroid, (doc.shape[0],1))

            # d = np.append(doc.todense(), [centroid], axis=0)

            cd_euc = distance.cdist(d, d, metric='euclidean')[-1]
            cd_cos = distance.cdist(d, d, metric='cosine')[-1]

            cd_euc = np.delete(cd_euc, -1)
            cd_cos = np.delete(cd_cos, -1)

            nans = np.isnan(cd_euc)
            cd_euc[nans] = 0
            nans = np.isnan(cd_cos)
            cd_cos[nans] = 0

            C.append(centroid)
            CD_euc.append(cd_euc)
            CD_cos.append(cd_cos)

        return C, CD_euc, CD_cos

    def euclidean_distance(self, x, y):
        """
        Euclidean distance between two points (arrays of points).
        :param x: Points x.
        :param y: Points y.
        :return: Euclidean distance between points x and y.
        """
        return np.sqrt(np.sum((x - y) ** 2))

    def preprocess(self):
        """
        Preprocessing of input Corpora. Creates word embeddings. Results are stored in internal variables.
        """
        tfidf_vectorizer = TfidfVectorizer(max_features=None, stop_words="english")

        # TOLE SEM PRESTAVIL GOR IN docs UPORABIL ZA
        # corpus_sentences = lemmatizeTokens(self.corpus, join=True)
        corpus_tokens = lemmatizeTokens(self.corpus, join=False)
        print(corpus_tokens)
        # Remove stopwords and string punctuations
        sw = stopwords.words("english")
        corpus_tokens = [[token for token in i if token not in string.punctuation and token not in sw] for i in corpus_tokens]
        print(corpus_tokens)
        # corpus_tokens = self.corpus.tokens
        # append source/prompt text
        # TODO: source_text rabimo sploh?
        # docs.append((lemmatizeTokens(self.source_texts, join=True)[0]))

        # WINDOW PARAMETERS #
        window_step = 10  # Steps of 10 words
        window_size_factor = 0.25  # 25% of average words in all essays
        avg_essays_words = sum([len(tokens) for tokens in corpus_tokens]) / len(corpus_tokens)
        window_size = int(avg_essays_words * window_size_factor)

        # Create corpus/essay parts
        for tokens in corpus_tokens:
            wsize = window_size
            step = window_step
            while len(tokens) <= wsize:  # If windows too big, reduce it by factor of 2; size and steps
                wsize = int(wsize / 2)
                step = int(max(1, step / 2))
            parts = []
            i = 0
            j = wsize
            p = " ".join(tokens[i:min(j, len(tokens))])
            if len(p.replace(" ", "")) > 1:  # Fix for sometimes appending empty string
                parts.append(p)
            while j <= len(tokens):
                i += step
                j += step
                p = " ".join(tokens[i:j])
                if len(p.replace(" ", "")) > 1:  # Fix for sometimes appending empty string
                    parts.append(p)

            if len(parts) == 0: # If no valid parts were added, add a placeholder so it doesn't crash
                parts.append("0")

            self.corpus_parts.append(parts)

        # print("CP -2")
        # print(self.corpus_parts[-2])
        # print(corpus_tokens[-2])

        # TODO: preveri, ce je pravilno

        glove_embeddings = None
        flair_embeddings = None
        if self.word_embeddings == globals.EMBEDDING_GLOVE_SPACY:
            glove_embeddings = spacy.load("en_vectors_web_lg")
            print("GloVe (SpaCy) vectors loaded!")
        elif self.word_embeddings == globals.EMBEDDING_GLOVE_FLAIR:
            flair_embeddings = DocumentPoolEmbeddings([WordEmbeddings("glove")])
            print("GloVe (Flair) vectors loaded!")

        self.tfidf_parts = []
        qwe = 1
        for parts in self.corpus_parts:
            if self.word_embeddings == globals.EMBEDDING_TFIDF:
                # Try/catch if string empty or contains stopwords
                try:
                    self.tfidf_parts.append(tfidf_vectorizer.fit_transform(parts))
                except Exception:
                    print("Warning: empty string or stopwords only in tfidf essay part! Appending zeros!")
                    self.tfidf_parts.append(np.zeros((len(parts),len(parts))))
            elif self.word_embeddings == globals.EMBEDDING_GLOVE_FLAIR:
                essay_word_embedding = []
                # vocab = nlp.vocab
                for essay_part in parts:
                    # FLAIR
                    essay_part_sentences = Sentence(essay_part)
                    if essay_part == "":
                        continue
                    flair_embeddings.embed(essay_part_sentences)
                    essay_word_embedding.append(np.array(essay_part_sentences.get_embedding().detach().numpy()))
                qwe += 1
                self.tfidf_parts.append(np.array(essay_word_embedding))
            else:

                ''' 
                king = np.array(nlp.vocab.get_vector("king"))
                queen = np.array(nlp.vocab.get_vector("queen"))
                man = np.array(nlp.vocab.get_vector("man"))
                woman = np.array(nlp.vocab.get_vector("woman"))

                print(np.sum(np.square(king-queen)))
                print(np.sum(np.square(man-woman)))
                print(np.sum(np.square(king-man)))
                print(np.sum(np.square(queen-woman)))
                print(np.sum(np.square(king-woman)))
                print(np.sum(np.square(queen-man)))
                '''
                essay_word_embedding = []
                # vocab = nlp.vocab
                for essay_part in parts:
                    # SPACY
                    essay_word_embedding.append(np.array(glove_embeddings(essay_part).vector))  # spacy
                    # FLAIR
                    # print(essay_part)
                    # essay_part_sentences = Sentence(essay_part)
                    # flair_embeddings.embed(essay_part_sentences)
                    # essay_word_embedding.append(np.array(essay_part_sentences.get_embedding().detach().numpy()))
                # print(essay_word_embedding)
                # print(essay_word_embedding)
                self.tfidf_parts.append(np.array(essay_word_embedding))

        self.essay_scores = list(np.round(self.grades))

        # print(self.tfidf_parts[0])

        print("Word embedding windows done")

    def calculate_distance_matrix(self, metric='euclidean'):
        """
        Calculates distance matrix of internal word embeddings with specified distance metric.
        Execution of preprocess() method is required beforehand.
        :param metric: 'euclidean' or 'cosine'. Specifies distance metric to use for calculation of distance matrix.
        :return: Return calculated distance matrices (for each essay).
        """
        D = []

        for doc in self.tfidf_parts:
            s = 0
            if issparse(doc):
                doc = doc.todense()
            d = distance.cdist(doc, doc, metric=metric)
            nans = np.isnan(d)
            d[nans] = 0
            '''d = np.zeros((doc.shape[0], doc.shape[0]))
            if doc.shape[0] == 1:
                d[0,0] = 1
            else:
                for i in range(doc.shape[0]):
                    for j in range(i+1, doc.shape[0]):
                        if i == j:
                            continue
                        else:
                            dist = distance_f(doc[i].toarray(), doc[j].toarray())
                            d[i,j] = dist
                            d[j,i] = dist
            '''
            D.append(d)
        # print(D)
        return D
