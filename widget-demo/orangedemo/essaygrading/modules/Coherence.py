from orangedemo.essaygrading.modules.BaseModule import BaseModule
from sklearn.feature_extraction.text import TfidfVectorizer
from orangedemo.essaygrading.modules.lemmatizer import lemmatizeTokens
from scipy.spatial import distance
import numpy as np
import math

class Coherence(BaseModule):

    def __init__(self, corpus, corpus_sentences, source_texts):
        super().__init__(corpus, corpus_sentences)

        self.source_texts = source_texts

        self.attributes = []
        self.corpus_parts = []
        self.tfidf_parts = []
        self.essay_scores = []
        self.distance_matrix = []

    def calculate_all(self, selected_attributes, attribute_dictionary, callback=None, proportions=None, i=None):

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
        averages = []
        minimums = []
        maximums = []
        for d in D:
            neighbour_distances = [d[i,i+1] for i in range(0, d.shape[0]-1)]
            if len(neighbour_distances) == 0:
                averages.append(0)
                minimums.append(0)
                maximums.append(0)
            else:
                averages.append(sum(neighbour_distances) / len(neighbour_distances)) #TODO: division by zero dodej max(i guess)
                minimums.append(min(neighbour_distances))
                maximums.append(max(neighbour_distances))

        indexes = np.divide(np.array(minimums), np.array(maximums))
        return averages, minimums, maximums, indexes

    def calculate_any_point_distances(self, D):
        averages = []
        maximums = []
        for d in D:
            n = d.shape[0]
            distances = np.triu(d)
            averages.append(np.sum(distances) / max(1, (n*(n-1)/2)))
            maximums.append(np.amax(distances))
        return averages, maximums

    def calculate_nn_distances(self, D):
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
        return clark_evans, averages, cumulative_freq # TODO: za cumfreq nism zihr :/

    def calculate_centroid_distances(self, C):
        averages = []
        minimums = []
        maximums = []

        for centroid_distance in C:
            averages.append(sum(centroid_distance) / len(centroid_distance))
            minimums.append(min(centroid_distance))
            maximums.append(max(centroid_distance))

        indexes = np.divide(np.array(minimums), np.array(maximums))

        return averages, minimums, maximums, indexes

    def calculate_standard_distance(self, C):
        # C = centroids
        standard_distances = []
        relative_distances = []
        for doc_i in range(len(self.tfidf_parts)):
            doc = self.tfidf_parts[doc_i]
            # formula n=doc.shape[1], N=doc.shape[0]
            s = 0
            # tole rabi kar dolgo
            #for k in range(doc.shape[1]):
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
                d_max = max(d_max, abs(self.euclidean_distance(np.array([i.todense()]), np.array([C[doc_i]]))))
                if d_max == 0:
                    d_max = 1
            relative_distances.append(standard_distances[-1]/d_max)

        return standard_distances, relative_distances

    def calculate_determinant(self, D):
        determinants = []
        for doc in D:
            determinants.append(np.linalg.det(doc))
        return determinants

    def calculate_morans_i(self, C):
        morans_i = []
        for doc_i in range(len(self.tfidf_parts)):
            doc = self.tfidf_parts[doc_i]
            N = doc.shape[0] # st. tock
            n = doc.shape[1] # st. komponent
            S = (N - 1)*2 # vsota utezi TODO: je to res: sosedov je n-1 ??
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
            # treba je narest v dveh delih, ker so notrnji >1 in <n elementi dvakrat v formuli (stevec), imenovalec je isti
            D = doc - C[doc_i]
            Di = np.delete(D, -1, axis=0)
            Dj = np.delete(D, 0, axis=0)
            D_numerator = np.sum(np.multiply(Di, Dj), axis=0)
            Dj = np.delete(D, -1, axis=0)
            Di = np.delete(D, 0, axis=0)
            D_numerator += np.sum(np.multiply(Di, Dj), axis=0)
            D_denominator = np.sum(np.power(D, 2), axis=0)
            m = np.sum(np.divide(D_numerator, D_denominator))
            if N < 2:
                m = 0
            else:
                m = (N/S) * (m/n)
            morans_i.append(m)

        return morans_i


    def calculate_gearys_c(self, C):
        gearys_c = []
        for doc_i in range(len(self.tfidf_parts)):
            doc = self.tfidf_parts[doc_i]
            N = doc.shape[0] # st. tock
            n = doc.shape[1] # st. komponent
            S = (N - 1)*2 # vsota utezi TODO: je to res: sosedov je n-1 ??
            c = 0 # sprotna vsota
            D = doc - C[doc_i]
            '''
            for k in range(doc.shape[1]):
                # utez je ena smao, ce sta soseda - torej pride v postev samo i in i+1
                numerator = 0
                denominator = 0
                for i in range(doc.shape[0]):
                    for j in range(doc.shape[0]):
                        if abs(i-j) == 1:
                            numerator += ((doc[i,k] - doc[j,k])**2)
                for i in range(doc.shape[0]):
                    for j in range(doc.shape[0]):
                        if abs(i-j) == 1:
                            denominator += (D[i,k]**2)

                if denominator != 0:
                    c += numerator / denominator
            '''
            # treba je narest v dveh delih, ker so notrnji >1 in <n elementi dvakrat v formuli (stevec), imenovalec je isti
            D = doc.todense()
            Di = np.delete(D, -1, axis=0)
            Dj = np.delete(D, 0, axis=0)
            D_numerator = np.sum(np.power(np.subtract(Di, Dj), 2), axis=0)
            D_denominator = np.sum(np.power(Di - C[doc_i], 2), axis=0)
            Dj = np.delete(D, -1, axis=0)
            Di = np.delete(D, 0, axis=0)
            D_numerator += np.sum(np.power(np.subtract(Di, Dj), 2), axis=0)
            D_denominator += np.sum(np.power(Di - C[doc_i], 2), axis=0)
            c = np.sum(np.divide(D_numerator, D_denominator))

            if N < 2:
                c = 0
            else:
                c = ((N-1)/2) * (c/n)
            gearys_c.append(c)

        return gearys_c

    # distance_threshold = array of avg distance between any two points
    # D_mat = distance matrix
    def calculate_getis_g(self, D_mat, distance_threshold):
        getis_g = []
        for doc_i in range(len(self.tfidf_parts)):
            doc = self.tfidf_parts[doc_i]
            N = doc.shape[0]  # st. tock
            n = doc.shape[1]  # st. komponent
            S = (N - 1) * 2  # vsota utezi TODO: je to res: sosedov je n-1 ??
            g = 0  # sprotna vsota
            D = doc.todense()
            d = distance_threshold[doc_i]
            '''
            for k in range(doc.shape[1]):
                numerator = 0
                denominator = 0
                for i in range(doc.shape[0]):
                    for j in range(doc.shape[0]):
                        denominator += (D[i,k] * D[j,k])
                        if D_mat[doc_i][i][j] <= d: # TODO: mybe i != j???
                            numerator += (D[i,k] * D[j,k])


                if denominator != 0:
                    g += numerator / denominator
            '''
            # nimam blage veze kaj je ta black magic z broadcastanjem, ampak dela :DDDD
            D = doc.todense()
            W = np.where(D_mat[doc_i] <= distance_threshold[doc_i], 1, 0) # matrika utezi (1 ali 0) veliksoti NxN - katere vsote pridejo v postev
            D = np.array(D)
            multiplied = np.multiply(D[None,:,:], D[:,None,:]) # broadcastanje - iz tega rata pol 3D array, vsak element (vrstica) z vsakim - 3D zato, ker je vec komponent
            weighted = multiplied * W[:,:,None] # braodcastamo in mnozimo se z utezmi
            D_numerator = np.sum(weighted, axis=(1,0)) #
            D_denominator = np.sum(multiplied, axis=(1,0))
            g = np.sum(np.divide(D_numerator, D_denominator))

            if N < 2:
                g = 0
            else:
                g = g/n
            getis_g.append(g)

        return getis_g

    def calculate_centroids(self):
        C = []
        CD_euc = [] # centroid distance matrix
        CD_cos = []

        for doc in self.tfidf_parts:
            cd_euc = []
            cd_cos = []

            centroid = np.sum(doc, axis=0).A[0] / doc.shape[0]

            #for i in range(doc.shape[0]):
            #    cd_euc.append(self.euclidean_distance(doc[i].toarray(), centroid))
            #    cd_cos.append(cosine_similarity(doc[i].toarray(), [centroid]).tolist()[0][0])



            c_mat = np.tile(centroid, (doc.shape[0],1))


            d = np.append(doc.todense(), [centroid], axis=0)

            cd_euc = distance.cdist(d, d, metric='euclidean')[-1]
            cd_cos = distance.cdist(d, d, metric='cosine')[-1]

            cd_euc = np.delete(cd_euc, -1)
            cd_cos = np.delete(cd_cos, -1)

            C.append(centroid)
            CD_euc.append(cd_euc)
            CD_cos.append(cd_cos)

        return C, CD_euc, CD_cos

    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def preprocess(self):
        tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words="english",
                                           use_idf=True)

        # TOLE SEM PRESTAVIL GOR IN docs UPORABIL ZA
        docs = lemmatizeTokens(self.corpus, join=True)
        # append source/prompt text
        # TODO: source_text rabimo sploh?
        #docs.append((lemmatizeTokens(self.source_texts, join=True)[0]))

        # WINDOW PARAMETERS #
        step = 10 # Steps of 10 words
        window_size_factor = 0.25 # 25% of average words in all essays
        avg_essays_words = sum([len(tokens) for tokens in docs]) / len(docs)
        window_size = int(avg_essays_words * window_size_factor)

        # Create corpus/essay parts
        for tokens in docs:
            parts = []
            i = 0
            j = window_size
            parts.append(tokens[i:min(j, len(tokens))])
            while j <= len(tokens):
                i += step
                j += step
                #parts.append(tokens[i:min(j, len(tokens))])
                parts.append(tokens[i:j])
            self.corpus_parts.append(parts)

        # TODO: preveri, ce je pravilno
        self.tfidf_parts = []
        for parts in self.corpus_parts:
            self.tfidf_parts.append(tfidf_vectorizer.fit_transform(parts))

        self.essay_scores = list(np.floor(self.corpus.X[:, 5] / 2))

        #print(self.tfidf_parts)

        print(self.tfidf_parts[0])

        print("TFIDF windows done")

    def calculate_distance_matrix(self, metric='euclidean'):
        D = []

        for doc in self.tfidf_parts:
            s = 0
            d = distance.cdist(doc.todense(), doc.todense(), metric=metric)
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
        #print(D)
        return D
