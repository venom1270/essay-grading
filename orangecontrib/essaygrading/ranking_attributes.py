from orangecontrib.essaygrading.widgets.OWScoreEssayPredictions import quadratic_weighted_kappa
import numpy as np


from Orange.data import Table, Domain, ContinuousVariable
from Orange.regression import RandomForestRegressionLearner
from Orange.regression import RidgeRegressionLearner

from sklearn.model_selection import KFold
from Orange.preprocess.score import RReliefF, UnivariateLinearRegression
from Orange.regression import RidgeRegressionLearner
from scipy import stats

TABLES = []
TABLES.append(Table.from_file("datasets/FAS/All/AGE+/DS1_AGE+_TFIDF.tab"))
TABLES.append(Table.from_file("datasets/FAS/All/AGE+/DS2A_AGE+_TFIDF.tab"))
TABLES.append(Table.from_file("datasets/FAS/All/AGE+/DS2B_AGE+_TFIDF.tab"))
TABLES.append(Table.from_file("datasets/FAS/All/SAGE/DS3_SAGE_TFIDF.tab"))
TABLES.append(Table.from_file("datasets/FAS/All/SAGE/DS4_SAGE_TFIDF.tab"))
TABLES.append(Table.from_file("datasets/FAS/All/SAGE/DS5_SAGE_TFIDF.tab"))
TABLES.append(Table.from_file("datasets/FAS/All/AGE+/DS6_AGE+_TFIDF.tab"))
TABLES.append(Table.from_file("datasets/FAS/All/AGE+/DS7_AGE+_TFIDF.tab"))
TABLES.append(Table.from_file("datasets/FAS/All/AGE+/DS8_AGE+_TFIDF.tab"))

kf = KFold(n_splits=10, shuffle=False)
global_avg_rangs = []
global_avg_kappas = []

for i, table in enumerate(TABLES):

    print("************************+ PROCESSING TABLE " + str(i+1) + " / " + str(len(TABLES)))

    kappas = []
    ranks = []
    fold = 1

    for train_i, test_i in kf.split(table):

        print("FOLD " + str(fold) + " / 10")
        fold += 1

        train_y = np.array(table.Y)[train_i]
        train_x = np.array([x for x in table.X])[train_i]

        test_y = np.array(table.Y)[test_i]
        test_x = np.array([x for x in table.X])[test_i]

        # GET ATTRIBUTE RANKING : RRELIEFF

        domain = table.domain
        data = Table.from_numpy(domain, np.array(train_x), np.array(train_y))
        #relief = RReliefF(n_iterations=200)
        #weights = relief.score_data(data, feature=False)
        ulr = RidgeRegressionLearner(alpha=0.02) #UnivariateLinearRegression()
        weights = ulr.score_data(data, None)
        # ULR JE DOST BOLJ LEGIT!
        print(weights)
        ranks.append(stats.rankdata(weights*-1))
        print(ranks[-1])
        # TALE IZBOR DELA, VENDAR JE V ZAPROEDJU RANGOV in ne v originalnem zaporedju
        top_50 = sorted(range(len(ranks[-1])), key=lambda i: ranks[-1][i])[:75]
        #print(top_50)
        train_x = np.array(train_x)[:, top_50]
        test_x = np.array(test_x)[:, top_50]
        #input()

        '''
        shuffle=False, RF, ULR
        BREZ IZBORA ATRIBUTOV
            GLOBAL KAPPA: 0.7471467598155617
            
        Z IZBOROM TOP 50
            GLOBAL KAPPA: 0.7354183139030254
            TOP 75: 0.7427307958667978
            TOP 75, shuffle=True: 0.7427265475787926
            
            
        RELIEF:
            BREZ: 0.7471467598155617
            TOP 50: 0.7220142889251973

        '''

        # END

        rf = RidgeRegressionLearner(alpha=0.02)  # (n_estimators=100, )#, min_samples_split=max(int(len(attributes[0])/3), 2))
        rf = rf.fit(train_x, train_y)

        # print("Predicting...")
        predictions = rf.predict(test_x)

        predictions = [p if p >= 0 and p <= 100 else 0 for p in predictions]
        kappas.append(quadratic_weighted_kappa(np.round(predictions), test_y))


    print(kappas)
    print("AVG KAPPA:")
    avg_kappa = np.average(np.array(kappas))
    print(avg_kappa)
    global_avg_kappas.append(avg_kappa)
    print("AVG RANKS:")
    avg_ranks = np.average(np.array(ranks), axis=0)
    print(avg_ranks)
    global_avg_rangs.append(avg_ranks)

    # input()

print("GLOBAL RANGS:")
print(global_avg_rangs)

# Take care of SAGE attributes
SAGE_rangs = []
for i in range(len(global_avg_rangs)):
    if len(global_avg_rangs[i]) > 103:
        SAGE_rangs.append(global_avg_rangs[i][103:])
        global_avg_rangs[i] = global_avg_rangs[i][:103]

AVG_RANG_AGE = np.average(np.array(global_avg_rangs), axis=0)
AVG_RANG_SAGE = np.average(np.array(SAGE_rangs), axis=0)
AVG_RANG = np.concatenate((AVG_RANG_AGE, AVG_RANG_SAGE))

print("FINAL RANG:")
print(AVG_RANG)

print("GLOBAL KAPPAS:")
print(global_avg_kappas)

AVG_KAPPA = np.average(np.array(global_avg_kappas))
print("GLOBAL KAPPA")
print(AVG_KAPPA)

attribute_names = [cv.name for cv in TABLES[4].domain.attributes]
print([(x,r) for r,x in sorted(zip(AVG_RANG,attribute_names))])


print("*** LATEX CODE ***")
print("\\begin{tabular}{l r}")
print("\\hline")
print("Atribut & Povprečen rank\\\\")
print("\\hline")

slovar = {
 'numberOfDifferentWords': "Število različnih besed",
 'clarkEvansNearestNeighbour': "Clark Evans-ov najbližji sosed",
 'numberOfCharactersNoSpaces': "Število znakov (brez presledkov)",
 'avgDistanceCentroidCos': "Povprečna razdalja do centroida (cos)",
 'numberOfCharacters': "Število znakov",
 'avgDistanceCentroidEuc': "Povprečna razdalja do centroida (euc)",
 'minDistanceCentroidCos': "Minimalna razdalja do centroida (cos)",
 'standardDistance': "Standardna razdalja",
 'minDistanceCentroidEuc': "Minimalna razdalja do centroida (euc)",
 'numberOfWords': "Število besed",
 'pos_NN': "Število samostalnikov, ednina (pos\\_NN)",
 'maxDistanceCentroidCos': "Maksimalna razdalja do centroida (cos)",
 'guiraudsIndex': "Guiraud's Index",
 'numberOfSentences': "Število stavkov",
 'pos_IN': "Število predlogov (pos\\_IN)",
 'pos_JJ': "Število pridevnikov (pos\\_JJ)",
 'pos_DT': "Število determinerjev TODO (pos\\_DT)",
 'moransI': "Moran's I",
 'maxDistanceCentroidEuc': "Maksimalna razdalja do centroida (euc)",
 'cosineSumOfCorrelationValues': "Korelacijske vrednosti kosinusne podobnosti",
 'avgDistanceAnyPointsCos': "Povprečna razdalja med vsemi točkami (cos)",
 'numberOfDifferentPosTags': "Število različnih oblikoskladenjskih oznak",
 'advancedGuiraudIndex': "Advanced Guiraud's Index",
 'avgDistanceAnyPointsEuc': "Povprečna razdalja med vsemi točkami (euc)",
 'pos_RB': "Število prislovov (pos\\_RB)",
 'getissG': "Getiss' G",
 'gearysC': "Geary's C",
 'maxDistanceAnyPointsCos': "Maksimalna radzalja med vsemi točkami (cos)",
 'pos_VB': "Število glagolov v osnovni obliki (pos\\_VB)",
 'maxDistanceAnyPointsEuc': "Maksimalna razdalja med vsemi točkami (euc)",
 'pos_TO': "Število pojavitev \\textit{to} (pos\\_TO)",
 'pos_VBN': "Število glagolov deležnikov v pretekliku (pos\\_VBN)",
 'pos_CC': "Število veznikov (pos\\_CC)",
 'typeTokenRatio': "Type-token ratio",
 'pos_VBD': "Število glagolov v pretekliku (pos\\_VBD)",
 'pos_VBG': "Število glagolov deležnikov v sedanjiku (pos\\_VBG)",
 'cosinePattern': "Cosine pattern",
 'pos_PRP$': "Število svojilnih zaimkov (pos\\_PRP\\$)",
 'pos_VBZ': "Število glagolov v sedanjiku tretje osebe (pos\\_VBZ)",
 'indexDistanceCentroidEuc': "Indeks razdalje centroidov (min/max) (euc)",
 'hapaxLegomena': "Hapax Legomena",
 'pos_PRP': "Število osebnih zaimkov (pos\\_PRP)",
 'pos_MD': "Število modalnih glagolov",
 'cosineTopEssaySimilarityAverage': "Kosinusna podobnost z najboljšimi eseji",
 'indexDistanceNeighbouringPointsCos': "Indeks razdalje sosednih točk (min/max) (cos)",
 'numberOfStopwords': "Število stopbesed TODO",
 'pos_NNP': "Število lastnih imen v ednini (pos\\_NNP)",
 'numberOfSpellcheckingErrors': "Število pravopisnih napak",
 'indexDistanceCentroidCos': "Indeks razdalje do centroidov (min/max) (cos)",
 'maxDistanceNeighbouringPointsEuc': "Maksimalna razdalja med sosednjimi točkami (euc)",
 'cumulativeFrequencyDistribution': "Kumulativna frekvenčna porazdelitev TODO",
 'pos_WDT': "Število WH-determinerjev (pos\\_WDT) TODO",
 'pos_VBP': "Število glagolov v ednini in sedanjiku, ki niso v tretji osebi (pos\\_VBP)",
 'indexDistanceNeighbouringPointsEuc': "Indeks razdalje med sosednjimi točkami (min/max) (euc)",
 'relativeDistance': "Relative distance",
 'scoreCosineSimilarityMax': "Ocena eseja z največjo podobnostjo",
 'maxDistanceNeighbouringPointsCos': "Maksimalna razdalja med sosednjimi točkami (cos)",
 'numberOfLongWords': "Število dolgih besed",
 'pos_CD': 60.58888888888888, 
 'pos_RP': 61.455555555555556, 
 'pos_WRB': 61.666666666666664, 
 'consistencyErrors': "Število nezadovoljivih primerov (SAGE)",
 'theDEstimate': "D estimate",
 'pos_NNS': 64.51111111111112, 
 'numberOfPunctuationErrors': "Število napačne rabe ločil",
 'sumErrors': "Vsota skladnostnih napak (SAGE)",
 'pos_POS': 67.63333333333333, 
 'pos_JJR': 68.8, 
 'numberOfLongSentences': "Število dolgih povedi",
 'simpleMeasureOfGobbledygook': "Simple measure of Gobbledygook (SMOG)",
 'pos_JJS': 70.24444444444445, 
 'semanticErrors': "Število semantičnih/skladnostnih napak (SAGE)",
 'minDistanceNeighbouringPointsCos': "Minimalna razdalja med sosednjimi točkami (cos)",
 'pos_RBR': 72.46666666666667, 
 'averageSentenceLength': "Povprečna dolžina povedi",
 'gunningFogIndex': "Gunning Fog index",
 'pos_RBS': 75.16666666666666, 
 'wordVariationIndex': "Word Variation index",
 'mostFrequentSentenceLength': "Najbolj pogosta dolžina povedi",
 'minDistanceNeighbouringPointsEuc': "minimalna razdalja med sosednjimi točkami (euc)",
 'averageWordLength': "Povprečna dolžina besed",
 'numberOfCapitalizationErrors': "Število napak z veliko začetnico",
 'lix': "LIX",
 'automatedReadabilityIndex': "Automated Readability Index",
 'pos_PDT': 76.64444444444445, 
 'mostFrequentWordLength': "Najbolj pogosta dolžina besede",
 'fleschKincaidGradeLevel': "Flesch-Kincaid Grade Level",
 'fleschReadingEase': "Flesch Reading Ease",
 'pos_EX': 78.4111111111111, 
 'avgDistanceNearestNeighbour': "Povprečna razdalja med najbljižjimi sosedi",
 'daleChallReadabilityFormula': "Dale-Chall Readability Formula",
 'correctVerbForm': "Število glagolskih besed",
 'yulesK': "Yule's K",
 'pos_FW': 82.6888888888889, 
 'numberOfShortWords': "Število kratkih besed",
 'avgDistanceNeighbouringPointsEuc': "Povprečna razdalja med sosednjimi točkami (euc)",
 'nominalRatio': "Nominal Ratio",
 'avgDistanceNeighbouringPointsCos': "Povprečna razdalja med sosednjimi točkami (cos)",
 'sentenceStructureTreeHeight': "Povprečna višina stavčnega drevesa TODO",
 'numberOfShortSentences': "Število kratkih povedi",
 'pos_UH': 93.24444444444444, 
 'determinantDistanceMatrix': "Determinanta matriek razdalj",
 'pos_SYM': 94.94444444444444, 
 'pos_WP': 96.43333333333334, 
 'pos_NNPS': 99.53333333333335, 
 'pos_LS': 103.45555555555555
}

i = 1
for r,x in sorted(zip(AVG_RANG,attribute_names)):
    print(str(i) + ". " + str(slovar[x]) + " & " + str(np.round(r,2)).replace(".", ",") + "\\\\")
    i += 1
    if i > 50:
        break
print("\\hline")
print("\\end{tabular}")

