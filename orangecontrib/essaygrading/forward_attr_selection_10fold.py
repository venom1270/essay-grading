from orangecontrib.essaygrading.widgets.OWScore import quadratic_weighted_kappa
from orangecontrib.essaygrading.widgets.OWAttributeSelection import calculateAttributes
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

from Orange.data import Table, Domain, ContinuousVariable
from Orange.regression import RandomForestRegressionLearner
from Orange.regression import LinearRegressionLearner

# table = Table.from_file("datasets/FAS/set1_train_2.tab")
#table = Table.from_file("datasets/FAS/set1_train_2.tab")
table = Table.from_file("datasets/FAS/set8_flair.tab")


TABLES = []
for s in ["DS1.tab", "DS2A.tab", "DS2B.tab", "DS3.tab", "DS4.tab", "DS5.tab", "DS6.tab", "DS7.tab", "DS8.tab"]:
    TABLES.append(Table.from_file("datasets/FAS/All/" + s))

RESULTS = []

for i, table in enumerate(TABLES):

    print("************************+ PROCESSING TABLE " + str(i+1) + " / 9")

    scores = np.array(table.Y)
    ALL_ATTRIBUTES = np.array([x for x in table.X])
    ALL_ATTRIBUTES = np.nan_to_num(ALL_ATTRIBUTES)
    # print(table)

    # print(ALL_ATTRIBUTES.shape)

    scaler = preprocessing.MinMaxScaler()
    ALL_ATTRIBUTES = scaler.fit_transform(ALL_ATTRIBUTES)

    # ALL_ATTRIBUTES = preprocessing.normalize(ALL_ATTRIBUTES.transpose()).transpose()

    # print(ALL_ATTRIBUTES)
    # print(max(ALL_ATTRIBUTES[:, 5]))
    # print(min(ALL_ATTRIBUTES[:, 5]))

    k = 10
    chunk = int(len(ALL_ATTRIBUTES) / k)

    attributes = np.array([ALL_ATTRIBUTES[:, 0]]).transpose()
    kept_attributes = []
    avg_kappa = 0

    train = []
    train_scores = []
    test = []
    test_scores = []

    for j in range(10):
        start = j * chunk
        end = min((j + 1) * chunk, len(ALL_ATTRIBUTES))
        train.append(np.concatenate((ALL_ATTRIBUTES[:start], ALL_ATTRIBUTES[end:])))
        train_scores.append(np.concatenate((scores[:start], scores[end:])))
        test.append(ALL_ATTRIBUTES[start:end])
        test_scores.append(scores[start:end])

    FOLDS_ATTRIBUTES = []
    ALL_KAPPAS = []
    for fold in range(k):

        print("STARTING NEW FOLD: " + str(fold) + "/" + str(k))

        added_attribute = None
        best_score = -1
        best_i = -1
        curr_kappa = 0
        kept_attributes = []
        avg_kappa = 0

        for iter in range(len(ALL_ATTRIBUTES[0])):

            print("ITERATION " + str(iter) + " / " + str(len(ALL_ATTRIBUTES[0])))
            # print(ALL_ATTRIBUTES[:, i])

            kappas = []
            kappas_attr = []
            #attributes = np.concatenate((attributes, np.array([ALL_ATTRIBUTES[:, i]]).transpose()), axis=1)


            #attributes = np.concatenate((attributes, np.array([ALL_ATTRIBUTES[:, i]]).transpose()), axis=1)
            #attributes = np.take(ALL_ATTRIBUTES, taken_attributes, axis=1)

            for i in range(len(ALL_ATTRIBUTES[0])):

                if i in kept_attributes:
                    # print(str(i) + " in kept attributest... continuing.")
                    continue

                taken_attributes = kept_attributes + [i]

                train_attributes = np.take(train[fold], taken_attributes, axis=1)
                test_attributes = np.take(test[fold], taken_attributes, axis=1)

                #rf = LinearRegressionLearner()  # (n_estimators=100, )#, min_samples_split=max(int(len(attributes[0])/3), 2))
                rf = RandomForestRegressionLearner(n_estimators=50, bootstrap=True, )#, min_samples_split=max(int(len(taken_attributes)/3), 2)))

                # print("Training...")
                rf = rf.fit(train_attributes, train_scores[fold])

                # print("Predicting...")
                predictions = rf.predict(test_attributes)

                # print("Results...")
                #print(np.round(predictions))
                #print(test_scores[fold])
                predictions = [p if p >= 0 and p <= 500 else 0 for p in predictions]
                kappas.append(quadratic_weighted_kappa(np.round(predictions), test_scores[fold]))
                kappas_attr.append(i)

            # TODO? to bi lohka naredu sproti v loopu...
            print(kappas)
            curr_kappa = np.max(kappas)
            curr_i = kappas_attr[np.argmax(kappas)]
            print(str(curr_kappa) + " >= " + str(avg_kappa))

            if curr_kappa >= best_score:
                best_score = curr_kappa
                best_i = curr_i
            else:
                best_score = 0
                best_i = -1

            attributes = attributes[:, :-1]

            if best_score >= avg_kappa:
                print("Attribute " + str(best_i) + " had highest score: " + str(best_score) + " | avg: " + str(avg_kappa))
                avg_kappa = best_score
                attributes = np.concatenate((attributes, np.array([ALL_ATTRIBUTES[:, best_i]]).transpose()), axis=1)
                kept_attributes.append(best_i)

                print(kept_attributes)

            else:
                print("No attribute increased score; BREAK;")
                ALL_KAPPAS.append(avg_kappa)
                break

        FOLDS_ATTRIBUTES.append(kept_attributes)


    print("*********** END *************")
    print(FOLDS_ATTRIBUTES)

    print(ALL_KAPPAS)
    print(np.average(ALL_KAPPAS))

    FINAL = set()
    for fa in FOLDS_ATTRIBUTES:
        FINAL = FINAL.union(set(fa))
    print("KEPT ATTRIBUTES: ")
    print(FINAL)
    print(sorted(FINAL))
    print(len(FINAL))
    RESULTS.append((np.average(ALL_KAPPAS), FINAL))
    for i in FINAL:
        print(table.domain.attributes[i])


print("************** FINISHED")
print(RESULTS)













'''
LINEAR REGRESSION
[(0.8596840950569167, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 87, 88, 89, 91, 93, 96, 99, 100, 101, 102, 103}),
 (0.7555963146413822, {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 79, 80, 84, 85, 86, 87, 88, 89, 91, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103}),
 (0.7073959956886671, {0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 96, 99, 100, 101, 102, 103}),
 (0.7281969283940342, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 90, 95, 96, 97, 100, 101, 102, 103}),
 (0.7954052118180869, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103}),
 (0.8377992837920971, {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 103}),
 (0.7961840222617229, {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 68, 69, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103}),
 (0.8148159085860505, {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 79, 80, 81, 83, 86, 88, 89, 91, 94, 96, 97, 98, 99, 100, 101, 103}),
 (0.803011716266567, {0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 84, 85, 87, 88, 89, 90, 91, 93, 97, 99, 100, 102, 103})]


0.78 POVPREČEJE!!!!

Vzel je večino atributov



'''


'''
RANDOM FOREST n=50

[(0.8497930142964554, {1, 2, 65, 67, 69, 4, 66, 8, 6, 74, 12, 13, 78, 79, 77, 23, 87, 89, 26, 91, 93, 94, 96, 35, 36, 37, 44, 55, 56}), (0.735819299283176, {64, 1, 67, 71, 7, 8, 74, 11, 12, 78, 17, 18, 19, 84, 20, 85, 23, 88, 89, 90, 21, 92, 95, 33, 100, 36, 39, 41, 44, 51, 56, 60}), (0.6789460926062295, {1, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 25, 26, 30, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 53, 54, 56, 57, 58, 62, 68, 73, 74, 87, 101}), (0.7172442887172391, {64, 0, 2, 67, 69, 70, 71, 7, 73, 12, 14, 86, 28, 94, 96, 35, 37, 38, 103, 40, 41, 42, 44, 45, 47, 48, 50, 53, 55, 56, 57, 58}), (0.7854956335687262, {1, 2, 5, 6, 7, 8, 11, 12, 17, 22, 33, 34, 37, 39, 40, 42, 43, 44, 51, 52, 57, 60, 61, 62, 66, 71, 73, 74, 82, 86, 89, 91, 93, 94, 98, 103}), (0.8225442300908672, {0, 1, 2, 66, 4, 5, 70, 64, 8, 73, 74, 6, 72, 77, 13, 15, 9, 12, 84, 85, 87, 88, 32, 33, 36, 42, 48, 49, 57, 60, 61}), (0.7871889336659799, {0, 1, 2, 6, 13, 14, 19, 28, 29, 31, 37, 38, 39, 42, 43, 44, 46, 48, 50, 51, 53, 54, 57, 58, 60, 64, 71, 72, 73, 74, 80, 86, 87, 89, 90, 91, 93, 95, 96, 97, 99, 101, 102}), (0.8104137494318531, {1, 7, 9, 11, 12, 14, 18, 21, 29, 36, 43, 44, 48, 49, 51, 52, 54, 57, 58, 60, 61, 63, 71, 73, 74, 75, 82, 83, 86, 88, 89, 91, 93, 96, 100, 103}), (0.7573395581517949, {0, 2, 68, 5, 6, 69, 72, 9, 10, 7, 12, 77, 16, 84, 4, 20, 24, 25, 91, 27, 29, 31, 98, 36, 39, 41, 43, 44, 45, 48, 60, 63})]



'''


'''

RANDOM FOREST n=50 se enkrat

numberOfCharacters
mostFrequentWordLength
averageWordLength
numberOfSentences
numberOfLongSentences
numberOfDifferentWords
fleschKincaidGradeLevel
daleChallReadabilityFormula
automatedReadabilityIndex
simpleMeasureOfGobbledygook
wordVariationIndex
guiraudsIndex
theDEstimate
hapaxLegomena
advancedGuiraudIndex
sentenceStructureTreeHeight
pos_DT
pos_EX
pos_JJR
pos_JJS
pos_LS
pos_PDT
pos_RB
pos_TO
pos_VBD
pos_VBN
pos_VBP
pos_WP
numberOfSpellcheckingErrors
numberOfCapitalizationErrors
numberOfPunctuationErrors
scoreCosineSimilarityMax
maxDistanceNeighbouringPointsEuc
indexDistanceNeighbouringPointsEuc
avgDistanceNeighbouringPointsCos
maxDistanceAnyPointsEuc
cumulativeFrequencyDistribution
minDistanceCentroidEuc
avgDistanceCentroidCos
standardDistance
moransI
************** FINISHED
[(0.8517894248554809, {2, 3, 6, 10, 12, 13, 16, 22, 25, 27, 28, 36, 37, 41, 42, 46, 48, 54, 55, 56, 57, 59, 60, 63, 70, 72, 80, 83, 84, 85, 87, 89, 90, 95, 97, 102}), (0.7277171954815793, {1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 20, 26, 34, 39, 42, 44, 46, 48, 58, 65, 66, 67, 70, 73, 74, 77, 82, 84, 86, 88, 91, 92, 95, 96, 98, 102}), (0.6840265990703795, {2, 6, 7, 8, 11, 12, 13, 14, 15, 16, 20, 25, 30, 32, 33, 36, 39, 42, 44, 45, 47, 50, 53, 54, 56, 57, 58, 60, 66, 68, 69, 70, 73, 74, 77, 102}), (0.7147095941325453, {0, 65, 2, 66, 67, 69, 72, 73, 12, 13, 18, 82, 87, 90, 28, 93, 30, 97, 36, 38, 103, 41, 42, 44, 47, 48, 49, 53, 54, 56, 57, 58}), (0.7737205209679991, {0, 1, 2, 5, 7, 8, 9, 10, 12, 17, 20, 23, 32, 40, 41, 42, 44, 47, 50, 51, 52, 55, 56, 57, 61, 64, 65, 68, 72, 73, 74, 77, 81, 84, 86, 92, 94, 98, 100}), (0.8190052456039514, {0, 1, 2, 66, 5, 6, 72, 73, 74, 12, 77, 14, 13, 79, 19, 85, 87, 91, 28, 94, 95, 35, 36, 37, 102, 42, 48, 49, 50, 51, 55, 57, 58, 63}), (0.7898545628643276, {1, 2, 5, 14, 17, 18, 20, 25, 26, 28, 29, 31, 35, 38, 39, 42, 43, 47, 48, 51, 54, 60, 69, 70, 72, 73, 74, 77, 78, 82, 85, 87, 88, 90, 91, 94, 95, 100, 102}), (0.807342754094502, {64, 1, 0, 3, 68, 7, 71, 9, 74, 11, 12, 13, 14, 15, 80, 18, 83, 85, 22, 88, 27, 97, 33, 98, 36, 34, 103, 42, 43, 46, 47, 60, 61, 63}), (0.767315904584518, {0, 5, 6, 7, 9, 12, 16, 17, 18, 19, 21, 25, 27, 28, 29, 31, 35, 36, 40, 41, 42, 48, 52, 57, 60, 62, 63, 66, 68, 69, 70, 71, 77, 78, 79, 84, 89, 91, 94, 98, 101})]


'''





''''
SET 8_ 100

.7918502202643174 >= 0.7977689243027889
No attribute increased score; BREAK;
*********** END *************
[[0, 2, 72, 54, 9, 69, 53, 51], [25, 4, 91, 42, 56, 69, 53], [25, 11, 30, 45, 74], [12, 82, 33, 8, 68, 49], [0, 28, 69, 86, 60, 50], [38, 14, 19, 11, 37, 99], [12, 69, 33, 101, 64, 14, 42, 5, 24], [25, 21, 72, 103, 69, 81], [62, 28, 61, 103, 71, 69, 58], [12, 68, 65, 48, 36, 47]]
KEPT ATTRIBUTES: 
{0, 2, 4, 5, 8, 9, 11, 12, 14, 19, 21, 24, 25, 28, 30, 33, 36, 37, 38, 42, 45, 47, 48, 49, 50, 51, 53, 54, 56, 58, 60, 61, 62, 64, 65, 68, 69, 71, 72, 74, 81, 82, 86, 91, 99, 101, 103}
[0, 2, 4, 5, 8, 9, 11, 12, 14, 19, 21, 24, 25, 28, 30, 33, 36, 37, 38, 42, 45, 47, 48, 49, 50, 51, 53, 54, 56, 58, 60, 61, 62, 64, 65, 68, 69, 71, 72, 74, 81, 82, 86, 91, 99, 101, 103]
47
numberOfCharacters
numberOfWords
numberOfLongWords
mostFrequentWordLength
numberOfShortSentences
numberOfLongSentences
averageSentenceLength
numberOfDifferentWords
gunningFogIndex
simpleMeasureOfGobbledygook
wordVariationIndex
typeTokenRatio
guiraudsIndex
hapaxLegomena
numberOfDifferentPosTags
pos_CC
pos_EX
pos_FW
pos_IN
pos_LS
pos_NNS
pos_NNPS
pos_PDT
pos_POS
pos_PRP
pos_PRP$
pos_RBR
pos_RBS
pos_SYM
pos_UH
pos_VBD
pos_VBG
pos_VBN
pos_VBZ
pos_WDT
numberOfSpellcheckingErrors
numberOfCapitalizationErrors
scoreCosineSimilarityMax
cosineTopEssaySimilarityAverage
cosineSumOfCorrelationValues
maxDistanceNeighbouringPointsCos
indexDistanceNeighbouringPointsCos
maxDistanceAnyPointsCos
minDistanceCentroidEuc
relativeDistance
moransI
getissG


'''

'''
ISTO KOT ZGORAJ (SET8), SAM DA IZPIŠEM AVG
0.7801292226111182
KEPT ATTRIBUTES: 
{0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 19, 20, 22, 23, 24, 25, 27, 28, 33, 36, 37, 43, 44, 48, 49, 52, 61, 62, 63, 65, 66, 68, 69, 71, 72, 81, 82, 84, 91, 97, 98, 101}
[0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 19, 20, 22, 23, 24, 25, 27, 28, 33, 36, 37, 43, 44, 48, 49, 52, 61, 62, 63, 65, 66, 68, 69, 71, 72, 81, 82, 84, 91, 97, 98, 101]
46
numberOfCharacters
numberOfCharactersNoSpaces
numberOfLongWords
mostFrequentWordLength
averageWordLength
numberOfSentences
numberOfShortSentences
numberOfLongSentences
mostFrequentSentenceLength
averageSentenceLength
numberOfDifferentWords
gunningFogIndex
fleschReadingEase
fleschKincaidGradeLevel
simpleMeasureOfGobbledygook
lix
nominalRatio
lexicalDiversity
typeTokenRatio
guiraudsIndex
theDEstimate
hapaxLegomena
pos_CC
pos_EX
pos_FW
pos_MD
pos_NN
pos_PDT
pos_POS
pos_RB
pos_VBG
pos_VBN
pos_VBP
pos_WDT
pos_WP
numberOfSpellcheckingErrors
numberOfCapitalizationErrors
scoreCosineSimilarityMax
cosineTopEssaySimilarityAverage
maxDistanceNeighbouringPointsCos
indexDistanceNeighbouringPointsCos
maxDistanceAnyPointsEuc
minDistanceCentroidEuc
indexDistanceCentroidCos
standardDistance
moransI

Process finished with exit code 0


'''