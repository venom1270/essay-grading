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

scores = np.array(table.Y)
ALL_ATTRIBUTES = np.array([x for x in table.X])
ALL_ATTRIBUTES = np.nan_to_num(ALL_ATTRIBUTES)
print(table)

print(ALL_ATTRIBUTES.shape)

scaler = preprocessing.MinMaxScaler()
ALL_ATTRIBUTES = scaler.fit_transform(ALL_ATTRIBUTES)

# ALL_ATTRIBUTES = preprocessing.normalize(ALL_ATTRIBUTES.transpose()).transpose()

print(ALL_ATTRIBUTES)
print(max(ALL_ATTRIBUTES[:, 5]))
print(min(ALL_ATTRIBUTES[:, 5]))

k = 10
chunk = int(len(ALL_ATTRIBUTES) / k)

USE_SKLEARN_RFE = False

if USE_SKLEARN_RFE:
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(ALL_ATTRIBUTES, scores)
    selector = RFE(rf, 50, step=1)
    selector.fit(ALL_ATTRIBUTES, scores)
    print(selector.support_)
    print(selector.ranking_)

    for i in range(len(selector.support_)):
        if selector.support_[i]:
            print(table.domain.attributes[i])

    exit()

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
            rf = RandomForestRegressionLearner(n_estimators=100, bootstrap=True, )#, min_samples_split=max(int(len(attributes[0])/3), 2)))

            # print("Training...")
            rf = rf.fit(train_attributes, train_scores[fold])

            # print("Predicting...")
            predictions = rf.predict(test_attributes)

            # print("Results...")
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
for i in FINAL:
    print(table.domain.attributes[i])


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